#!/usr/bin/env python3
"""
Consumer Worker for processing chunks and storing them in Vector DB.
Uses a shared pre-compiled LangGraph singleton inherited via fork.
"""

import json
import multiprocessing
import os
import signal
import sys
import time
import traceback
from typing import Any, Callable, List

from config import settings
from services.job_service import JobService
from services.parquet_service import init_schema
from services.redis_service import get_redis_client
from utils.metrics import FileMetrics
from utils.trace_utils import get_logger, set_trace_id

log = get_logger("ingest_consumer")


def current_time() -> int:
    return int(time.time())


def consumer_worker(queue_name: str, shared_data: Any, parq_lock: Any) -> None:
    """Main consumer worker function using Zero-Memory staging."""
    from services.parquet_service import get_staged_chunks, stage_chunks
    from workers.consumer_graph import run_consumer_graph

    try:
        r = get_redis_client()
        timestamps = {}
        chunk_buffer = []
        processed_files = set()  # Dedup: track files already finalized

        log.info(f"🚀 Started shared-graph consumer for queue: {queue_name}_output (Staged Mode)")

        while not shared_data["shutdown_flag"]:
            res = r.blpop(f"{queue_name}_output", timeout=5)

            # TTL check for stale staged data
            now = current_time()
            for file_path, first_seen in list(timestamps.items()):
                if now - first_seen > settings.STAGED_CHUNK_TTL:
                    log.warning(f"⌛ TTL expired for {file_path}, cleaning up staged data")
                    with parq_lock:
                        get_staged_chunks(file_path, purge=True)
                    timestamps.pop(file_path, None)

            if res:
                _, job_raw = res
                data = json.loads(job_raw)
                source_file = data.get("source_file")
                trace_id = data.get("trace_id")

                if trace_id:
                    set_trace_id(trace_id)

                if data.get("type") == "file_end":
                    expected = data.get("expected_chunks", 0)
                    sentinel_job_id = data.get("job_id")

                    # Dedup: skip if already processed this file
                    if source_file in processed_files:
                        log.warning(f"⚠️ [{queue_name}] Duplicate file_end for {source_file}, skipping (already processed)")
                        continue
                    processed_files.add(source_file)

                    # 1. Flush any remaining chunks for this file from buffer
                    if chunk_buffer:
                        with parq_lock:
                            stage_chunks(chunk_buffer)
                        chunk_buffer = []

                    # 2. RETRIEVE FROM PERSISTENT STAGING
                    timestamps.pop(source_file, None)
                    log.info(f"📨 [{queue_name}] Received file_end for {source_file} (job_id={sentinel_job_id})")
                    metrics = FileMetrics(worker="consumer", file=source_file, queue=queue_name)

                    with parq_lock:
                        final_chunks = get_staged_chunks(source_file, purge=True)
                        log.info(f"📨 [{queue_name}] Retrieved {len(final_chunks)} chunks for {source_file}")
                        # Attempt to find trace_id in chunks if not in sentinel
                        f_trace_id = trace_id or (final_chunks[0].get("trace_id") if final_chunks else None)
                        graph_ok = run_consumer_graph(source_file, expected, final_chunks, metrics, trace_id=f_trace_id, job_id=sentinel_job_id)
                        if not graph_ok:
                            log.error(f"💥 Consumer graph failed for {source_file}")
                            try:
                                from services.job_service import STATUS_INGEST_FAILED
                                # Use sentinel job_id if available, otherwise look up by filename
                                jid = sentinel_job_id
                                pdf, md = None, None
                                if not jid:
                                    job_res, _ = JobService._execute_with_retry(
                                        "SELECT id, pdf_path, md_path FROM ingestion_lifecycle WHERE original_filename = ? ORDER BY finalized_at DESC LIMIT 1",
                                        (source_file,), fetch=True
                                    )
                                    if job_res:
                                        jid, pdf, md = job_res
                                else:
                                    path_res, _ = JobService._execute_with_retry(
                                        "SELECT pdf_path, md_path FROM ingestion_lifecycle WHERE id = ?",
                                        (jid,), fetch=True
                                    )
                                    if path_res:
                                        pdf, md = path_res
                                if jid:
                                    JobService.transition_job(jid, STATUS_INGEST_FAILED, error="Consumer graph failed")
                                    log.info(f"📮 Transitioned {source_file} to INGEST_FAILED after graph failure")
                            except Exception as inner_e:
                                log.error(f"Failed to record consumer failure for {source_file}: {inner_e}")
                    continue

                if source_file not in timestamps:
                    timestamps[source_file] = current_time()

                # 3. BUFFER CHUNKS
                chunk_buffer.append(data)
                if len(chunk_buffer) >= settings.CONSUMER_BATCH_SIZE:
                    with parq_lock:
                        stage_chunks(chunk_buffer)
                    chunk_buffer = []

    except Exception as e:
        log.error(f"💥 Critical consumer error: {e}")
        log.error(traceback.format_exc())
    finally:
        log.info(f"✅ Exiting worker for {queue_name}")


CHILD_PROCESSES = []


def make_sigint_handler(processes: List[multiprocessing.Process], ppid: int, shared_data: Any) -> Callable[[int, Any], None]:
    def handler(signum: int, frame: Any) -> None:
        if os.getpid() != ppid:
            return
        shared_data["shutdown_flag"] = True
        log.info(f"[Parent {os.getpid()}] SIGINT received. Sending to children...")
        for p in processes:
            p.join()
        sys.exit(0)

    return handler


def main() -> None:
    # 1. CORE PATH VALIDATION & CONFIRMATION
    log.info("🔍 [Startup Audit] Verifying System Configuration:")

    config_manifest = [
        ("DEFAULT_DOC_INGEST_ROOT", settings.DEFAULT_DOC_INGEST_ROOT),
        ("LLM_PATH", settings.LLM_PATH),
        ("SUPERVISOR_LLM_ENDPOINTS", settings.SUPERVISOR_LLM_ENDPOINTS),
        ("EMBEDDING_ENDPOINTS", settings.EMBEDDING_ENDPOINTS),
        ("WHISPER_MODEL_ENDPOINTS", settings.WHISPER_MODEL_ENDPOINTS),
        ("OCR_ENDPOINTS", settings.OCR_ENDPOINTS),
        ("VECTOR_DB_URL", settings.VECTOR_DB_URL),
        ("VECTOR_DB_USE_GRPC", settings.VECTOR_DB_USE_GRPC),
        ("VECTOR_DB_TIMEOUT", settings.VECTOR_DB_TIMEOUT),
        ("VECTOR_DB_BATCH_SIZE", settings.VECTOR_DB_BATCH_SIZE),
    ]
    for name, value in config_manifest:
        if value is not None and value != "NOT_SET":
            str_val = str(value).strip()
            
            # Determine Mode and Icon
            mode_label = ""
            icon = "✅"
            
            if name in ["LLM_PATH", "SUPERVISOR_LLM_ENDPOINTS", "EMBEDDING_ENDPOINTS", "WHISPER_MODEL_ENDPOINTS", "OCR_ENDPOINTS", "VECTOR_DB_URL"]:
                if str_val.startswith(("http://", "https://")):
                    mode_label = " [MODE: REMOTE]"
                    icon = "📡"
                elif name == "OCR_ENDPOINTS" and str_val == "LOCAL":
                    mode_label = " [MODE: LOCAL]"
                    icon = "🏠"
                elif os.path.exists(str_val):
                    mode_label = " [MODE: LOCAL]"
                    icon = "🏠"
                else:
                    icon = "❌"
            
            # Special hint for gRPC default
            if name == "VECTOR_DB_USE_GRPC" and os.getenv("VECTOR_DB_USE_GRPC") is None:
                mode_label = " (Default: gRPC)"

            log.info(f"   {icon} {name:25} : {str_val}{mode_label}")
        else:
            log.info(f"   ⚠️ {name:25} : NOT CONFIGURED (Optional)")

    parent_pid = os.getpid()
    init_schema()

    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass

    # 1. PRE-COMPILE CONSUMER GRAPH IN PARENT
    from workers.consumer_graph import get_consumer_app

    log.info("🏗️ Pre-compiling Consumer LangGraph in parent process...")
    get_consumer_app()

    # 2. Setup Shared Manager
    with multiprocessing.Manager() as manager:
        shared_dict = manager.dict({"shutdown_flag": False})
        parq_lock = manager.Lock()

        signal.signal(signal.SIGINT, make_sigint_handler(CHILD_PROCESSES, parent_pid, shared_dict))

        for queue_name in settings.QUEUE_NAMES:
            p = multiprocessing.Process(target=consumer_worker, args=(queue_name, shared_dict, parq_lock))
            p.start()
            CHILD_PROCESSES.append(p)

        for p in CHILD_PROCESSES:
            p.join()


if __name__ == "__main__":
    main()
