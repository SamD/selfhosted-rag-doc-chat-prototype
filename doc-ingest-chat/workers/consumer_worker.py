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
from config.settings import CHUNK_TIMEOUT, QUEUE_NAMES
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

        log.info(f"🚀 Started shared-graph consumer for queue: {queue_name} (Staged Mode)")

        while not shared_data["shutdown_flag"]:
            res = r.blpop(queue_name, timeout=5)

            # TTL check for stale staged data
            now = current_time()
            for file_path, first_seen in list(timestamps.items()):
                if now - first_seen > CHUNK_TIMEOUT:
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

                    # 1. Flush any remaining chunks for this file from buffer
                    if chunk_buffer:
                        with parq_lock:
                            stage_chunks(chunk_buffer)
                        chunk_buffer = []

                    # 2. RETRIEVE FROM PERSISTENT STAGING
                    timestamps.pop(source_file, None)
                    log.info(f"📨 [{queue_name}] Received file_end for {source_file}")
                    metrics = FileMetrics(worker="consumer", file=source_file, queue=queue_name)

                    with parq_lock:
                        final_chunks = get_staged_chunks(source_file, purge=True)
                        log.info(f"📨 [{queue_name}] Retrieved {len(final_chunks)} chunks for {source_file}")
                        # Attempt to find trace_id in chunks if not in sentinel
                        f_trace_id = trace_id or (final_chunks[0].get("trace_id") if final_chunks else None)
                        run_consumer_graph(source_file, expected, final_chunks, metrics, trace_id=f_trace_id)
                    continue

                if source_file not in timestamps:
                    timestamps[source_file] = current_time()

                # 3. BUFFER CHUNKS
                chunk_buffer.append(data)
                if len(chunk_buffer) >= 50:
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
        ("SUPERVISOR_LLM_PATH", settings.SUPERVISOR_LLM_PATH),
        ("EMBEDDING_MODEL_PATH", settings.EMBEDDING_MODEL_PATH),
        ("WHISPER_MODEL_PATH", settings.WHISPER_MODEL_PATH),
    ]
    for name, value in config_manifest:
        if value and value != "NOT_SET":
            status = "✅" if os.path.exists(value) or value.startswith("http") else "❌"
            log.info(f"   {status} {name:25} : {value}")
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

        for queue_name in QUEUE_NAMES:
            p = multiprocessing.Process(target=consumer_worker, args=(queue_name, shared_dict, parq_lock))
            p.start()
            CHILD_PROCESSES.append(p)

        for p in CHILD_PROCESSES:
            p.join()


if __name__ == "__main__":
    main()
