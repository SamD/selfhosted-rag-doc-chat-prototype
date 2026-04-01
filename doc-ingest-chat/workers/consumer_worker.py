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
from collections import defaultdict
from typing import Any, Callable, List

from config.settings import CHUNK_TIMEOUT, MAX_CHROMA_BATCH_SIZE, MAX_CHUNKS, QUEUE_NAMES
from services.parquet_service import append_chunks, init_schema
from services.redis_service import get_redis_client
from utils.consumer_utils import store_chunks_in_db
from utils.logging_config import setup_logging
from utils.metrics import FileMetrics

log = setup_logging("ingest_consumer.log", include_default_filters=True)


def current_time() -> int:
    return int(time.time())


def consumer_worker(queue_name: str, shared_data: Any, parq_lock: Any) -> None:
    """Main consumer worker function."""
    from workers.consumer_graph import run_consumer_graph

    try:
        r = get_redis_client()
        buffer = defaultdict(list)
        timestamps = {}

        log.info(f"🚀 Started shared-graph consumer for queue: {queue_name}")

        while not shared_data["shutdown_flag"]:
            # Use a small timeout to allow frequent shutdown_flag checks
            res = r.blpop(queue_name, timeout=5)

            # Periodically clean up stale buffers (TTL check)
            now = current_time()
            for file_path, first_seen in list(timestamps.items()):
                if now - first_seen > CHUNK_TIMEOUT:
                    log.warning(f"⌛ TTL expired for {file_path}, discarding buffer")
                    buffer.pop(file_path, None)
                    timestamps.pop(file_path, None)

            if res:
                _, job_raw = res
                data = json.loads(job_raw)
                source_file = data.get("source_file")

                if data.get("type") == "file_end":
                    expected = data.get("expected_chunks", 0)
                    final_chunks = buffer.pop(source_file, [])
                    timestamps.pop(source_file, None)

                    log.info(f"📨 [{queue_name}] Received file_end for {source_file} (finalizing)")
                    metrics = FileMetrics(worker="consumer", file=source_file, queue=queue_name)
                    with parq_lock:
                        run_consumer_graph(source_file, expected, final_chunks, metrics)
                    continue

                if source_file not in timestamps:
                    timestamps[source_file] = current_time()

                buffer[source_file].append(data)

                if len(buffer[source_file]) >= MAX_CHROMA_BATCH_SIZE:
                    try:
                        active_batch = buffer[source_file]
                        with parq_lock:
                            append_chunks(active_batch)
                            store_chunks_in_db(source_file, active_batch)
                        buffer[source_file] = []
                    except Exception as e:
                        log.error(f"💥 Incremental ingestion failed for {source_file}: {e}")

                if len(buffer[source_file]) >= MAX_CHUNKS:
                    log.warning(f"🛘 Max chunks exceeded for {source_file} - discarding")
                    buffer.pop(source_file, None)
                    timestamps.pop(source_file, None)
                    continue

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
