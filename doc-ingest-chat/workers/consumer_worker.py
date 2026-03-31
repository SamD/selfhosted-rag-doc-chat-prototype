#!/usr/bin/env python3
"""
Consumer Worker for processing chunks and storing them in Vector DB.
Refactored for ZERO-MEMORY archival via DuckDB.
"""

import json
import multiprocessing
import os
import signal
import sys
import time
import traceback
from collections import defaultdict
from itertools import cycle
from typing import Any, Callable, List

from config.settings import CHUNK_TIMEOUT, MAX_CHROMA_BATCH_SIZE, MAX_CHUNKS, QUEUE_NAMES
from services.parquet_service import append_chunks, init_schema
from services.redis_service import get_redis_client
from utils.consumer_utils import store_chunks_in_db
from utils.logging_config import setup_logging
from utils.metrics import FileMetrics

queue_lock = multiprocessing.Lock()
queue_cycle = cycle(QUEUE_NAMES)
# parquet_lock not needed for DuckDB writes as we have retry logic
parquet_lock = multiprocessing.Lock()

log = setup_logging("ingest_consumer.log", include_default_filters=True)

def get_next_queue() -> str:
    global queue_lock, queue_cycle
    with queue_lock:
        return next(queue_cycle)

def current_time() -> int:
    return int(time.time())

def consumer_worker(queue_name: str, shared_data: Any, parq_lock: Any) -> None:
    """Main consumer worker function."""
    from workers.consumer_graph import run_consumer_graph
    
    try:
        r = get_redis_client()
        # buffer: Only holds the CURRENT active batch
        buffer = defaultdict(list)
        timestamps = {}

        log.info(f"🚀 Started zero-memory consumer for queue: {queue_name}")

        while True:
            now = current_time()
            # Buffer TTL cleanup
            for file, first_seen in list(timestamps.items()):
                if now - first_seen > CHUNK_TIMEOUT:
                    log.info(f"⌛ TTL expired for {file}, discarding buffer")
                    buffer.pop(file, None)
                    timestamps.pop(file, None)
                    # update_failed_files(file)
                    pass

            item = r.blpop(queue_name, timeout=5)
            
            if shared_data["shutdown_flag"]:
                log.info("\n👋 SHUTDOWN_FLAG set exiting ...")
                break

            if not item:
                continue

            try:
                data = json.loads(item[1])
            except Exception as e:
                log.info(f"⚠️ [{queue_name}] Skipping malformed Redis entry: {e}")
                continue

            source_file = data.get("source_file")
            if not source_file:
                continue

            if data.get("type") == "file_end":
                expected = data["expected_chunks"]
                log.info(f"📨 [{queue_name}] Received file_end for {source_file} (finalizing remaining chunks)")
                
                # Retrieve remaining chunks from memory
                remaining_chunks = buffer.pop(source_file, [])
                timestamps.pop(source_file, None)
                
                metrics = FileMetrics(worker="consumer", file=source_file, queue=queue_name)
                # Finalize: Process last chunks + Export Parquet + Update DuckDB Status
                run_consumer_graph(source_file, expected, remaining_chunks, metrics)

            else:
                if source_file not in timestamps:
                    timestamps[source_file] = current_time()
                
                buffer[source_file].append(data)

                # Incremental write when batch size is reached
                if len(buffer[source_file]) >= MAX_CHROMA_BATCH_SIZE:
                    log.info(f"📦 [{queue_name}] Batch threshold reached for {source_file} ({MAX_CHROMA_BATCH_SIZE} chunks). Ingesting...")
                    try:
                        active_batch = buffer[source_file]
                        
                        # 1. Persist to DuckDB (Disk bound) - SAFETY FIRST
                        # This ensures we have the data captured even if Qdrant/VRAM fails
                        append_chunks(active_batch)
                        
                        # 2. Ingest to Qdrant (CPU/GPU bound)
                        store_chunks_in_db(source_file, active_batch)
                        
                        # 3. CRITICAL: Clear memory IMMEDIATELY
                        buffer[source_file] = []
                        
                    except Exception as e:
                        log.error(f"💥 Incremental ingestion failed for {source_file}: {e}")
                
                # Global safety limit
                if len(buffer[source_file]) >= MAX_CHUNKS:
                    log.warning(f"🛘 Max chunks exceeded for {source_file} - discarding")
                    buffer.pop(source_file, None)
                    timestamps.pop(source_file, None)
                    # update_failed_files(source_file)
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

    with multiprocessing.Manager() as manager:
        shared_dict = manager.dict({"shutdown_flag": False})
        signal.signal(signal.SIGINT, make_sigint_handler(CHILD_PROCESSES, parent_pid, shared_dict))

        for i in range(len(QUEUE_NAMES)):
            next_queue = get_next_queue()
            p = multiprocessing.Process(target=consumer_worker, args=(next_queue, shared_dict, parquet_lock))
            p.start()
            CHILD_PROCESSES.append(p)
            
        log.info(f"🚀 Started {len(QUEUE_NAMES)} consumer workers for queues: {QUEUE_NAMES}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sys.exit(0)

if __name__ == "__main__":
    main()
