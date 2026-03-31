#!/usr/bin/env python3
"""
Producer Worker for processing documents and enqueuing them into Redis.
This version uses LangGraph for state-machine based orchestration of the ingestion flow.
"""

import gc
import multiprocessing
import os
import signal
import time
import uuid
from multiprocessing import Lock, Value

import redis

# Import configuration
from config.settings import (
    ALL_SUPPORTED_EXT,
    DEBUG_IMAGE_DIR,
    INGEST_FOLDER,
    MAX_CHROMA_BATCH_SIZE_LIMIT,
    QUEUE_NAMES,
    REDIS_HOST,
    REDIS_PORT,
)
from PIL import Image

# Ingest State Management
from services.job_service import init_job_db, is_file_processed
from utils.file_utils import normalize_rel_path
from utils.logging_config import setup_logging

os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)

Image.MAX_IMAGE_PIXELS = 500_000_000
MAX_CHROMA_BATCH_SIZE = MAX_CHROMA_BATCH_SIZE_LIMIT

# Queue Rotation State (Shared across processes)
queue_names = QUEUE_NAMES
queue_lock = None
queue_index = None

def get_next_queue():
    """
    Returns the next queue in a round-robin fashion.
    Thread-safe and Process-safe via global locks initialized in init_worker.
    """
    global queue_lock, queue_index
    with queue_lock:
        i = queue_index.value
        queue_index.value = (i + 1) % len(queue_names)
        return queue_names[i]

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
log = setup_logging("ingest_producer.log", include_default_filters=True)

def run_ingest(job_tuple):
    """
    The target function for the multiprocessing Pool.
    Delegates the entire ingestion lifecycle to the LangGraph StateMachine.
    """
    # Import here to avoid circular dependencies during process serialization
    from workers.producer_graph import run_ingest_graph
    return run_ingest_graph(job_tuple)

def init_worker(lock_obj, index_obj):
    """
    Initializes shared state for each worker process in the Pool.
    Ensures that the queue rotation lock and index are accessible.
    """
    global queue_lock, queue_index
    queue_lock = lock_obj
    queue_index = index_obj

# Shared primitives for the multiprocessing pool
lock = Lock()
index = Value("i", 0)
SHUTDOWN = multiprocessing.Event()

def signal_handler(sig, frame):
    """Graceful shutdown handler for OS signals."""
    log.warning(f"💥 Received signal {sig}, initiating shutdown...")
    SHUTDOWN.set()

def run_tree_watcher(scan_interval=30):
    """
    Main loop that watches the INGEST_FOLDER for new files.
    - Scans directory recursively.
    - Checks DuckDB to skip already processed files.
    - Dispatches new work to the multiprocessing pool.
    """
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Use 'fork' for better compatibility with shared CUDA contexts (if applicable)
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass

    # Ensure the relational state tracking table exists
    init_job_db()

    while not SHUTDOWN.is_set():
        try:
            jobs = []
            job_id_base = str(uuid.uuid4())[:8] # Correlation ID prefix for the batch
            counter = 0

            # Scan the ingestion directory
            for root, _, files in os.walk(INGEST_FOLDER):
                for fname in files:
                    if fname.lower().endswith(ALL_SUPPORTED_EXT):
                        full_path = os.path.join(root, fname)
                        full_path = full_path.decode() if isinstance(full_path, bytes) else full_path
                        _ingest_folder = INGEST_FOLDER.decode() if isinstance(INGEST_FOLDER, bytes) else INGEST_FOLDER
                        rel_path = normalize_rel_path(os.path.relpath(full_path, _ingest_folder))
                        
                        # PERSISTENCE CHECK: Skip files already marked 'completed' in DuckDB
                        if is_file_processed(rel_path):
                            continue
                            
                        jobs.append((f"{job_id_base}-{counter}", full_path, rel_path))
                        counter += 1

            if jobs:
                log.info(f"📦 Found {len(jobs)} new file(s) to ingest")
                success = 0

                pool = None
                try:
                    # Spawn a pool of workers to process files in parallel
                    pool = multiprocessing.Pool(
                        processes=min(4, os.cpu_count()),
                        initializer=init_worker,
                        initargs=(lock, index),
                        maxtasksperchild=1, # Refresh process after each job to clear memory/GPU cache
                    )

                    # Execute the LangGraph workflow for each file
                    for result in pool.imap_unordered(run_ingest, jobs, chunksize=1):
                        if SHUTDOWN.is_set():
                            log.warning("⛔ Shutdown requested mid-processing")
                            break
                        if result:
                            success += 1

                except Exception as e:
                    log.error(f"Error during multiprocessing: {e}")
                finally:
                    if pool:
                        pool.terminate()
                        pool.join()
                        log.info(f"✅ Ingested {success}/{len(jobs)} file(s) this cycle")

            else:
                log.info("🔍 No new files found this cycle")

            if not SHUTDOWN.is_set():
                time.sleep(scan_interval)

        except Exception as e:
            log.error(f"Unhandled error in producer loop: {e}")
            time.sleep(5)

    log.info("🛑 Tree walker exiting cleanly.")

def main():
    run_tree_watcher()

if __name__ == "__main__":
    main()
