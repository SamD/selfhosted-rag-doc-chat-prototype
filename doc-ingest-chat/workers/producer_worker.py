#!/usr/bin/env python3
"""
Producer Worker for processing documents and enqueuing them into Redis.
This version uses LangGraph for state-machine based orchestration of the ingestion flow.
"""

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

# Shared State (across processes)
queue_names = QUEUE_NAMES
queue_lock = None
queue_index = None
gpu_lock = None # Global lock for Supervisor LLM access

def get_next_queue():
    """
    Returns the next queue in a round-robin fashion.
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
    Target for the multiprocessing Pool.
    """
    from workers.producer_graph import run_ingest_graph
    # We pass the global gpu_lock to the graph runner
    return run_ingest_graph(job_tuple, gpu_lock_obj=gpu_lock)

def init_worker(q_lock_obj, q_idx_obj, g_lock_obj):
    """
    Initializes shared state for each worker process.
    """
    global queue_lock, queue_index, gpu_lock
    queue_lock = q_lock_obj
    queue_index = q_idx_obj
    gpu_lock = g_lock_obj

# Shared primitives
lock = Lock()
index = Value("i", 0)
global_gpu_lock = Lock() # NEW: Created in parent, shared with all children
SHUTDOWN = multiprocessing.Event()

def signal_handler(sig, frame):
    log.warning(f"💥 Received signal {sig}, initiating shutdown...")
    SHUTDOWN.set()

def run_tree_watcher(scan_interval=30):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass

    init_job_db()

    while not SHUTDOWN.is_set():
        try:
            jobs = []
            job_id_base = str(uuid.uuid4())[:8]
            counter = 0

            for root, _, files in os.walk(INGEST_FOLDER):
                for fname in files:
                    if fname.lower().endswith(ALL_SUPPORTED_EXT):
                        full_path = os.path.join(root, fname)
                        full_path = full_path.decode() if isinstance(full_path, bytes) else full_path
                        _ingest_folder = INGEST_FOLDER.decode() if isinstance(INGEST_FOLDER, bytes) else INGEST_FOLDER
                        rel_path = normalize_rel_path(os.path.relpath(full_path, _ingest_folder))
                        
                        if is_file_processed(rel_path):
                            continue
                            
                        jobs.append((f"{job_id_base}-{counter}", full_path, rel_path))
                        counter += 1

            if jobs:
                log.info(f"📦 Found {len(jobs)} new file(s) to ingest")
                success = 0

                pool = None
                try:
                    pool = multiprocessing.Pool(
                        processes=min(4, os.cpu_count()),
                        initializer=init_worker,
                        initargs=(lock, index, global_gpu_lock), # PASS LOCK HERE
                        maxtasksperchild=1,
                    )

                    for result in pool.imap_unordered(run_ingest, jobs, chunksize=1):
                        if SHUTDOWN.is_set():
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
