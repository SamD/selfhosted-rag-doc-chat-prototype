#!/usr/bin/env python3
"""
Producer Worker for processing documents and enqueuing them into Redis.
Uses a shared pre-compiled LangGraph singleton inherited via fork.
"""

import multiprocessing
import os
import signal
import time
import uuid
from multiprocessing import Manager, Pool

from config.settings import (
    ALL_SUPPORTED_EXT,
    DEBUG_IMAGE_DIR,
    INGEST_FOLDER,
    QUEUE_NAMES,
)
from PIL import Image
from services.job_service import init_job_db, is_file_processed
from utils.file_utils import normalize_rel_path
from utils.logging_config import setup_logging

os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)
Image.MAX_IMAGE_PIXELS = 500_000_000

# Shared State (Global in worker processes)
queue_lock = None
queue_index = None
gpu_lock = None


def get_next_queue():
    global queue_lock, queue_index
    with queue_lock:
        i = queue_index.value
        queue_index.value = (i + 1) % len(QUEUE_NAMES)
        return QUEUE_NAMES[i]


# Moved Redis instantiation to worker tasks to avoid fork-safety issues
log = setup_logging("ingest_producer.log", include_default_filters=True)


def run_ingest(job_tuple):
    from workers.producer_graph import run_ingest_graph

    # We pass the global gpu_lock to the graph runner
    return run_ingest_graph(job_tuple, gpu_lock_obj=gpu_lock)


def init_worker(q_lock_obj, q_idx_obj, g_lock_obj):
    global queue_lock, queue_index, gpu_lock
    queue_lock = q_lock_obj
    queue_index = q_idx_obj
    gpu_lock = g_lock_obj


SHUTDOWN = multiprocessing.Event()


def signal_handler(sig, frame):
    log.warning(f"💥 Received signal {sig}, initiating shutdown...")
    SHUTDOWN.set()


def main(scan_interval=30):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass

    # 1. PRE-COMPILE THE PRODUCER GRAPH IN PARENT
    from workers.producer_graph import get_producer_app

    log.info("🏗️ Pre-compiling Producer LangGraph in parent process...")
    get_producer_app()

    init_job_db()

    # 2. Setup Manager for shared locks and indices
    with Manager() as manager:
        global_gpu_lock = manager.Lock()
        q_lock = manager.Lock()
        q_idx = manager.Value("i", 0)

        # Persistent Pool to avoid constant re-spawning
        num_workers = min(4, os.cpu_count() or 1)
        log.info(f"🚀 Starting persistent Producer Pool with {num_workers} workers")

        with Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(q_lock, q_idx, global_gpu_lock),
            maxtasksperchild=10,  # Increased for better efficiency (re-loads models less often)
        ) as pool:
            while not SHUTDOWN.is_set():
                try:
                    jobs = []
                    job_id_base = str(uuid.uuid4())[:8]
                    counter = 0

                    for root, _, files in os.walk(INGEST_FOLDER):
                        for fname in files:
                            if fname.lower().endswith(ALL_SUPPORTED_EXT):
                                full_path = os.path.join(root, fname)
                                _ingest_folder = INGEST_FOLDER.decode() if isinstance(INGEST_FOLDER, bytes) else INGEST_FOLDER
                                rel_path = normalize_rel_path(os.path.relpath(full_path, _ingest_folder))

                                if is_file_processed(rel_path):
                                    continue

                                jobs.append((f"{job_id_base}-{counter}", full_path, rel_path))
                                counter += 1

                    if jobs:
                        log.info(f"📦 Found {len(jobs)} new file(s) to ingest")
                        success = 0

                        for result in pool.imap_unordered(run_ingest, jobs, chunksize=1):
                            if SHUTDOWN.is_set():
                                break
                            if result:
                                success += 1
                        log.info(f"✅ Ingested {success}/{len(jobs)} file(s) this cycle")
                    else:
                        log.info("🔍 No new files found this cycle")

                    if not SHUTDOWN.is_set():
                        time.sleep(scan_interval)

                except Exception as e:
                    log.error(f"Unhandled error in producer loop: {e}")
                    time.sleep(5)

    log.info("🛑 Tree walker exiting cleanly.")


if __name__ == "__main__":
    main()
