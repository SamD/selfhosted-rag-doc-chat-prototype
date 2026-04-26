#!/usr/bin/env python3
"""
Producer Worker for processing documents and enqueuing them into Redis.
Refactored for database-driven state machine lifecycle.
"""

import multiprocessing
import os
import random
import shutil
import signal
import time
from multiprocessing import Manager, Pool

from config import settings
from PIL import Image
from services.job_service import STATUS_CONSUMING, STATUS_INGEST_FAILED, STATUS_INGESTING, STATUS_PREPROCESSING_COMPLETE, JobService, init_job_db
from utils.logging_config import setup_logging

os.makedirs(settings.DEBUG_IMAGE_DIR, exist_ok=True)
Image.MAX_IMAGE_PIXELS = 500_000_000

# Shared State (Global in worker processes)
queue_lock = None
queue_index = None
gpu_lock = None


def get_next_queue():
    global queue_lock, queue_index
    with queue_lock:
        i = queue_index.value
        queue_index.value = (i + 1) % len(settings.QUEUE_NAMES)
        return settings.QUEUE_NAMES[i]


log = setup_logging("ingest_producer.log", include_default_filters=True)


def producer_worker_task(dummy_arg):
    """
    Worker process loop: Polls DB for PREPROCESSING_COMPLETE jobs, claims them, and performs work.
    """
    from workers.producer_graph import run_ingest_graph

    # 0. STARTUP JITTER: Prevent 'Thundering Herd' on cold start
    time.sleep(random.random() * 2.0)

    while not SHUTDOWN.is_set():
        # 1. ATOMIC CLAIM
        job = JobService.claim_job(STATUS_PREPROCESSING_COMPLETE, STATUS_INGESTING)
        if not job:
            # JITTERED POLLING: Prevent workers from syncing up their heartbeats
            time.sleep(5 + (random.random() * 2.0))
            continue

        job_id = job["id"]
        pdf_path = job["pdf_path"]
        md_path = job["md_path"]
        filename = job["original_filename"]

        log.info(f"👷 Producer (PID {os.getpid()}) claimed job {job_id} [{filename}]")

        try:
            # 2. MOVE TO CONSUMING ISOLATION
            # Both files move in tandem
            cons_pdf_path = os.path.join(settings.CONSUMING_DIR, filename)
            cons_md_path = os.path.join(settings.CONSUMING_DIR, os.path.basename(md_path))

            if os.path.exists(pdf_path):
                shutil.move(pdf_path, cons_pdf_path)
            if os.path.exists(md_path):
                shutil.move(md_path, cons_md_path)

            # Update DB with current location
            JobService.transition_job(job_id, STATUS_INGESTING, new_pdf_path=cons_pdf_path, new_md_path=cons_md_path)

            # 3. PERFORM CHUNKING & ENQUEUING (via Graph)
            # The graph needs (job_id, full_path, rel_path) - we pass the MD path as the primary work file
            job_tuple = (job_id, cons_md_path, filename)
            success = run_ingest_graph(job_tuple, gpu_lock_obj=gpu_lock)

            if success:
                # 4. TRANSITION TO CONSUMING
                # Chunks are in Redis, Consumer is picking them up
                JobService.transition_job(job_id, STATUS_CONSUMING)
                log.info(f"✅ Finished Ingesting (Producer phase): {filename}")
            else:
                raise Exception("Producer Graph returned failure.")

        except Exception as e:
            log.error(f"💥 Fatal error in producer worker for {filename}: {e}")
            # MOVE TO FAILED
            failed_pdf_path = os.path.join(settings.FAILED_DIR, filename)
            failed_md_path = os.path.join(settings.FAILED_DIR, os.path.basename(md_path))

            try:
                if os.path.exists(cons_pdf_path):
                    shutil.move(cons_pdf_path, failed_pdf_path)
                if os.path.exists(cons_md_path):
                    shutil.move(cons_md_path, failed_md_path)
            except Exception:
                pass

            JobService.transition_job(job_id, STATUS_INGEST_FAILED, new_pdf_path=failed_pdf_path, new_md_path=failed_md_path, error=str(e))


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

    # Pre-compile Graph
    from workers.producer_graph import get_producer_app

    get_producer_app()
    init_job_db()

    with Manager() as manager:
        global_gpu_lock = manager.Lock()
        q_lock = manager.Lock()
        q_idx = manager.Value("i", 0)

        num_workers = min(4, os.cpu_count() or 1)
        log.info(f"🚀 Starting Producer Pool with {num_workers} workers (DB-driven)")

        with Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(q_lock, q_idx, global_gpu_lock),
            maxtasksperchild=5,
        ) as pool:
            # Dispatch workers to their loops
            for _ in range(num_workers):
                pool.apply_async(producer_worker_task, (None,))

            # Keep main loop alive for monitoring
            while not SHUTDOWN.is_set():
                time.sleep(scan_interval)

    log.info("🛑 Producer Controller exiting cleanly.")


if __name__ == "__main__":
    main()
