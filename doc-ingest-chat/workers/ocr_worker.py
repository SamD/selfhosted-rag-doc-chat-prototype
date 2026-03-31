#!/usr/bin/env python3
"""
OCR Worker for processing images and extracting text.
Refactored to use LangGraph for processing orchestration.
"""

import json
import os
import traceback
from multiprocessing import Lock, Pool

from config.settings import REDIS_OCR_JOB_QUEUE
from PIL import Image
from services.redis_service import get_redis_client
from utils.logging_config import setup_logging, setup_pdf_logging

log = setup_logging("ingest_ocr_worker.log", include_default_filters=True)

def worker_task(job):
    """
    Process a single OCR job by invoking the LangGraph.
    """
    from workers.ocr_graph import run_ocr_graph
    
    try:
        success = run_ocr_graph(job)
        if not success:
            log.warning(f"⚠️ OCR Graph reported failure for {job.get('rel_path')}")
    except Exception as e:
        log.error(f"💥 Critical failure in worker_task: {e}")
        log.error(traceback.format_exc())

def init_worker(lock):
    """Initialize worker with lock if needed (shared between processes)."""
    global queue_lock
    queue_lock = lock

def dispatcher(p):
    """
    Main loop that pops jobs from Redis and dispatches them to the worker pool.
    """
    redis_client = get_redis_client()
    log.info(f"🛰️ OCR Dispatcher listening on {REDIS_OCR_JOB_QUEUE}...")
    
    while True:
        try:
            _, job_raw = redis_client.brpop(REDIS_OCR_JOB_QUEUE)
            job = json.loads(job_raw)
            # Dispatch to process pool asynchronously
            p.apply_async(worker_task, args=(job,))
        except json.JSONDecodeError:
            log.error(f"💥 Malformed Job: {job_raw}")
        except Exception as e:
            log.error(f"💥 Dispatcher error: {e}")

def main():
    """Main OCR worker entry point."""
    setup_pdf_logging()

    # Set PIL image max pixels to handle large documents
    Image.MAX_IMAGE_PIXELS = 500_000_000

    lock = Lock()
    try:
        # Scale workers based on CPU count (keeping 1-2 processes for GPU/CPU heavy OCR)
        num_workers = min(2, os.cpu_count() or 1)
        log.info(f"🚀 Spawning {num_workers} OCR worker processes")
        
        pool = Pool(processes=num_workers, initializer=init_worker, initargs=(lock,), maxtasksperchild=1)
        dispatcher(pool)
    except KeyboardInterrupt:
        log.info("💥 CTRL+C received, shutting down OCR pool")
        pool.terminate()
        pool.join()
        log.info("✅ OCR worker pool terminated cleanly")

if __name__ == "__main__":
    main()
