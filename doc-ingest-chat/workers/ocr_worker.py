#!/usr/bin/env python3
"""
OCR Worker for processing images and extracting text.
Uses a shared pre-compiled LangGraph singleton to avoid redundant compilation.
"""

import json
import multiprocessing
import os
import signal
import traceback
from multiprocessing import Manager, Pool

from config.settings import REDIS_OCR_JOB_QUEUE
from PIL import Image
from services.redis_service import get_redis_client
from utils.logging_config import setup_logging, setup_pdf_logging

log = setup_logging("ingest_ocr_worker.log", include_default_filters=True)


def worker_task(job):
    """
    Process a single OCR job. The LangGraph app is already inherited
    from the parent process via fork.
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
    """Initialize worker with shared lock."""
    global queue_lock
    queue_lock = lock


SHUTDOWN = multiprocessing.Event()


def signal_handler(sig, frame):
    log.warning(f"💥 Received signal {sig}, initiating OCR shutdown...")
    SHUTDOWN.set()


def dispatcher(p, shared_state):
    """Main loop that pops jobs and dispatches to pool."""
    redis_client = get_redis_client()
    log.info(f"🛰️ OCR Dispatcher listening on {REDIS_OCR_JOB_QUEUE}...")

    while not SHUTDOWN.is_set():
        try:
            # Use timeout to check SHUTDOWN event frequently
            res = redis_client.brpop(REDIS_OCR_JOB_QUEUE, timeout=5)
            if res:
                _, job_raw = res
                job = json.loads(job_raw)
                p.apply_async(worker_task, args=(job,))
        except json.JSONDecodeError:
            log.error(f"💥 Malformed Job: {job_raw}")
        except Exception as e:
            log.error(f"💥 Dispatcher error: {e}")


def main():
    """Main OCR worker entry point with pre-compilation."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    setup_pdf_logging()
    Image.MAX_IMAGE_PIXELS = 500_000_000

    # 1. PRE-COMPILE THE GRAPH IN PARENT
    from workers.ocr_graph import get_ocr_app

    log.info("🏗️ Pre-compiling OCR LangGraph in parent process...")
    get_ocr_app()

    # 2. Setup Shared Manager
    with Manager() as manager:
        shared_state = manager.dict()
        lock = manager.Lock()

        try:
            num_workers = min(2, os.cpu_count() or 1)
            log.info(f"🚀 Spawning {num_workers} OCR workers (maxtasksperchild=1)")

            with Pool(processes=num_workers, initializer=init_worker, initargs=(lock,), maxtasksperchild=1) as pool:
                dispatcher(pool, shared_state)
        except Exception as e:
            log.error(f"💥 OCR Worker encountered error: {e}")
        finally:
            log.info("✅ OCR worker pool terminated cleanly")


if __name__ == "__main__":
    # Ensure we use fork for memory inheritance
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()
