#!/usr/bin/env python3
"""
GateKeeper Worker for monitoring staging directory and normalizing content.
Refactored for database-driven state machine lifecycle.
"""

import multiprocessing
import os
import random
import shutil
import signal
import time
from pathlib import Path

from config import settings
from services.job_service import STATUS_INGEST_FAILED, STATUS_NEW, STATUS_PREPROCESSING, STATUS_PREPROCESSING_COMPLETE, JobService, init_job_db
from utils.logging_config import setup_logging
from workers.gatekeeper_logic import (
    gatekeeper_extract_and_normalize,
    get_slug,
    log_gatekeeper_result,
)

log = setup_logging("gatekeeper.log", include_default_filters=True)

SHUTDOWN = multiprocessing.Event()


def signal_handler(sig, frame):
    log.warning(f"💥 Received signal {sig}, initiating shutdown...")
    SHUTDOWN.set()


def gatekeeper_worker_process():
    """
    Worker process loop: Polls DB for NEW jobs, claims them, and performs work.
    """
    # 0. STARTUP JITTER: Prevent 'Thundering Herd' on cold start
    time.sleep(random.random() * 2.0)

    while not SHUTDOWN.is_set():
        # 1. ATOMIC CLAIM
        job = JobService.claim_job(STATUS_NEW, STATUS_PREPROCESSING)
        if not job:
            # JITTERED POLLING: Prevent workers from syncing up their heartbeats
            time.sleep(5 + (random.random() * 2.0))
            continue

        job_id = job["id"]
        original_pdf_path = job["pdf_path"]
        filename = job["original_filename"]
        slug = get_slug(Path(filename).stem)

        log.info(f"👷 Worker (PID {os.getpid()}) claimed job {job_id} [{filename}]")

        try:
            # 2. MOVE TO PREPROCESSING isolations
            prep_pdf_path = os.path.join(settings.PREPROCESSING_DIR, filename)
            prep_md_path = os.path.join(settings.PREPROCESSING_DIR, f"{slug}.md")

            # Atomic move before logic
            if os.path.exists(original_pdf_path):
                shutil.move(original_pdf_path, prep_pdf_path)

            # Update DB with current location
            JobService.transition_job(job_id, STATUS_PREPROCESSING, new_pdf_path=prep_pdf_path, new_md_path=prep_md_path)

            # 3. PERFORM NORMALIZATION
            # We allow 3 attempts inside the logic or handled here
            success = False
            metadata = None
            for attempt in range(1, 4):
                log.info(f"🔄 Normalization attempt {attempt} for {slug}")
                success, metadata = gatekeeper_extract_and_normalize(job_id, prep_pdf_path, prep_md_path)
                if success:
                    break
                log.warning(f"⚠️ Attempt {attempt} failed for {slug}")

            if success:
                # 4. MOVE TO INGESTION (HANDOFF TO PRODUCER)
                ingest_pdf_path = os.path.join(settings.INGESTION_DIR, filename)
                ingest_md_path = os.path.join(settings.INGESTION_DIR, f"{slug}.md")

                shutil.move(prep_pdf_path, ingest_pdf_path)
                shutil.move(prep_md_path, ingest_md_path)

                # UPDATE TO PREPROCESSING_COMPLETE
                JobService.transition_job(job_id, STATUS_PREPROCESSING_COMPLETE, new_pdf_path=ingest_pdf_path, new_md_path=ingest_md_path)
                # Keep legacy log for compatibility
                log_gatekeeper_result(slug, "SUCCESS", metadata=metadata)
                log.info(f"✅ Finished Preprocessing: {slug}")
            else:
                raise Exception("Normalization failed after all attempts.")

        except Exception as e:
            log.error(f"💥 Fatal error in worker for {filename}: {e}")
            # MOVE TO FAILED
            failed_pdf_path = os.path.join(settings.FAILED_DIR, filename)
            failed_md_path = os.path.join(settings.FAILED_DIR, f"{slug}.md.partial")

            try:
                if os.path.exists(prep_pdf_path):
                    shutil.move(prep_pdf_path, failed_pdf_path)
                if os.path.exists(prep_md_path):
                    shutil.move(prep_md_path, failed_md_path)
            except Exception:
                pass

            JobService.transition_job(job_id, STATUS_INGEST_FAILED, new_pdf_path=failed_pdf_path, new_md_path=failed_md_path, error=str(e))
            log_gatekeeper_result(slug, "FAILURE", error_msg=str(e))


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Ensure lifecycle folders
    settings.ensure_folders()
    init_job_db()

    log.info(f"🚀 GateKeeper Controller started, monitoring {settings.STAGING_DIR}")

    # Launch worker pool
    num_workers = 2
    pool = multiprocessing.Pool(processes=num_workers)

    # We use apply_async to keep the main loop free for Discovery
    for _ in range(num_workers):
        pool.apply_async(gatekeeper_worker_process)

    while not SHUTDOWN.is_set():
        try:
            # DISCOVERY PHASE: Scan STAGING_DIR and create NEW records
            for fname in os.listdir(settings.STAGING_DIR):
                if SHUTDOWN.is_set():
                    break

                full_path = os.path.join(settings.STAGING_DIR, fname)
                if os.path.isfile(full_path) and not fname.startswith("."):
                    # Check if already in DB to avoid duplicates
                    # We can use a fast query here or rely on id generation
                    # create_job handles checking if it's new
                    JobService.create_job(full_path)

            time.sleep(20)

        except Exception as e:
            log.error(f"Unhandled error in gatekeeper controller: {e}")
            time.sleep(5)

    log.info("🛑 GateKeeper Controller shutting down...")
    pool.terminate()
    pool.join()
    log.info("🛑 GateKeeper Worker exiting cleanly.")


if __name__ == "__main__":
    main()
