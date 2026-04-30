#!/usr/bin/env python3
"""
GateKeeper Worker for monitoring staging directory and normalizing content.
Refactored for database-driven state machine lifecycle and multi-type content handlers.
"""

import multiprocessing
import os
import random
import shutil
import signal
import time
from pathlib import Path

from config import settings
from services.job_service import (
    STATUS_INGEST_FAILED,
    STATUS_NEW,
    STATUS_PREPROCESSING,
    STATUS_PREPROCESSING_COMPLETE,
    JobService,
    init_job_db,
)
from utils.trace_utils import get_logger, set_trace_id
from workers.gatekeeper_logic import (
    gatekeeper_extract_and_normalize,
    get_slug,
    log_gatekeeper_result,
)

log = get_logger("gatekeeper")

SHUTDOWN = multiprocessing.Event()


def signal_handler(sig, frame):
    log.warning(f"💥 Received signal {sig}, initiating shutdown...")
    SHUTDOWN.set()


def gatekeeper_process_job(job: dict) -> bool:
    """Processes a single claimed job from the lifecycle database."""
    job_id = job["id"]
    # 'pdf_path' in DB currently stores the path to any original file (PDF, MP4, etc.)
    original_file_path = job["pdf_path"]
    filename = job["original_filename"]
    trace_id = job.get("trace_id", "UNKNOWN")
    set_trace_id(trace_id)

    slug = get_slug(Path(filename).stem)

    log.info(f"👷 Worker (PID {os.getpid()}) claimed job {job_id} [{filename}]")

    try:
        # 2. MOVE TO PREPROCESSING isolations
        prep_file_path = os.path.join(settings.PREPROCESSING_DIR, filename)
        prep_md_path = os.path.join(settings.PREPROCESSING_DIR, f"{slug}.md")

        # Atomic move before logic
        if os.path.exists(original_file_path):
            shutil.move(original_file_path, prep_file_path)

        # Update DB with current location
        JobService.transition_job(job_id, STATUS_PREPROCESSING, new_pdf_path=prep_file_path, new_md_path=prep_md_path)

        # 3. PERFORM NORMALIZATION
        success = False
        metadata = None
        for attempt in range(1, 4):
            log.info(f"🔄 Normalization attempt {attempt} for {slug}")
            success, metadata = gatekeeper_extract_and_normalize(job_id, prep_file_path, prep_md_path)
            if success:
                break
            log.warning(f"⚠️ Attempt {attempt} failed for {slug}")

        if success:
            # 4. MOVE TO INGESTION (HANDOFF TO PRODUCER)
            ingest_file_path = os.path.join(settings.INGESTION_DIR, filename)
            ingest_md_path = os.path.join(settings.INGESTION_DIR, f"{slug}.md")

            shutil.move(prep_file_path, ingest_file_path)
            shutil.move(prep_md_path, ingest_md_path)

            # UPDATE TO PREPROCESSING_COMPLETE
            JobService.transition_job(job_id, STATUS_PREPROCESSING_COMPLETE, new_pdf_path=ingest_file_path, new_md_path=ingest_md_path)
            # Keep legacy log for compatibility
            log_gatekeeper_result(slug, "SUCCESS", metadata=metadata)
            log.info(f"✅ Finished Preprocessing: {slug}")
            return True
        else:
            raise Exception("Normalization failed after all attempts.")

    except Exception as e:
        from utils.exceptions import ConfigurationError

        log.error(f"💥 Fatal error in worker for {filename}: {e}")

        # Identify specific configuration failure (like missing Whisper path)
        error_reason = f"Configuration Missing: {str(e)}" if isinstance(e, ConfigurationError) else str(e)
        if isinstance(e, ConfigurationError):
            log.warning(f"🛑 Skipping {filename} and recording failure due to missing environment config.")

        # MOVE TO FAILED
        failed_file_path = os.path.join(settings.FAILED_DIR, filename)
        failed_md_path = os.path.join(settings.FAILED_DIR, f"{slug}.md.partial")

        try:
            if os.path.exists(prep_file_path):
                shutil.move(prep_file_path, failed_file_path)
            if os.path.exists(prep_md_path):
                shutil.move(prep_md_path, failed_md_path)
        except Exception:
            pass

        # RECORD SPECIFIC REASON IN DUCKDB
        JobService.transition_job(job_id, STATUS_INGEST_FAILED, new_pdf_path=failed_file_path, new_md_path=failed_md_path, error=error_reason)
        log_gatekeeper_result(slug, "FAILURE", error_msg=error_reason)
        return False


def gatekeeper_worker_process():
    """
    Worker process loop: Polls DB for NEW jobs, claims them, and performs work.
    Supports PDF, MP4, and other media types via modular handlers.
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

        gatekeeper_process_job(job)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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

    # Ensure lifecycle folders
    settings.ensure_folders()
    init_job_db()

    # DEEP ASSET AUDIT: Whisper
    if settings.WHISPER_MODEL_PATH != "NOT_SET":
        if not os.path.exists(settings.WHISPER_MODEL_PATH):
            log.warning(f"⚠️  WHISPER_MODEL_PATH is set but the directory does not exist: {settings.WHISPER_MODEL_PATH}")
        else:
            missing = []
            for req_file in settings.WHISPER_REQUIRED_FILES:
                if not os.path.exists(os.path.join(settings.WHISPER_MODEL_PATH, req_file)):
                    missing.append(req_file)
            if missing:
                log.warning(f"⚠️  WHISPER_MODEL_PATH is INCOMPLETE. Missing: {', '.join(missing)}")
                log.warning("⚠️  Media files (.mp4/.mp3) will fail until these assets are provided.")

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
                    # create_job handles checking if it's new based on filename
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
