#!/usr/bin/env python3
"""
GateKeeper Worker for monitoring staging directory and normalizing content.
"""

import os
import shutil
import signal
import time
from pathlib import Path

from config import settings
from utils.logging_config import setup_logging
from workers.gatekeeper_logic import (
    gatekeeper_extract_and_normalize,
    generate_slug,
    log_gatekeeper_result,
)

log = setup_logging("gatekeeper.log", include_default_filters=True)

SHUTDOWN = False


def signal_handler(sig, frame):
    global SHUTDOWN
    log.warning(f"💥 Received signal {sig}, initiating shutdown...")
    SHUTDOWN = True


def gatekeeper_process_file(file_path):
    p = Path(file_path)
    slug = generate_slug(file_path)

    try:
        # 1. Media Detection & Tiered Extraction
        ext = p.suffix.lower()

        # 2. Tiered Processing
        if ext in [".mp4", ".mkv", ".mov", ".mp3", ".wav", ".flac", ".m4a"]:
            # TODO: Implement batching for media
            log.warning(f"Media batching not yet implemented: {file_path}")
            log_gatekeeper_result(slug, "SKIPPED", error_msg="Media batching not yet implemented")
            return False
        elif ext in [".html", ".htm"]:
            # TODO: Implement batching for web
            log.warning(f"Web batching not yet implemented: {file_path}")
            log_gatekeeper_result(slug, "SKIPPED", error_msg="Web batching not yet implemented")
            return False
        elif ext == ".pdf":
            # 3. The 3-Attempt Normalization Loop (for the whole document)
            for attempt in range(1, 4):
                try:
                    log.info(f"🔄 Full document normalization attempt {attempt} for {slug}")
                    success, metadata = gatekeeper_extract_and_normalize(file_path)
                    if success:
                        # Success: Log and move processed file
                        log_gatekeeper_result(slug, "SUCCESS", metadata=metadata)
                        processed_dir = Path(settings.STAGING_FOLDER) / "processed"
                        processed_dir.mkdir(exist_ok=True)
                        shutil.move(file_path, processed_dir / p.name)
                        return True
                    else:
                        raise Exception("Normalization returned False")

                except Exception as e:
                    log.error(f"❌ Attempt {attempt} failed for {slug}: {e}")
                    if attempt == 3:
                        # Move to 'failed' folder
                        log_gatekeeper_result(slug, "FAILURE", error_msg=str(e))
                        failed_dir = Path(settings.STAGING_FOLDER) / "failed"
                        failed_dir.mkdir(exist_ok=True)
                        shutil.move(file_path, failed_dir / p.name)
                        return False
                    continue
        else:
            log.warning(f"Unsupported file type: {file_path}")
            log_gatekeeper_result(slug, "SKIPPED", error_msg=f"Unsupported file type: {ext}")
            return False

    except Exception as fatal_error:
        log.error(f"💥 Fatal error processing {file_path}: {fatal_error}")
        log_gatekeeper_result(slug, "ERROR", error_msg=str(fatal_error))
        # Move to 'failed' folder
        failed_dir = Path(settings.STAGING_FOLDER) / "failed"
        failed_dir.mkdir(exist_ok=True)
        shutil.move(file_path, failed_dir / p.name)
        return False


def main(scan_interval=10):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    staging_folder = settings.STAGING_FOLDER
    os.makedirs(staging_folder, exist_ok=True)

    log.info(f"🚀 GateKeeper Worker started, monitoring {staging_folder}")

    while not SHUTDOWN:
        try:
            files_to_process = []
            for root, dirs, files in os.walk(staging_folder):
                if "processed" in dirs:
                    dirs.remove("processed")
                if "failed" in dirs:
                    dirs.remove("failed")

                for fname in files:
                    full_path = os.path.join(root, fname)
                    files_to_process.append(full_path)

            if files_to_process:
                log.info(f"📦 Found {len(files_to_process)} file(s) in staging")
                for file_path in files_to_process:
                    if SHUTDOWN:
                        break
                    gatekeeper_process_file(file_path)
            else:
                log.debug("🔍 No new files in staging")

            if not SHUTDOWN:
                time.sleep(scan_interval)

        except Exception as e:
            log.error(f"Unhandled error in gatekeeper loop: {e}")
            time.sleep(5)

    log.info("🛑 GateKeeper Worker exiting cleanly.")


if __name__ == "__main__":
    main()
