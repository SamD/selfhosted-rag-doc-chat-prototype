#!/usr/bin/env python3
"""
OCR Worker for processing images and extracting text.
"""
import base64
import json
import os
import signal
import traceback
from multiprocessing import Lock, Pool

import numpy as np
import pytesseract
from config.settings import (
    DEBUG_IMAGE_DIR,
    REDIS_OCR_JOB_QUEUE,
    TESSDATA_PREFIX,
    TESSERACT_LANGS,
    TESSERACT_OEM,
    TESSERACT_PSM,
    TESSERACT_USE_SCRIPT_LATIN,
)
from PIL import Image
from services.redis_service import get_redis_client
from utils.logging_config import setup_logging, setup_pdf_logging
from utils.text_utils import is_invalid_text
from utils.metrics import JobMetrics

log = setup_logging("ingest_ocr_worker.log", include_default_filters=True)

def safe_image_save(pil_image, path, format=None):
    """
    Save a PIL image to disk, creating directories as needed and logging the result.

    Args:
        pil_image (PIL.Image.Image): The image to save.
        path (str): Path to save the image (should include extension).
        format (str, optional): Image format (e.g., 'PNG', 'JPEG').

    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        pil_image.save(path, format=format)
        log.info(f"✅ Saved image: {path}")
        return True

    except Exception as e:
        log.info(f"💥 Failed to save image to {path}: {e}")
        return False


def save_bad_image(np_image, debug_image_path, log_prefix):
    """Save bad image for debugging."""
    log.info(f"{log_prefix} ⚠️  OCR returned empty/short text")
    log.info(f"np_image.shape = {np_image.shape}, dtype = {np_image.dtype}")
    log.info(f"---- Saving invalid page for debugging: {debug_image_path}")
    # Convert np_image back to PIL
    pil_image = Image.fromarray(np_image)
    safe_image_save(pil_image, debug_image_path)


def fallback_to_tesseract(np_image, rel_path, page_num):
    """Fallback to Tesseract OCR."""
    import time

    log.info(f"↪️ Falling back to Tesseract for {rel_path} page {page_num}")
    env = os.environ.copy()
    if TESSDATA_PREFIX:
        env["TESSDATA_PREFIX"] = TESSDATA_PREFIX

    config_parts = [f"--psm {TESSERACT_PSM}", f"--oem {TESSERACT_OEM}"]
    if TESSERACT_USE_SCRIPT_LATIN:
        config_parts.append("-c tessedit_script=Latin")
    config = " ".join(config_parts)

    start_time = time.perf_counter()
    try:
        text = pytesseract.image_to_string(
            np_image,
            lang=TESSERACT_LANGS,
            config=config,
            env=env,
        ).strip()
    except TypeError:
        # Some pytesseract versions do not accept env kwarg
        text = pytesseract.image_to_string(
            np_image,
            lang=TESSERACT_LANGS,
            config=config,
        ).strip()

    tesseract_time_ms = (time.perf_counter() - start_time) * 1000.0

    if is_invalid_text(text):
        log.info(" ⚠️  Tesseract returned empty/short text")
        return None, "notext_tesseract", tesseract_time_ms

    return text.strip(), "tesseract", tesseract_time_ms


def ocr_image_with_fallback(np_image, rel_path, page_num, use_gpu):
    """Process image with OCR fallback."""
    doc_id = os.path.basename(rel_path).replace('/', '_').replace('\\', '_')
    debug_image_path = os.path.join(DEBUG_IMAGE_DIR, f"{doc_id}_page_{page_num}.png")
    log_prefix = f"[Doc {rel_path}] Page {page_num}]"

    try:
        text, engine, tesseract_time_ms = fallback_to_tesseract(np_image, rel_path, page_num)
        if text:
            log.info(f"{log_prefix} ✅ Tesseract fallback success with {use_gpu} ({len(text.strip())} chars)")
            return text, engine, tesseract_time_ms
        # else save bad or empty image
        log.info(f"⚠️ Tesseract fallback failed as well for {rel_path}, page {page_num} - [{use_gpu}]")
        save_bad_image(np_image, debug_image_path, log_prefix)
        return None, "notext_" + engine, tesseract_time_ms

    finally:
        signal.alarm(0)
        del np_image
        if 'text' in locals():
            del text


def worker_task(job):
    """Process OCR job."""
    import time

    # Initialize metrics collection
    job_id = job.get("job_id", "unknown")
    metrics = JobMetrics(worker="ocr", job_id=job_id)

    try:
        with metrics.timer("total_processing"):
            reply_key = job["reply_key"]
            rel_path = job["rel_path"]
            page_num = job["page_num"]

            shape = tuple(job["image_shape"])
            dtype = job["image_dtype"]

            # Track image decode time
            with metrics.timer("image_decode"):
                np_image = np.frombuffer(base64.b64decode(job["image_base64"]), dtype=dtype).reshape(shape)

            log.info(f"📥 Job: {rel_path}, page {page_num}, job_id {job_id}")

            use_gpu = True
            log.info(f"🚀 Using {'GPU' if use_gpu else 'CPU'} for {rel_path}, page {page_num}")
            text, engine, tesseract_time_ms = ocr_image_with_fallback(np_image, rel_path, page_num, use_gpu=use_gpu)

            # Add metrics fields
            metrics.add_field("tesseract_execution_time_ms", tesseract_time_ms)
            metrics.add_field("engine", engine)
            metrics.add_field("text_length", len(text) if text else 0)
            metrics.add_field("file", rel_path)
            metrics.add_field("page", page_num)
            metrics.add_field("success", not (engine and engine.startswith("notext")))

            response = {
                "text": text,
                "rel_path": rel_path,
                "page_num": page_num,
                "engine": engine,
                "job_id": job_id,
            }

    except Exception:
        log.info("❌ Unhandled OCR failure")
        log.info(traceback.format_exc())

        job_id = "unknown"
        rel_path = "unknown"
        page_num = -1
        reply_key = None

        if 'job' in locals():
            job_id = job.get("job_id", "unknown")
            rel_path = job.get("rel_path", "unknown")
            page_num = job.get("page_num", -1)
            reply_key = job.get("reply_key")

        response = {
            "text": "",
            "rel_path": rel_path,
            "page_num": page_num,
            "engine": "error",
            "job_id": job_id
        }

        # ❌ DO NOT lpush here
        if not reply_key:
            log.info("❌ No reply_key available — the producer will hang indefinitely!")

    finally:
        # Emit metrics
        metrics.emit(log)

        if reply_key:
            redis_client = get_redis_client()
            redis_client.lpush(reply_key, json.dumps(response))
            redis_client.expire(reply_key, 300)
            log.info(f"📤 Response sent to {reply_key}")


def init_worker(lock):
    """Initialize worker with lock."""
    global queue_lock
    queue_lock = lock


def dispatcher(p):
    """Dispatch jobs to worker pool."""
    redis_client = get_redis_client()
    while True:
        _, job_raw = redis_client.brpop(REDIS_OCR_JOB_QUEUE)
        try:
            job = json.loads(job_raw)
        except json.JSONDecodeError:
            log.info("ERROR: Malformed Job:", job_raw)
            continue
        p.apply_async(worker_task, args=(job,))


def main():
    """Main OCR worker function."""
    # Set up logging
    setup_pdf_logging()
    
    # Set PIL image max pixels
    Image.MAX_IMAGE_PIXELS = 500_000_000
    
    lock = Lock()
    try:
        num_workers = min(2, os.cpu_count() or 1)
        log.info(f"🚀 Spawning {num_workers} OCR worker processes (via Pool)")
        pool = Pool(processes=num_workers, initializer=init_worker, initargs=(lock,), maxtasksperchild=1)
        dispatcher(pool)
    except KeyboardInterrupt:
        log.info("💥 CTRL+C received, shutting down OCR pool")
        pool.terminate()
        pool.join()
        log.info("✅ OCR worker pool terminated cleanly")


if __name__ == "__main__":
    main() 