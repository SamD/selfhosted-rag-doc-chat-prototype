#!/usr/bin/env python3
"""
Utility functions for OCR processing.
"""

import logging
import os
import time

import pytesseract
from config.settings import (
    TESSDATA_PREFIX,
    TESSERACT_LANGS,
    TESSERACT_OEM,
    TESSERACT_PSM,
    TESSERACT_USE_SCRIPT_LATIN,
)
from PIL import Image

log = logging.getLogger("ingest.ocr_utils")


def safe_image_save(pil_image, path, format=None):
    """Save a PIL image to disk safely."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pil_image.save(path, format=format)
        log.info(f"✅ Saved image: {path}")
        return True
    except Exception as e:
        log.info(f"💥 Failed to save image to {path}: {e}")
        return False


def save_bad_image(np_image, debug_image_path, log_prefix):
    """Save problematic image for debugging OCR failures."""
    log.info(f"{log_prefix} ⚠️ Saving invalid page for debugging: {debug_image_path}")
    pil_image = Image.fromarray(np_image)
    safe_image_save(pil_image, debug_image_path)


def run_tesseract(np_image, rel_path, page_num):
    """
    Executes Tesseract OCR on a NumPy image array.
    Configures Tesseract based on project settings (PSM, OEM, Scripts).
    """
    log.info(f"↪️ Running Tesseract for {rel_path} page {page_num}")

    env = os.environ.copy()
    if TESSDATA_PREFIX:
        env["TESSDATA_PREFIX"] = TESSDATA_PREFIX

    config_parts = [f"--psm {TESSERACT_PSM}", f"--oem {TESSERACT_OEM}"]
    if TESSERACT_USE_SCRIPT_LATIN:
        config_parts.append("-c tessedit_script=Latin")
    config = " ".join(config_parts)

    start_time = time.perf_counter()
    try:
        # Some versions of pytesseract do not support the env kwarg
        try:
            text = pytesseract.image_to_string(
                np_image,
                lang=TESSERACT_LANGS,
                config=config,
                env=env,
            ).strip()
        except TypeError:
            text = pytesseract.image_to_string(
                np_image,
                lang=TESSERACT_LANGS,
                config=config,
            ).strip()

        execution_time_ms = (time.perf_counter() - start_time) * 1000.0

        if not text or not text.strip():
            return None, "notext_tesseract", execution_time_ms

        return text, "tesseract", execution_time_ms

    except Exception as e:
        log.error(f"💥 Tesseract execution failed: {e}")
        return None, "error_tesseract", 0.0
