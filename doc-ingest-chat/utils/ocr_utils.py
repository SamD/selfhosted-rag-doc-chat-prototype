#!/usr/bin/env python3
"""
Utility functions for OCR processing.
Uses Docling (EasyOCR backend) for robust extraction.
"""

import base64
import json
import logging
import os
import time
import traceback
import uuid
from typing import Optional, Tuple

import cv2
import numpy as np
import redis
from config.settings import MAX_OCR_DIM, REDIS_HOST, REDIS_OCR_JOB_QUEUE, REDIS_PORT
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from PIL import Image

log = logging.getLogger("ingest.ocr_utils")

# Global Docling Converter (Lazy initialized)
_DOCLING_CONVERTER = None
_REDIS_CLIENT_CACHE = None


def get_redis_client():
    """Lazy initializer for the Redis client to ensure fork safety."""
    global _REDIS_CLIENT_CACHE
    if _REDIS_CLIENT_CACHE is None:
        _REDIS_CLIENT_CACHE = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _REDIS_CLIENT_CACHE


def preprocess_image(pil_image):
    """Preprocess image for OCR."""
    if pil_image is None:
        log.error("💥 preprocess_image received None")
        return None
    w, h = pil_image.size
    if max(w, h) > MAX_OCR_DIM:
        scale = MAX_OCR_DIM / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    np_image = np.array(pil_image)
    if len(np_image.shape) == 3:
        if np_image.shape[2] == 4:  # RGBA
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2GRAY)
        else:  # RGB
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    return np_image


def send_image_to_ocr(np_image, rel_path, page_num):
    """Send image to OCR service and wait for response."""
    start_time = time.perf_counter()
    job_id = str(uuid.uuid4())
    reply_key = f"ocr_reply:{job_id}"
    job = {
        "job_id": job_id,
        "rel_path": rel_path,
        "page_num": page_num,
        "image_shape": np_image.shape,
        "image_dtype": str(np_image.dtype),
        "image_base64": base64.b64encode(np_image.tobytes()).decode(),
        "reply_key": reply_key,
    }
    redis_client = get_redis_client()
    redis_client.lpush(REDIS_OCR_JOB_QUEUE, json.dumps(job))
    result = redis_client.blpop(reply_key, timeout=300)
    if not result:
        raise TimeoutError(f"OCR timeout for {rel_path} page {page_num}")
    ocr_roundtrip_ms = (time.perf_counter() - start_time) * 1000.0
    _, data = result
    result = json.loads(data)
    return (
        result.get("text"),
        result.get("rel_path"),
        result.get("page_num"),
        result.get("engine"),
        result.get("job_id"),
        ocr_roundtrip_ms,
    )


def get_docling_converter():
    """
    Lazy initializer for Docling DocumentConverter with EasyOCR backend.
    """
    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        try:
            log.info("🚀 Initializing Docling Converter (EasyOCR backend)...")

            # OCR Options (EasyOCR)
            ocr_options = EasyOcrOptions(
                lang=["en"],
                use_gpu=os.getenv("DEVICE", "cpu").lower() == "cuda",
            )

            # Use PdfPipelineOptions for both to ensure compatibility with StandardPdfPipeline
            # which Docling often uses even for image-based inputs.
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = ocr_options
            # Disable extra extractions for raw text speed
            pipeline_options.do_table_structure = False

            _DOCLING_CONVERTER = DocumentConverter(
                allowed_formats=[InputFormat.IMAGE, InputFormat.PDF],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                    InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
                },
            )
        except Exception as e:
            log.error(f"💥 Failed to initialize Docling: {e}")
            log.error(traceback.format_exc())
            return None
    return _DOCLING_CONVERTER


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


def run_ocr(np_image, rel_path, page_num) -> Tuple[Optional[str], str, float]:
    """
    Executes OCR using Docling (EasyOCR backend) on a NumPy image array.
    """
    log.info(f"🔄 Running Docling (EasyOCR) for {rel_path} page {page_num}")

    converter = get_docling_converter()
    if not converter:
        return None, "error_docling_not_init", 0.0

    import tempfile

    start_time = time.perf_counter()
    try:
        # Docling works best with file paths. We save the NumPy array to a temp file.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_image = Image.fromarray(np_image)
            pil_image.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = converter.convert(tmp_path)
            # Result is a converted document. We export to raw text.
            full_text = result.document.export_to_text().strip()

            execution_time_ms = (time.perf_counter() - start_time) * 1000.0

            if not full_text:
                return None, "notext_docling", execution_time_ms

            return full_text, "docling_easyocr", execution_time_ms
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        log.error(f"💥 Docling execution failed: {e}")
        log.error(traceback.format_exc())
        return None, "error_docling", 0.0
