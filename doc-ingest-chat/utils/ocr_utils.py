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
from config.settings import MAX_OCR_DIM, OCR_ENDPOINTS, REDIS_HOST, REDIS_OCR_JOB_QUEUE, REDIS_PORT
from PIL import Image
from utils.trace_utils import get_logger, set_trace_id

log = get_logger("ingest.ocr_utils")

# Global Docling Converter (Lazy initialized)
_DOCLING_CONVERTER = None
_REDIS_CLIENT_CACHE = None


def get_redis_client():
    """Lazy initializer for the Redis client to ensure fork safety."""
    global _REDIS_CLIENT_CACHE
    if _REDIS_CLIENT_CACHE is None:
        _REDIS_CLIENT_CACHE = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
            socket_connect_timeout=5, socket_timeout=None,
        )
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


def send_image_to_ocr(np_image, rel_path, page_num, trace_id: str = None):
    """Send image to OCR service and wait for response. Always uses Redis queue."""
    if trace_id:
        set_trace_id(trace_id)

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
        "trace_id": trace_id,
    }
    try:
        redis_client = get_redis_client()
        redis_client.lpush(REDIS_OCR_JOB_QUEUE, json.dumps(job))
    except Exception as e:
        log.error(f"❌ Failed to submit OCR job to Redis: {e}")
        raise RuntimeError(f"Redis submission failed: {e}")

    # HEARTBEAT POLLING: Instead of one long block, we poll and log status
    result = None
    wait_timeout = 300
    start_wait = time.time()

    while (time.time() - start_wait) < wait_timeout:
        # Check for response in 30-second increments
        result = redis_client.blpop(reply_key, timeout=30)
        if result:
            break

        elapsed = int(time.time() - start_wait)
        log.info(f"⏳ Waiting for OCR... {rel_path} P{page_num} ({elapsed}s elapsed)")

    if not result:
        raise TimeoutError(f"OCR timeout after {wait_timeout}s for {rel_path} page {page_num}")

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
    Enforces strict offline mode for air-gapped environments.
    """
    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                AcceleratorDevice,
                AcceleratorOptions,
                EasyOcrOptions,
                PdfPipelineOptions,
            )
            from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

            # 1. ENFORCE AIR-GAP: Kill internet dependency for Docling/HuggingFace
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            # USE THE BAKED-IN CACHE: Do not point to /tmp
            CACHE_PATH = "/usr/local/model_cache"
            os.environ["HF_HOME"] = CACHE_PATH
            os.environ["TORCH_HOME"] = CACHE_PATH
            os.environ["XDG_CACHE_HOME"] = CACHE_PATH
            os.environ["EASYOCR_MODULE_PATH"] = CACHE_PATH

            # Silence internal library noise
            logging.getLogger("docling").setLevel(logging.WARNING)
            logging.getLogger("easyocr").setLevel(logging.WARNING)
            logging.getLogger("docling_parse").setLevel(logging.WARNING)

            log.info(f"🚀 Initializing Docling Converter (STRICT OFFLINE MODE) cache: {CACHE_PATH}")

            # Modern Docling 2.x Acceleration Setup
            device_type = os.getenv("DEVICE", "cpu").lower()
            accel_device = AcceleratorDevice.CUDA if device_type == "cuda" else AcceleratorDevice.CPU
            log.info(f"⚡ Setting Docling acceleration to: {accel_device}")

            # OCR Options (EasyOCR)
            # download_enabled=False ensures it doesn't try to fetch languages from net
            ocr_options = EasyOcrOptions(lang=["en"], download_enabled=False)

            # Use PdfPipelineOptions for both to ensure compatibility with StandardPdfPipeline
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = ocr_options
            # Disable extra extractions for raw text speed
            pipeline_options.do_table_structure = False

            # 2. LOCAL ARTIFACTS: Pointing to HF_HOME for local loading
            # This implicitly disables model downloads in Docling 2.x
            pipeline_options.artifacts_path = os.getenv("HF_HOME", "/usr/local/model_cache")

            # Apply modern acceleration settings
            pipeline_options.accelerator_options = AcceleratorOptions(
                device=accel_device
            )

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


def run_remote_ocr(np_image, rel_path, page_num, url, trace_id: str = None) -> Tuple[Optional[str], str, float]:
    """
    Executes OCR by sending the image to a remote docling-serve instance.
    """
    import io

    import requests

    if trace_id:
        set_trace_id(trace_id)

    log.info(f"🛰️ Sending remote OCR request to {url} for {rel_path} P{page_num}")

    start_time = time.perf_counter()
    try:
        # Convert NumPy to PNG in memory
        pil_image = Image.fromarray(np_image)
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)

        # Use 'files' as the primary key as indicated by the error message, 
        # but try 'file' if the server expects a single file.
        files = {"files": (f"page_{page_num}.png", buf, "image/png")}
        
        # docling-serve (FastAPI) handles multiple values for the same key as a list.
        # We pass them as a list of tuples to ensure requests repeats the keys correctly.
        data = [
            ("ocr_engine", "easyocr"),
            ("ocr_lang", "en"),
            ("to_formats", "md"),
            ("include_images", "false")
        ]

        log.info(f"📤 Remote OCR Payload: URL={url}, Field='files', Params={data}")
        response = requests.post(url, files=files, data=data, timeout=300)

        # Fallback to 'file' (singular) if the plural version is still rejected with a 422
        if response.status_code == 422:
            log.warning("⚠️ Server rejected 'files' field. Retrying with 'file' singular...")
            buf.seek(0)
            files_singular = {"file": (f"page_{page_num}.png", buf, "image/png")}
            response = requests.post(url, files=files_singular, data=data, timeout=300)

        execution_time_ms = (time.perf_counter() - start_time) * 1000.0

        if response.status_code == 200:
            result = response.json()
            
            # --- SIMPLIFIED RECURSIVE EXTRACTION ---
            def find_text(obj, key_name=None):
                """Visit every node. If it's a string and parent key is a text key, return it."""
                # Whitelist of keys known to contain the final markdown/text
                TEXT_KEYS = ["md", "markdown", "text", "content", "body", "md_content", "text_content"]
                
                def is_text_key(k):
                    if not k:
                        return False
                    k_lower = k.lower()
                    return k_lower in TEXT_KEYS or "markdown" in k_lower or k_lower.endswith("_content")

                if isinstance(obj, str):
                    if is_text_key(key_name) and len(obj.strip()) > 10:
                        return obj
                    return None
                
                if isinstance(obj, dict):
                    # Check our direct children first
                    for k, v in obj.items():
                        if is_text_key(k) and isinstance(v, str) and len(v.strip()) > 10:
                            return v
                    # Recurse deeper
                    for k, v in obj.items():
                        res = find_text(v, k)
                        if res:
                            return res
                elif isinstance(obj, list):
                    for item in obj:
                        res = find_text(item, key_name)
                        if res:
                            return res
                return None

            text = find_text(result)

            if not text:
                log.warning(f"⚠️ Remote OCR succeeded but text extraction failed. FULL RESPONSE: {json.dumps(result)}")
                return None, "remote_ocr_no_text", execution_time_ms

            log.info(f"✅ Remote OCR succeeded for {rel_path} P{page_num} ({len(text)} chars)")
            return text, "remote_docling_serve", execution_time_ms
        else:
            log.error(f"💥 Remote OCR failed with status {response.status_code}: {response.text}")
            return None, f"remote_ocr_error_{response.status_code}", execution_time_ms

    except Exception as e:
        log.error(f"💥 Remote OCR exception: {e}")
        return None, "remote_ocr_exception", 0.0


def run_ocr(np_image, rel_path, page_num, trace_id: str = None) -> Tuple[Optional[str], str, float]:
    """
    Executes OCR using Docling (EasyOCR backend) on a NumPy image array.
    """
    if trace_id:
        set_trace_id(trace_id)

    if OCR_ENDPOINTS.startswith(("http://", "https://")):
        return run_remote_ocr(np_image, rel_path, page_num, OCR_ENDPOINTS, trace_id)

    log.info(f"🔄 Running Docling (EasyOCR) for {rel_path} page {page_num}")

    converter = get_docling_converter()
    if not converter:
        return None, "error_docling_not_init", 0.0

    # Ensure a context-aware filename for the temporary file
    # This prevents 'docling-parse' from logging generic 'Processing document tmp...' nonsense.
    clean_name = os.path.basename(rel_path).replace(".", "_")
    context_filename = f"page_{page_num}_{clean_name}.png"
    tmp_path = os.path.join("/tmp", context_filename)

    start_time = time.perf_counter()
    try:
        # Save the NumPy array to a context-aware file
        pil_image = Image.fromarray(np_image)
        pil_image.save(tmp_path)

        try:
            result = converter.convert(tmp_path)
            # Result is a converted document. We export to markdown for better structure preservation.
            full_text = result.document.export_to_markdown().strip()

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
