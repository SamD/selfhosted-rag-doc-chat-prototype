#!/usr/bin/env python3
"""
Utility functions for the producer worker and graph.
Refactored for PARALLEL extraction, OCR jobs, and Document Context injection.
"""

import base64
import gc
import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pdfplumber
import redis
import torch
import whisperx
from bs4 import BeautifulSoup
from charset_normalizer import from_path
from config.settings import (
    COMPUTE_TYPE,
    DEVICE,
    EMBEDDING_MODEL_PATH,
    MAX_OCR_DIM,
    MEDIA_BATCH_SIZE,
    REDIS_HOST,
    REDIS_OCR_JOB_QUEUE,
    REDIS_PORT,
    SUPPORTED_MEDIA_EXT,
)
from pdf2image import convert_from_path
from PIL import Image
from processors.text_processor import split_doc
from transformers import AutoTokenizer
from utils.text_utils import is_gibberish, is_low_quality, is_visibly_corrupt

log = logging.getLogger("ingest.producer_utils")

# Global tokenizer and redis client (Lazy initialized per process)
_CACHED_TOKENIZER = None
_REDIS_CLIENT_CACHE = None


def get_redis_client():
    """Lazy initializer for the Redis client to ensure fork safety."""
    global _REDIS_CLIENT_CACHE
    if _REDIS_CLIENT_CACHE is None:
        _REDIS_CLIENT_CACHE = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _REDIS_CLIENT_CACHE


def get_tokenizer():
    """Lazy initializer for the shared tokenizer."""
    global _CACHED_TOKENIZER
    if _CACHED_TOKENIZER is None:
        log.info(f"🚀 Loading tokenizer from {EMBEDDING_MODEL_PATH}")
        _CACHED_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH, use_fast=True, trust_remote_code=True, local_files_only=True)
    return _CACHED_TOKENIZER


try:
    Resample = Image.Resampling
except AttributeError:
    Resample = Image

device = DEVICE
batch_size = MEDIA_BATCH_SIZE
compute_type = COMPUTE_TYPE


def get_whisper_model():
    return whisperx.load_model("large-v2", device, compute_type=compute_type)


def extract_text_from_media(filepath):
    if not filepath.lower().endswith(SUPPORTED_MEDIA_EXT):
        raise ValueError(f"Unsupported file type: {filepath}")
    log.info(f" 🎥 Processing media {filepath}")
    try:
        audio = whisperx.load_audio(filepath)
        model = get_whisper_model()
        result = model.transcribe(audio, batch_size=batch_size)
        return result["segments"]
    except Exception as e:
        log.error(f"Transcription failed for {filepath}: {e}", exc_info=True)
        return None
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        if "model" in locals():
            del model


def is_bad_ocr(text):
    tokenizer = get_tokenizer()
    return not text or not text.strip() or is_gibberish(text) or is_visibly_corrupt(text) or is_low_quality(text, tokenizer)


def preprocess_image(pil_image):
    w, h = pil_image.size
    if max(w, h) > MAX_OCR_DIM:
        scale = MAX_OCR_DIM / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        pil_image = pil_image.resize(new_size, Resample.LANCZOS)
    np_image = np.array(pil_image)
    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    return np_image


def extract_text_from_html(full_path: str) -> str:
    try:
        match = from_path(full_path).best()
        if not match:
            raise ValueError(f"[ERROR] Could not detect encoding for: {full_path}")
        html = str(match)
        soup = BeautifulSoup(html, "html5lib")
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text
    except Exception as e:
        log.error(f"[ERROR] extract_text_from_html failed for {full_path}: {e}", exc_info=True)
        return None


def send_image_to_ocr(np_image, rel_path, page_num):
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
    return (result.get("text"), result.get("rel_path"), result.get("page_num"), result.get("engine"), result.get("job_id"), ocr_roundtrip_ms)


def blocking_push_with_backpressure(rclient, queue_name: str, entries: list[str], max_queue_length: int = 1000, poll_interval: float = 0.5, warn_after: float = 10.0, rel_path: str = "unknown"):
    push_script = rclient.register_script("""
    local queue = KEYS[1]
    local max_len = tonumber(ARGV[1])
    local new_items = {}
    for i = 2, #ARGV do
        table.insert(new_items, ARGV[i])
    end
    local current_len = redis.call("LLEN", queue)
    if current_len + #new_items <= max_len then
        for _, item in ipairs(new_items) do
            redis.call("RPUSH", queue, item)
        end
        return 1
    else
        return 0
    end
    """)
    start_wait = time.time()
    warned = False
    total_wait_time = 0
    while True:
        result = push_script(keys=[queue_name], args=[max_queue_length] + entries)
        if result == 1:
            return
        if not warned and (time.time() - start_wait) > warn_after:
            warned = True
            log.warning(f"⏳ Backpressure triggered on queue {queue_name} for {rel_path}")
        time.sleep(poll_interval)
        total_wait_time += poll_interval


def _process_single_page(full_path, rel_path, page_num, file_type, chunk_callback, metrics, document_id):
    """Internal helper to process a single PDF page (text or OCR)."""
    try:
        # Use a fresh pdfplumber handle per page to be thread-safe
        with pdfplumber.open(full_path) as pdf:
            page = pdf.pages[page_num - 1]
            try:
                text = page.extract_text()
                text = (text or "").strip()
            except Exception:
                text = ""

            if not text or is_bad_ocr(text):
                # Fallback to OCR
                pill_image = convert_from_path(full_path, dpi=300, first_page=page_num, last_page=page_num)[0]
                np_image = preprocess_image(pill_image)
                result = send_image_to_ocr(np_image, rel_path, page_num)
                text, _, _, engine, _, ocr_ms = result

                if metrics:
                    metrics.add_ocr_operation(page=page_num, ocr_roundtrip_time_ms=ocr_ms, engine=engine, success=bool(text))

                if not text or not text.strip():
                    return False

                chunk_texts, _ = split_doc(text.strip(), rel_path, file_type, page_num=page_num, document_id=document_id)
                if chunk_callback:
                    chunk_callback([(c, engine, page_num) for c in chunk_texts])
                return True
            else:
                # Normal text extraction
                chunk_texts, _ = split_doc(text.strip(), rel_path, file_type, page_num=page_num, document_id=document_id)
                if chunk_callback:
                    chunk_callback([(c, "pdfplumber", page_num) for c in chunk_texts])
                return True
    except Exception as e:
        log.error(f"💥 Failed page {page_num} of {rel_path}: {e}")
        return False


def process_pdf_by_page(full_path, rel_path, file_type, chunk_callback=None, metrics=None, document_id=None):
    """
    Orchestrates the parallel extraction of a PDF.
    """
    try:
        with pdfplumber.open(full_path) as pdf:
            total_pages = len(pdf.pages)

        log.info(f"📂 Processing {total_pages} pages of {rel_path} in parallel")

        # Parallelize page extraction
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_process_single_page, full_path, rel_path, p + 1, file_type, chunk_callback, metrics, document_id) for p in range(total_pages)]
            # Wait for all pages to complete
            for future in futures:
                future.result()

    except Exception as e:
        log.error(f"💥 Failed to open PDF {rel_path}: {e}")
    return [], []


def fallback_ocr(full_path, rel_path=None, job_id=None, metrics=None, chunk_callback=None, document_id=None):
    """
    Force OCR on every page of a PDF in parallel.
    """
    return process_pdf_by_page(full_path, rel_path, "pdf", chunk_callback=chunk_callback, metrics=metrics, document_id=document_id)
