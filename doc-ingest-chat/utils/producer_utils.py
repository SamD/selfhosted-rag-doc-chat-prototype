#!/usr/bin/env python3
"""
Utility functions for the producer worker and graph.
Refactored for PARALLEL extraction, OCR jobs, and Document Context injection.
"""

import gc
import json
import logging
import re
import time

import redis
import torch
from bs4 import BeautifulSoup
from charset_normalizer import from_path
from config.settings import (
    COMPUTE_TYPE,
    DEVICE,
    EMBEDDING_MODEL_PATH,
    MEDIA_BATCH_SIZE,
    REDIS_HOST,
    REDIS_PORT,
    SUPPORTED_MEDIA_EXT,
)
from transformers import AutoTokenizer

log = logging.getLogger("ingest.producer_utils")

# Global tokenizer and redis client (Lazy initialized per process)
_CACHED_TOKENIZER = None
_REDIS_CLIENT_CACHE = None


def get_tokenizer():
    """Lazy initializer for the shared tokenizer."""
    global _CACHED_TOKENIZER
    if _CACHED_TOKENIZER is None:
        log.info(f"🚀 Loading tokenizer from {EMBEDDING_MODEL_PATH}")
        _CACHED_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
    return _CACHED_TOKENIZER


def get_redis_client():
    """Lazy initializer for the Redis client to ensure fork safety."""
    global _REDIS_CLIENT_CACHE
    if _REDIS_CLIENT_CACHE is None:
        _REDIS_CLIENT_CACHE = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _REDIS_CLIENT_CACHE


def get_whisper_model():
    import whisperx

    return whisperx.load_model("large-v2", DEVICE, compute_type=COMPUTE_TYPE)


def extract_text_from_media(filepath):
    import whisperx

    if not filepath.lower().endswith(SUPPORTED_MEDIA_EXT):
        raise ValueError(f"Unsupported file type: {filepath}")
    log.info(f" 🎥 Processing media {filepath}")
    try:
        audio = whisperx.load_audio(filepath)
        model = get_whisper_model()
        result = model.transcribe(audio, batch_size=MEDIA_BATCH_SIZE)
        return result["segments"]
    except Exception as e:
        log.error(f"Transcription failed for {filepath}: {e}", exc_info=True)
        return None
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        if "model" in locals():
            del model


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


def handle_error(state: dict, error_msg: str, logger: logging.Logger) -> dict:
    """Helper for consistent error state updates in LangGraph."""
    logger.error(f"💥 {error_msg}")
    return {**state, "status": "failed", "error": error_msg}


def send_file_end_sentinel(rclient, queue_name: str, source_file: str, total_chunks: int):
    """Sends the sentinel message to indicate the end of document processing."""
    sentinel = {
        "source_file": source_file,
        "type": "file_end",
        "total_chunks": total_chunks,
    }
    rclient.rpush(queue_name, json.dumps(sentinel))
    log.info(f"🏁 Sent file_end sentinel for {source_file} ({total_chunks} chunks)")
