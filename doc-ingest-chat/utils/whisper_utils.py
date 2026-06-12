#!/usr/bin/env python3
"""
Utility functions for Media (WhisperX) processing.
Delegates to a dedicated WhisperX worker via Redis queue to avoid dependency conflicts.
"""

import json
import os
import time
import uuid
from typing import Generator

import redis
from config.settings import REDIS_HOST, REDIS_PORT, REDIS_WHISPER_JOB_QUEUE
from utils.trace_utils import get_logger, set_trace_id

log = get_logger("ingest.whisper_utils")

_REDIS_CLIENT_CACHE = None


def get_redis_client():
    """Lazy initializer for the Redis client."""
    global _REDIS_CLIENT_CACHE
    if _REDIS_CLIENT_CACHE is None:
        _REDIS_CLIENT_CACHE = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
            socket_connect_timeout=5, socket_timeout=None,
        )
    return _REDIS_CLIENT_CACHE


def send_media_to_whisperx(file_path: str, language: str = "en", mime_type: str = None, trace_id: str = None) -> Generator[str, None, None]:
    """
    Send media file path to WhisperX service and yield transcription segments.
    Always uses Redis queue to ensure the flow remains unchanged.
    """
    if trace_id:
        set_trace_id(trace_id)

    job_id = str(uuid.uuid4())

    # We send the absolute path so the worker (sharing volumes) can find it
    abs_file_path = os.path.abspath(file_path)

    job = {
        "job_id": job_id,
        "file_path": abs_file_path,
        "language": language,
        "mime_type": mime_type,
        "trace_id": trace_id,
    }

    log.info(f"📤 Sending {file_path} to WhisperX worker (Job: {job_id})")

    try:
        redis_client = get_redis_client()
        redis_client.lpush(f"{REDIS_WHISPER_JOB_QUEUE}_input", json.dumps(job))
    except Exception as e:
        log.error(f"❌ Failed to submit WhisperX job to Redis: {e}")
        raise RuntimeError(f"Redis submission failed: {e}")

    # Read from shared reply queue, match by job_id
    reply_queue = f"{REDIS_WHISPER_JOB_QUEUE.replace('_processing_job', '_reply')}_output"
    wait_timeout = 1800  # 30 minutes for long videos
    start_wait = time.time()

    segments_received = 0

    while (time.time() - start_wait) < wait_timeout:
        res = redis_client.blpop(reply_queue, timeout=30)
        if res:
            _, data_raw = res
            data = json.loads(data_raw)

            # Check if this reply belongs to our job
            if data.get("job_id") != job_id:
                # Not our reply, put it back for another caller
                redis_client.lpush(reply_queue, data_raw)
                continue

            if data.get("type") == "segment":
                segments_received += 1
                yield data.get("text")
            elif data.get("type") == "done":
                log.info(f"✅ WhisperX transcription complete for {file_path} ({segments_received} segments)")
                break
            elif data.get("type") == "error":
                error_msg = data.get("error", "Unknown error")
                log.error(f"❌ WhisperX worker reported error: {error_msg}")
                raise RuntimeError(f"WhisperX error: {error_msg}")
        else:
            elapsed = int(time.time() - start_wait)
            log.info(f"⏳ Waiting for WhisperX... {file_path} ({elapsed}s elapsed)")

    else:
        log.error(f"⏰ WhisperX timeout for {file_path}")
        raise TimeoutError(f"WhisperX transcription timed out after {wait_timeout}s")

