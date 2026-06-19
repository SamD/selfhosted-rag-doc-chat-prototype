#!/usr/bin/env python3
"""
Utility functions for Media (WhisperX) processing.
Delegates to a dedicated WhisperX worker via Redis queue to avoid dependency conflicts.
"""

import asyncio
import json
import os
import time
import uuid
from datetime import timedelta
from typing import Generator

import redis
from config.settings import REDIS_HOST, REDIS_PORT, REDIS_WHISPER_JOB_QUEUE, TEMPORAL_HOST, TEMPORAL_PORT, TEMPORAL_SERVER_URL, USE_TEMPORAL_WHISPER
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


def send_media_to_whisperx_temporal(file_path: str, language: str = "en", mime_type: str = None, trace_id: str = None) -> Generator[str, None, None]:
    """
    Send media file path to WhisperX service using Temporal Activities.
    Creates a Temporal Client, executes the transcribe_media workflow synchronously,
    and yields segment text from the returned TranscriptionResult.
    Raises RuntimeError on transcription failure (matching existing error contract).
    """
    if trace_id:
        set_trace_id(trace_id)

    from models.transcription_input import TranscriptionInput

    job_id = str(uuid.uuid4())
    
    # We send the absolute path so the worker (sharing volumes) can find it
    abs_file_path = os.path.abspath(file_path)

    # Create TranscriptionInput for Temporal workflow
    input_data = TranscriptionInput(
        file_path=abs_file_path,
        language=language,
        mime_type=mime_type
    )

    log.info(f"📤 Sending {file_path} to WhisperX Temporal Activity (Job: {job_id})")

    try:
        from config.settings import TEMPORAL_WHISPER_TASK_QUEUE
        from temporal_worker.workflows import TranscribeWorkflow
        from temporalio.client import Client as TemporalClient

        # Bridge async Temporal client with asyncio.run() because this generator is synchronous
        async def _run():
            target_host = f"{TEMPORAL_HOST}:{TEMPORAL_PORT}"
            if TEMPORAL_HOST == "localhost" and TEMPORAL_PORT == 7233:
                target_host = TEMPORAL_SERVER_URL
            client = await TemporalClient.connect(
                target_host=target_host,
            )
            try:
                return await client.execute_workflow(
                    TranscribeWorkflow,
                    input_data,
                    id=job_id,
                    task_queue=TEMPORAL_WHISPER_TASK_QUEUE,
                    execution_timeout=timedelta(minutes=90),  # 90 minutes
                )
            except Exception as e:
                print(f"[WHISPER] execute_workflow failed: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise

        result = asyncio.run(_run())
        print(f"[WHISPER] Got result: {result}", flush=True)

        # Yield segments from the result
        for segment in result.segments:
            yield segment["text"]

        log.info(f"✅ WhisperX Temporal transcription complete for {file_path} ({len(result.segments)} segments)")

    except Exception as e:
        log.error(f"❌ WhisperX Temporal Activity failed: {e}")
        raise RuntimeError(f"WhisperX Temporal error: {e}")


def send_media_to_whisperx(file_path: str, language: str = "en", mime_type: str = None, trace_id: str = None) -> Generator[str, None, None]:
    """
    Send media file path to WhisperX service and yield transcription segments.
    Checks USE_TEMPORAL_WHISPER flag and dispatches to either Temporal or Redis path.
    When flag is false, behavior is identical to current Redis implementation.
    """
    if trace_id:
        set_trace_id(trace_id)

    if USE_TEMPORAL_WHISPER:
        log.info("🔄 Using Temporal path for WhisperX transcription")
        yield from send_media_to_whisperx_temporal(file_path, language, mime_type, trace_id)
        return

    # Original Redis-based implementation
    job_id = str(uuid.uuid4())
    reply_key = f"whisper_reply:{job_id}"

    # We send the absolute path so the worker (sharing volumes) can find it
    abs_file_path = os.path.abspath(file_path)

    job = {
        "job_id": job_id,
        "file_path": abs_file_path,
        "reply_key": reply_key,
        "language": language,
        "mime_type": mime_type,
        "trace_id": trace_id,
    }

    log.info(f"📤 Sending {file_path} to WhisperX worker (Job: {job_id})")

    try:
        redis_client = get_redis_client()
        redis_client.lpush(REDIS_WHISPER_JOB_QUEUE, json.dumps(job))
    except Exception as e:
        log.error(f"❌ Failed to submit WhisperX job to Redis: {e}")
        raise RuntimeError(f"Redis submission failed: {e}")

    # WhisperX jobs can take a long time.
    wait_timeout = 1800  # 30 minutes for long videos
    start_wait = time.time()

    segments_received = 0

    while (time.time() - start_wait) < wait_timeout:
        res = redis_client.blpop(reply_key, timeout=30)
        if res:
            _, data_raw = res
            data = json.loads(data_raw)

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

