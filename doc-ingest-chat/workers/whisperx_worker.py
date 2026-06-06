#!/usr/bin/env python3
"""
Dedicated WhisperX Worker.
Listens for transcription jobs and pushes segments back to Redis.
"""

import torch

# ROOT CAUSE FIX: pyannote.audio (used by WhisperX) requires TF32 to be disabled 
# for reproducibility. By setting this explicitly at the very top, we satisfy 
# the library's state requirements and prevent it from triggering the 
# ReproducibilityWarning during its own initialization.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

import json
import logging
import os
import signal
import time
import traceback

import redis
from utils.trace_utils import get_logger, set_trace_id

# Minimal config for the worker - can be overridden by ENV
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_WHISPER_JOB_QUEUE = os.getenv("REDIS_WHISPER_JOB_QUEUE", "whisper_processing_job")

# WhisperX specific settings
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
BATCH_SIZE = int(os.getenv("MEDIA_BATCH_SIZE", 8))
WHISPER_MODEL_ENDPOINTS = os.getenv("WHISPER_MODEL_ENDPOINTS", "/models/whisper")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = get_logger("whisperx_worker")

SHUTDOWN = False


class RemoteWhisper:
    """
    Wrapper for remote Whisper server.
    Uses direct requests to allow for non-standard path structures.
    """

    def __init__(self, base_url: str):
        self.url = base_url
        log.info(f"🌐 Remote Whisper Target: {self.url}")

    def transcribe_file(self, file_path: str, language: str = "en", mime_type: str = None):
        """Transcribe file using remote API via direct POST."""
        import mimetypes

        import requests

        if not mime_type:
            mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"

        with open(file_path, "rb") as audio_file:
            files = {"file": (os.path.basename(file_path), audio_file, mime_type)}
            data = {
                "temperature": "0.0",
                "response_format": "json",
                "language": language,
            }

            log.info(f"📤 Sending POST to {self.url} (mime: {mime_type})...")
            response = requests.post(self.url, files=files, data=data, timeout=300)
            response.raise_for_status()

            result = response.json()
            text = result.get("text", "")
            return {"segments": [{"text": text}]}


def signal_handler(sig, frame):
    global SHUTDOWN
    log.warning(f"💥 Received signal {sig}, initiating WhisperX shutdown...")
    SHUTDOWN = True


def worker_loop():
    """Main loop that pops jobs and performs transcription."""
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
            socket_connect_timeout=5, socket_timeout=None,
        )
        log.info(f"🛰️ WhisperX Worker listening on {REDIS_WHISPER_JOB_QUEUE}...")
        log.info(f"🚀 Using Device: {DEVICE}, Compute Type: {COMPUTE_TYPE}, Model Path: {WHISPER_MODEL_ENDPOINTS}")

        # Lazy load whisperx to avoid overhead if redis fails
        import torchaudio

        # Monkey-patch torchaudio if needed (for older versions in some images)
        if not hasattr(torchaudio, "AudioMetaData"):
            from typing import NamedTuple

            class AudioMetaData(NamedTuple):
                sample_rate: int
                num_frames: int
                num_channels: int
                bits_per_sample: int
                encoding: str

            torchaudio.AudioMetaData = AudioMetaData
            log.info("🩹 Applied monkey-patch for torchaudio.AudioMetaData")

        # Load model once
        if WHISPER_MODEL_ENDPOINTS.startswith(("http://", "https://")):
            log.info(f"🏗️ Connecting to remote Whisper at {WHISPER_MODEL_ENDPOINTS}...")
            model = RemoteWhisper(base_url=WHISPER_MODEL_ENDPOINTS)
        else:
            import whisperx

            log.info(f"🏗️ Loading WhisperX model from {WHISPER_MODEL_ENDPOINTS}...")
            model = whisperx.load_model(WHISPER_MODEL_ENDPOINTS, DEVICE, compute_type=COMPUTE_TYPE)
        log.info("✅ Model loaded successfully.")

        while not SHUTDOWN:
            try:
                res = redis_client.brpop(REDIS_WHISPER_JOB_QUEUE, timeout=5)
                if res:
                    _, job_raw = res
                    job = json.loads(job_raw)
                    job_id = job.get("job_id")
                    file_path = job.get("file_path")
                    reply_key = job.get("reply_key")
                    language = job.get("language", "en")
                    trace_id = job.get("trace_id")
                    mime_type = job.get("mime_type")

                    if trace_id:
                        set_trace_id(trace_id)

                    mode = "REMOTE" if isinstance(model, RemoteWhisper) else "LOCAL"
                    log.info(f"🎬 Processing job {job_id} ({language}) [MODE: {mode}]: {file_path}")

                    try:
                        if not os.path.exists(file_path):
                            raise FileNotFoundError(f"File not found: {file_path}")

                        if isinstance(model, RemoteWhisper):
                            result = model.transcribe_file(file_path, language=language, mime_type=mime_type)
                        else:
                            import whisperx

                            audio = whisperx.load_audio(file_path)
                            result = model.transcribe(audio, batch_size=BATCH_SIZE, language=language)

                        for segment in result["segments"]:
                            redis_client.rpush(
                                reply_key, json.dumps({"type": "segment", "text": segment["text"]})
                            )

                        redis_client.rpush(reply_key, json.dumps({"type": "done"}))
                        log.info(f"✅ Job {job_id} complete.")

                    except Exception as e:
                        log.error(f"💥 Error processing job {job_id}: {e}")
                        redis_client.rpush(reply_key, json.dumps({"type": "error", "error": str(e)}))

            except json.JSONDecodeError:
                log.error("💥 Malformed Job received")
            except Exception as e:
                log.error(f"💥 Worker loop error: {e}")
                time.sleep(1)

    except Exception as e:
        log.error(f"💥 Fatal initialization error: {e}")
        traceback.print_exc()

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    worker_loop()

if __name__ == "__main__":
    main()
