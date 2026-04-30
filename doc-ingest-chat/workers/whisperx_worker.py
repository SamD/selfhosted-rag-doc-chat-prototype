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
BATCH_SIZE = int(os.getenv("MEDIA_BATCH_SIZE", 16))
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "/models/whisper")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = get_logger("whisperx_worker")

SHUTDOWN = False

def signal_handler(sig, frame):
    global SHUTDOWN
    log.warning(f"💥 Received signal {sig}, initiating WhisperX shutdown...")
    SHUTDOWN = True

def worker_loop():
    """Main loop that pops jobs and performs transcription."""
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        log.info(f"🛰️ WhisperX Worker listening on {REDIS_WHISPER_JOB_QUEUE}...")
        log.info(f"🚀 Using Device: {DEVICE}, Compute Type: {COMPUTE_TYPE}, Model Path: {WHISPER_MODEL_PATH}")

        # Lazy load whisperx to avoid overhead if redis fails
        import torchaudio
        import whisperx

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
        log.info(f"🏗️ Loading WhisperX model from {WHISPER_MODEL_PATH}...")
        model = whisperx.load_model(WHISPER_MODEL_PATH, DEVICE, compute_type=COMPUTE_TYPE)
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
                    
                    if trace_id:
                        set_trace_id(trace_id)

                    log.info(f"🎬 Processing job {job_id} ({language}): {file_path}")

                    try:
                        if not os.path.exists(file_path):
                            raise FileNotFoundError(f"File not found: {file_path}")

                        audio = whisperx.load_audio(file_path)
                        result = model.transcribe(audio, batch_size=BATCH_SIZE, language=language)

                        for segment in result["segments"]:
                            redis_client.rpush(reply_key, json.dumps({
                                "type": "segment",
                                "text": segment["text"]
                            }))
                        
                        redis_client.rpush(reply_key, json.dumps({"type": "done"}))
                        log.info(f"✅ Job {job_id} complete.")

                    except Exception as e:
                        log.error(f"💥 Error processing job {job_id}: {e}")
                        redis_client.rpush(reply_key, json.dumps({
                            "type": "error",
                            "error": str(e)
                        }))

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
