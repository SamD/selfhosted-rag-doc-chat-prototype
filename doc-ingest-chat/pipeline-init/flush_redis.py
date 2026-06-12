#!/usr/bin/env python3
"""Flush all pipeline Redis queues on startup."""

import os
import sys
import time

import redis

REDIS_HOST = os.environ.get("REDIS_HOST", "192.168.30.67")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6380"))

# Static queues (OCR, Whisper)
STATIC_QUEUES = [
    "ocr_processing_job_input",
    "ocr_processing_job_output",
    "ocr_reply_input",
    "ocr_reply_output",
    "whisper_processing_job_input",
    "whisper_processing_job_output",
    "whisper_reply_input",
    "whisper_reply_output",
    "retype_llm_job_input",
    "retype_llm_job_output",
    "retype_llm_reply_input",
    "retype_llm_reply_output",
]


def get_chunk_queues():
    """Build chunk ingest queue names from QUEUE_NAMES env var (matches producer/consumer)."""
    queue_names_str = os.environ.get("QUEUE_NAMES", "chunk_ingest_queue:0")
    queue_names = [q.strip() for q in queue_names_str.split(",") if q.strip()]
    chunk_queues = []
    for q in queue_names:
        chunk_queues.append(f"{q}_input")
        chunk_queues.append(f"{q}_output")
    return chunk_queues


def flush_queues():
    print(f"🔌 Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    for attempt in range(1, 31):
        try:
            r.ping()
            break
        except redis.ConnectionError:
            if attempt == 30:
                print("❌ Failed to connect to Redis after 30 attempts")
                sys.exit(1)
            print(f"   Waiting for Redis... (attempt {attempt}/30)")
            time.sleep(2)

    print("✅ Connected to Redis")
    all_queues = STATIC_QUEUES + get_chunk_queues()
    total = 0
    for queue in all_queues:
        length = r.llen(queue)
        if length > 0:
            r.delete(queue)
            print(f"   🗑️  Flushed {queue} ({length} messages)")
        else:
            print(f"   ✓ {queue} (empty)")
        total += length

    print(f"✅ Flushed {total} messages from {len(all_queues)} queues")


if __name__ == "__main__":
    flush_queues()
