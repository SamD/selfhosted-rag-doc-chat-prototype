#!/usr/bin/env python3
"""Temporal Activity Worker entry point for WhisperX transcription."""
import asyncio
import logging
import signal

from temporalio.client import Client
from temporalio.worker import Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest.temporal_worker")

async def main():
    import sys
    print(f"DEBUG: Starting temporal worker", flush=True, file=sys.stderr)
    # Build Temporal connection from TEMPORAL_HOST:PORT with legacy fallback
    from config.settings import (
        TEMPORAL_HOST,
        TEMPORAL_PORT,
        TEMPORAL_SERVER_URL,
        TEMPORAL_WHISPER_TASK_QUEUE,
    )
    print(f"DEBUG: TEMPORAL_HOST={TEMPORAL_HOST} PORT={TEMPORAL_PORT}", flush=True, file=sys.stderr)

    target_host = f"{TEMPORAL_HOST}:{TEMPORAL_PORT}"
    if TEMPORAL_HOST == "localhost" and TEMPORAL_PORT == 7233:
        target_host = TEMPORAL_SERVER_URL  # legacy fallback
    print(f"DEBUG: Connecting to {target_host}", flush=True, file=sys.stderr)

    client = await Client.connect(target_host=target_host)
    print(f"DEBUG: Connected!", flush=True, file=sys.stderr)
    
    from temporal_worker.activities import transcribe_media
    from temporal_worker.workflows import TranscribeWorkflow
    
    worker = Worker(
        client=client,
        task_queue=TEMPORAL_WHISPER_TASK_QUEUE,
        workflows=[TranscribeWorkflow],
        activities=[transcribe_media],
        max_concurrent_activities=1,
    )
    
    stop = asyncio.Event()
    def signal_handler(sig, frame):
        print(f"[WORKER] Signal received, setting stop", flush=True)
        log.warning("Received SIGTERM, shutting down gracefully...")
        stop.set()
    
    signal.signal(signal.SIGTERM, signal_handler)
    
    async def heartbeat():
        while not stop.is_set():
            print(f"[HEARTBEAT] Worker alive, waiting for tasks...", flush=True)
            await asyncio.sleep(10)
    
    asyncio.create_task(heartbeat())
    
    async with worker:
        print(f"[WORKER] Entering async with worker, task_queue={TEMPORAL_WHISPER_TASK_QUEUE}", flush=True)
        log.info("Temporal WhisperX Activity Worker started")
        await stop.wait()
    
    log.info("Temporal WhisperX Activity Worker stopped")

if __name__ == "__main__":
    asyncio.run(main())