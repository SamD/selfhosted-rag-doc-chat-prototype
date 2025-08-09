#!/usr/bin/env python3
"""
Consumer Worker for processing chunks and storing them in ChromaDB.
"""
import json
import multiprocessing
import os
import signal
import sys
import time
import traceback
from collections import defaultdict
from itertools import cycle

from config.settings import CHUNK_TIMEOUT, MAX_CHROMA_BATCH_SIZE, MAX_CHUNKS, MAX_TOKENS, PARQUET_FILE, QUEUE_NAMES
from more_itertools import chunked
from services.database import get_db
from services.parquet_service import write_to_parquet
from services.redis_service import get_redis_client
from utils.file_utils import update_failed_files, update_ingested_files
from utils.logging_config import setup_logging

# from workers.queue_manager import QueueManager  # No need for queue manager in consumer



queue_lock = multiprocessing.Lock()
queue_cycle = cycle(QUEUE_NAMES)
parquet_lock = multiprocessing.Lock()

log = setup_logging("ingest_consumer.log", include_default_filters=True)

def get_next_queue():
    global queue_lock
    global queue_cycle

    with queue_lock:
        return next(queue_cycle)


def current_time():
    """Get current timestamp."""
    return int(time.time())


def consumer_worker(queue_name, shared_data, parq_lock):
    """Main consumer worker function."""
    try:
        r = get_redis_client()
        db = get_db()
        buffer = defaultdict(list)
        timestamps = {}
        success_counter = 0

        log.info(f"🚀 Started consumer for queue: {queue_name}")

        while True:
            now = current_time()
            for file, first_seen in list(timestamps.items()):
                if now - first_seen > CHUNK_TIMEOUT:
                    log.info(f"⌛ TTL expired for {file}, discarding buffer")
                    buffer.pop(file, None)
                    timestamps.pop(file, None)
                    update_failed_files(file)

            idle_counter = getattr(consumer_worker, "_idle_counter", 0)
            item = r.blpop(queue_name, timeout=5)
            if not item:
                idle_counter += 1
                if idle_counter % 12 == 0:
                    log.info(f"🕒 [{queue_name}] Idle waiting for chunks...")
                consumer_worker._idle_counter = idle_counter
                continue
            else:
                consumer_worker._idle_counter = 0

            if shared_data['shutdown_flag']:
                log.info("\n👋 SHUTDOWN_FLAG set exiting ...")
                break

            try:
                data = json.loads(item[1])
            except Exception as e:
                log.info(f"⚠️ [{queue_name}] Skipping malformed Redis entry: {e}")
                continue

            if data.get("type") == "file_end":
                source_file = data["source_file"]
                expected = data["expected_chunks"]

                log.info(f"📨 [{queue_name}] Received file_end for {source_file} (expecting {expected} chunks)")

                chunks = buffer.get(source_file, [])

                log.info(f"🧩 [{queue_name}] Buffer for {source_file} contains {len(chunks)} chunks")

                valid_chunks = []
                for entry in chunks:
                    if isinstance(entry["chunk"], str):
                        token_len = len(entry["chunk"])
                        if token_len > MAX_TOKENS:
                            log.info(f"⚠️ Chunk {entry['id']} from {entry['source_file']} exceeds {MAX_TOKENS} tokens ({token_len}) — dropping")
                            continue  # Drop this chunk
                        else:
                            valid_chunks.append(entry)
                    else:
                        log.info(f"⚠️ Chunk {entry['id']} is not a string — skipping")

                chunks = valid_chunks

                if len(chunks) != expected:
                    log.info(f"❌ [{queue_name}] File {source_file} incomplete (expected {expected}, got {len(chunks)})")
                    buffer.pop(source_file, None)
                    timestamps.pop(source_file, None)
                    update_failed_files(source_file)

                    db.delete(where={"source_file": source_file})
                    continue
                else:
                    try:
                        # Extract fields just once
                        all_texts = [entry["chunk"] for entry in chunks]
                        all_metadatas = [
                            {
                                "source_file": entry["source_file"],
                                "type": entry["type"],
                                "engine": entry.get("engine", "unknown"),
                                "hash": entry["hash"],
                                "chunk_index": entry.get("chunk_index", i),
                                "id": entry["id"],
                                "page": int(entry["page"]) if isinstance(entry.get("page"), (int, str)) and str(entry["page"]).isdigit() else -1
                            }
                            for i, entry in enumerate(chunks)
                        ]
                        all_ids = [entry["id"] for entry in chunks]

                        # Split and ingest in safe batches
                        for texts_batch, metas_batch, ids_batch in zip(
                                chunked(all_texts, MAX_CHROMA_BATCH_SIZE),
                                chunked(all_metadatas, MAX_CHROMA_BATCH_SIZE),
                                chunked(all_ids, MAX_CHROMA_BATCH_SIZE),
                        ):
                            db.add_texts(
                                texts_batch,
                                metadatas=metas_batch,
                                ids=ids_batch,
                            )

                        count = db._collection.count()
                        if count == 0:
                            raise RuntimeError(f"💥 Chroma persist failed — 0 documents after ingesting {source_file}")
                        log.info(f"✅ [{queue_name}] Persisted {source_file} — Chroma doc count: {count}")

                        success_counter += 1
                        if success_counter % 10 == 0:
                            log.info(f"📝 [{queue_name}] Archived {len(chunks)} chunks from {source_file} to chunks.parquet")

                        try:
                            write_to_parquet(chunks, PARQUET_FILE, parq_lock)
                            log.info(f"📝 Archived {len(chunks)} chunks from {source_file} to chunks.parquet")
                        except Exception as e:
                            log.info(f"💥 Failed to write Parquet file for {source_file}: {e}")
                            db.delete(where={"source_file": source_file})
                            update_failed_files(source_file)
                            continue

                        update_ingested_files(source_file)

                    except Exception as e:
                        log.info(f"💥 [{queue_name}] Failed to write {source_file} to Chroma: {e}\n{traceback.format_exc()}")
                        db.delete(where={"source_file": source_file})
                        update_failed_files(source_file)

                    finally:
                        buffer.pop(source_file, None)
                        timestamps.pop(source_file, None)

            else:
                required_keys = {"id", "chunk", "source_file", "type", "hash"}
                if not required_keys.issubset(data):
                    log.info(f"⚠️ [{queue_name}] Skipping malformed chunk — missing keys: {required_keys - set(data)}")
                    continue

                source_file = data["source_file"]
                log.info(f"📥 [{queue_name}] Received chunk {data['id']} from {source_file}")

                if len(buffer[source_file]) >= MAX_CHUNKS:
                    log.info(f"🛘 [{queue_name}] Max chunks exceeded for {source_file}")
                    buffer.pop(source_file, None)
                    timestamps.pop(source_file, None)
                    update_failed_files(source_file)
                    continue

                if source_file not in timestamps:
                    timestamps[source_file] = current_time()
                buffer[source_file].append(data)
    finally:
        log.info("✅ Exiting")


CHILD_PROCESSES = []
def make_sigint_handler(processes, ppid, shared_data):
    """Create signal handler for graceful shutdown."""
    def handler(signum, frame):
        if os.getpid() != ppid:
            # Prevent child processes from handling this
            return

        shared_data['shutdown_flag'] = True
        log.info(f"[Parent {os.getpid()}] SIGINT received. Sending to children...")

        for p in processes:
            log.info(f"[Parent] Joining PID {p.pid}")
            p.join()

        log.info("[Parent] All children joined. Exiting.")
        sys.exit(0)

    return handler



def main():
    parent_pid = os.getpid()


    with multiprocessing.Manager() as manager:
        shared_dict = manager.dict({'shutdown_flag': False})
        signal.signal(signal.SIGINT, make_sigint_handler(CHILD_PROCESSES, parent_pid, shared_dict))


        for i in range(len(QUEUE_NAMES)):
            next_queue = get_next_queue()
            p = multiprocessing.Process(target=consumer_worker, args=(next_queue,shared_dict,parquet_lock))
            p.start()
            CHILD_PROCESSES.append(p)
        log.info(f"🚀 Started {len(QUEUE_NAMES)} consumer workers for queues: {QUEUE_NAMES}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("[Parent] Caught KeyboardInterrupt outside handler")
            sys.exit(0)


if __name__ == "__main__":
    main()