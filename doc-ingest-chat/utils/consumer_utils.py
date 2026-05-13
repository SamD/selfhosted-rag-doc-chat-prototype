#!/usr/bin/env python3
"""
Utility functions for the consumer worker and graph.
"""

import gc
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import torch
from config.settings import MAX_CHROMA_BATCH_SIZE, USE_QDRANT
from more_itertools import chunked
from services.database import get_vectorstore

log = logging.getLogger("ingest.consumer_utils")


def store_chunks_in_db(source_file: str, chunks: List[Dict[str, Any]], metrics: Optional[Any] = None) -> int:
    """
    Stores a list of chunks in the Vector DB (Chroma or Qdrant).
    Handles batching, basic validation, and retries for transient errors.
    """
    # Filter out chunks that were already stored during incremental ingestion
    chunks_to_store = [c for c in chunks if not c.get("_already_stored")]

    if not chunks_to_store:
        log.info(f"⏭️ All {len(chunks)} chunks for {source_file} already in DB")
        return 0

    db_type = "Qdrant" if USE_QDRANT else "ChromaDB"
    log.info(f"📤 Writing {len(chunks_to_store)} chunks → {db_type} for {source_file}")

    try:
        db = get_vectorstore()

        # Extract fields with safe defaults
        all_texts = [entry.get("chunk", "[EMPTY CHUNK]") for entry in chunks_to_store]

        # Safety net: validate each text before embedding (catches tokenizer/remote discrepancy)
        from config.settings import MAX_TOKENS
        from utils.text_utils import get_tokenizer as _get_tok
        _tokenizer = _get_tok()
        _valid_texts = []
        for text in all_texts:
            if _tokenizer and len(text) > MAX_TOKENS * 5:
                _tokens = len(_tokenizer.encode(text, add_special_tokens=True))
                if _tokens > MAX_TOKENS:
                    log.error(f"❌ Chunk passed validation but exceeds token budget at embed time: {_tokens} tokens, {len(text)} chars. Skipping.")
                    raise RuntimeError(f"Chunk token count ({_tokens}) exceeds MAX_TOKENS ({MAX_TOKENS}) at embed time")
            _valid_texts.append(text)
        all_texts = _valid_texts
        all_metadatas = [
            {
                "source_file": entry.get("source_file", source_file),
                "type": entry.get("type", "unknown"),
                "engine": entry.get("engine", "llamacpp"),
                "hash": entry.get("hash", "unknown"),
                "chunk_index": entry.get("chunk_index", i),
                "id": entry.get("id", f"missing_id_{i}"),
                "page": int(entry.get("page", -1)) if str(entry.get("page", "")).isdigit() else -1,
            }
            for i, entry in enumerate(chunks_to_store)
        ]
        all_ids = [m["id"] for m in all_metadatas]

        # Split and ingest in safe batches
        batches_count = 0

        # RETRY CONFIG
        MAX_RETRIES = 5
        BASE_DELAY = 2.0  # seconds

        def add_batch_with_retry(texts, metas, ids, batch_idx):
            last_err = None
            for attempt in range(MAX_RETRIES):
                try:
                    db.add_texts(texts, metadatas=metas, ids=ids)
                    return True
                except Exception as e:
                    last_err = e
                    # Transient errors often include timeouts or connection issues
                    err_msg = str(e).lower()
                    is_transient = any(
                        x in err_msg
                        for x in ["timeout", "timed out", "connection", "rate limit", "overloaded", "503", "504"]
                    )

                    if not is_transient:
                        log.error(f"❌ Non-transient error in batch {batch_idx}: {e}")
                        raise e

                    delay = (BASE_DELAY * (2**attempt)) + (random.random() * 1.0)
                    log.warning(
                        f"⚠️ Batch {batch_idx} failed (Attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

            log.error(f"💥 Batch {batch_idx} failed after {MAX_RETRIES} attempts.")
            raise last_err

        # If metrics provided, wrap in timer
        timer_ctx = metrics.timer("chromadb_embedding") if metrics else open(os.devnull, "w")

        with (timer_ctx if metrics else open(os.devnull, "w")):
            for i, (texts_batch, metas_batch, ids_batch) in enumerate(
                zip(
                    chunked(all_texts, MAX_CHROMA_BATCH_SIZE),
                    chunked(all_metadatas, MAX_CHROMA_BATCH_SIZE),
                    chunked(all_ids, MAX_CHROMA_BATCH_SIZE),
                )
            ):
                add_batch_with_retry(texts_batch, metas_batch, ids_batch, i)
                batches_count += 1

        count = db.get_collection_count()
        if count == 0:
            raise RuntimeError(f"💥 Vector DB persist failed — 0 documents after ingesting {source_file}")

        log.info(f"✅ Persisted {source_file} — Vector DB doc count: {count}")
        return batches_count
    except Exception as e:
        db_type = "Qdrant" if USE_QDRANT else "ChromaDB"
        log.error(f"💥 Failed to write {source_file} to {db_type}: {e}")
        raise e
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
