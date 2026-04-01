#!/usr/bin/env python3
"""
Utility functions for the consumer worker and graph.
"""

import gc
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from config.settings import MAX_CHROMA_BATCH_SIZE, USE_QDRANT
from more_itertools import chunked
from services.database import get_db

log = logging.getLogger("ingest.consumer_utils")


def store_chunks_in_db(source_file: str, chunks: List[Dict[str, Any]], metrics: Optional[Any] = None) -> int:
    """
    Stores a list of chunks in the Vector DB (Chroma or Qdrant).
    Handles batching and basic validation.
    """
    # Filter out chunks that were already stored during incremental ingestion
    chunks_to_store = [c for c in chunks if not c.get("_already_stored")]

    if not chunks_to_store:
        log.info(f"⏭️ All {len(chunks)} chunks for {source_file} already in DB")
        return 0

    try:
        db = get_db()

        # Extract fields
        all_texts = [entry["chunk"] for entry in chunks_to_store]
        all_metadatas = [
            {
                "source_file": entry["source_file"],
                "type": entry["type"],
                "engine": entry.get("engine", "unknown"),
                "hash": entry["hash"],
                "chunk_index": entry.get("chunk_index", i),
                "id": entry["id"],
                "page": int(entry["page"]) if isinstance(entry.get("page"), (int, str)) and str(entry["page"]).isdigit() else -1,
            }
            for i, entry in enumerate(chunks_to_store)
        ]
        all_ids = [entry["id"] for entry in chunks_to_store]

        # Split and ingest in safe batches
        batches_count = 0

        # If metrics provided, wrap in timer
        timer_ctx = metrics.timer("chromadb_embedding") if metrics else open(os.devnull, "w")

        if metrics:
            with timer_ctx:
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
                    batches_count += 1
        else:
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
