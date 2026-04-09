#!/usr/bin/env python3
"""
LangGraph implementation of the producer ingestion workflow.
Includes deterministic document_id generation and per-chunk enrichment.
"""

import hashlib
import json
import logging
import os
from typing import Any, Optional, TypedDict

from config.settings import SUPPORTED_MEDIA_EXT
from langgraph.graph import END, StateGraph
from processors.text_processor import TextProcessor, make_chunk_id, split_doc, split_markdown_doc
from services.job_service import STATUS_ENQUEUED, STATUS_FAILED, STATUS_PROCESSING, update_job_status
from services.redis_service import get_redis_client
from utils.producer_utils import (
    blocking_push_with_backpressure,
    extract_text_from_html,
    extract_text_from_media,
)
from workers.producer_worker import get_next_queue

log = logging.getLogger("ingest.producer_graph")

# Global cache for the compiled graph
_COMPILED_PRODUCER_APP = None

# Global reference for the multiprocessing GPU lock
_GPU_LOCK = None


class IngestState(TypedDict):
    """Represents the internal state of a single file ingestion job."""

    job_id: str
    full_path: str
    rel_path: str
    file_type: str
    total_chunks_sent: int
    queue_name: Optional[str]
    status: str
    error: Optional[str]
    document_id: Optional[str]
    file_metadata: Optional[dict]
    metrics: Optional[Any]
    pages_processed: int


def scan_file_node(state: IngestState) -> IngestState:
    """Initial validation and type detection for normalized documents."""
    full_path = state["full_path"]
    rel_path = state["rel_path"]
    log.info(f"🔍 [Node: Scan] {rel_path}")
    update_job_status(rel_path, STATUS_PROCESSING, state["job_id"])

    if not os.path.exists(full_path):
        return {**state, "status": STATUS_FAILED, "error": "File does not exist"}

    file_ext = os.path.splitext(full_path)[1].lower()
    if file_ext != ".md":
        # We only process .md files from the gatekeeper now
        # But we might still support media/html later if they bypass gatekeeper
        if file_ext in (".html", ".htm"):
            file_type = "html"
        elif file_ext in SUPPORTED_MEDIA_EXT:
            file_type = "media"
        else:
            return {**state, "status": STATUS_FAILED, "error": f"Producer only processes normalized .md files (got {file_ext})"}
    else:
        file_type = "markdown"

    # CALCULATE DOCUMENT ID (Deterministic hash of file bytes)
    try:
        with open(full_path, "rb") as f:
            file_bytes = f.read()
        document_id = TextProcessor.get_document_id(file_bytes)
        log.info(f"🆔 [Node: Scan] Generated ID: {document_id} for {rel_path}")
    except Exception as e:
        log.error(f"💥 Failed to generate ID for {rel_path}: {e}")
        return {**state, "status": STATUS_FAILED, "error": f"ID generation failed: {e}"}

    return {**state, "file_type": file_type, "document_id": document_id, "status": STATUS_PROCESSING, "queue_name": get_next_queue()}


def stream_chunks_to_redis(chunks_with_engine, state: IngestState) -> int:
    if not chunks_with_engine:
        return 0
    rel_path = state["rel_path"]
    file_type = state["file_type"]
    queue_name = state["queue_name"]
    start_idx = state["total_chunks_sent"]
    entries = []
    for i, (chunk, engine, page) in enumerate(chunks_with_engine):
        idx = start_idx + i
        entry = {
            "chunk": chunk,
            "id": make_chunk_id(rel_path, idx, chunk),
            "source_file": rel_path,
            "document_id": state["document_id"],
            "type": file_type,
            "hash": hashlib.md5(chunk.encode()).hexdigest(),
            "engine": engine,
            "page": page,
            "chunk_index": idx,
        }
        entries.append(json.dumps(TextProcessor.normalize_metadata(entry)))
    redis_client = get_redis_client()
    blocking_push_with_backpressure(rclient=redis_client, queue_name=queue_name, entries=entries, max_queue_length=50000, rel_path=rel_path)
    return len(entries)


def markdown_extract_node(state: IngestState) -> IngestState:
    """Extracts and splits content from normalized Markdown files."""
    if state["status"] == STATUS_FAILED:
        return state
    full_path = state["full_path"]
    rel_path = state["rel_path"]
    queue_name = state["queue_name"]

    log.info(f"📝 [Node: Markdown Extract] {rel_path}")
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks, metadatas = split_markdown_doc(text, rel_path)
        if not chunks:
            return {**state, "status": STATUS_FAILED, "error": "Empty Markdown or split failed"}

        total_chunks = len(chunks)
        entries = []
        for idx, (chunk_text, meta) in enumerate(zip(chunks, metadatas)):
            # Combine with document_id and ensure payload matches schema
            payload = {
                "chunk": chunk_text,
                "id": make_chunk_id(rel_path, idx, chunk_text),
                "source_file": rel_path,
                "document_id": state["document_id"],
                "type": "markdown",
                "hash": hashlib.md5(chunk_text.encode()).hexdigest(),
                "engine": "gatekeeper_normalized",
                "page": meta.get("page", -1),
                "chunk_index": idx,
                "metadata": meta,  # Full YAML and header context
            }
            entries.append(json.dumps(TextProcessor.normalize_metadata(payload)))

        redis_client = get_redis_client()
        blocking_push_with_backpressure(rclient=redis_client, queue_name=queue_name, entries=entries, max_queue_length=50000, rel_path=rel_path)

        return {**state, "total_chunks_sent": total_chunks, "status": STATUS_PROCESSING}
    except Exception as e:
        log.error(f"💥 Failed to process Markdown {rel_path}: {e}")
        return {**state, "status": STATUS_FAILED, "error": str(e)}


def html_extract_node(state: IngestState) -> IngestState:
    if state["status"] == STATUS_FAILED:
        return state
    full_path = state["full_path"]
    rel_path = state["rel_path"]
    document_id = state["document_id"]

    log.info(f"🌐 [Node: HTML Extract] {rel_path}")
    text = extract_text_from_html(full_path)
    if not text or len(text.strip()) < 10:
        return {**state, "status": STATUS_FAILED, "error": "Empty HTML"}
    chunks, _ = split_doc(text, rel_path, "html", page_num=-1, document_id=document_id)
    chunks_with_engine = [(c, "html", -1) for c in chunks]
    count = stream_chunks_to_redis(chunks_with_engine, state)
    return {**state, "total_chunks_sent": count, "pages_processed": 1}


def media_extract_node(state: IngestState) -> IngestState:
    if state["status"] == STATUS_FAILED:
        return state
    full_path = state["full_path"]
    rel_path = state["rel_path"]
    document_id = state["document_id"]

    log.info(f"🎥 [Node: Media Extract] {rel_path}")
    try:
        segments = extract_text_from_media(full_path)
        if not segments:
            return {**state, "status": STATUS_FAILED, "error": "Transcription failed"}
        text = " ".join([s["text"] for s in segments])
        if not text or len(text.strip()) < 10:
            return {**state, "status": STATUS_FAILED, "error": "Empty text"}
        chunks, _ = split_doc(text, rel_path, "media", page_num=-1, document_id=document_id)
        chunks_with_engine = [(c, "whisper", -1) for c in chunks]
        count = stream_chunks_to_redis(chunks_with_engine, state)
        return {**state, "total_chunks_sent": count, "pages_processed": 1}
    except Exception as e:
        return {**state, "status": STATUS_FAILED, "error": str(e)}


def send_sentinel_node(state: IngestState) -> IngestState:
    if state["status"] == STATUS_FAILED:
        return state
    rel_path = state["rel_path"]
    queue_name = state["queue_name"]
    total_sent = state["total_chunks_sent"]
    try:
        redis_client = get_redis_client()
        blocking_push_with_backpressure(rclient=redis_client, queue_name=queue_name, entries=[json.dumps({"type": "file_end", "source_file": rel_path, "expected_chunks": total_sent})], max_queue_length=50000, rel_path=rel_path)
        return {**state, "status": STATUS_ENQUEUED}
    except Exception as e:
        return {**state, "status": STATUS_FAILED, "error": str(e)}


def finalize_state_node(state: IngestState) -> IngestState:
    rel_path = state["rel_path"]
    status = state["status"]
    error = state["error"]
    document_id = state["document_id"]
    log.info(f"🏁 [Node: Finalize] {rel_path}: {status}")
    update_job_status(rel_path, status, state["job_id"], error, document_id)
    return state


def create_producer_graph():
    workflow = StateGraph(IngestState)
    workflow.add_node("scan_file_node", scan_file_node)
    workflow.add_node("markdown_extract_node", markdown_extract_node)
    workflow.add_node("html_extract_node", html_extract_node)
    workflow.add_node("media_extract_node", media_extract_node)
    workflow.add_node("send_sentinel_node", send_sentinel_node)
    workflow.add_node("finalize_state_node", finalize_state_node)
    workflow.set_entry_point("scan_file_node")

    def route_scan(s):
        if s["status"] == STATUS_FAILED:
            return "finalize_state_node"
        if s["file_type"] == "markdown":
            return "markdown_extract_node"
        if s["file_type"] == "html":
            return "html_extract_node"
        return "media_extract_node"

    workflow.add_conditional_edges(
        "scan_file_node",
        route_scan,
        {
            "markdown_extract_node": "markdown_extract_node",
            "html_extract_node": "html_extract_node",
            "media_extract_node": "media_extract_node",
            "finalize_state_node": "finalize_state_node",
        },
    )

    workflow.add_conditional_edges("markdown_extract_node", lambda s: "finalize_state_node" if s["status"] == STATUS_FAILED else "send_sentinel_node", {"send_sentinel_node": "send_sentinel_node", "finalize_state_node": "finalize_state_node"})

    workflow.add_conditional_edges("html_extract_node", lambda s: "finalize_state_node" if s["status"] == STATUS_FAILED else "send_sentinel_node", {"send_sentinel_node": "send_sentinel_node", "finalize_state_node": "finalize_state_node"})

    workflow.add_conditional_edges("media_extract_node", lambda s: "finalize_state_node" if s["status"] == STATUS_FAILED else "send_sentinel_node", {"send_sentinel_node": "send_sentinel_node", "finalize_state_node": "finalize_state_node"})

    workflow.add_edge("send_sentinel_node", "finalize_state_node")
    workflow.add_edge("finalize_state_node", END)
    return workflow.compile()


def get_producer_app():
    global _COMPILED_PRODUCER_APP
    if _COMPILED_PRODUCER_APP is None:
        log.info("🔨 Compiling Shared Producer LangGraph...")
        _COMPILED_PRODUCER_APP = create_producer_graph()
    return _COMPILED_PRODUCER_APP


def run_ingest_graph(job_tuple: tuple, gpu_lock_obj=None) -> bool:
    global _GPU_LOCK
    _GPU_LOCK = gpu_lock_obj
    job_id, source_file, rel_path = job_tuple

    initial_state: IngestState = {
        "job_id": job_id,
        "full_path": source_file,
        "rel_path": rel_path,
        "file_type": "unknown",
        "total_chunks_sent": 0,
        "queue_name": None,
        "status": STATUS_PROCESSING,
        "error": None,
        "document_id": None,
        "file_metadata": None,
        "metrics": None,
        "pages_processed": 0,
    }

    try:
        app = get_producer_app()
        final_state = app.invoke(initial_state)
        return final_state["status"] == STATUS_ENQUEUED
    except Exception as e:
        log.error(f"💥 Graph fatal error: {e}")
        return False
