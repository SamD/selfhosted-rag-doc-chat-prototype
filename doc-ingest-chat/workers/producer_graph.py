#!/usr/bin/env python3
"""
LangGraph implementation of the producer ingestion workflow.
Refactored for STREAMING extraction and SINGLE compilation.
"""

import hashlib
import json
import os
import threading
from typing import Any, Optional, TypedDict

from config.settings import SUPPORTED_MEDIA_EXT
from langgraph.graph import END, StateGraph
from processors.text_processor import TextProcessor, make_chunk_id, split_doc
from services.job_service import STATUS_ENQUEUED, STATUS_FAILED, STATUS_PROCESSING, update_job_status
from services.redis_service import get_redis_client
from utils.logging_config import setup_logging
from utils.producer_utils import (
    blocking_push_with_backpressure,
    extract_text_from_html,
    extract_text_from_media,
    fallback_ocr,
    process_pdf_by_page,
)
from utils.text_utils import is_valid_pdf
from workers.producer_worker import get_next_queue

log = setup_logging("producer_graph.log")

# Global cache for the compiled graph
_COMPILED_PRODUCER_APP = None
_APP_LOCK = threading.Lock()

class IngestState(TypedDict):
    """Represents the internal state of a single file ingestion job."""
    job_id: str
    full_path: str
    rel_path: str
    file_type: str
    total_chunks_sent: int
    queue_name: Optional[str]
    requires_ocr: bool
    status: str
    error: Optional[str]
    metrics: Optional[Any]
    pages_processed: int

def scan_file_node(state: IngestState) -> IngestState:
    full_path = state["full_path"]
    rel_path = state["rel_path"]
    log.info(f"🔍 [Node: Scan] {rel_path}")
    update_job_status(rel_path, STATUS_PROCESSING, state["job_id"])
    if not os.path.exists(full_path):
        return {**state, "status": STATUS_FAILED, "error": "File does not exist"}
    file_ext = os.path.splitext(full_path)[1].lower()
    if file_ext == ".pdf":
        file_type = "pdf"
    elif file_ext in (".html", ".htm"):
        file_type = "html"
    elif file_ext in SUPPORTED_MEDIA_EXT:
        file_type = "media"
    else:
        return {**state, "status": STATUS_FAILED, "error": f"Unsupported file type: {file_ext}"}
    return {**state, "file_type": file_type, "status": STATUS_PROCESSING, "queue_name": get_next_queue()}

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
            "type": file_type,
            "hash": hashlib.md5(chunk.encode()).hexdigest(),
            "engine": engine,
            "page": page,
            "chunk_index": idx,
        }
        entries.append(json.dumps(TextProcessor.normalize_metadata(entry)))
    redis_client = get_redis_client()
    blocking_push_with_backpressure(
        rclient=redis_client,
        queue_name=queue_name,
        entries=entries,
        max_queue_length=50000,
        rel_path=rel_path
    )
    return len(entries)

def pdf_extract_node(state: IngestState) -> IngestState:
    full_path = state["full_path"]
    rel_path = state["rel_path"]
    log.info(f"📄 [Node: PDF Extract] {rel_path} (Parallel Streaming)")
    if not is_valid_pdf(full_path):
        return {**state, "status": STATUS_FAILED, "error": "Invalid PDF"}
    total_sent = 0
    pages_seen = set()
    counter_lock = threading.Lock()
    def on_chunks(batch):
        nonlocal total_sent
        with counter_lock:
            count = stream_chunks_to_redis(batch, {**state, "total_chunks_sent": total_sent})
            total_sent += count
            for _, _, p in batch:
                pages_seen.add(p)
    try:
        process_pdf_by_page(full_path, rel_path, "pdf", chunk_callback=on_chunks)
        if total_sent == 0:
            return {**state, "requires_ocr": True}
        return {**state, "total_chunks_sent": total_sent, "pages_processed": len(pages_seen), "requires_ocr": False}
    except Exception:
        return {**state, "requires_ocr": True}

def html_extract_node(state: IngestState) -> IngestState:
    full_path = state["full_path"]
    rel_path = state["rel_path"]
    log.info(f"🌐 [Node: HTML Extract] {rel_path}")
    text = extract_text_from_html(full_path)
    if not text or len(text.strip()) < 10:
        return {**state, "status": STATUS_FAILED, "error": "Empty or short HTML content"}
    chunks, _ = split_doc(text, rel_path, "html", page_num=-1)
    chunks_with_engine = [(c, "html", -1) for c in chunks]
    count = stream_chunks_to_redis(chunks_with_engine, state)
    return {**state, "total_chunks_sent": count, "pages_processed": 1}

def media_extract_node(state: IngestState) -> IngestState:
    full_path = state["full_path"]
    rel_path = state["rel_path"]
    log.info(f"🎥 [Node: Media Extract] {rel_path}")
    try:
        segments = extract_text_from_media(full_path)
        if not segments:
            return {**state, "status": STATUS_FAILED, "error": "Transcription failed"}
        text = " ".join([s["text"] for s in segments])
        if not text or len(text.strip()) < 10:
            return {**state, "status": STATUS_FAILED, "error": "Transcription produced empty text"}
        chunks, _ = split_doc(text, rel_path, "media", page_num=-1)
        chunks_with_engine = [(c, "whisper", -1) for c in chunks]
        count = stream_chunks_to_redis(chunks_with_engine, state)
        return {**state, "total_chunks_sent": count, "pages_processed": 1}
    except Exception as e:
        return {**state, "status": STATUS_FAILED, "error": f"Transcription failed: {e}"}

def fallback_ocr_node(state: IngestState) -> IngestState:
    full_path = state["full_path"]
    rel_path = state["rel_path"]
    log.info(f"📸 [Node: OCR Fallback] {rel_path} (Parallel Streaming)")
    total_sent = 0
    pages_seen = set()
    counter_lock = threading.Lock()
    def on_chunks(batch):
        nonlocal total_sent
        with counter_lock:
            count = stream_chunks_to_redis(batch, {**state, "total_chunks_sent": total_sent})
            total_sent += count
            for _, _, p in batch:
                pages_seen.add(p)
    try:
        fallback_ocr(full_path, rel_path, state["job_id"], chunk_callback=on_chunks)
        if total_sent == 0:
            return {**state, "status": STATUS_FAILED, "error": "OCR failed to produce any usable text"}
        return {**state, "total_chunks_sent": total_sent, "pages_processed": len(pages_seen)}
    except Exception as e:
        return {**state, "status": STATUS_FAILED, "error": f"OCR failed: {e}"}

def send_sentinel_node(state: IngestState) -> IngestState:
    if state["status"] == STATUS_FAILED:
        return state
    rel_path = state["rel_path"]
    queue_name = state["queue_name"]
    total_sent = state["total_chunks_sent"]
    try:
        redis_client = get_redis_client()
        blocking_push_with_backpressure(
            rclient=redis_client,
            queue_name=queue_name,
            entries=[json.dumps({"type": "file_end", "source_file": rel_path, "expected_chunks": total_sent})],
            max_queue_length=50000,
            rel_path=rel_path
        )
        return {**state, "status": STATUS_ENQUEUED}
    except Exception as e:
        return {**state, "status": STATUS_FAILED, "error": f"Sentinel failed: {e}"}

def finalize_state_node(state: IngestState) -> IngestState:
    rel_path = state["rel_path"]
    status = state["status"]
    error = state["error"]
    log.info(f"🏁 [Node: Finalize] {rel_path}: {status}")
    update_job_status(rel_path, status, state["job_id"], error)
    return state

def create_producer_graph():
    workflow = StateGraph(IngestState)
    workflow.add_node("scan_file_node", scan_file_node)
    workflow.add_node("pdf_extract_node", pdf_extract_node)
    workflow.add_node("html_extract_node", html_extract_node)
    workflow.add_node("media_extract_node", media_extract_node)
    workflow.add_node("fallback_ocr_node", fallback_ocr_node)
    workflow.add_node("send_sentinel_node", send_sentinel_node)
    workflow.add_node("finalize_state_node", finalize_state_node)
    workflow.set_entry_point("scan_file_node")
    
    def route_scan(s):
        if s["status"] == STATUS_FAILED:
            return "finalize_state_node"
        if s["file_type"] == "pdf":
            return "pdf_extract_node"
        if s["file_type"] == "html":
            return "html_extract_node"
        return "media_extract_node"
        
    workflow.add_conditional_edges("scan_file_node", route_scan, {
        "pdf_extract_node": "pdf_extract_node",
        "html_extract_node": "html_extract_node",
        "media_extract_node": "media_extract_node",
        "finalize_state_node": "finalize_state_node"
    })
    workflow.add_conditional_edges("pdf_extract_node", lambda s: "finalize_state_node" if s["status"] == STATUS_FAILED else ("fallback_ocr_node" if s["requires_ocr"] else "send_sentinel_node"), {
        "fallback_ocr_node": "fallback_ocr_node",
        "send_sentinel_node": "send_sentinel_node",
        "finalize_state_node": "finalize_state_node"
    })
    workflow.add_conditional_edges("html_extract_node", lambda s: "finalize_state_node" if s["status"] == STATUS_FAILED else "send_sentinel_node", {
        "send_sentinel_node": "send_sentinel_node",
        "finalize_state_node": "finalize_state_node"
    })
    workflow.add_conditional_edges("media_extract_node", lambda s: "finalize_state_node" if s["status"] == STATUS_FAILED else "send_sentinel_node", {
        "send_sentinel_node": "send_sentinel_node",
        "finalize_state_node": "finalize_state_node"
    })
    workflow.add_conditional_edges("fallback_ocr_node", lambda s: "finalize_state_node" if s["status"] == STATUS_FAILED else "send_sentinel_node", {
        "send_sentinel_node": "send_sentinel_node",
        "finalize_state_node": "finalize_state_node"
    })
    workflow.add_edge("send_sentinel_node", "finalize_state_node")
    workflow.add_edge("finalize_state_node", END)
    return workflow.compile()

def get_producer_app():
    global _COMPILED_PRODUCER_APP
    if _COMPILED_PRODUCER_APP is None:
        with _APP_LOCK:
            if _COMPILED_PRODUCER_APP is None:
                log.info("🔨 Compiling Producer LangGraph...")
                _COMPILED_PRODUCER_APP = create_producer_graph()
    return _COMPILED_PRODUCER_APP

def run_ingest_graph(job_tuple: tuple) -> bool:
    job_id, source_file, rel_path = job_tuple
    initial_state: IngestState = {
        "job_id": job_id,
        "full_path": source_file,
        "rel_path": rel_path,
        "file_type": "unknown",
        "total_chunks_sent": 0,
        "queue_name": None,
        "requires_ocr": False,
        "status": STATUS_PROCESSING,
        "error": None,
        "metrics": None,
        "pages_processed": 0
    }
    try:
        app = get_producer_app()
        final_state = app.invoke(initial_state)
        return final_state["status"] == STATUS_ENQUEUED
    except Exception as e:
        log.error(f"💥 Graph fatal error: {e}")
        return False
