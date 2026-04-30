#!/usr/bin/env python3
"""
LangGraph implementation of the producer ingestion workflow.
Includes deterministic document_id generation and per-chunk enrichment.
"""

import hashlib
import json
import os
from typing import List, Optional, TypedDict

from config.settings import MAX_TOKENS, SUPPORTED_MEDIA_EXT
from langgraph.graph import END, StateGraph
from processors.text_processor import TextProcessor, make_chunk_id, split_markdown_doc
from services.job_service import STATUS_FAILED
from services.redis_service import get_redis_client
from utils.producer_utils import (
    blocking_push_with_backpressure,
    handle_error,
    send_file_end_sentinel,
)
from utils.trace_utils import get_logger, get_trace_id, set_trace_id

log = get_logger("ingest.producer_graph")


class IngestState(TypedDict):
    """Represents the internal state of the ingestion graph."""

    full_path: str
    rel_path: str
    job_id: str
    trace_id: Optional[str]
    queue_name: str
    document_id: Optional[str]
    file_type: str
    chunks: List[str]
    metadata: List[dict]
    status: str
    error: Optional[str]


def scan_file_node(state: IngestState) -> IngestState:
    """Initial validation and type detection for normalized documents."""
    full_path = state["full_path"]
    rel_path = state["rel_path"]
    
    # Retrieve trace_id from DB if not present in state
    trace_id = state.get("trace_id")
    if not trace_id:
        try:
            from services.job_service import JobService
            query = "SELECT trace_id FROM ingestion_lifecycle WHERE id = ?"
            res, _ = JobService._execute_with_retry(query, (state["job_id"],), fetch=True)
            if res:
                trace_id = res[0]
        except Exception:
            trace_id = "UNKNOWN"
    
    set_trace_id(trace_id)
    log.info(f"🔍 [Node: Scan] {rel_path}")

    if not os.path.exists(full_path):
        return {**state, "status": STATUS_FAILED, "error": f"File does not exist: {full_path}"}

    file_ext = os.path.splitext(full_path)[1].lower()
    if file_ext != ".md":
        # We only process .md files from the gatekeeper now
        if file_ext in (".html", ".htm"):
            file_type = "html"
        elif file_ext in SUPPORTED_MEDIA_EXT:
            file_type = "media"
        else:
            return {**state, "status": STATUS_FAILED, "error": f"Unsupported extension: {file_ext}"}
    else:
        file_type = "markdown"

    return {**state, "file_type": file_type, "status": "processing"}


def _push_chunks_to_redis(state: IngestState, chunks_with_engine: List[tuple]) -> int:
    """Helper to format and push chunks to partitioned Redis queues."""
    queue_name = state["queue_name"]
    rel_path = state["rel_path"]
    file_type = state["file_type"]

    entries = []
    start_idx = 0
    doc_id = state["document_id"]
    trace_id = state.get("trace_id") or get_trace_id()
    # Mandatory RAG prefix for Vector DB consistency
    enrichment_prefix = f"passage: [{doc_id}] "

    for i, (chunk_text, engine, page) in enumerate(chunks_with_engine):
        idx = start_idx + i

        # WE PREPEND THE PREFIX HERE: This is what the Consumer will validate
        # and what Qdrant will store.
        final_chunk = f"{enrichment_prefix}{chunk_text}"

        entry = {
            "chunk": final_chunk,
            "id": make_chunk_id(rel_path, idx, final_chunk, document_id=doc_id),
            "source_file": rel_path,
            "document_id": doc_id,
            "trace_id": trace_id,
            "source_type": file_type,
            "chunk_format": "MD" if file_type == "markdown" else "TEXT",
            "hash": hashlib.md5(final_chunk.encode()).hexdigest(),
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

    log.info(f"📝 [Node: Markdown Extract] {rel_path}")
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Deterministic document_id from normalized content
        doc_id = TextProcessor.get_document_id(text.encode())

        # Use new prefix-aware splitting to prevent Consumer dropping oversized chunks
        try:
            chunks, metadatas = split_markdown_doc(text, rel_path, budget=MAX_TOKENS, prefix="passage: ", document_id=doc_id)
        except Exception as e:
            from utils.exceptions import ConfigurationError

            if isinstance(e, ConfigurationError):
                error_msg = f"Configuration Missing: {str(e)}"
                log.error(f"🛑 {error_msg}")
                return {**state, "status": STATUS_FAILED, "error": error_msg}
            raise e
        if not chunks:
            return {**state, "status": STATUS_FAILED, "error": "Empty Markdown or split failed"}

        total_chunks = len(chunks)

        # In Markdown mode, headers are preserved in metadatas, but we use a simple tuple format
        # compatible with the _push helper
        chunks_with_engine = []
        for i, chunk in enumerate(chunks):
            # Extract page from metadata if available (added by Gatekeeper in YAML)
            page = metadatas[i].get("page", 1)
            chunks_with_engine.append((chunk, "llamacpp", page))

        pushed_count = _push_chunks_to_redis({**state, "document_id": doc_id}, chunks_with_engine)
        log.info(f"🚀 [Node: Markdown] Pushed {pushed_count}/{total_chunks} chunks for {rel_path}")

        return {**state, "status": "processing", "document_id": doc_id, "chunks": chunks}
    except Exception as e:
        return handle_error(state, f"Markdown processing error: {e}", log)


def html_extract_node(state: IngestState) -> IngestState:
    """HTML processing node (Placeholder for now)."""
    if state["status"] == STATUS_FAILED:
        return state
    return {**state, "status": STATUS_FAILED, "error": "HTML extraction not yet implemented in LangGraph mode"}


def media_extract_node(state: IngestState) -> IngestState:
    """Media (Video/Audio) processing node (Placeholder for now)."""
    if state["status"] == STATUS_FAILED:
        return state
    return {**state, "status": STATUS_FAILED, "error": "Media transcription not yet implemented in LangGraph mode"}


def send_sentinel_node(state: IngestState) -> IngestState:
    """Sends the file_end sentinel to the Consumer."""
    if state["status"] == STATUS_FAILED:
        return state

    log.info(f"🏁 [Node: Finalize] Sending sentinel for {state['rel_path']}")
    try:
        redis_client = get_redis_client()
        send_file_end_sentinel(
            rclient=redis_client,
            queue_name=state["queue_name"],
            source_file=state["rel_path"],
            total_chunks=len(state["chunks"]),
        )
        return {**state, "status": "processing"}
    except Exception as e:
        return handle_error(state, f"Sentinel error: {e}", log)


def finalize_state_node(state: IngestState) -> IngestState:
    """Terminal node. (State transitions are managed by the calling Worker)."""
    return state


def get_producer_app():
    """Compiles the Producer LangGraph."""
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

    workflow.add_conditional_edges("scan_file_node", route_scan, {"markdown_extract_node": "markdown_extract_node", "html_extract_node": "html_extract_node", "media_extract_node": "media_extract_node", "finalize_state_node": "finalize_state_node"})

    workflow.add_edge("markdown_extract_node", "send_sentinel_node")
    workflow.add_edge("html_extract_node", "send_sentinel_node")
    workflow.add_edge("media_extract_node", "send_sentinel_node")

    workflow.add_edge("send_sentinel_node", "finalize_state_node")
    workflow.add_edge("finalize_state_node", END)
    return workflow.compile()


def run_ingest_graph(job_tuple, gpu_lock_obj=None) -> bool:
    """Invokes the compiled LangGraph for a single document."""
    job_id, full_path, rel_path = job_tuple
    from workers.producer_worker import get_next_queue

    initial_state: IngestState = {
        "full_path": full_path,
        "rel_path": rel_path,
        "job_id": job_id,
        "queue_name": get_next_queue(),
        "chunks": [],
        "metadata": [],
        "status": "pending",
        "error": None,
        "document_id": None,
        "file_type": "unknown",
    }

    try:
        app = get_producer_app()
        final_state = app.invoke(initial_state)
        return final_state["status"] == "processing"
    except Exception as e:
        log.error(f"💥 Graph fatal error: {e}")
        return False
