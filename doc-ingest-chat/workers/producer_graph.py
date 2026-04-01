#!/usr/bin/env python3
"""
LangGraph implementation of the producer ingestion workflow.
Includes a Supervisor Agent (Qwen2.5) for Document Context Enrichment.
Optimized with a Multiprocessing GPU Lock to prevent deadlocks.
"""

import hashlib
import json
import logging
import os
from typing import Any, Optional, TypedDict

import pdfplumber
from config.settings import SUPPORTED_MEDIA_EXT
from langgraph.graph import END, StateGraph
from processors.text_processor import TextProcessor, make_chunk_id, split_doc
from services.job_service import STATUS_ENQUEUED, STATUS_FAILED, STATUS_PROCESSING, update_job_status
from services.redis_service import get_redis_client
from utils.llm_setup import get_supervisor_llm
from utils.producer_utils import (
    blocking_push_with_backpressure,
    extract_text_from_html,
    extract_text_from_media,
    fallback_ocr,
    process_pdf_by_page,
)
from utils.text_utils import is_valid_pdf
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
    requires_ocr: bool
    status: str
    error: Optional[str]
    doc_preview: Optional[str]
    doc_context: Optional[str]
    metrics: Optional[Any]
    pages_processed: int

def scan_file_node(state: IngestState) -> IngestState:
    """Initial validation and type detection."""
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

def preview_node(state: IngestState) -> IngestState:
    """Extracts a small sample of text for the Supervisor Agent to analyze."""
    full_path = state["full_path"]
    file_type = state["file_type"]
    preview_text = ""
    
    try:
        if file_type == "pdf":
            with pdfplumber.open(full_path) as pdf:
                for i in range(min(10, len(pdf.pages))):
                    text = pdf.pages[i].extract_text()
                    if text and len(text.strip()) > 20:
                        preview_text += text + "\n"
                    if len(preview_text) > 500:
                        break
        elif file_type == "html":
            preview_text = extract_text_from_html(full_path)
            
        if not preview_text or len(preview_text.strip()) < 20:
            log.error(f"❌ [Node: Preview] Failed for {state['rel_path']}: No extractable text found.")
            return {**state, "status": STATUS_FAILED, "error": "Insufficient text for context"}

        return {**state, "doc_preview": (preview_text or "")[:2000]}
    except Exception as e:
        log.error(f"💥 [Node: Preview] Critical failure: {e}")
        return {**state, "status": STATUS_FAILED, "error": f"Preview failed: {e}"}

def supervisor_node(state: IngestState) -> IngestState:
    """Uses Qwen2.5-1.5B with EXCLUSIVE GPU lock to generate context."""
    if state["status"] == STATUS_FAILED:
        return state

    preview = state.get("doc_preview")
    rel_path = state["rel_path"]
    
    import time
    start_time = time.perf_counter()
    
    # CRITICAL: Use the global multiprocessing lock to prevent GPU race conditions
    global _GPU_LOCK
    # If no multiprocessing lock is available (e.g. unit tests), we fall back to no-op context
    # instead of a useless threading lock.
    from contextlib import nullcontext
    lock_ctx = _GPU_LOCK if _GPU_LOCK else nullcontext()
    
    log.info(f"🕵️ [Supervisor] Waiting for GPU lock for {rel_path}...")
    with lock_ctx:
        log.info(f"🕵️ [Supervisor] Analyzing {rel_path}...")
        try:
            llm = get_supervisor_llm()
            prompt = (
                "<|im_start|>system\nYou are a precise metadata extractor. Describe the document in ONE concise sentence. "
                "ONLY output the sentence.<|im_end|>\n"
                f"<|im_start|>user\nText sample:\n\n{preview}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
            response = llm.invoke(prompt)
            raw_resp = (response or "").strip()
            
            if not raw_resp or len(raw_resp) < 5:
                 return {**state, "status": STATUS_FAILED, "error": "LLM failed"}

            context = raw_resp.split("\n")[0].replace("This document is ", "")
            duration = time.perf_counter() - start_time
            
            msg = f"🕵️ [Supervisor] Final Context for {rel_path} (took {duration:.2f}s): {context}"
            logging.getLogger("ingest").info(msg)
            print(msg, flush=True)
            
            return {**state, "doc_context": context}
        except Exception as e:
            log.error(f"🕵️ [Supervisor] 💥 Node failed: {e}")
            return {**state, "status": STATUS_FAILED, "error": str(e)}

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
    if state["status"] == STATUS_FAILED:
        return state
    full_path = state["full_path"]
    rel_path = state["rel_path"]
    doc_context = state["doc_context"]
    log.info(f"📄 [Node: PDF Extract] {rel_path} (Context: {doc_context})")
    if not is_valid_pdf(full_path):
        return {**state, "status": STATUS_FAILED, "error": "Invalid PDF"}
    total_sent = 0
    pages_seen = set()
    def on_chunks(batch):
        nonlocal total_sent
        enriched_batch = []
        for chunk, engine, page in batch:
            enriched_text = f"[Document: {doc_context}] {chunk}"
            enriched_batch.append((enriched_text, engine, page))
        if total_sent == 0 and enriched_batch:
            sample = enriched_batch[0][0]
            msg = f"✨ [Enrichment Sample] {rel_path} Chunk 0: {sample[:150]}..."
            logging.getLogger("ingest").info(msg)
            print(msg, flush=True)
        count = stream_chunks_to_redis(enriched_batch, {**state, "total_chunks_sent": total_sent})
        total_sent += count
        for _, _, p in batch:
            pages_seen.add(p)
    try:
        process_pdf_by_page(full_path, rel_path, "pdf", chunk_callback=on_chunks, doc_context=doc_context)
        if total_sent == 0:
            return {**state, "requires_ocr": True}
        return {**state, "total_chunks_sent": total_sent, "pages_processed": len(pages_seen), "requires_ocr": False}
    except Exception:
        return {**state, "requires_ocr": True}

def html_extract_node(state: IngestState) -> IngestState:
    if state["status"] == STATUS_FAILED:
        return state
    full_path = state["full_path"]
    rel_path = state["rel_path"]
    doc_context = state["doc_context"]
    log.info(f"🌐 [Node: HTML Extract] {rel_path}")
    text = extract_text_from_html(full_path)
    if not text or len(text.strip()) < 10:
        return {**state, "status": STATUS_FAILED, "error": "Empty HTML"}
    chunks, _ = split_doc(text, rel_path, "html", page_num=-1, doc_context=doc_context)
    enriched_chunks = [f"[Document: {doc_context}] {c}" if not c.startswith("[Document:") else c for c in chunks]
    chunks_with_engine = [(c, "html", -1) for c in enriched_chunks]
    count = stream_chunks_to_redis(chunks_with_engine, state)
    return {**state, "total_chunks_sent": count, "pages_processed": 1}

def media_extract_node(state: IngestState) -> IngestState:
    if state["status"] == STATUS_FAILED:
        return state
    full_path = state["full_path"]
    rel_path = state["rel_path"]
    doc_context = state["doc_context"]
    log.info(f"🎥 [Node: Media Extract] {rel_path}")
    try:
        segments = extract_text_from_media(full_path)
        if not segments:
            return {**state, "status": STATUS_FAILED, "error": "Transcription failed"}
        text = " ".join([s["text"] for s in segments])
        if not text or len(text.strip()) < 10:
            return {**state, "status": STATUS_FAILED, "error": "Empty text"}
        chunks, _ = split_doc(text, rel_path, "media", page_num=-1, doc_context=doc_context)
        enriched_chunks = [f"[Document: {doc_context}] {c}" if not c.startswith("[Document:") else c for c in chunks]
        chunks_with_engine = [(c, "whisper", -1) for c in enriched_chunks]
        count = stream_chunks_to_redis(chunks_with_engine, state)
        return {**state, "total_chunks_sent": count, "pages_processed": 1}
    except Exception as e:
        return {**state, "status": STATUS_FAILED, "error": str(e)}

def fallback_ocr_node(state: IngestState) -> IngestState:
    if state["status"] == STATUS_FAILED:
        return state
    full_path = state["full_path"]
    rel_path = state["rel_path"]
    doc_context = state["doc_context"]
    log.info(f"📸 [Node: OCR Fallback] {rel_path}")
    total_sent = 0
    pages_seen = set()
    def on_chunks(batch):
        nonlocal total_sent
        enriched_batch = []
        for chunk, engine, page in batch:
            enriched_text = f"[Document: {doc_context}] {chunk}"
            enriched_batch.append((enriched_text, engine, page))
        count = stream_chunks_to_redis(enriched_batch, {**state, "total_chunks_sent": total_sent})
        total_sent += count
        for _, _, p in batch:
            pages_seen.add(p)
    try:
        fallback_ocr(full_path, rel_path, state["job_id"], chunk_callback=on_chunks, doc_context=doc_context)
        if total_sent == 0:
            return {**state, "status": STATUS_FAILED, "error": "OCR failed"}
        return {**state, "total_chunks_sent": total_sent, "pages_processed": len(pages_seen)}
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
        blocking_push_with_backpressure(
            rclient=redis_client,
            queue_name=queue_name,
            entries=[json.dumps({"type": "file_end", "source_file": rel_path, "expected_chunks": total_sent})],
            max_queue_length=50000,
            rel_path=rel_path
        )
        return {**state, "status": STATUS_ENQUEUED}
    except Exception as e:
        return {**state, "status": STATUS_FAILED, "error": str(e)}

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
    workflow.add_node("preview_node", preview_node)
    workflow.add_node("supervisor_node", supervisor_node)
    workflow.add_node("pdf_extract_node", pdf_extract_node)
    workflow.add_node("html_extract_node", html_extract_node)
    workflow.add_node("media_extract_node", media_extract_node)
    workflow.add_node("fallback_ocr_node", fallback_ocr_node)
    workflow.add_node("send_sentinel_node", send_sentinel_node)
    workflow.add_node("finalize_state_node", finalize_state_node)
    workflow.set_entry_point("scan_file_node")
    workflow.add_edge("scan_file_node", "preview_node")
    workflow.add_edge("preview_node", "supervisor_node")
    
    def route_scan(s):
        if s["status"] == STATUS_FAILED:
            return "finalize_state_node"
        if s["file_type"] == "pdf":
            return "pdf_extract_node"
        if s["file_type"] == "html":
            return "html_extract_node"
        return "media_extract_node"
        
    workflow.add_conditional_edges(
        "supervisor_node", 
        route_scan, 
        {
            "pdf_extract_node": "pdf_extract_node", 
            "html_extract_node": "html_extract_node", 
            "media_extract_node": "media_extract_node", 
            "finalize_state_node": "finalize_state_node"
        }
    )
    
    workflow.add_conditional_edges(
        "pdf_extract_node", 
        lambda s: "finalize_state_node" if s["status"] == STATUS_FAILED else (
            "fallback_ocr_node" if s["requires_ocr"] else "send_sentinel_node"
        ), 
        {
            "fallback_ocr_node": "fallback_ocr_node", 
            "send_sentinel_node": "send_sentinel_node", 
            "finalize_state_node": "finalize_state_node"
        }
    )
    
    workflow.add_conditional_edges(
        "html_extract_node", 
        lambda s: "finalize_state_node" if s["status"] == STATUS_FAILED else "send_sentinel_node", 
        {
            "send_sentinel_node": "send_sentinel_node", 
            "finalize_state_node": "finalize_state_node"
        }
    )
    
    workflow.add_conditional_edges(
        "media_extract_node", 
        lambda s: "finalize_state_node" if s["status"] == STATUS_FAILED else "send_sentinel_node", 
        {
            "send_sentinel_node": "send_sentinel_node", 
            "finalize_state_node": "finalize_state_node"
        }
    )
    
    workflow.add_conditional_edges(
        "fallback_ocr_node", 
        lambda s: "finalize_state_node" if s["status"] == STATUS_FAILED else "send_sentinel_node", 
        {
            "send_sentinel_node": "send_sentinel_node", 
            "finalize_state_node": "finalize_state_node"
        }
    )
    
    workflow.add_edge("send_sentinel_node", "finalize_state_node")
    workflow.add_edge("finalize_state_node", END)
    return workflow.compile()

def get_producer_app():
    global _COMPILED_PRODUCER_APP
    if _COMPILED_PRODUCER_APP is None:
        log.info("🔨 Compiling Supervisor Producer LangGraph...")
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
        "requires_ocr": False, 
        "status": STATUS_PROCESSING, 
        "error": None, 
        "doc_preview": None, 
        "doc_context": "General Document", 
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
