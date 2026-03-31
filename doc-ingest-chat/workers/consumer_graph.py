#!/usr/bin/env python3
"""
LangGraph implementation of the consumer batch processing workflow.
Refactored for ZERO-MEMORY archival via DuckDB.
"""

import threading
from typing import Any, Dict, List, Optional, TypedDict

import logging
from config.settings import MAX_TOKENS
from langgraph.graph import END, StateGraph
from services.job_service import STATUS_COMPLETED, STATUS_FAILED, update_job_status
from services.parquet_service import commit_to_parquet
from utils.consumer_utils import store_chunks_in_db
from utils.metrics import FileMetrics

log = logging.getLogger("ingest.consumer_graph")

# Global cache for the compiled graph
_COMPILED_CONSUMER_APP = None
_APP_LOCK = threading.Lock()

class ConsumerState(TypedDict):
    """Represents the state for finalization of a file."""
    source_file: str
    expected_chunks: int
    chunks: List[Dict[str, Any]] # Remaining chunks not yet in VDB/DuckDB
    metrics: Optional[FileMetrics]
    status: str
    error: Optional[str]

def validate_remaining_chunks_node(state: ConsumerState) -> ConsumerState:
    """Validates remaining chunks and token limits."""
    source_file = state["source_file"]
    chunks = state["chunks"]
    
    log.info(f"🧐 [Node: Final Validate] {source_file} (Processing remaining {len(chunks)} chunks)")
    
    # We no longer fail if len(chunks) != expected here because 
    # many chunks were already processed incrementally.
    
    valid_chunks = []
    for entry in chunks:
        if not isinstance(entry.get("chunk"), str):
            continue
        if len(entry["chunk"]) > MAX_TOKENS:
            continue
        valid_chunks.append(entry)
        
    return {**state, "chunks": valid_chunks, "status": "processing"}

def store_final_chunks_node(state: ConsumerState) -> ConsumerState:
    """Stores any remaining chunks in Vector DB and DuckDB."""
    if state["status"] == STATUS_FAILED:
        return state
        
    try:
        from services.parquet_service import append_chunks
        
        # 1. Write to DuckDB (Persistent table) - SAFETY FIRST
        append_chunks(state["chunks"])
        
        # 2. Write to Qdrant (Vector DB)
        batches_count = store_chunks_in_db(state["source_file"], state["chunks"], state["metrics"])
        
        if state["metrics"]:
            state["metrics"].add_counter("batches_processed", batches_count)
            state["metrics"].add_counter("chunks_stored", len(state["chunks"]))
            
        return {**state, "status": "processing"}
    except Exception as e:
        log.error(f"💥 Final storage node failed: {e}")
        # update_failed_files(state["source_file"])
        return {**state, "status": STATUS_FAILED, "error": f"Storage error: {e}"}

def archival_export_node(state: ConsumerState) -> ConsumerState:
    """Triggers DuckDB to export its internal table to Parquet."""
    if state["status"] == STATUS_FAILED:
        return state
        
    try:
        # This is the key fix: It tells DuckDB to export the ALREADY SAVED data to Parquet.
        # Zero chunks are held in Python memory during this step.
        commit_to_parquet()
        # update_ingested_files(state["source_file"])
        return {**state, "status": STATUS_COMPLETED}
    except Exception as e:
        log.error(f"💥 Final archival export failed: {e}")
        return {**state, "status": STATUS_FAILED, "error": f"Parquet export failed: {e}"}

def finalize_consumer_node(state: ConsumerState) -> ConsumerState:
    """Final node to update the global Job status in DuckDB."""
    source_file = state["source_file"]
    status = state["status"]
    error = state["error"]
    
    log.info(f"🏁 [Node: Finalize Consumer] {source_file}: {status}")
    update_job_status(source_file, status, error_message=error)
    
    if state["metrics"]:
        state["metrics"].emit(log)
        
    return state

def create_consumer_graph():
    workflow = StateGraph(ConsumerState)
    workflow.add_node("validate_remaining_chunks_node", validate_remaining_chunks_node)
    workflow.add_node("store_final_chunks_node", store_final_chunks_node)
    workflow.add_node("archival_export_node", archival_export_node)
    workflow.add_node("finalize_consumer_node", finalize_consumer_node)
    workflow.set_entry_point("validate_remaining_chunks_node")
    workflow.add_edge("validate_remaining_chunks_node", "store_final_chunks_node")
    workflow.add_edge("store_final_chunks_node", "archival_export_node")
    workflow.add_edge("archival_export_node", "finalize_consumer_node")
    workflow.add_edge("finalize_consumer_node", END)
    return workflow.compile()

def get_consumer_app():
    global _COMPILED_CONSUMER_APP
    if _COMPILED_CONSUMER_APP is None:
        with _APP_LOCK:
            if _COMPILED_CONSUMER_APP is None:
                log.info("🔨 Compiling Consumer LangGraph...")
                _COMPILED_CONSUMER_APP = create_consumer_graph()
    return _COMPILED_CONSUMER_APP

def run_consumer_graph(source_file: str, expected_chunks: int, chunks: List[Dict[str, Any]], metrics: Optional[FileMetrics]) -> bool:
    """Invokes the cached graph to finalize a document."""
    initial_state: ConsumerState = {
        "source_file": source_file,
        "expected_chunks": expected_chunks,
        "chunks": chunks,
        "metrics": metrics,
        "status": "pending",
        "error": None
    }
    try:
        app = get_consumer_app()
        final_state = app.invoke(initial_state)
        return final_state["status"] == STATUS_COMPLETED
    except Exception as e:
        log.error(f"💥 Consumer Graph fatal error for {source_file}: {e}")
        update_job_status(source_file, STATUS_FAILED, error_message=str(e))
        return False
