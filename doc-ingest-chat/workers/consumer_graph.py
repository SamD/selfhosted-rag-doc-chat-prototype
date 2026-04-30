#!/usr/bin/env python3
"""
LangGraph implementation of the consumer batch processing workflow.
Refactored for ZERO-MEMORY archival and database-driven lifecycle finalization.
"""

import os
import shutil
from typing import Any, Dict, List, Optional, TypedDict

from config import settings
from langgraph.graph import END, StateGraph
from processors.text_processor import validate_chunk
from services.job_service import (
    STATUS_INGEST_FAILED,
    STATUS_INGEST_SUCCESS,
    JobService,
)
from services.parquet_service import commit_to_parquet
from utils.consumer_utils import store_chunks_in_db
from utils.metrics import FileMetrics
from utils.producer_utils import get_tokenizer
from utils.trace_utils import get_logger, set_trace_id

log = get_logger("ingest.consumer_graph")

# Global cache for the compiled graph
_COMPILED_CONSUMER_APP = None


class ConsumerState(TypedDict):
    """Represents the state for finalization of a file."""

    source_file: str
    trace_id: Optional[str]
    expected_chunks: int
    chunks: List[Dict[str, Any]]
    metrics: Optional[FileMetrics]
    status: str
    error: Optional[str]
    job_id: Optional[str]


def validate_remaining_chunks_node(state: ConsumerState) -> ConsumerState:
    """Validates remaining chunks and finds job ID in DuckDB with lock retry."""
    source_file = state["source_file"]
    chunks = state["chunks"]
    
    trace_id = state.get("trace_id")
    if trace_id:
        set_trace_id(trace_id)

    log.info(f"🧐 [Node: Final Validate] {source_file}")

    # Find Job ID using protected retry helper
    job_id = None
    try:
        query = "SELECT id FROM ingestion_lifecycle WHERE original_filename = ? AND status != ?"
        res, _ = JobService._execute_with_retry(query, (source_file, STATUS_INGEST_FAILED), fetch=True)
        if res:
            job_id = res[0]
    except Exception as e:
        log.warning(f"⚠️ Could not find job ID for {source_file} in lifecycle DB: {e}")

    tokenizer = get_tokenizer()
    processed_chunks = []
    for entry in chunks:
        chunk_text = entry.get("chunk")
        # Now returns the chunk itself (truncated if needed) instead of a bool
        final_chunk = validate_chunk(chunk_text, tokenizer)
        if final_chunk:
            entry["chunk"] = final_chunk
            processed_chunks.append(entry)

    return {**state, "chunks": processed_chunks, "status": "processing", "job_id": job_id}


def store_final_chunks_node(state: ConsumerState) -> ConsumerState:
    """Stores remaining chunks in Vector DB and DuckDB."""
    if state["status"] == STATUS_INGEST_FAILED:
        return state

    try:
        from services.parquet_service import append_chunks

        if state["chunks"]:
            append_chunks(state["chunks"])
            store_chunks_in_db(state["source_file"], state["chunks"], state["metrics"])

        return {**state, "status": "processing"}
    except Exception as e:
        log.error(f"💥 Final storage node failed: {e}")
        return {**state, "status": STATUS_INGEST_FAILED, "error": f"Storage error: {e}"}


def archival_export_node(state: ConsumerState) -> ConsumerState:
    """Triggers DuckDB export to Parquet."""
    if state["status"] == STATUS_INGEST_FAILED:
        return state

    try:
        commit_to_parquet()
        return {**state, "status": "completed"}
    except Exception as e:
        log.error(f"💥 Final archival export failed: {e}")
        return {**state, "status": STATUS_INGEST_FAILED, "error": f"Parquet export failed: {e}"}


def finalize_consumer_node(state: ConsumerState) -> ConsumerState:
    """Moves files to SUCCESS/FAILED folders and updates final status using atomic retries."""
    source_file = state["source_file"]
    job_id = state.get("job_id")
    status = state["status"]
    error = state["error"]

    final_lifecycle_status = STATUS_INGEST_SUCCESS if status == "completed" else STATUS_INGEST_FAILED
    log.info(f"🏁 [Node: Finalize Consumer] {source_file}: {final_lifecycle_status}")

    if job_id:
        try:
            # 1. Look up current paths with lock protection
            query_paths = "SELECT pdf_path, md_path FROM ingestion_lifecycle WHERE id = ?"
            paths, _ = JobService._execute_with_retry(query_paths, (job_id,), fetch=True)

            if paths:
                old_pdf, old_md = paths
                dest_dir = settings.SUCCESS_DIR if final_lifecycle_status == STATUS_INGEST_SUCCESS else settings.FAILED_DIR

                new_pdf = os.path.join(dest_dir, os.path.basename(old_pdf)) if old_pdf else None
                new_md = os.path.join(dest_dir, os.path.basename(old_md)) if old_md else None

                # MOVE PDF: Force overwrite if exists to prevent OSError
                if old_pdf and os.path.exists(old_pdf):
                    if new_pdf and os.path.exists(new_pdf):
                        os.remove(new_pdf)
                    shutil.move(old_pdf, new_pdf)
                    log.info(f"🚚 Moved PDF to {new_pdf}")

                # MOVE MD: Force overwrite if exists
                if old_md and os.path.exists(old_md):
                    if new_md and os.path.exists(new_md):
                        os.remove(new_md)
                    shutil.move(old_md, new_md)
                    log.info(f"🚚 Moved Markdown to {new_md}")

                # Update DB with lock protection
                JobService.transition_job(job_id, final_lifecycle_status, new_pdf_path=new_pdf, new_md_path=new_md, error=error)
                log.info(f"✅ Document {job_id} transitioned to {final_lifecycle_status}")
        except Exception as e:
            log.error(f"💥 Error during final file move: {e}")
    else:
        log.warning(f"⚠️ Skipping file move for {source_file}, no job_id found.")

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
        _COMPILED_CONSUMER_APP = create_consumer_graph()
    return _COMPILED_CONSUMER_APP


def run_consumer_graph(source_file: str, expected_chunks: int, chunks: List[Dict[str, Any]], metrics: Optional[FileMetrics], trace_id: str = None) -> bool:
    """Invokes the cached graph to finalize a document."""
    initial_state: ConsumerState = {"source_file": source_file, "trace_id": trace_id, "expected_chunks": expected_chunks, "chunks": chunks, "metrics": metrics, "status": "pending", "error": None, "job_id": None}
    try:
        app = get_consumer_app()
        final_state = app.invoke(initial_state)
        return final_state["status"] == "completed"
    except Exception as e:
        log.error(f"💥 Consumer Graph fatal error for {source_file}: {e}")
        return False
