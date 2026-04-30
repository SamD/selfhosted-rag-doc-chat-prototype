#!/usr/bin/env python3
"""
LangGraph implementation of the OCR worker workflow.
Uses Docling (EasyOCR) as the primary OCR engine.
"""

import base64
import json
import os
from typing import List, Optional, TypedDict

import numpy as np
from config.settings import DEBUG_IMAGE_DIR
from langgraph.graph import END, StateGraph
from services.redis_service import get_redis_client
from utils.ocr_utils import run_ocr, save_bad_image
from utils.trace_utils import get_logger, set_trace_id

log = get_logger("ingest.ocr_graph")

# Global cache for the compiled graph
_COMPILED_OCR_APP = None


class OCRState(TypedDict):
    """Represents the internal state of an OCR job."""

    job_id: str
    trace_id: Optional[str]
    rel_path: str
    page_num: int
    image_base64: str
    image_shape: List[int]
    image_dtype: str
    reply_key: str
    np_image: Optional[np.ndarray]
    text: Optional[str]
    engine: str
    execution_time_ms: float
    status: str
    error: Optional[str]


def decode_image_node(state: OCRState) -> OCRState:
    """Decodes base64 string back into a NumPy array."""
    trace_id = state.get("trace_id")
    if trace_id:
        set_trace_id(trace_id)
        
    try:
        shape = tuple(state["image_shape"])
        dtype = state["image_dtype"]
        np_image = np.frombuffer(base64.b64decode(state["image_base64"]), dtype=dtype).reshape(shape)
        log.info(f"📥 [OCR Node: Decode] {state['rel_path']} page {state['page_num']}")
        return {**state, "np_image": np_image, "status": "processing"}
    except Exception as e:
        log.error(f"💥 [OCR Node: Decode] Failed for {state['rel_path']}: {e}")
        return {**state, "status": "failed", "error": f"Decode error: {e}"}


def ocr_node(state: OCRState) -> OCRState:
    """Executes Docling (EasyOCR) OCR on the decoded image."""
    if state["status"] == "failed":
        return state

    try:
        log.info(f"🔄 [OCR Node: OCR] Running Docling for {state['rel_path']} page {state['page_num']}")
        text, engine, time_ms = run_ocr(state["np_image"], state["rel_path"], state["page_num"])

        if not text:
            log.warning(f"⚠️ [OCR Node: OCR] No text extracted for {state['rel_path']} page {state['page_num']}")
            doc_id = os.path.basename(state["rel_path"]).replace("/", "_").replace("\\", "_")
            debug_image_path = os.path.join(DEBUG_IMAGE_DIR, f"{doc_id}_page_{state['page_num']}.png")
            save_bad_image(state["np_image"], debug_image_path, f"[Doc {state['rel_path']}][Page {state['page_num']}]")
            return {**state, "text": "", "engine": engine, "execution_time_ms": time_ms, "status": "failed", "error": "No text extracted"}

        log.info(f"✅ [OCR Node: OCR] Succeeded for {state['rel_path']} page {state['page_num']} ({len(text)} chars)")
        return {**state, "text": text, "engine": engine, "execution_time_ms": time_ms, "status": "success"}
    except Exception as e:
        log.error(f"💥 [OCR Node: OCR] Failed for {state['rel_path']}: {e}")
        return {**state, "status": "failed", "error": f"OCR error: {e}"}


def respond_node(state: OCRState) -> OCRState:
    """Sends the result back to Redis via LPUSH, regardless of success/failure."""
    response = {"text": state.get("text", ""), "rel_path": state["rel_path"], "page_num": state["page_num"], "engine": state.get("engine", "unknown"), "job_id": state["job_id"], "error": state.get("error"), "status": state["status"]}
    try:
        redis_client = get_redis_client()
        redis_client.lpush(state["reply_key"], json.dumps(response))
        redis_client.expire(state["reply_key"], 300)
        log.info(f"📤 [OCR Node: Respond] Sent to {state['reply_key']} status={state['status']} engine={state.get('engine', 'unknown')}")
    except Exception as e:
        log.error(f"💥 [OCR Node: Respond] Failed to send Redis response: {e}")
    return state


def create_ocr_graph():
    """Initializes and compiles the OCR StateGraph."""
    workflow = StateGraph(OCRState)
    workflow.add_node("decode_image_node", decode_image_node)
    workflow.add_node("ocr_node", ocr_node)
    workflow.add_node("respond_node", respond_node)

    workflow.set_entry_point("decode_image_node")
    workflow.add_edge("decode_image_node", "ocr_node")
    workflow.add_edge("ocr_node", "respond_node")
    workflow.add_edge("respond_node", END)

    return workflow.compile()


def get_ocr_app():
    """Singleton getter for the compiled graph."""
    global _COMPILED_OCR_APP
    if _COMPILED_OCR_APP is None:
        log.info("🔨 Compiling OCR LangGraph...")
        _COMPILED_OCR_APP = create_ocr_graph()
    return _COMPILED_OCR_APP

def run_ocr_graph(job: dict) -> bool:
    """Invokes the cached graph for a single OCR job."""
    initial_state: OCRState = {
        "job_id": job.get("job_id", "unknown"),
        "trace_id": job.get("trace_id"),
        "rel_path": job["rel_path"],
        "page_num": job["page_num"],
        "image_base64": job["image_base64"],
        "image_shape": job["image_shape"],
        "image_dtype": job["image_dtype"],
        "reply_key": job["reply_key"],
        "np_image": None,
        "text": None,
        "engine": "unknown",
        "execution_time_ms": 0.0,
        "status": "pending",
        "error": None,
    }
    try:
        app = get_ocr_app()
        final_state = app.invoke(initial_state)
        return final_state["status"] == "success"
    except Exception as e:
        log.error(f"💥 OCR Graph fatal error: {e}")
        return False
