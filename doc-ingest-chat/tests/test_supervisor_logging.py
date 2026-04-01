import logging
from unittest.mock import MagicMock, patch

from workers.producer_graph import IngestState, supervisor_node


def test_supervisor_logging_emitted(caplog):
    """
    Ensures that the supervisor node emits logs with the 🕵️ icon
    """
    state: IngestState = {
        "rel_path": "test_doc.pdf",
        "doc_preview": "This is a long enough text to trigger the supervisor agent logic in the graph.",
        "job_id": "test-job-123",
        "full_path": "/tmp/test_doc.pdf",
        "file_type": "pdf",
        "total_chunks_sent": 0,
        "queue_name": "q1",
        "requires_ocr": False,
        "status": "processing",
        "error": None,
        "metrics": None,
        "pages_processed": 0
    }

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "This document is a test case for logging."

    caplog.set_level(logging.INFO)

    with patch("workers.producer_graph.get_supervisor_llm", return_value=mock_llm):
        supervisor_node(state)

    log_text = caplog.text
    assert "🕵️ [Supervisor] Waiting for GPU lock for test_doc.pdf" in log_text
    assert "🕵️ [Supervisor] Analyzing test_doc.pdf" in log_text
    assert "🕵️ [Supervisor] Final Context for test_doc.pdf" in log_text

def test_supervisor_skipping_logged(caplog):
    """Ensures skipping is logged correctly when no preview exists."""
    # Note: supervisor_node now only runs if status is not FAILED.
    # But preview_node marks it failed if preview is missing.
    state: IngestState = {
        "rel_path": "empty.pdf",
        "doc_preview": "", 
        "status": "failed", # Marked by preview_node
        "error": "Insufficient text"
    }
    
    caplog.set_level(logging.INFO)
    supervisor_node(state)
    
    # Node should return immediately without extra logging
    assert "🕵️ [Supervisor] Analyzing" not in caplog.text
