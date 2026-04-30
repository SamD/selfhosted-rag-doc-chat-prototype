import unittest
from unittest.mock import MagicMock, patch

from workers.producer_graph import (
    finalize_state_node,
    markdown_extract_node,
    run_ingest_graph,
    scan_file_node,
    send_sentinel_node,
)


def test_scan_file_node_success():
    # Mock state and deps
    state = {
        "full_path": "/tmp/test.md",
        "rel_path": "test.md",
        "job_id": "job123",
        "status": "pending",
        "trace_id": "trace-123"
    }

    with patch("os.path.exists", return_value=True):
        new_state = scan_file_node(state)
        assert new_state["status"] == "processing"
        assert new_state["file_type"] == "markdown"


@patch("workers.producer_graph.split_markdown_doc")
@patch("workers.producer_graph.get_redis_client")
@patch("workers.producer_graph.blocking_push_with_backpressure")
def test_markdown_extract_node_success(mock_push, mock_redis, mock_split):
    state = {"full_path": "/tmp/test.md", "rel_path": "test.md", "queue_name": "q1", "document_id": "DOC_123", "file_type": "markdown", "status": "processing", "trace_id": "trace-123"}

    mock_split.return_value = (["chunk1"], [{"page": 1}])

    with patch("builtins.open", unittest.mock.mock_open(read_data="content")):
        new_state = markdown_extract_node(state)
        assert new_state["status"] == "processing"
        assert "chunks" in new_state
        assert len(new_state["chunks"]) == 1
        assert mock_push.called
        # Verify trace_id in payload
        _, kwargs = mock_push.call_args
        entries = kwargs["entries"]
        import json
        entry = json.loads(entries[0])
        assert entry["trace_id"] == "trace-123"


@patch("workers.producer_graph.get_redis_client")
@patch("workers.producer_graph.send_file_end_sentinel")
def test_send_sentinel_node(mock_send, mock_redis):
    state = {"rel_path": "test.md", "queue_name": "q1", "chunks": ["chunk1"], "status": "processing"}

    new_state = send_sentinel_node(state)
    assert new_state["status"] == "processing"
    assert mock_send.called


def test_finalize_state_node_success():
    state = {"rel_path": "test.md", "job_id": "job123", "status": "processing", "error": None}
    # Node should simply return state now as worker handles transitions
    new_state = finalize_state_node(state)
    assert new_state == state


@patch("workers.producer_graph.get_producer_app")
@patch("workers.producer_worker.get_next_queue", return_value="test_q")
def test_run_ingest_graph_integration(mock_q, mock_get):
    job_tuple = ("job1", "src", "rel")
    mock_app = MagicMock()
    mock_app.invoke.return_value = {"status": "processing"}
    mock_get.return_value = mock_app

    # Mock JobService lookup for trace_id
    with patch("services.job_service.JobService._execute_with_retry", return_value=(["trace-abc"], ["trace_id"])):
        assert run_ingest_graph(job_tuple) is True
