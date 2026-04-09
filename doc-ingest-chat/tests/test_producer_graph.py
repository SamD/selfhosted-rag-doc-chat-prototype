from unittest.mock import MagicMock, patch

import pytest
from services.job_service import STATUS_ENQUEUED, STATUS_PROCESSING
from workers.producer_graph import run_ingest_graph


@pytest.fixture
def mock_state():
    return {
        "job_id": "test-job",
        "full_path": "/tmp/test.md",
        "rel_path": "test.md",
        "file_type": "markdown",
        "total_chunks_sent": 0,
        "queue_name": "q1",
        "status": STATUS_PROCESSING,
        "error": None,
        "document_id": "DOC_A1B2",
        "file_metadata": None,
        "metrics": None,
        "pages_processed": 0,
    }


def test_scan_file_node_success(mock_state):
    from workers.producer_graph import scan_file_node

    with patch("os.path.exists", return_value=True), patch("workers.producer_graph.update_job_status"), patch("workers.producer_graph.get_next_queue", return_value="q1"), patch("builtins.open", new_callable=MagicMock) as mock_file:
        mock_file.return_value.__enter__.return_value.read.return_value = b"data"

        with patch("processors.text_processor.TextProcessor.get_document_id", return_value="DOC_A1B2"):
            state = scan_file_node(mock_state)
            assert state["file_type"] == "markdown"
            assert state["queue_name"] == "q1"
            assert state["document_id"] == "DOC_A1B2"


@patch("workers.producer_graph.split_markdown_doc")
@patch("workers.producer_graph.get_redis_client")
@patch("workers.producer_graph.blocking_push_with_backpressure")
def test_markdown_extract_node_success(mock_push, mock_redis, mock_split, mock_state):
    from workers.producer_graph import markdown_extract_node

    mock_split.return_value = (["chunk1"], [{"ID": "id", "Slug": "slug"}])

    with patch("builtins.open", new_callable=MagicMock) as mock_file:
        mock_file.return_value.__enter__.return_value.read.return_value = "content"
        state = markdown_extract_node(mock_state)
        assert state["total_chunks_sent"] == 1
        assert state["status"] == STATUS_PROCESSING
        mock_push.assert_called_once()


@patch("workers.producer_graph.get_redis_client")
@patch("workers.producer_graph.blocking_push_with_backpressure")
def test_send_sentinel_node(mock_push, mock_redis, mock_state):
    from workers.producer_graph import send_sentinel_node

    state = send_sentinel_node(mock_state)
    assert state["status"] == STATUS_ENQUEUED
    mock_push.assert_called_once()


def test_run_ingest_graph_integration():
    job_tuple = ("job1", "src", "rel")
    with patch("workers.producer_graph.get_producer_app") as mock_get:
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"status": STATUS_ENQUEUED}
        mock_get.return_value = mock_app
        assert run_ingest_graph(job_tuple) is True
