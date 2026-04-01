from unittest.mock import MagicMock, patch

import pytest
from services.job_service import STATUS_ENQUEUED, STATUS_PROCESSING
from workers.producer_graph import run_ingest_graph


@pytest.fixture
def mock_state():
    return {
        "job_id": "test-job",
        "full_path": "/tmp/test.pdf",
        "rel_path": "test.pdf",
        "file_type": "pdf",
        "total_chunks_sent": 0,
        "queue_name": "q1",
        "requires_ocr": False,
        "status": STATUS_PROCESSING,
        "error": None,
        "document_id": "DOC_A1B2",
        "metrics": None,
        "pages_processed": 0,
    }


def test_scan_file_node_success(mock_state):
    from workers.producer_graph import scan_file_node

    with patch("os.path.exists", return_value=True), patch("workers.producer_graph.update_job_status"), patch("workers.producer_graph.get_next_queue", return_value="q1"), patch("builtins.open", new_callable=MagicMock) as mock_file:
        mock_file.return_value.__enter__.return_value.read.return_value = b"data"

        with patch("processors.text_processor.TextProcessor.get_document_id", return_value="DOC_A1B2"):
            state = scan_file_node(mock_state)
            assert state["file_type"] == "pdf"
            assert state["queue_name"] == "q1"
            assert state["document_id"] == "DOC_A1B2"


@patch("workers.producer_graph.process_pdf_by_page")
def test_pdf_extract_node_success(mock_process, mock_state):
    from workers.producer_graph import pdf_extract_node

    # We patch stream_chunks_to_redis because it's called inside the callback
    with patch("workers.producer_graph.stream_chunks_to_redis", return_value=1) as mock_stream:

        def side_effect(path, rel, ftype, chunk_callback, **kwargs):
            chunk_callback([("text", "engine", 1)])

        mock_process.side_effect = side_effect

        with patch("workers.producer_graph.is_valid_pdf", return_value=True):
            state = pdf_extract_node(mock_state)
            assert state["total_chunks_sent"] == 1
            assert state["pages_processed"] == 1
            mock_stream.assert_called_once()


@patch("workers.producer_graph.get_redis_client")
@patch("workers.producer_graph.blocking_push_with_backpressure")
def test_send_sentinel_node(mock_push, mock_redis, mock_state):
    from workers.producer_graph import send_sentinel_node

    state = send_sentinel_node(mock_state)
    assert state["status"] == STATUS_ENQUEUED
    mock_push.assert_called_once()


@patch("workers.producer_graph.process_pdf_by_page")
def test_pdf_extract_node_context_enrichment(mock_process, mock_state):
    """
    CRITICAL TEST: Ensures that the [DOC_HASH] anchor is
    correctly handled by the node callback.
    """
    from workers.producer_graph import pdf_extract_node

    mock_state["document_id"] = "DOC_A1B2"
    enriched_content = f"passage: [{mock_state['document_id']}] raw chunk content"

    # We catch the data being sent to stream_chunks_to_redis
    with patch("workers.producer_graph.stream_chunks_to_redis", return_value=1) as mock_stream:

        def side_effect(path, rel, ftype, chunk_callback, **kwargs):
            # Simulate a utility function yielding an enriched chunk
            chunk_callback([(enriched_content, "pdfplumber", 1)])

        mock_process.side_effect = side_effect

        with patch("workers.producer_graph.is_valid_pdf", return_value=True):
            pdf_extract_node(mock_state)

            # Verify that the chunk passed to stream_chunks_to_redis was received correctly
            sent_batch = mock_stream.call_args[0][0]
            result_text = sent_batch[0][0]

            assert result_text == enriched_content
            assert mock_stream.called


def test_run_ingest_graph_integration():
    job_tuple = ("job1", "src", "rel")
    with patch("workers.producer_graph.get_producer_app") as mock_get:
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"status": STATUS_ENQUEUED}
        mock_get.return_value = mock_app
        assert run_ingest_graph(job_tuple) is True
