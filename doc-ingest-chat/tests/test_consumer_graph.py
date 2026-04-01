from unittest.mock import MagicMock, patch

import pytest
from services.job_service import STATUS_COMPLETED
from workers.consumer_graph import run_consumer_graph


@pytest.fixture
def mock_consumer_state():
    return {"source_file": "test.pdf", "expected_chunks": 1, "chunks": [{"chunk": "text", "id": "id1", "source_file": "test.pdf", "hash": "h1", "type": "pdf"}], "metrics": None, "status": "pending", "error": None}


def test_validate_remaining_chunks_node_success(mock_consumer_state):
    from workers.consumer_graph import validate_remaining_chunks_node

    state = validate_remaining_chunks_node(mock_consumer_state)
    assert state["status"] == "processing"
    assert len(state["chunks"]) == 1


@patch("workers.consumer_graph.store_chunks_in_db")
@patch("services.parquet_service.append_chunks")
def test_store_final_chunks_node_success(mock_append, mock_store, mock_consumer_state):
    from workers.consumer_graph import store_final_chunks_node

    mock_consumer_state["status"] = "processing"
    mock_store.return_value = 1
    state = store_final_chunks_node(mock_consumer_state)
    assert state["status"] == "processing"
    mock_store.assert_called_once()
    mock_append.assert_called_once()


@patch("workers.consumer_graph.commit_to_parquet")
def test_archival_export_node_success(mock_commit, mock_consumer_state):
    from workers.consumer_graph import archival_export_node

    mock_consumer_state["status"] = "processing"
    state = archival_export_node(mock_consumer_state)
    assert state["status"] == STATUS_COMPLETED
    mock_commit.assert_called_once()


def test_run_consumer_graph_integration():
    with patch("workers.consumer_graph.get_consumer_app") as mock_get:
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"status": STATUS_COMPLETED}
        mock_get.return_value = mock_app
        assert run_consumer_graph("f1", 1, [], None) is True
