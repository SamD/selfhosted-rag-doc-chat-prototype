import json
from unittest.mock import MagicMock, patch

from workers.consumer_worker import consumer_worker


@patch("services.parquet_service.stage_chunks")
def test_consumer_worker_incremental_respects_lock(mock_stage):
    """
    Ensures that the staging logic in the consumer worker
    uses the multiprocessing lock during file_end flush.
    """
    mock_redis = MagicMock()
    # First message is a chunk, second is file_end
    chunk_data = {"source_file": "f1", "chunk": "c1", "id": "i1", "hash": "h1", "type": "pdf"}
    sentinel_data = {"type": "file_end", "source_file": "f1", "expected_chunks": 1}

    shared_data = {"shutdown_flag": False}
    mock_parq_lock = MagicMock()

    messages = [("q", json.dumps(chunk_data).encode()), ("q", json.dumps(sentinel_data).encode())]

    def blpop_side_effect(*args, **kwargs):
        if messages:
            return messages.pop(0)
        shared_data["shutdown_flag"] = True
        return None

    mock_redis.blpop.side_effect = blpop_side_effect

    with patch("workers.consumer_worker.get_redis_client", return_value=mock_redis):
        # Patch the actual location where it's used or where it's imported
        with patch("workers.consumer_graph.run_consumer_graph"):
            with patch("services.parquet_service.get_staged_chunks", return_value=[]):
                consumer_worker("test-queue", shared_data, mock_parq_lock)

    # ASSERTION: The lock was used during file_end flush
    assert mock_parq_lock.__enter__.call_count >= 1
    mock_parq_lock.__exit__.assert_called()
