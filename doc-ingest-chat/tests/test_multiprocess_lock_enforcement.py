import json
from unittest.mock import MagicMock, patch

from workers.consumer_worker import consumer_worker


@patch("workers.consumer_worker.store_chunks_in_db")
@patch("workers.consumer_worker.append_chunks")
def test_consumer_worker_incremental_respects_lock(mock_append, mock_store):
    """
    Ensures that the incremental ingestion logic in the consumer worker
    uses the multiprocessing lock.
    """
    mock_redis = MagicMock()
    chunk_data = {"source_file": "f1", "chunk": "c1", "id": "i1", "hash": "h1", "type": "pdf"}

    shared_data = {"shutdown_flag": False}
    mock_parq_lock = MagicMock()

    # We trigger shutdown after the first pop
    def blpop_side_effect(*args, **kwargs):
        if not hasattr(blpop_side_effect, "called"):
            blpop_side_effect.called = True
            return ("q", json.dumps(chunk_data).encode())
        shared_data["shutdown_flag"] = True
        return None

    mock_redis.blpop.side_effect = blpop_side_effect

    # Set batch size to 1 to trigger immediate write
    with patch("workers.consumer_worker.MAX_CHROMA_BATCH_SIZE", 1), patch("workers.consumer_worker.get_redis_client", return_value=mock_redis):
        consumer_worker("test-queue", shared_data, mock_parq_lock)

    # ASSERTION: The lock was used during the incremental write
    mock_parq_lock.__enter__.assert_called()
    mock_parq_lock.__exit__.assert_called()
