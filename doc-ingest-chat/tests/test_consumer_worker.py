import os
import sys
from itertools import cycle
from unittest.mock import MagicMock, patch

# Set required environment variables before importing
os.environ.setdefault("INGEST_FOLDER", "/tmp/test")
os.environ.setdefault("CHROMA_DATA_DIR", "/tmp/chroma")
os.environ.setdefault("EMBEDDING_MODEL_PATH", "intfloat/e5-large-v2")
os.environ.setdefault("LLM_PATH", "/tmp/test.gguf")

# Ensure the worker module is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../workers")))
import consumer_worker


def test_get_next_queue_cycles():
    # Patch queue_lock and queue_cycle with a cycle iterator
    with patch.object(consumer_worker, "queue_lock"), patch.object(consumer_worker, "queue_cycle", cycle(["q1", "q2"])):
        result1 = consumer_worker.get_next_queue()
        result2 = consumer_worker.get_next_queue()
        assert result1 in ["q1", "q2"]
        assert result2 in ["q1", "q2"]
        assert result1 != result2


def test_current_time_returns_int():
    assert isinstance(consumer_worker.current_time(), int)


@patch("consumer_worker.get_redis_client")
@patch("consumer_worker.get_db")
def test_consumer_worker_handles_shutdown(mock_get_db, mock_get_redis):
    # Setup
    mock_redis = MagicMock()
    mock_get_redis.return_value = mock_redis
    mock_db = MagicMock()
    mock_get_db.return_value = mock_db
    shared_data = {"shutdown_flag": True}
    # Should exit immediately
    consumer_worker.consumer_worker("test_queue", shared_data, MagicMock())
    # No exceptions should be raised
