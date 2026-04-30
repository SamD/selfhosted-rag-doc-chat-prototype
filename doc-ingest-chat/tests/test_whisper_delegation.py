import json
from unittest.mock import MagicMock, patch

import pytest
from utils.whisper_utils import send_media_to_whisperx


def test_send_media_to_whisperx_success():
    """Test successful transcription streaming with language support."""
    mock_redis = MagicMock()
    # Mock responses: 2 segments then "done"
    mock_redis.blpop.side_effect = [
        ("key", json.dumps({"type": "segment", "text": "Hello"})),
        ("key", json.dumps({"type": "segment", "text": "World"})),
        ("key", json.dumps({"type": "done"}))
    ]
    
    with patch("utils.whisper_utils.get_redis_client", return_value=mock_redis), \
         patch("os.path.abspath", side_effect=lambda x: f"/abs/{x}"):
        
        segments = list(send_media_to_whisperx("test.mp4", language="fr", trace_id="trace-abc"))
        
        assert segments == ["Hello", "World"]
        assert mock_redis.lpush.called
        # Check if language and trace_id were passed in the JSON payload
        args, _ = mock_redis.lpush.call_args
        job_data = json.loads(args[1])
        assert job_data["language"] == "fr"
        assert job_data["trace_id"] == "trace-abc"
        assert mock_redis.blpop.call_count == 3

def test_send_media_to_whisperx_worker_error():
    """Test error reported by WhisperX worker."""
    mock_redis = MagicMock()
    mock_redis.blpop.return_value = ("key", json.dumps({"type": "error", "error": "CUDA OOM"}))
    
    with patch("utils.whisper_utils.get_redis_client", return_value=mock_redis), \
         patch("os.path.abspath", side_effect=lambda x: f"/abs/{x}"):
        
        with pytest.raises(RuntimeError) as excinfo:
            list(send_media_to_whisperx("test.mp4"))
        
        assert "WhisperX error: CUDA OOM" in str(excinfo.value)

def test_send_media_to_whisperx_timeout():
    """Test timeout waiting for WhisperX response."""
    mock_redis = MagicMock()
    mock_redis.blpop.return_value = None # Simulate timeout
    
    # Speed up timeout for test
    with patch("utils.whisper_utils.get_redis_client", return_value=mock_redis), \
         patch("os.path.abspath", side_effect=lambda x: f"/abs/{x}"), \
         patch("time.time", side_effect=[0, 0, 2000, 2000, 2000]): # Enough values for start_wait and loop checks
        
        with pytest.raises(TimeoutError) as excinfo:
            list(send_media_to_whisperx("test.mp4"))
        
        assert "timed out after 1800s" in str(excinfo.value)

def test_send_media_to_whisperx_redis_failure():
    """Test Redis connection failure during job submission."""
    mock_redis = MagicMock()
    mock_redis.lpush.side_effect = Exception("Connection refused")
    
    with patch("utils.whisper_utils.get_redis_client", return_value=mock_redis), \
         patch("os.path.abspath", side_effect=lambda x: f"/abs/{x}"):
        
        with pytest.raises(RuntimeError) as excinfo:
            list(send_media_to_whisperx("test.mp4"))
        
        assert "Redis submission failed" in str(excinfo.value)
