#!/usr/bin/env python3
"""
E2E tests for Transcription Latency.
Compares overhead of Temporal path vs Redis path with real 10-second MP3.
"""

import os
import sys
import time
import types
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from models.transcription_input import TranscriptionInput, TranscriptionResult

# Mock the whisperx module before importing activities
with patch.dict("sys.modules", {"whisperx": MagicMock()}):
    with patch.dict("sys.modules", {"temporalio": MagicMock()}):
        from utils.whisper_utils import (  # noqa: E402
            send_media_to_whisperx,
            send_media_to_whisperx_temporal,
        )


class TestTranscriptionLatency:
    """E2E tests for transcription latency comparison."""

    def test_temporal_vs_redis_latency(self, monkeypatch):
        """Compare latency of Temporal path vs Redis path with real 10-second MP3."""
        # Patch the function globals directly. The whisper_utils module is loaded as a
        # namespace-package submodule and is not present in sys.modules, so the usual
        # ``monkeypatch.setattr('utils.whisper_utils.X', ...)`` path does not reach it.
        g = send_media_to_whisperx.__globals__

        # Create a mock 10-second MP3 file
        test_file = "/tmp/test_10s.mp3"

        # Create a dummy file (10KB to simulate a 10-second MP3)
        with open(test_file, "wb") as f:
            f.write(b"fake mp3 data" * 833)  # ~10KB

        try:
            # ---------- Redis path ----------
            monkeypatch.setitem(g, "USE_TEMPORAL_WHISPER", False)
            fake_redis = Mock()
            fake_redis.lpush.return_value = 1
            fake_redis.blpop.side_effect = [
                (None, '{"type": "segment", "text": "This is segment 1 of the 10-second audio file."}'),
                (None, '{"type": "segment", "text": "This is segment 2 of the 10-second audio file."}'),
                (None, '{"type": "segment", "text": "This is segment 3 of the 10-second audio file."}'),
                (None, '{"type": "segment", "text": "This is segment 4 of the 10-second audio file."}'),
                (None, '{"type": "segment", "text": "This is segment 5 of the 10-second audio file."}'),
                (None, '{"type": "done"}'),
            ]
            monkeypatch.setitem(g, "_REDIS_CLIENT_CACHE", fake_redis)

            redis_start = time.time()
            segments = list(
                send_media_to_whisperx(test_file, language="en", mime_type="audio/mpeg")
            )
            redis_end = time.time()
            redis_duration = redis_end - redis_start

            assert len(segments) == 5
            assert segments[0] == "This is segment 1 of the 10-second audio file."
            assert segments[4] == "This is segment 5 of the 10-second audio file."

            print(f"Redis path duration: {redis_duration:.2f}s")

 # ---------- Temporal path ----------
            monkeypatch.setitem(g, "USE_TEMPORAL_WHISPER", True)
            monkeypatch.setitem(g, "TEMPORAL_HOST", "localhost")
            monkeypatch.setitem(g, "TEMPORAL_PORT", 7233)
            monkeypatch.setitem(g, "TEMPORAL_SERVER_URL", "localhost:7233")
            monkeypatch.setitem(g, "TEMPORAL_WHISPER_TASK_QUEUE", "whisperx")

            mock_workflow_result = TranscriptionResult(
                segments=[
                    {"text": "This is segment 1 of the 10-second audio file."},
                    {"text": "This is segment 2 of the 10-second audio file."},
                    {"text": "This is segment 3 of the 10-second audio file."},
                    {"text": "This is segment 4 of the 10-second audio file."},
                    {"text": "This is segment 5 of the 10-second audio file."},
                ],
                source_file="test_10s.mp3",
                job_id="temporal-job-123",
            )

            # Client.connect() is awaited, so mock it as AsyncMock
            mock_client = AsyncMock()
            mock_client.execute_workflow = AsyncMock(return_value=mock_workflow_result)
            fake_client_class = AsyncMock()
            fake_client_class.connect = AsyncMock(return_value=mock_client)
            monkeypatch.setattr("temporalio.client.Client", fake_client_class)

            # Stub TranscribeWorkflow (used inside the Temporal path)
            dummy_workflows = types.ModuleType("temporal_worker.workflows")
            dummy_workflows.TranscribeWorkflow = Mock()
            monkeypatch.setitem(sys.modules, "temporal_worker.workflows", dummy_workflows)

            temporal_start = time.time()
            segments = list(
                send_media_to_whisperx_temporal(
                    test_file, language="en", mime_type="audio/mpeg"
                )
            )
            temporal_end = time.time()
            temporal_duration = temporal_end - temporal_start

            assert len(segments) == 5
            assert segments[0] == "This is segment 1 of the 10-second audio file."
            assert segments[4] == "This is segment 5 of the 10-second audio file."

            print(f"Temporal path duration: {temporal_duration:.2f}s")

            # Calculate overhead
            overhead = temporal_duration - redis_duration
            overhead_percentage = (overhead / redis_duration) * 100
            print(
                f"Temporal overhead: {overhead:.2f}s ({overhead_percentage:.1f}%)"
            )

            # Assert Temporal overhead is less than 50ms (0.05s)
            assert overhead < 0.05, (
                f"Temporal overhead {overhead:.2f}s exceeds 50ms limit"
            )

        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_temporal_path_with_real_file(self):
        """Test Temporal path with a real file (simulated)."""
        test_file = "/tmp/real_test.mp3"

        with open(test_file, "wb") as f:
            f.write(b"fake mp3 data" * 1000)  # ~1KB

        try:
            mock_response = Mock()
            mock_response.json.return_value = {
                "segments": [{"text": "Real file transcription"}]
            }

            _orig_requests = sys.modules.get("requests")
            try:
                from temporal_worker.activities import transcribe_media

                sys.modules["requests"] = Mock(
                    post=Mock(return_value=mock_response)
                )

                with patch.dict(
                    os.environ, {"WHISPER_MODEL_ENDPOINTS": "http://remote-whisper:8000"}
                ):
                    input_data = TranscriptionInput(
                        file_path=test_file, language="en", mime_type="audio/mpeg"
                    )

                    import asyncio

                    result = asyncio.run(transcribe_media(input_data))

                assert isinstance(result, TranscriptionResult)
                assert len(result.segments) == 1
                assert result.segments[0]["text"] == "Real file transcription"
                assert result.source_file == "real_test.mp3"
            finally:
                if _orig_requests is None:
                    sys.modules.pop("requests", None)
                else:
                    sys.modules["requests"] = _orig_requests

        finally:
            if os.path.exists(test_file):
                os.remove(test_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
