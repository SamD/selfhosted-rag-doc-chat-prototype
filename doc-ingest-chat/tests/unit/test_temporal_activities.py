#!/usr/bin/env python3
"""
Unit tests for Temporal Activities.
Tests transcribe_media() Activity with mock RemoteWhisper.
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Dataclasses don't need any external deps
from models.transcription_input import TranscriptionInput, TranscriptionResult

# Fixtures -------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _minimal_env(tmp_path: Path):
    """Ensure minimal env vars exist so settings lazy-load without crashing."""
    defaults = {
        "EMBEDDING_ENDPOINTS": "/tmp/emb",
        "LLM_PATH": "/tmp/llm.gguf",
        "SUPERVISOR_LLM_ENDPOINTS": "/tmp/sup.gguf",
        "DEFAULT_DOC_INGEST_ROOT": str(tmp_path),
    }
    for k, v in defaults.items():
        if k not in os.environ:
            os.environ[k] = v
    yield


# Dataclass tests ------------------------------------------------------------

class TestTranscriptionInput:
    """Test TranscriptionInput dataclass."""

    def test_defaults(self):
        input_data = TranscriptionInput(file_path="/test/file.mp3")
        assert input_data.file_path == "/test/file.mp3"
        assert input_data.language == "en"
        assert input_data.mime_type is None

    def test_with_values(self):
        input_data = TranscriptionInput(
            file_path="/test/file.mp3",
            language="es",
            mime_type="audio/mpeg",
        )
        assert input_data.language == "es"
        assert input_data.mime_type == "audio/mpeg"


class TestTranscriptionResult:
    """Test TranscriptionResult dataclass."""

    def test_creation(self):
        segments = [{"text": "Hello world"}, {"text": "How are you?"}]
        result = TranscriptionResult(
            segments=segments,
            source_file="test.mp3",
            job_id="test-job-123",
        )
        assert result.segments == segments
        assert result.source_file == "test.mp3"
        assert result.job_id == "test-job-123"


# Activity tests -------------------------------------------------------------

class TestTranscribeMediaActivity:
    """Test transcribe_media Activity."""

    @pytest.mark.parametrize("whisper_path", [
        "/nonexistent/model",  # local path → triggers whisperx code path
        "http://remote:8000",  # remote URL → triggers requests code path
    ])
    def test_activity_rejects_missing_file(self, whisper_path: str):
        input_data = TranscriptionInput(file_path="/does/not/exist.mp3")

        with patch.dict(os.environ, {"WHISPER_MODEL_ENDPOINTS": whisper_path}):
            # Import inside test to ensure mocks are active
            from temporal_worker.activities import transcribe_media
            with pytest.raises(FileNotFoundError) as exc_info:
                asyncio.run(transcribe_media(input_data))
            assert "File not found" in str(exc_info.value)

    def test_remote_whisper_path(self, tmp_path: Path):
        """Verify remote WhisperX HTTP path (no whisperx import needed)."""
        # Create a real temp file so os.path.exists passes
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\x00" * 100)

        mock_response = Mock()
        mock_response.json.return_value = {
            "segments": [{"text": "Hello world"}, {"text": "How are you?"}],
        }
        mock_response.raise_for_status = Mock()

        input_data = TranscriptionInput(
            file_path=str(audio_file), language="en", mime_type="audio/mpeg"
        )

        _orig_requests = sys.modules.get('requests')
        try:
            with patch.dict(os.environ, {
                "WHISPER_MODEL_ENDPOINTS": "http://remote:8000/inference",
            }):
                from temporal_worker.activities import transcribe_media
                sys.modules['requests'] = Mock(post=Mock(return_value=mock_response))
                result = asyncio.run(transcribe_media(input_data))
                assert isinstance(result, TranscriptionResult)
                assert len(result.segments) == 2
                assert result.segments[0]["text"] == "Hello world"
                assert result.source_file == "test.mp3"
        finally:
            if _orig_requests is None:
                sys.modules.pop('requests', None)
            else:
                sys.modules['requests'] = _orig_requests

    def test_local_whisperx_path(self, tmp_path: Path):
        """Verify local whisperx code path is reachable."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        mock_model_inst = Mock()
        mock_model_inst.transcribe.return_value = {
            "segments": [{"text": "Local output"}],
        }
        mock_model_class = Mock(return_value=mock_model_inst)
        mock_whisperx = MagicMock()
        mock_whisperx.load_audio.return_value = b"audio_data"
        mock_whisperx.load_model = mock_model_class

        input_data = TranscriptionInput(
            file_path=str(audio_file), language="en", mime_type="audio/wav"
        )

        # Store original so we restore it
        _orig_whisperx = sys.modules.get('whisperx')

        try:
            with patch.dict(os.environ, {
                "WHISPER_MODEL_ENDPOINTS": str(tmp_path / "whisper_model"),
                "DEVICE": "cpu",
                "COMPUTE_TYPE": "float32",
                "MEDIA_BATCH_SIZE": "8",
            }):
                # Import inside test to ensure mocks are active
                from temporal_worker.activities import transcribe_media
                # Patch at sys.modules level — since activities.py does `import whisperx`
                # inside the function, it will resolve to our mock
                sys.modules['whisperx'] = mock_whisperx
                result = asyncio.run(transcribe_media(input_data))

            assert len(result.segments) == 1
            assert result.segments[0]["text"] == "Local output"
            mock_whisperx.load_audio.assert_called_once_with(str(audio_file))
        finally:
            if _orig_whisperx is None:
                sys.modules.pop('whisperx', None)
            else:
                sys.modules['whisperx'] = _orig_whisperx
