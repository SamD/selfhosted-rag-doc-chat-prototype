#!/usr/bin/env python3
"""
Tests for DocumentProcessor document processing functionality.
"""

import os
import sys
from unittest.mock import patch

import pytest

# Set required environment variables before importing settings
os.environ.setdefault("DEFAULT_DOC_INGEST_ROOT", "/tmp/test")
os.environ.setdefault("CHROMA_DATA_DIR", "/tmp/chroma")
os.environ.setdefault("EMBEDDING_MODEL_PATH", "intfloat/e5-large-v2")
os.environ.setdefault("LLM_PATH", "/tmp/test.gguf")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processors.document_processor import DocumentProcessor, extract_text_from_html

# ---------------------------------------------------------------------------
# extract_text_from_html
# ---------------------------------------------------------------------------


def test_extract_text_from_html_returns_text(tmp_path):
    """Returns extracted text from a well-formed HTML file."""
    html_file = tmp_path / "page.html"
    html_file.write_text("<html><body><h1>Hello</h1><p>World</p></body></html>", encoding="utf-8")

    result = DocumentProcessor.extract_text_from_html(str(html_file))

    assert result is not None
    assert "Hello" in result
    assert "World" in result


def test_extract_text_from_html_collapses_blank_lines(tmp_path):
    """Multiple consecutive blank lines are collapsed to a single blank line."""
    html_file = tmp_path / "spaced.html"
    html_file.write_text("<html><body><p>A</p><p></p><p></p><p>B</p></body></html>", encoding="utf-8")

    result = DocumentProcessor.extract_text_from_html(str(html_file))

    assert result is not None
    assert "\n\n\n" not in result  # No triple (or more) newlines


def test_extract_text_from_html_strips_tags(tmp_path):
    """HTML tags are removed from the output."""
    html_file = tmp_path / "tags.html"
    html_file.write_text("<html><body><b>Bold</b> and <em>italic</em></body></html>", encoding="utf-8")

    result = DocumentProcessor.extract_text_from_html(str(html_file))

    assert result is not None
    assert "<b>" not in result
    assert "<em>" not in result
    assert "Bold" in result
    assert "italic" in result


def test_extract_text_from_html_returns_none_on_missing_file():
    """Returns None when the file does not exist."""
    result = DocumentProcessor.extract_text_from_html("/nonexistent/path/file.html")
    assert result is None


def test_extract_text_from_html_module_alias(tmp_path):
    """Module-level extract_text_from_html alias works identically."""
    html_file = tmp_path / "alias.html"
    html_file.write_text("<html><body><p>Alias test</p></body></html>", encoding="utf-8")

    result = extract_text_from_html(str(html_file))

    assert result is not None
    assert "Alias test" in result


# ---------------------------------------------------------------------------
# extract_text_from_media
# ---------------------------------------------------------------------------


def test_extract_text_from_media_raises_on_unsupported_extension():
    """Raises ValueError for file extensions not in SUPPORTED_MEDIA_EXT."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentProcessor.extract_text_from_media("document.pdf")


@patch("utils.whisper_utils.send_media_to_whisperx")
def test_extract_text_from_media_returns_segments_on_success(mock_send):
    """Returns transcription segments when whisperx succeeds."""
    mock_send.return_value = iter(["Hello world"])
    
    result = DocumentProcessor.extract_text_from_media("audio.mp3")
    
    assert result == ["Hello world"]
    mock_send.assert_called_once_with("audio.mp3")

@patch("utils.whisper_utils.send_media_to_whisperx")
def test_extract_text_from_media_returns_none_on_whisperx_error(mock_send):
    """Returns None when whisperx raises an exception."""
    mock_send.side_effect = RuntimeError("audio error")
    
    result = DocumentProcessor.extract_text_from_media("audio.wav")

    assert result is None


# ---------------------------------------------------------------------------
# Module-level alias
# ---------------------------------------------------------------------------


def test_module_alias_extract_text_from_html():
    """Module-level extract_text_from_html is the same object as the static method."""
    assert extract_text_from_html is DocumentProcessor.extract_text_from_html
