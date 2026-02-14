#!/usr/bin/env python3
"""
Tests for DocumentProcessor document processing functionality.
"""

import base64
import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Set required environment variables before importing settings
os.environ.setdefault("INGEST_FOLDER", "/tmp/test")
os.environ.setdefault("CHROMA_DATA_DIR", "/tmp/chroma")
os.environ.setdefault("EMBEDDING_MODEL_PATH", "intfloat/e5-large-v2")
os.environ.setdefault("LLM_PATH", "/tmp/test.gguf")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processors.document_processor import DocumentProcessor, extract_text_from_html

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_image(width: int, height: int) -> Image.Image:
    """Create a solid-colour RGB PIL image."""
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))


def _make_mock_redis(reply_payload: dict) -> MagicMock:
    """Return a mock Redis client that responds to blpop with the given payload."""
    mock_redis = MagicMock()
    mock_redis.blpop.return_value = (b"ocr_reply:job", json.dumps(reply_payload).encode())
    return mock_redis


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


def test_extract_text_from_html_returns_none_when_encoding_detection_fails(tmp_path):
    """Returns None when charset-normalizer cannot detect encoding."""
    html_file = tmp_path / "empty.html"
    html_file.write_bytes(b"")

    with patch("processors.document_processor.from_path") as mock_from_path:
        mock_from_path.return_value.best.return_value = None
        result = DocumentProcessor.extract_text_from_html(str(html_file))

    assert result is None


def test_extract_text_from_html_module_alias(tmp_path):
    """Module-level extract_text_from_html alias works identically."""
    html_file = tmp_path / "alias.html"
    html_file.write_text("<html><body><p>Alias test</p></body></html>", encoding="utf-8")

    result = extract_text_from_html(str(html_file))

    assert result is not None
    assert "Alias test" in result


# ---------------------------------------------------------------------------
# extract_text_with_pdfplumber
# ---------------------------------------------------------------------------


def _make_pdf_mock(pages_text: list) -> MagicMock:
    """Build a pdfplumber mock that yields pages with the given text strings."""
    mock_pdf = MagicMock()
    mock_pages = []
    for text in pages_text:
        page = MagicMock()
        page.extract_text.return_value = text
        mock_pages.append(page)
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = mock_pages
    return mock_pdf


def test_extract_text_with_pdfplumber_returns_text():
    """Returns concatenated text from all pages."""
    mock_pdf = _make_pdf_mock(["Page one text here.", "Page two text here."])
    with patch("pdfplumber.open", return_value=mock_pdf):
        result = DocumentProcessor.extract_text_with_pdfplumber("doc.pdf")

    assert result is not None
    assert "Page one" in result
    assert "Page two" in result


def test_extract_text_with_pdfplumber_returns_none_on_short_text():
    """Returns None when the extracted text is shorter than 10 characters."""
    mock_pdf = _make_pdf_mock(["Hi"])  # too short
    with patch("pdfplumber.open", return_value=mock_pdf):
        result = DocumentProcessor.extract_text_with_pdfplumber("doc.pdf")

    assert result is None


def test_extract_text_with_pdfplumber_returns_none_on_empty_text():
    """Returns None when all pages produce empty text (scanned PDF)."""
    mock_pdf = _make_pdf_mock([None, None])
    with patch("pdfplumber.open", return_value=mock_pdf):
        result = DocumentProcessor.extract_text_with_pdfplumber("doc.pdf")

    assert result is None


def test_extract_text_with_pdfplumber_returns_none_on_open_error():
    """Returns None when pdfplumber.open raises an exception."""
    with patch("pdfplumber.open", side_effect=RuntimeError("corrupt PDF")):
        result = DocumentProcessor.extract_text_with_pdfplumber("bad.pdf")

    assert result is None


# ---------------------------------------------------------------------------
# extract_text_from_media
# ---------------------------------------------------------------------------


def test_extract_text_from_media_raises_on_unsupported_extension():
    """Raises ValueError for file extensions not in SUPPORTED_MEDIA_EXT."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentProcessor.extract_text_from_media("document.pdf")


def test_extract_text_from_media_raises_on_unknown_extension():
    """Raises ValueError for an arbitrary unsupported extension."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentProcessor.extract_text_from_media("file.xyz")


def test_extract_text_from_media_returns_segments_on_success():
    """Returns transcription segments when whisperx succeeds."""
    mock_whisperx = MagicMock()
    mock_whisperx.load_audio.return_value = MagicMock()
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"segments": [{"text": "Hello world"}]}
    mock_whisperx.load_model.return_value = mock_model

    with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
        result = DocumentProcessor.extract_text_from_media("audio.mp3")

    assert result == [{"text": "Hello world"}]


def test_extract_text_from_media_returns_none_on_whisperx_error():
    """Returns None when whisperx raises an exception."""
    mock_whisperx = MagicMock()
    mock_whisperx.load_audio.side_effect = RuntimeError("audio error")

    with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
        result = DocumentProcessor.extract_text_from_media("audio.wav")

    assert result is None


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------


def test_preprocess_image_returns_2d_grayscale_array():
    """Output is a 2-D (H, W) grayscale numpy array."""
    img = _make_rgb_image(100, 80)
    result = DocumentProcessor.preprocess_image(img)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 2


def test_preprocess_image_small_image_not_resized():
    """Images smaller than MAX_OCR_DIM are not resized."""
    with patch("processors.document_processor.MAX_OCR_DIM", 3000):
        img = _make_rgb_image(200, 150)
        result = DocumentProcessor.preprocess_image(img)

    assert result.shape == (150, 200)


def test_preprocess_image_large_image_is_downscaled():
    """Images larger than MAX_OCR_DIM on their longest side are downscaled."""
    max_dim = 100
    with patch("processors.document_processor.MAX_OCR_DIM", max_dim):
        img = _make_rgb_image(400, 200)  # width is the longest side
        result = DocumentProcessor.preprocess_image(img)

    assert max(result.shape) <= max_dim


def test_preprocess_image_preserves_aspect_ratio():
    """Downscaled images preserve approximate aspect ratio."""
    max_dim = 100
    with patch("processors.document_processor.MAX_OCR_DIM", max_dim):
        img = _make_rgb_image(400, 200)  # 2:1 aspect ratio
        result = DocumentProcessor.preprocess_image(img)

    w, h = result.shape[1], result.shape[0]
    assert abs(w / h - 2.0) < 0.1


def test_preprocess_image_output_dtype():
    """Output array dtype is uint8 (standard image dtype)."""
    img = _make_rgb_image(50, 50)
    result = DocumentProcessor.preprocess_image(img)
    assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# send_image_to_ocr
# ---------------------------------------------------------------------------


def _make_np_image(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def test_send_image_to_ocr_pushes_job_to_redis():
    """A JSON job is pushed to the ocr_processing_job queue."""
    np_image = _make_np_image()
    reply_payload = {"text": "extracted", "rel_path": "doc.pdf", "page_num": 1, "engine": "tesseract", "job_id": "xyz"}
    mock_redis = _make_mock_redis(reply_payload)

    DocumentProcessor.send_image_to_ocr(np_image, "doc.pdf", 1, mock_redis)

    assert mock_redis.lpush.call_count == 1
    queue_name, raw_job = mock_redis.lpush.call_args[0]
    assert queue_name == "ocr_processing_job"
    job = json.loads(raw_job)
    assert job["rel_path"] == "doc.pdf"
    assert job["page_num"] == 1


def test_send_image_to_ocr_job_includes_image_data():
    """The job payload encodes image shape, dtype, and base64 pixel data."""
    np_image = _make_np_image(32, 48)
    reply_payload = {"text": "ok", "rel_path": "f.pdf", "page_num": 2, "engine": "easyocr", "job_id": "abc"}
    mock_redis = _make_mock_redis(reply_payload)

    DocumentProcessor.send_image_to_ocr(np_image, "f.pdf", 2, mock_redis)

    _, raw_job = mock_redis.lpush.call_args[0]
    job = json.loads(raw_job)
    assert job["image_shape"] == list(np_image.shape)
    assert job["image_dtype"] == str(np_image.dtype)
    # Verify the base64 round-trips back to the original bytes
    decoded = base64.b64decode(job["image_base64"])
    assert decoded == np_image.tobytes()


def test_send_image_to_ocr_waits_on_reply_key():
    """blpop is called with the reply key derived from the job_id."""
    np_image = _make_np_image()
    reply_payload = {"text": "result", "rel_path": "a.pdf", "page_num": 1, "engine": "tesseract", "job_id": "j1"}
    mock_redis = _make_mock_redis(reply_payload)

    DocumentProcessor.send_image_to_ocr(np_image, "a.pdf", 1, mock_redis)

    blpop_key = mock_redis.blpop.call_args[0][0]
    _, raw_job = mock_redis.lpush.call_args[0]
    job = json.loads(raw_job)
    assert blpop_key == f"ocr_reply:{job['job_id']}"


def test_send_image_to_ocr_returns_tuple_from_reply():
    """Return value is a 5-tuple matching the reply payload fields."""
    np_image = _make_np_image()
    reply_payload = {"text": "hello", "rel_path": "b.pdf", "page_num": 3, "engine": "easyocr", "job_id": "j99"}
    mock_redis = _make_mock_redis(reply_payload)

    text, rel_path, page_num, engine, job_id = DocumentProcessor.send_image_to_ocr(np_image, "b.pdf", 3, mock_redis)

    assert text == "hello"
    assert rel_path == "b.pdf"
    assert page_num == 3
    assert engine == "easyocr"
    assert job_id == "j99"


def test_send_image_to_ocr_raises_timeout_when_blpop_returns_none():
    """TimeoutError is raised when blpop returns None (timeout expired)."""
    np_image = _make_np_image()
    mock_redis = MagicMock()
    mock_redis.blpop.return_value = None

    with pytest.raises(TimeoutError, match="OCR timeout"):
        DocumentProcessor.send_image_to_ocr(np_image, "slow.pdf", 5, mock_redis)


def test_send_image_to_ocr_unique_job_ids():
    """Each call generates a distinct job_id."""
    np_image = _make_np_image()
    reply_payload = {"text": "t", "rel_path": "c.pdf", "page_num": 1, "engine": "e", "job_id": "x"}

    job_ids = []
    for _ in range(3):
        mock_redis = _make_mock_redis(reply_payload)
        DocumentProcessor.send_image_to_ocr(np_image, "c.pdf", 1, mock_redis)
        _, raw_job = mock_redis.lpush.call_args[0]
        job_ids.append(json.loads(raw_job)["job_id"])

    assert len(set(job_ids)) == 3


# ---------------------------------------------------------------------------
# process_pdf_by_page
# ---------------------------------------------------------------------------


def _make_pdfplumber_ctx(pages_text: list) -> MagicMock:
    """Return a pdfplumber context manager mock yielding pages."""
    mock_pdf = MagicMock()
    mock_pages = []
    for text in pages_text:
        page = MagicMock()
        page.extract_text.return_value = text
        mock_pages.append(page)
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = mock_pages
    return mock_pdf


def test_process_pdf_by_page_good_text_uses_split_doc():
    """Pages with good text are passed to TextProcessor.split_doc."""
    mock_pdf = _make_pdfplumber_ctx(["This is good readable text on page one."])
    mock_tokenizer = MagicMock()
    mock_redis = MagicMock()
    split_result = (["chunk_a"], [{"source_file": "doc.pdf"}])

    with patch("pdfplumber.open", return_value=mock_pdf), \
         patch("processors.document_processor.is_bad_ocr", return_value=False), \
         patch("processors.document_processor.TextProcessor.split_doc", return_value=split_result) as mock_split:

        chunks, metas = DocumentProcessor.process_pdf_by_page("doc.pdf", "doc.pdf", "pdf", mock_redis, mock_tokenizer)

    mock_split.assert_called_once()
    assert chunks == ["chunk_a"]
    assert len(metas) == 1


def test_process_pdf_by_page_accumulates_all_pages():
    """Chunks and metadatas from multiple pages are all returned."""
    mock_pdf = _make_pdfplumber_ctx(["Page one text.", "Page two text."])
    mock_tokenizer = MagicMock()
    mock_redis = MagicMock()
    split_result = (["chunk"], [{"source_file": "doc.pdf"}])

    with patch("pdfplumber.open", return_value=mock_pdf), \
         patch("processors.document_processor.is_bad_ocr", return_value=False), \
         patch("processors.document_processor.TextProcessor.split_doc", return_value=split_result):

        chunks, metas = DocumentProcessor.process_pdf_by_page("doc.pdf", "doc.pdf", "pdf", mock_redis, mock_tokenizer)

    assert len(chunks) == 2  # one chunk per page
    assert len(metas) == 2


def test_process_pdf_by_page_empty_page_triggers_ocr_fallback():
    """Pages with empty text trigger the OCR fallback path."""
    mock_pdf = _make_pdfplumber_ctx([""])
    mock_tokenizer = MagicMock()
    mock_redis = MagicMock()
    ocr_result = ("OCR extracted text", "doc.pdf", 1, "tesseract", "job1")
    split_result = (["ocr_chunk"], [{"source_file": "doc.pdf"}])
    fake_pil = _make_rgb_image(100, 100)

    with patch("pdfplumber.open", return_value=mock_pdf), \
         patch("processors.document_processor.is_bad_ocr", return_value=False), \
         patch("processors.document_processor.convert_from_path", return_value=[fake_pil]), \
         patch.object(DocumentProcessor, "preprocess_image", return_value=np.zeros((100, 100), dtype=np.uint8)), \
         patch.object(DocumentProcessor, "send_image_to_ocr", return_value=ocr_result), \
         patch("processors.document_processor.TextProcessor.split_doc", return_value=split_result):

        chunks, metas = DocumentProcessor.process_pdf_by_page("doc.pdf", "doc.pdf", "pdf", mock_redis, mock_tokenizer)

    assert chunks == ["ocr_chunk"]


def test_process_pdf_by_page_bad_ocr_text_triggers_fallback():
    """Pages where is_bad_ocr returns True trigger the OCR fallback."""
    mock_pdf = _make_pdfplumber_ctx(["ÃÂ garbled text ÂÃ"])
    mock_tokenizer = MagicMock()
    mock_redis = MagicMock()
    ocr_result = ("Clean OCR text here.", "doc.pdf", 1, "tesseract", "job2")
    split_result = (["clean_chunk"], [{"source_file": "doc.pdf"}])
    fake_pil = _make_rgb_image(100, 100)

    with patch("pdfplumber.open", return_value=mock_pdf), \
         patch("processors.document_processor.is_bad_ocr", side_effect=[True, False]), \
         patch("processors.document_processor.convert_from_path", return_value=[fake_pil]), \
         patch.object(DocumentProcessor, "preprocess_image", return_value=np.zeros((100, 100), dtype=np.uint8)), \
         patch.object(DocumentProcessor, "send_image_to_ocr", return_value=ocr_result), \
         patch("processors.document_processor.TextProcessor.split_doc", return_value=split_result):

        chunks, _ = DocumentProcessor.process_pdf_by_page("doc.pdf", "doc.pdf", "pdf", mock_redis, mock_tokenizer)

    assert chunks == ["clean_chunk"]


def test_process_pdf_by_page_ocr_returns_garbage_skips_page():
    """When OCR result is itself bad, the page is skipped."""
    mock_pdf = _make_pdfplumber_ctx([""])
    mock_tokenizer = MagicMock()
    mock_redis = MagicMock()
    ocr_result = ("ÃÃÂ garbage", "doc.pdf", 1, "tesseract", "job3")
    fake_pil = _make_rgb_image(100, 100)

    with patch("pdfplumber.open", return_value=mock_pdf), \
         patch("processors.document_processor.is_bad_ocr", return_value=True), \
         patch("processors.document_processor.convert_from_path", return_value=[fake_pil]), \
         patch.object(DocumentProcessor, "preprocess_image", return_value=np.zeros((100, 100), dtype=np.uint8)), \
         patch.object(DocumentProcessor, "send_image_to_ocr", return_value=ocr_result):

        chunks, metas = DocumentProcessor.process_pdf_by_page("doc.pdf", "doc.pdf", "pdf", mock_redis, mock_tokenizer)

    assert chunks == []
    assert metas == []


def test_process_pdf_by_page_pdfplumber_open_failure_returns_empty():
    """When pdfplumber cannot open the file, returns empty lists without raising."""
    with patch("pdfplumber.open", side_effect=RuntimeError("bad file")):
        chunks, metas = DocumentProcessor.process_pdf_by_page("bad.pdf", "bad.pdf", "pdf", MagicMock(), MagicMock())

    assert chunks == []
    assert metas == []


def test_process_pdf_by_page_ocr_exception_continues_to_next_page():
    """An exception during OCR is caught; processing continues with remaining pages."""
    mock_pdf = _make_pdfplumber_ctx(["", "Good second page text here."])
    mock_tokenizer = MagicMock()
    mock_redis = MagicMock()
    split_result = (["page2_chunk"], [{"source_file": "doc.pdf"}])
    def is_bad_side_effect(text, tok):
        return not text.strip()  # empty pages are "bad"

    with patch("pdfplumber.open", return_value=mock_pdf), \
         patch("processors.document_processor.is_bad_ocr", side_effect=is_bad_side_effect), \
         patch("processors.document_processor.convert_from_path", side_effect=RuntimeError("render failed")), \
         patch("processors.document_processor.TextProcessor.split_doc", return_value=split_result):

        chunks, _ = DocumentProcessor.process_pdf_by_page("doc.pdf", "doc.pdf", "pdf", mock_redis, mock_tokenizer)

    assert chunks == ["page2_chunk"]


# ---------------------------------------------------------------------------
# process_pdf_by_page_nofallback
# ---------------------------------------------------------------------------


def test_process_pdf_by_page_nofallback_returns_chunks():
    """Returns chunks and metadata for all non-empty pages."""
    mock_pdf = _make_pdfplumber_ctx(["First page text.", "Second page text."])
    mock_tokenizer = MagicMock()
    split_result = (["chunk"], [{"source_file": "doc.pdf"}])

    with patch("pdfplumber.open", return_value=mock_pdf), \
         patch("processors.document_processor.TextProcessor.split_doc", return_value=split_result):

        chunks, metas = DocumentProcessor.process_pdf_by_page_nofallback("doc.pdf", "doc.pdf", "pdf", mock_tokenizer)

    assert len(chunks) == 2
    assert len(metas) == 2


def test_process_pdf_by_page_nofallback_skips_empty_pages():
    """Empty or None pages are skipped; no chunks produced for them."""
    mock_pdf = _make_pdfplumber_ctx(["Good text here.", None, "   "])
    mock_tokenizer = MagicMock()
    split_result = (["chunk"], [{"source_file": "doc.pdf"}])

    with patch("pdfplumber.open", return_value=mock_pdf), \
         patch("processors.document_processor.TextProcessor.split_doc", return_value=split_result):

        chunks, metas = DocumentProcessor.process_pdf_by_page_nofallback("doc.pdf", "doc.pdf", "pdf", mock_tokenizer)

    assert len(chunks) == 1  # only the first non-empty page


def test_process_pdf_by_page_nofallback_no_ocr_called():
    """OCR-related functions are never called."""
    mock_pdf = _make_pdfplumber_ctx(["Page text."])
    mock_tokenizer = MagicMock()
    split_result = (["c"], [{}])

    with patch("pdfplumber.open", return_value=mock_pdf), \
         patch("processors.document_processor.TextProcessor.split_doc", return_value=split_result), \
         patch("processors.document_processor.convert_from_path") as mock_convert:

        DocumentProcessor.process_pdf_by_page_nofallback("doc.pdf", "doc.pdf", "pdf", mock_tokenizer)

    mock_convert.assert_not_called()


def test_process_pdf_by_page_nofallback_all_empty_returns_empty():
    """All-empty PDF returns empty chunk and metadata lists."""
    mock_pdf = _make_pdfplumber_ctx([None, None, ""])
    mock_tokenizer = MagicMock()

    with patch("pdfplumber.open", return_value=mock_pdf):
        chunks, metas = DocumentProcessor.process_pdf_by_page_nofallback("empty.pdf", "empty.pdf", "pdf", mock_tokenizer)

    assert chunks == []
    assert metas == []


# ---------------------------------------------------------------------------
# Module-level alias
# ---------------------------------------------------------------------------


def test_module_alias_extract_text_from_html():
    """Module-level extract_text_from_html is the same object as the static method."""
    assert extract_text_from_html is DocumentProcessor.extract_text_from_html
