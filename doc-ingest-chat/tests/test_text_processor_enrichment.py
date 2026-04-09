#!/usr/bin/env python3
"""
Tests for TextProcessor fast hash-based Document Enrichment.
Ensures the [DOC_HASH] format is strictly maintained.
"""

import os
import sys
from unittest.mock import MagicMock, patch

# Set required environment variables before importing settings
os.environ.setdefault("INGEST_FOLDER", "/tmp/test")
os.environ.setdefault("EMBEDDING_MODEL_PATH", "intfloat/e5-large-v2")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processors.text_processor import TextProcessor


def test_get_document_id():
    """Verify that document ID is a deterministic 8-char hash."""
    file_bytes = b"test content"
    doc_id = TextProcessor.get_document_id(file_bytes)
    assert doc_id.startswith("DOC_")
    assert len(doc_id) == 12  # DOC_ + 8 chars


def test_split_doc_with_hash_enrichment():
    """
    Verify that [DOC_HASH] is prepended to every chunk.
    """
    text = "This is some test content."
    rel_path = "test_doc.pdf"
    document_id = "DOC_A1B2"

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]

    with patch("processors.text_processor.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch.object(TextProcessor, "make_chunk", return_value=(3, "passage: [DOC_A1B2] chunk content")):
            chunks, metadata = TextProcessor.split_doc(text=text, rel_path=rel_path, file_type="pdf", document_id=document_id, overlap=0)

    assert len(chunks) == 1
    assert chunks[0] == "passage: [DOC_A1B2] chunk content"


def test_split_doc_without_document_id():
    """
    Verify fallback when no document_id is provided.
    """
    text = "Some text content."
    rel_path = "test_doc.pdf"

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]

    with patch("processors.text_processor.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch.object(TextProcessor, "make_chunk", return_value=(3, "passage: [DOC_UNKNOWN] original content")):
            chunks, _ = TextProcessor.split_doc(text=text, rel_path=rel_path, file_type="pdf", document_id=None, overlap=0)

    assert len(chunks) == 1
    assert chunks[0] == "passage: [DOC_UNKNOWN] original content"


def test_split_doc_preserves_metadata():
    """
    Ensure metadata remains correct with new enrichment.
    """
    text = "Content"
    rel_path = "doc.pdf"
    doc_id = "DOC_1234"

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1]

    with patch("processors.text_processor.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch.object(TextProcessor, "make_chunk", return_value=(1, "passage: [DOC_1234] Content")):
            chunks, metadata = TextProcessor.split_doc(text=text, rel_path=rel_path, file_type="pdf", document_id=doc_id, page_num=5, overlap=0)

    assert len(metadata) == 1
    assert metadata[0]["source_file"] == rel_path
    assert metadata[0]["page"] == 5
    assert chunks[0] == "passage: [DOC_1234] Content"
