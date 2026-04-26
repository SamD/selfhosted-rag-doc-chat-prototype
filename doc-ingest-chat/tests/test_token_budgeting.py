from unittest.mock import MagicMock, patch

import pytest
from config.settings import MAX_TOKENS
from processors.text_processor import split_markdown_doc, validate_chunk


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()

    # Mock simple whitespace tokenization for tests
    def encode_mock(text, add_special_tokens=False, **kwargs):
        tokens = text.split()
        if add_special_tokens:
            return [101] + tokens + [102]  # CLS + data + SEP
        return tokens

    tokenizer.encode.side_effect = encode_mock
    # Mock decode to return the joined words
    tokenizer.decode.side_effect = lambda tokens, **kwargs: " ".join([str(t) for t in tokens if t not in [101, 102]])
    return tokenizer


def test_markdown_splitting_safety_margin(mock_tokenizer):
    """
    Ensures that the Markdown splitter produces chunks that
    account for the RAG prefix and stay under the 512 limit.
    """
    # Create a long piece of unique text that definitely needs splitting
    sentences = [f"Unique-sentence-number-{i}-providing-enough-context-for-splitting-tests" for i in range(100)]
    md_content = "# Header\n\n" + "\n\n".join(sentences)

    doc_id = "DOC_TEST_123"
    prefix = "passage: "

    with patch("processors.text_processor.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        chunks, metas = split_markdown_doc(md_content, "test.md", tokenizer=mock_tokenizer, budget=MAX_TOKENS, prefix=prefix, document_id=doc_id)

    assert len(chunks) > 1

    for chunk in chunks:
        # Full enrichment check
        enriched_chunk = f"{prefix}[{doc_id}] {chunk}"
        tokens = mock_tokenizer.encode(enriched_chunk, add_special_tokens=True)
        assert len(tokens) <= MAX_TOKENS


def test_validate_chunk_strictness(mock_tokenizer):
    """Verifies that oversized chunks are truncated, not dropped."""
    # 511 'words'
    boundary_text = "word " * 511

    # 1. Base length is 511. Encode(add_special_tokens=False) returns 511.
    assert len(mock_tokenizer.encode(boundary_text, add_special_tokens=False)) == 511

    # 2. Validator adds 2 special tokens -> 513.
    # It should return a truncated string of length 510 + special tokens = 512.
    final_chunk = validate_chunk(boundary_text, mock_tokenizer)

    # Verify the result is a string and has the correct length
    assert isinstance(final_chunk, str)
    final_tokens = mock_tokenizer.encode(final_chunk, add_special_tokens=True)
    assert len(final_tokens) <= MAX_TOKENS


def test_splitter_validator_parity(mock_tokenizer):
    """Splitter output MUST pass through the Validator without further truncation."""
    dense_text = "Dense-sentence-repeated-to-test-boundaries " * 500
    doc_id = "DOC_PARITY"

    with patch("processors.text_processor.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        chunks, _ = split_markdown_doc(dense_text, "parity.md", tokenizer=mock_tokenizer, document_id=doc_id)

    assert len(chunks) > 1

    for chunk in chunks:
        # The Producer now prepends the prefix
        stored_chunk = f"passage: [{doc_id}] {chunk}"

        # The Validator should return the string UNTOUCHED because it's already safe
        validated_chunk = validate_chunk(stored_chunk, mock_tokenizer)
        assert validated_chunk == stored_chunk
