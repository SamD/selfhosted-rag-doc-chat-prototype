from unittest.mock import MagicMock

import pytest
from processors.text_processor import split_markdown_doc, validate_chunk


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer that simulates token counts."""
    tokenizer = MagicMock()
    # Simulate: 1 character = 1 token for simple testing
    tokenizer.encode.side_effect = lambda text, **kwargs: list(range(len(text)))
    # Simple decode: join tokens back (tokens are just indices)
    tokenizer.decode.side_effect = lambda tokens, **kwargs: "x" * len(tokens)
    return tokenizer

def test_zero_loss_sub_splitting_markdown(mock_tokenizer):
    """Verify that split_markdown_doc sub-splits instead of truncating."""
    # Create a document with a block of text that exceeds the internal safe budget
    # Internal budget is min(450, 512-prefix)
    massive_block = "A" * 600 
    doc = f"---\ntitle: test\n---\n# Header\n{massive_block}"
    
    # MAX_TOKENS is 512. Prefix will take some.
    chunks, metadata = split_markdown_doc(doc, "test.md", tokenizer=mock_tokenizer)
    
    # 1. Should have multiple chunks due to sub-splitting
    assert len(chunks) > 1
    
    # 2. Reconstructed text should match total length (approx)
    total_len = sum(len(c) for c in chunks)
    assert total_len >= 600

def test_zero_loss_validator(mock_tokenizer):
    """Verify that validate_chunk sub-splits instead of truncating."""
    massive_block = "B" * 1000 # Well over 512 tokens
    
    sub_chunks = validate_chunk(massive_block, mock_tokenizer)
    
    # 1. Should return multiple chunks
    assert len(sub_chunks) > 1
    
    # 2. Reconstructed text should match original length
    total_len = sum(len(sc) for sc in sub_chunks)
    assert total_len == 1000
