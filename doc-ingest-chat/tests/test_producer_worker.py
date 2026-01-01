import os
import sys
from unittest.mock import Mock, patch

# Set required environment variables before importing
os.environ.setdefault("INGEST_FOLDER", "/tmp/test")
os.environ.setdefault("CHROMA_DATA_DIR", "/tmp/chroma")
os.environ.setdefault("EMBEDDING_MODEL_PATH", "intfloat/e5-large-v2")
os.environ.setdefault("LLM_PATH", "/tmp/test.gguf")

# Ensure the worker module is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../workers")))

# Mock the tokenizer before importing producer_worker
with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
    mock_tokenizer.return_value = Mock()
    import producer_worker


def test_is_bad_ocr_true():
    # Should return True for empty or gibberish text
    with patch("producer_worker.is_gibberish", return_value=True), patch("producer_worker.is_visibly_corrupt", return_value=False), patch("producer_worker.is_low_quality", return_value=False):
        assert producer_worker.is_bad_ocr("nonsense") is True


def test_is_bad_ocr_false():
    # Should return False for good text
    with patch("producer_worker.is_gibberish", return_value=False), patch("producer_worker.is_visibly_corrupt", return_value=False), patch("producer_worker.is_low_quality", return_value=False):
        assert producer_worker.is_bad_ocr("This is valid text.") is False


def test_md5_from_int_list():
    result = producer_worker.md5_from_int_list([1, 2, 3])
    assert isinstance(result, str)
    assert len(result) == 32
