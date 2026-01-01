import os
import sys

# Set required environment variables before importing
os.environ.setdefault("INGEST_FOLDER", "/tmp/test")
os.environ.setdefault("CHROMA_DATA_DIR", "/tmp/chroma")
os.environ.setdefault("EMBEDDING_MODEL_PATH", "intfloat/e5-large-v2")
os.environ.setdefault("LLM_PATH", "/tmp/test.gguf")

# Ensure the worker module is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../workers")))
import ocr_worker


def test_safe_image_save_success(tmp_path):
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="red")
    path = tmp_path / "test.png"
    assert ocr_worker.safe_image_save(img, str(path)) is True


def test_safe_image_save_failure():
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="red")
    # Invalid path should fail
    assert ocr_worker.safe_image_save(img, "/invalid/path/test.png") is False


def test_fallback_to_tesseract_invalid(monkeypatch):
    monkeypatch.setattr(ocr_worker, "is_invalid_text", lambda x: True)
    import numpy as np

    arr = np.zeros((10, 10), dtype=np.uint8)
    text, engine, time_ms = ocr_worker.fallback_to_tesseract(arr, "file", 1)
    assert text is None
    assert engine.startswith("notext_")
    assert isinstance(time_ms, (int, float))
