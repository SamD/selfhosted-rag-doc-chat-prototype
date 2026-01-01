#!/usr/bin/env python3
import importlib


def _set_minimal_env(monkeypatch, tmp_path):
    ingest = tmp_path / "ingest"
    chroma = tmp_path / "chroma"
    e5 = tmp_path / "e5-model"
    llama = tmp_path / "llama-model"
    for p in (ingest, chroma, e5, llama):
        p.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("INGEST_FOLDER", str(ingest))
    monkeypatch.setenv("CHROMA_DATA_DIR", str(chroma))
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", str(e5))
    monkeypatch.setenv("LLM_PATH", str(llama))

    # Be explicit about Latin-mode defaults used in tests
    monkeypatch.setenv("ALLOW_LATIN_EXTENDED", "true")
    monkeypatch.setenv("LATIN_SCRIPT_MIN_RATIO", "0.7")


def test_import_text_utils_startup(monkeypatch, tmp_path):
    _set_minimal_env(monkeypatch, tmp_path)

    # Import settings after env is configured
    from config import settings as settings_module

    importlib.reload(settings_module)

    assert settings_module.ALLOW_LATIN_EXTENDED is True

    # Now import text_utils which depends on settings
    from utils import text_utils

    importlib.reload(text_utils)

    # Basic sanity check using Latin text with diacritics
    latin_text = "Lȳdia āctā est; fāmā clārā, rērum doctissima."
    assert text_utils.is_gibberish(latin_text) is False
    assert text_utils.is_invalid_text("Lorem ipsum dolor sit amet, consectetur.") is False


def test_import_ocr_worker_startup(monkeypatch, tmp_path):
    _set_minimal_env(monkeypatch, tmp_path)

    # Import settings then ocr_worker to ensure import-time config is safe
    from config import settings as settings_module

    importlib.reload(settings_module)

    from workers import ocr_worker

    importlib.reload(ocr_worker)

    assert hasattr(ocr_worker, "fallback_to_tesseract")
