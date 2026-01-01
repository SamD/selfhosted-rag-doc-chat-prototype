"""
Tests for vector database configuration and automatic port selection.
"""

import os
import sys
from unittest.mock import patch

# Ensure the config module is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_qdrant_profile_sets_correct_default_port():
    """Test that VECTOR_DB_PROFILE=qdrant sets default port to 6333."""
    with patch.dict(os.environ, {"VECTOR_DB_PROFILE": "qdrant", "VECTOR_DB_HOST": "vector-db", "EMBEDDING_MODEL_PATH": "/fake/path", "INGEST_FOLDER": "/fake/ingest", "CHROMA_DATA_DIR": "/fake/chroma", "LLM_PATH": "/fake/llm.gguf"}, clear=True):
        # Reload config to pick up environment changes
        import importlib

        from config import settings

        importlib.reload(settings)

        # Verify Qdrant settings
        assert settings.VECTOR_DB_PROFILE == "qdrant"
        assert settings.USE_QDRANT is True
        assert settings.VECTOR_DB_PORT == 6333
        assert settings.VECTOR_DB_HOST == "vector-db"


def test_chroma_profile_sets_correct_default_port():
    """Test that VECTOR_DB_PROFILE=chroma sets default port to 8000."""
    with patch.dict(os.environ, {"VECTOR_DB_PROFILE": "chroma", "VECTOR_DB_HOST": "vector-db", "EMBEDDING_MODEL_PATH": "/fake/path", "INGEST_FOLDER": "/fake/ingest", "CHROMA_DATA_DIR": "/fake/chroma", "LLM_PATH": "/fake/llm.gguf"}, clear=True):
        # Reload config to pick up environment changes
        import importlib

        from config import settings

        importlib.reload(settings)

        # Verify ChromaDB settings
        assert settings.VECTOR_DB_PROFILE == "chroma"
        assert settings.USE_QDRANT is False
        assert settings.VECTOR_DB_PORT == 8000
        assert settings.VECTOR_DB_HOST == "vector-db"


def test_explicit_port_overrides_default():
    """Test that explicitly setting VECTOR_DB_PORT overrides the default."""
    with patch.dict(
        os.environ,
        {
            "VECTOR_DB_PROFILE": "qdrant",
            "VECTOR_DB_HOST": "vector-db",
            "VECTOR_DB_PORT": "9999",  # Explicit override
            "EMBEDDING_MODEL_PATH": "/fake/path",
            "INGEST_FOLDER": "/fake/ingest",
            "CHROMA_DATA_DIR": "/fake/chroma",
            "LLM_PATH": "/fake/llm.gguf",
        },
        clear=True,
    ):
        # Reload config to pick up environment changes
        import importlib

        from config import settings

        importlib.reload(settings)

        # Verify explicit port is used
        assert settings.VECTOR_DB_PORT == 9999


def test_backward_compatibility_with_chroma_host():
    """Test that old CHROMA_HOST/PORT variables still work."""
    with patch.dict(os.environ, {"CHROMA_HOST": "old-chromadb", "CHROMA_PORT": "7777", "EMBEDDING_MODEL_PATH": "/fake/path", "INGEST_FOLDER": "/fake/ingest", "CHROMA_DATA_DIR": "/fake/chroma", "LLM_PATH": "/fake/llm.gguf"}, clear=True):
        # Reload config to pick up environment changes
        import importlib

        from config import settings

        importlib.reload(settings)

        # Verify backward compatibility aliases work
        assert settings.CHROMA_HOST == settings.VECTOR_DB_HOST
        assert settings.CHROMA_PORT == settings.VECTOR_DB_PORT
        assert settings.CHROMA_COLLECTION == settings.VECTOR_DB_COLLECTION


def test_profile_is_case_insensitive():
    """Test that VECTOR_DB_PROFILE is case-insensitive."""
    with patch.dict(
        os.environ,
        {
            "VECTOR_DB_PROFILE": "QDRANT",  # Uppercase
            "VECTOR_DB_HOST": "vector-db",
            "EMBEDDING_MODEL_PATH": "/fake/path",
            "INGEST_FOLDER": "/fake/ingest",
            "CHROMA_DATA_DIR": "/fake/chroma",
            "LLM_PATH": "/fake/llm.gguf",
        },
        clear=True,
    ):
        # Reload config to pick up environment changes
        import importlib

        from config import settings

        importlib.reload(settings)

        # Verify lowercase conversion
        assert settings.VECTOR_DB_PROFILE == "qdrant"
        assert settings.USE_QDRANT is True


def test_default_collection_name():
    """Test that VECTOR_DB_COLLECTION has a sensible default."""
    with patch.dict(os.environ, {"VECTOR_DB_PROFILE": "qdrant", "VECTOR_DB_HOST": "vector-db", "EMBEDDING_MODEL_PATH": "/fake/path", "INGEST_FOLDER": "/fake/ingest", "CHROMA_DATA_DIR": "/fake/chroma", "LLM_PATH": "/fake/llm.gguf"}, clear=True):
        # Reload config to pick up environment changes
        import importlib

        from config import settings

        importlib.reload(settings)

        # Verify collection name defaults are reasonable
        assert settings.VECTOR_DB_COLLECTION is not None
        assert isinstance(settings.VECTOR_DB_COLLECTION, str)
        assert len(settings.VECTOR_DB_COLLECTION) > 0
