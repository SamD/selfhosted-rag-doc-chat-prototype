"""
Simple tests for vector database settings that actually work.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_use_qdrant_flag_with_qdrant_profile():
    """Test that USE_QDRANT is True when VECTOR_DB_PROFILE is qdrant."""
    # Set minimal required environment
    os.environ["VECTOR_DB_PROFILE"] = "qdrant"
    os.environ["INGEST_FOLDER"] = "/tmp/test"
    os.environ["EMBEDDING_MODEL_PATH"] = "intfloat/e5-large-v2"
    os.environ["LLM_PATH"] = "/tmp/test.gguf"
    os.environ["CHROMA_DATA_DIR"] = "/tmp/chroma"

    # Reload settings module
    import importlib

    from config import settings

    importlib.reload(settings)

    assert settings.VECTOR_DB_PROFILE == "qdrant"
    assert settings.USE_QDRANT is True


def test_use_qdrant_flag_with_chroma_profile():
    """Test that USE_QDRANT is False when VECTOR_DB_PROFILE is chroma."""
    # Set minimal required environment
    os.environ["VECTOR_DB_PROFILE"] = "chroma"
    os.environ["INGEST_FOLDER"] = "/tmp/test"
    os.environ["EMBEDDING_MODEL_PATH"] = "intfloat/e5-large-v2"
    os.environ["LLM_PATH"] = "/tmp/test.gguf"
    os.environ["CHROMA_DATA_DIR"] = "/tmp/chroma"

    # Reload settings module
    import importlib

    from config import settings

    importlib.reload(settings)

    assert settings.VECTOR_DB_PROFILE == "chroma"
    assert settings.USE_QDRANT is False


def test_default_port_for_qdrant():
    """Test that default port is 6333 for Qdrant when not explicitly set."""
    # Set minimal required environment WITHOUT explicit port
    os.environ["VECTOR_DB_PROFILE"] = "qdrant"
    os.environ["INGEST_FOLDER"] = "/tmp/test"
    os.environ["EMBEDDING_MODEL_PATH"] = "intfloat/e5-large-v2"
    os.environ["LLM_PATH"] = "/tmp/test.gguf"
    os.environ["CHROMA_DATA_DIR"] = "/tmp/chroma"

    # Remove explicit port if set
    os.environ.pop("VECTOR_DB_PORT", None)

    # Reload settings module
    import importlib

    from config import settings

    importlib.reload(settings)

    assert settings.VECTOR_DB_PORT == 6333


def test_default_port_for_chroma():
    """Test that default port is 8000 for ChromaDB when not explicitly set."""
    # Set minimal required environment WITHOUT explicit port
    os.environ["VECTOR_DB_PROFILE"] = "chroma"
    os.environ["INGEST_FOLDER"] = "/tmp/test"
    os.environ["EMBEDDING_MODEL_PATH"] = "intfloat/e5-large-v2"
    os.environ["LLM_PATH"] = "/tmp/test.gguf"
    os.environ["CHROMA_DATA_DIR"] = "/tmp/chroma"

    # Remove explicit port if set
    os.environ.pop("VECTOR_DB_PORT", None)

    # Reload settings module
    import importlib

    from config import settings

    importlib.reload(settings)

    assert settings.VECTOR_DB_PORT == 8000


def test_explicit_port_override():
    """Test that explicitly setting VECTOR_DB_PORT overrides the default."""
    # Set explicit port
    os.environ["VECTOR_DB_PROFILE"] = "qdrant"
    os.environ["VECTOR_DB_PORT"] = "9999"
    os.environ["INGEST_FOLDER"] = "/tmp/test"
    os.environ["EMBEDDING_MODEL_PATH"] = "intfloat/e5-large-v2"
    os.environ["LLM_PATH"] = "/tmp/test.gguf"
    os.environ["CHROMA_DATA_DIR"] = "/tmp/chroma"

    # Reload settings module
    import importlib

    from config import settings

    importlib.reload(settings)

    assert settings.VECTOR_DB_PORT == 9999
