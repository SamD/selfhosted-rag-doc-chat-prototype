"""
Tests for VectorStoreWrapper compatibility layer.
"""

import os
import sys
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set required environment variables
os.environ.setdefault("INGEST_FOLDER", "/tmp/test")
os.environ.setdefault("CHROMA_DATA_DIR", "/tmp/chroma")
os.environ.setdefault("EMBEDDING_MODEL_PATH", "intfloat/e5-large-v2")
os.environ.setdefault("LLM_PATH", "/tmp/test.gguf")
os.environ.setdefault("VECTOR_DB_HOST", "vector-db")
os.environ.setdefault("VECTOR_DB_COLLECTION", "test_collection")


def test_wrapper_add_texts_delegates_to_vectorstore():
    """Test that add_texts calls the underlying vectorstore."""
    from services.database import VectorStoreWrapper

    mock_vectorstore = Mock()
    mock_vectorstore.add_texts.return_value = ["id1", "id2"]

    wrapper = VectorStoreWrapper(mock_vectorstore, "chroma")
    result = wrapper.add_texts(["text1", "text2"], metadatas=[{"k": "v"}], ids=["id1", "id2"])

    mock_vectorstore.add_texts.assert_called_once_with(["text1", "text2"], metadatas=[{"k": "v"}], ids=["id1", "id2"])
    assert result == ["id1", "id2"]


def test_wrapper_as_retriever_delegates_to_vectorstore():
    """Test that as_retriever calls the underlying vectorstore."""
    from services.database import VectorStoreWrapper

    mock_vectorstore = Mock()
    mock_retriever = Mock()
    mock_vectorstore.as_retriever.return_value = mock_retriever

    wrapper = VectorStoreWrapper(mock_vectorstore, "chroma")
    result = wrapper.as_retriever(search_kwargs={"k": 5})

    mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
    assert result == mock_retriever


def test_wrapper_delete_chromadb_with_where():
    """Test that delete with where clause works for ChromaDB."""
    from services.database import VectorStoreWrapper

    mock_vectorstore = Mock()
    wrapper = VectorStoreWrapper(mock_vectorstore, "chroma")

    wrapper.delete(where={"source_file": "test.pdf"})

    mock_vectorstore.delete.assert_called_once_with(where={"source_file": "test.pdf"})


def test_wrapper_delete_chromadb_with_ids():
    """Test that delete with IDs works for ChromaDB."""
    from services.database import VectorStoreWrapper

    mock_vectorstore = Mock()
    wrapper = VectorStoreWrapper(mock_vectorstore, "chroma")

    wrapper.delete(ids=["id1", "id2"])

    mock_vectorstore.delete.assert_called_once_with(ids=["id1", "id2"])


def test_wrapper_delete_qdrant_with_where():
    """Test that delete with where clause works for Qdrant."""
    from services.database import VectorStoreWrapper

    mock_vectorstore = Mock()
    mock_client = Mock()
    mock_vectorstore.client = mock_client

    wrapper = VectorStoreWrapper(mock_vectorstore, "qdrant")

    with patch.dict(os.environ, {"VECTOR_DB_COLLECTION": "test_collection"}):
        # Reload to pick up env var
        import importlib

        from config import settings

        importlib.reload(settings)

        wrapper.delete(where={"source_file": "test.pdf"})

    # Verify Qdrant client.delete was called
    mock_client.delete.assert_called_once()
    call_args = mock_client.delete.call_args
    assert "collection_name" in call_args.kwargs
    assert "points_selector" in call_args.kwargs


def test_wrapper_get_collection_count_chromadb():
    """Test that get_collection_count works for ChromaDB."""
    from services.database import VectorStoreWrapper

    mock_vectorstore = Mock()
    mock_collection = Mock()
    mock_collection.count.return_value = 42
    mock_vectorstore._collection = mock_collection

    wrapper = VectorStoreWrapper(mock_vectorstore, "chroma")
    count = wrapper.get_collection_count()

    assert count == 42
    mock_collection.count.assert_called_once()


def test_wrapper_get_collection_count_qdrant():
    """Test that get_collection_count works for Qdrant."""
    from services.database import VectorStoreWrapper

    mock_vectorstore = Mock()
    mock_client = Mock()
    mock_result = Mock()
    mock_result.count = 100
    mock_client.count.return_value = mock_result
    mock_vectorstore.client = mock_client

    wrapper = VectorStoreWrapper(mock_vectorstore, "qdrant")

    with patch.dict(os.environ, {"VECTOR_DB_COLLECTION": "test_collection"}):
        # Reload to pick up env var
        import importlib

        from config import settings

        importlib.reload(settings)

        count = wrapper.get_collection_count()

    assert count == 100
    mock_client.count.assert_called_once()
