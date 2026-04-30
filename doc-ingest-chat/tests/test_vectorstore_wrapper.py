import os
import sys
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.database import VectorStoreWrapper


def test_wrapper_add_texts_delegates_to_vectorstore():
    """Test that add_texts calls the underlying vectorstore."""
    mock_vectorstore = Mock(spec=["add_texts"])
    mock_vectorstore.add_texts.return_value = ["id1", "id2"]

    wrapper = VectorStoreWrapper(mock_vectorstore, "chroma")
    result = wrapper.add_texts(["text1", "text2"], metadatas=[{"k": "v"}], ids=["id1", "id2"])

    assert result == ["id1", "id2"]
    args, kwargs = mock_vectorstore.add_texts.call_args
    assert args[0] == ["text1", "text2"]
    assert kwargs["metadatas"] == [{"k": "v"}]
    assert kwargs["ids"] == ["id1", "id2"]


def test_wrapper_similarity_search_delegates_to_vectorstore():
    """Test that similarity_search calls the underlying vectorstore."""
    mock_vectorstore = Mock(spec=["similarity_search"])
    mock_vectorstore.similarity_search.return_value = ["doc1", "doc2"]

    wrapper = VectorStoreWrapper(mock_vectorstore, "qdrant")
    result = wrapper.similarity_search("query", k=2)

    assert result == ["doc1", "doc2"]
    mock_vectorstore.similarity_search.assert_called_once_with("query", 2)


def test_wrapper_as_retriever_delegates_to_vectorstore():
    """Test that as_retriever calls the underlying vectorstore."""
    mock_vectorstore = Mock(spec=["as_retriever"])
    mock_retriever = Mock()
    mock_vectorstore.as_retriever.return_value = mock_retriever

    wrapper = VectorStoreWrapper(mock_vectorstore, "chroma")
    result = wrapper.as_retriever(search_kwargs={"k": 1})

    assert result == mock_retriever
    mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 1})


def test_wrapper_delete_chromadb_with_where():
    """Test that delete with where clause works for ChromaDB."""
    mock_vectorstore = Mock(spec=["delete"])
    wrapper = VectorStoreWrapper(mock_vectorstore, "chroma")

    wrapper.delete(where={"source_file": "test.pdf"})

    mock_vectorstore.delete.assert_called_once_with(None, where={"source_file": "test.pdf"})


def test_wrapper_delete_chromadb_with_ids():
    """Test that delete with IDs works for ChromaDB."""
    mock_vectorstore = Mock(spec=["delete"])
    wrapper = VectorStoreWrapper(mock_vectorstore, "chroma")

    wrapper.delete(ids=["id1", "id2"])

    mock_vectorstore.delete.assert_called_once_with(["id1", "id2"])


def test_wrapper_delete_qdrant_with_where():
    """Test that delete with where clause works for Qdrant."""
    mock_vectorstore = Mock(spec=["delete", "client"])
    mock_client = Mock()
    mock_vectorstore.client = mock_client

    wrapper = VectorStoreWrapper(mock_vectorstore, "qdrant")

    with patch.dict(os.environ, {"VECTOR_DB_COLLECTION": "test_collection"}):
        wrapper.delete(where={"source_file": "test.pdf"})

    # Verify that the client.delete was called (since our wrapper handles Qdrant 'where' via client)
    mock_client.delete.assert_called_once()


def test_wrapper_get_collection_count_chromadb():
    """Test that get_collection_count works for ChromaDB."""
    mock_vectorstore = Mock(spec=["_collection"])
    mock_collection = Mock()
    mock_collection.count.return_value = 42
    mock_vectorstore._collection = mock_collection

    wrapper = VectorStoreWrapper(mock_vectorstore, "chroma")
    count = wrapper.get_collection_count()

    assert count == 42
    mock_collection.count.assert_called_once()


def test_wrapper_get_collection_count_qdrant():
    """Test that get_collection_count works for Qdrant."""
    mock_vectorstore = Mock(spec=["client", "collection_name"])
    mock_client = Mock()
    mock_result = Mock()
    mock_result.points_count = 100
    mock_client.get_collection.return_value = mock_result
    mock_vectorstore.client = mock_client
    mock_vectorstore.collection_name = "test_col"

    wrapper = VectorStoreWrapper(mock_vectorstore, "qdrant")
    count = wrapper.get_collection_count()

    assert count == 100
    mock_client.get_collection.assert_called_once_with("test_col")
