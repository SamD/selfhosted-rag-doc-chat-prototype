"""
Tests for database service, including Qdrant collection creation and port configuration.
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


@patch("utils.logging_config.setup_logging", return_value=Mock())
def test_qdrant_collection_creation_when_missing(mock_logging):
    """Test that Qdrant collection is created when it doesn't exist."""
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.models import Distance, VectorParams

    # Set environment for Qdrant
    os.environ["VECTOR_DB_PROFILE"] = "qdrant"
    os.environ.pop("VECTOR_DB_PORT", None)  # Use default

    # Reload settings
    import importlib

    from config import settings

    importlib.reload(settings)

    # Mock QdrantClient - create a proper UnexpectedResponse exception
    mock_client = Mock()
    # Create a mock response object
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.text = '{"status":{"error":"Not found: Collection `test_collection` doesn\'t exist!"}}'
    mock_response.content = b'{"status":{"error":"Not found: Collection `test_collection` doesn\'t exist!"}}'
    # Create UnexpectedResponse with proper attributes
    error = UnexpectedResponse.for_response(mock_response)
    error.status_code = 404  # Ensure status_code is set
    mock_client.get_collection.side_effect = error

    # Mock embeddings
    mock_embeddings = Mock()
    mock_embeddings.embed_query.return_value = [0.1] * 1024  # e5-large-v2 dimension

    with patch("services.database.QdrantClient", return_value=mock_client), patch("services.database.DatabaseService._get_embeddings", return_value=mock_embeddings), patch("services.database.QdrantVectorStore") as mock_qdrant_store:
        from services.database import DatabaseService

        DatabaseService.get_qdrant()

        # Verify collection creation was attempted
        mock_client.get_collection.assert_called_once_with("test_collection")
        mock_client.create_collection.assert_called_once()

        # Verify create_collection was called with correct parameters
        call_args = mock_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert isinstance(call_args.kwargs["vectors_config"], VectorParams)
        assert call_args.kwargs["vectors_config"].size == 1024
        assert call_args.kwargs["vectors_config"].distance == Distance.COSINE

        # Verify QdrantVectorStore was initialized
        mock_qdrant_store.assert_called_once()


@patch("utils.logging_config.setup_logging", return_value=Mock())
def test_qdrant_collection_reuse_when_exists(mock_logging):
    """Test that existing Qdrant collection is reused."""
    # Set environment for Qdrant
    os.environ["VECTOR_DB_PROFILE"] = "qdrant"
    os.environ.pop("VECTOR_DB_PORT", None)

    # Reload settings
    import importlib

    from config import settings

    importlib.reload(settings)

    # Mock QdrantClient with existing collection
    mock_client = Mock()
    mock_collection_info = Mock()
    mock_client.get_collection.return_value = mock_collection_info

    # Mock embeddings
    mock_embeddings = Mock()

    with patch("services.database.QdrantClient", return_value=mock_client), patch("services.database.DatabaseService._get_embeddings", return_value=mock_embeddings), patch("services.database.QdrantVectorStore") as mock_qdrant_store:
        from services.database import DatabaseService

        DatabaseService.get_qdrant()

        # Verify collection was checked
        mock_client.get_collection.assert_called_once_with("test_collection")

        # Verify collection was NOT created
        mock_client.create_collection.assert_not_called()

        # Verify QdrantVectorStore was initialized
        mock_qdrant_store.assert_called_once()


@patch("utils.logging_config.setup_logging", return_value=Mock())
def test_qdrant_uses_correct_port(mock_logging):
    """Test that Qdrant connection uses the correct port (6333 by default)."""
    # Set environment for Qdrant
    os.environ["VECTOR_DB_PROFILE"] = "qdrant"
    os.environ.pop("VECTOR_DB_PORT", None)  # Use default

    # Reload settings
    import importlib

    from config import settings

    importlib.reload(settings)

    # Mock QdrantClient
    mock_client = Mock()
    mock_client.get_collection.return_value = Mock()

    # Mock embeddings
    mock_embeddings = Mock()

    with patch("services.database.QdrantClient", return_value=mock_client) as mock_qdrant_client, patch("services.database.DatabaseService._get_embeddings", return_value=mock_embeddings), patch("services.database.QdrantVectorStore"):
        from services.database import DatabaseService

        DatabaseService.get_qdrant()

        # Verify QdrantClient was called with correct port
        mock_qdrant_client.assert_called_once()
        call_kwargs = mock_qdrant_client.call_args.kwargs
        assert call_kwargs["host"] == "vector-db"
        assert call_kwargs["port"] == 6333


@patch("utils.logging_config.setup_logging", return_value=Mock())
def test_chromadb_uses_correct_port(mock_logging):
    """Test that ChromaDB connection uses the correct port (8000 by default)."""
    # Set environment for ChromaDB
    os.environ["VECTOR_DB_PROFILE"] = "chroma"
    os.environ.pop("VECTOR_DB_PORT", None)  # Use default

    # Reload settings
    import importlib

    from config import settings

    importlib.reload(settings)

    # Reload database module to pick up new settings
    import services.database as database_module

    importlib.reload(database_module)

    # Mock chromadb and embeddings after reload
    mock_chroma_client = Mock()

    with patch.object(database_module.DatabaseService, "_get_embeddings", return_value=Mock()), patch("services.database.chromadb.HttpClient", return_value=mock_chroma_client) as mock_http_client, patch("services.database.Chroma"):
        database_module.DatabaseService.get_chromadb()

        # Verify HttpClient was called with correct port
        mock_http_client.assert_called_once()
        call_kwargs = mock_http_client.call_args.kwargs
        assert call_kwargs["host"] == "vector-db"
        assert call_kwargs["port"] == 8000


@patch("utils.logging_config.setup_logging", return_value=Mock())
def test_qdrant_collection_creation_handles_embedding_dimension(mock_logging):
    """Test that collection creation correctly determines embedding dimension."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    # Set environment for Qdrant
    os.environ["VECTOR_DB_PROFILE"] = "qdrant"

    # Reload settings
    import importlib

    from config import settings

    importlib.reload(settings)

    # Mock QdrantClient
    mock_client = Mock()
    # Create a mock response object for 404
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.text = '{"status":{"error":"Not found: Collection `test_collection` doesn\'t exist!"}}'
    mock_response.content = b'{"status":{"error":"Not found: Collection `test_collection` doesn\'t exist!"}}'
    error = UnexpectedResponse.for_response(mock_response)
    error.status_code = 404  # Ensure status_code is set
    mock_client.get_collection.side_effect = error

    # Mock embeddings with different dimensions
    test_dimensions = [512, 768, 1024, 1536]
    for dim in test_dimensions:
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * dim

        with patch("services.database.QdrantClient", return_value=mock_client), patch("services.database.DatabaseService._get_embeddings", return_value=mock_embeddings), patch("services.database.QdrantVectorStore"):
            from services.database import DatabaseService

            DatabaseService.get_qdrant()

            # Verify create_collection was called with correct dimension
            call_args = mock_client.create_collection.call_args
            assert call_args.kwargs["vectors_config"].size == dim

            # Reset mock for next iteration
            mock_client.reset_mock()


@patch("utils.logging_config.setup_logging", return_value=Mock())
def test_qdrant_handles_non_404_errors(mock_logging):
    """Test that non-404 errors are re-raised during collection check."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    # Set environment for Qdrant
    os.environ["VECTOR_DB_PROFILE"] = "qdrant"

    # Reload settings
    import importlib

    from config import settings

    importlib.reload(settings)

    # Mock QdrantClient with 500 error (not 404)
    mock_client = Mock()
    # Create a mock response object for 500 error
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = '{"status":{"error":"Internal server error"}}'
    mock_response.content = b'{"status":{"error":"Internal server error"}}'
    error = UnexpectedResponse.for_response(mock_response)
    error.status_code = 500  # Ensure status_code is set
    mock_client.get_collection.side_effect = error

    # Mock embeddings
    mock_embeddings = Mock()

    with patch("services.database.QdrantClient", return_value=mock_client), patch("services.database.DatabaseService._get_embeddings", return_value=mock_embeddings):
        from services.database import DatabaseService

        # Should raise the exception, not try to create collection
        try:
            DatabaseService.get_qdrant()
            assert False, "Should have raised UnexpectedResponse"
        except UnexpectedResponse as e:
            assert e.status_code == 500

        # Verify create_collection was NOT called
        mock_client.create_collection.assert_not_called()


@patch("utils.logging_config.setup_logging", return_value=Mock())
def test_get_db_returns_qdrant_when_configured(mock_logging):
    """Test that get_db returns Qdrant when USE_QDRANT is True."""
    # Set environment for Qdrant
    os.environ["VECTOR_DB_PROFILE"] = "qdrant"

    # Reload settings
    import importlib

    from config import settings

    importlib.reload(settings)

    # Reload database module to pick up new settings
    import services.database as database_module

    importlib.reload(database_module)

    with patch.object(database_module.DatabaseService, "get_qdrant", return_value=Mock()) as mock_get_qdrant, patch.object(database_module.DatabaseService, "get_chromadb") as mock_get_chroma:
        database_module.get_db()

        # Verify get_qdrant was called, not get_chromadb
        mock_get_qdrant.assert_called_once()
        mock_get_chroma.assert_not_called()


@patch("utils.logging_config.setup_logging", return_value=Mock())
def test_get_db_returns_chromadb_when_configured(mock_logging):
    """Test that get_db returns ChromaDB when USE_QDRANT is False."""
    # Set environment for ChromaDB
    os.environ["VECTOR_DB_PROFILE"] = "chroma"

    # Reload settings
    import importlib

    from config import settings

    importlib.reload(settings)

    # Reload database module to pick up new settings
    import services.database as database_module

    importlib.reload(database_module)

    with patch.object(database_module.DatabaseService, "get_qdrant") as mock_get_qdrant, patch.object(database_module.DatabaseService, "get_chromadb", return_value=Mock()) as mock_get_chroma:
        database_module.get_db()

        # Verify get_chromadb was called, not get_qdrant
        mock_get_chroma.assert_called_once()
        mock_get_qdrant.assert_not_called()
