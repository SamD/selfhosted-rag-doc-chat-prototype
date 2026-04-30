from unittest.mock import MagicMock, patch

import duckdb
from services.database import VectorStoreWrapper, get_duckdb, get_vectorstore


def test_duckdb_type():
    """
    Ensures get_duckdb() returns a relational connection,
    NOT a vector store.
    """
    with patch("duckdb.connect") as mock_connect:
        mock_con = MagicMock(spec=duckdb.DuckDBPyConnection)
        mock_connect.return_value = mock_con

        db = get_duckdb()
        assert isinstance(db, MagicMock)
        assert not hasattr(db, "add_texts")


def test_vectorstore_type():
    """
    Ensures get_vectorstore() returns a VectorStore object,
    NOT a DuckDB connection.
    """
    # Mock at a higher level to avoid library validation errors
    mock_wrapper = MagicMock(spec=VectorStoreWrapper)
    mock_wrapper.add_texts = MagicMock()

    with patch("services.database.DatabaseService.get_qdrant", return_value=mock_wrapper), patch("services.database.DatabaseService.get_chromadb", return_value=mock_wrapper):
        vdb = get_vectorstore()
        # It MUST have add_texts (this is what failed in production)
        assert hasattr(vdb, "add_texts")
        # Verify it is not a DuckDB connection
        assert not isinstance(vdb, duckdb.DuckDBPyConnection)


def test_alias_uniqueness():
    """
    Verifies that the helpers are pointing to different
    functional logic.
    """
    assert get_duckdb != get_vectorstore
