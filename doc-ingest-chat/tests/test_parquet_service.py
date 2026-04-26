from unittest.mock import MagicMock, patch

from services.parquet_service import ParquetService, append_chunks, commit_to_parquet, get_staged_chunks, init_schema, stage_chunk


def test_init_schema():
    with patch.object(ParquetService, "_execute_protected_query") as mock_retry:
        init_schema()
        # Verify that both tables were created via retry helper
        assert mock_retry.call_count >= 2


def test_append_chunks():
    with patch("duckdb.connect") as mock_connect:
        mock_con = MagicMock()
        mock_connect.return_value = mock_con
        append_chunks([{"id": "1", "chunk": "t", "source_file": "f", "type": "p", "chunk_index": 0, "hash": "h", "page": 1}])
        mock_con.register.assert_called_once()
        mock_con.execute.assert_called_once()


def test_commit_to_parquet():
    with patch.object(ParquetService, "_execute_protected_query") as mock_retry:
        commit_to_parquet()
        mock_retry.assert_called_once()


def test_stage_chunk_logic():
    """Verifies that stage_chunk uses the retry helper and correct SQL."""
    with patch.object(ParquetService, "_execute_protected_query") as mock_retry:
        test_entry = {"id": "chunk_1", "source_file": "test.pdf", "document_id": "DOC_123", "chunk": "hello world"}

        stage_chunk(test_entry)

        # ASSERTION 1: It must use the retry helper
        mock_retry.assert_called_once()

        # ASSERTION 2: It must use INSERT OR REPLACE
        args = mock_retry.call_args[0]
        assert "INSERT OR REPLACE INTO staged_chunks" in args[0]
        assert args[1][0] == "chunk_1"


def test_get_staged_chunks_retry():
    """Verifies that retrieval also uses the retry helper."""
    with patch.object(ParquetService, "_execute_protected_query") as mock_retry:
        mock_retry.side_effect = [
            [('{"id": "c1"}',)],  # Result for SELECT
            True,  # Result for DELETE
        ]

        chunks = get_staged_chunks("test.pdf", purge=True)
        assert len(chunks) == 1

        # ASSERTION: Both SELECT and DELETE must be protected by retry
        assert mock_retry.call_count == 2
