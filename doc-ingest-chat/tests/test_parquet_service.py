from unittest.mock import MagicMock, patch

from services.parquet_service import append_chunks, commit_to_parquet, init_schema


def test_init_schema():
    with patch("services.job_service.JobService._execute_with_retry") as mock_retry:
        init_schema()
        mock_retry.assert_called_once()


def test_append_chunks():
    with patch("duckdb.connect") as mock_connect:
        mock_con = MagicMock()
        mock_connect.return_value = mock_con
        append_chunks([{"id": "1", "chunk": "t", "source_file": "f", "type": "p", "chunk_index": 0, "hash": "h", "page": 1}])
        mock_con.register.assert_called_once()
        mock_con.execute.assert_called_once()


def test_commit_to_parquet():
    with patch("services.job_service.JobService._execute_with_retry") as mock_retry:
        commit_to_parquet()
        mock_retry.assert_called_once()
