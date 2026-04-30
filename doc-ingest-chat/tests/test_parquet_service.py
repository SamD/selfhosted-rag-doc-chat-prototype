from unittest.mock import patch

import duckdb
from services.database import DatabaseService
from services.parquet_service import append_chunks, commit_to_parquet, get_staged_chunks, init_schema, stage_chunk


def test_init_schema():
    with patch("services.database.DatabaseService.init_db") as mock_init:
        init_schema()
        mock_init.assert_called_once()


def test_append_chunks(tmp_path):
    db_file = str(tmp_path / "append_test.duckdb")
    with patch("config.settings.DUCKDB_FILE", db_file):
        DatabaseService.init_db(db_path=db_file)

        append_chunks([{"id": "1", "chunk": "t", "source_file": "f", "type": "p", "chunk_index": 0, "hash": "h", "page": 1, "document_id": "d1"}])

        con = duckdb.connect(db_file)
        res = con.execute("SELECT count(*) FROM parquet_chunks").fetchone()
        assert res[0] == 1
        con.close()


def test_commit_to_parquet(tmp_path):
    db_file = str(tmp_path / "parquet_test.duckdb")
    with patch("config.settings.DUCKDB_FILE", db_file):
        DatabaseService.init_db(db_path=db_file)
        # Verify call succeeds without CatalogException
        commit_to_parquet()


def test_stage_chunk_logic(tmp_path):
    """Verifies that stage_chunk correctly persists to DuckDB."""
    db_file = str(tmp_path / "stage_test.duckdb")
    with patch("config.settings.DUCKDB_FILE", db_file):
        DatabaseService.init_db(db_path=db_file)

        test_entry = {"id": "chunk_1", "source_file": "test.pdf", "document_id": "DOC_123", "chunk": "hello world"}
        stage_chunk(test_entry)

        con = duckdb.connect(db_file)
        res = con.execute("SELECT count(*) FROM staged_chunks").fetchone()
        assert res[0] == 1
        con.close()


def test_get_staged_chunks_retry(tmp_path):
    """Verifies that retrieval correctly gets data from DuckDB."""
    db_file = str(tmp_path / "get_test.duckdb")
    with patch("config.settings.DUCKDB_FILE", db_file):
        DatabaseService.init_db(db_path=db_file)

        test_entry = {"id": "c1", "source_file": "test.pdf", "document_id": "d1", "chunk": "data"}
        stage_chunk(test_entry)

        chunks = get_staged_chunks("test.pdf", purge=True)
        assert len(chunks) == 1
        assert chunks[0]["id"] == "c1"


def test_duckdb_type_filtering(tmp_path):
    """
    Verifies that we can insert and differentiate chunks by type (PDF vs MP4).
    """
    db_file = str(tmp_path / "type_test.duckdb")
    with patch("config.settings.DUCKDB_FILE", db_file):
        DatabaseService.init_db(db_path=db_file)

        pdf_chunk = {"id": "p1", "chunk": "pdf data", "source_file": "a.pdf", "type": "pdf", "chunk_index": 0, "document_id": "DOC1"}
        mp4_chunk = {"id": "m1", "chunk": "mp4 data", "source_file": "b.mp4", "type": "mp4", "chunk_index": 0, "document_id": "DOC2"}

        append_chunks([pdf_chunk, mp4_chunk])

        con = duckdb.connect(db_file)
        pdf_count = con.execute("SELECT count(*) FROM parquet_chunks WHERE type = 'pdf'").fetchone()[0]
        mp4_count = con.execute("SELECT count(*) FROM parquet_chunks WHERE type = 'mp4'").fetchone()[0]
        con.close()

        assert pdf_count == 1
        assert mp4_count == 1
