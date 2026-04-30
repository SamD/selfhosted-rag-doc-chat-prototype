from unittest.mock import patch

import duckdb
from services.database import DatabaseService


def test_fresh_db_initialization(tmp_path):
    """
    Verifies that a fresh database is correctly initialized from schema.sql.
    """
    db_file = str(tmp_path / "init_test.duckdb")

    # Patch the DUCKDB_FILE setting to use our test file
    with patch("services.database.settings.DUCKDB_FILE", db_file):
        # Trigger initialization
        DatabaseService.init_db()

        # Verify tables exist
        con = duckdb.connect(db_file)

        # 1. Check lifecycle table
        res = con.execute("SELECT count(*) FROM ingestion_lifecycle").fetchone()
        assert res is not None

        # 2. Check parquet_chunks table
        res = con.execute("SELECT count(*) FROM parquet_chunks").fetchone()
        assert res is not None

        # 3. Check staged_chunks table
        res = con.execute("SELECT count(*) FROM staged_chunks").fetchone()
        assert res is not None

        # 4. Check a specific column to ensure schema matches
        cols = [desc[0] for desc in con.execute("SELECT * FROM ingestion_lifecycle LIMIT 0").description]
        assert "md_path" in cols
        assert "worker_id" in cols

        con.close()


def test_idempotent_db_initialization(tmp_path):
    """
    Verifies that calling init_db multiple times does not cause errors.
    """
    db_file = str(tmp_path / "idempotent_test.duckdb")

    with patch("services.database.settings.DUCKDB_FILE", db_file):
        # Call multiple times
        DatabaseService.init_db()
        DatabaseService.init_db()
        DatabaseService.init_db()

        # If it reaches here without an exception, it is idempotent.
        assert True
