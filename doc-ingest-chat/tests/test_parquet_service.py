#!/usr/bin/env python3
"""
Tests for ParquetService data storage operations.
"""

import os
import sys
import threading
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
import pytest

# Set required environment variables before importing settings
os.environ.setdefault("INGEST_FOLDER", "/tmp/test")
os.environ.setdefault("CHROMA_DATA_DIR", "/tmp/chroma")
os.environ.setdefault("EMBEDDING_MODEL_PATH", "intfloat/e5-large-v2")
os.environ.setdefault("LLM_PATH", "/tmp/test.gguf")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.parquet_service import ParquetService, init_schema, write_to_parquet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary DuckDB file path."""
    return str(tmp_path / "test.duckdb")


@pytest.fixture
def tmp_parquet(tmp_path):
    """Temporary Parquet output path."""
    return str(tmp_path / "out.parquet")


@pytest.fixture
def full_entries():
    """Two fully-populated chunk entries."""
    return [
        {
            "id": "abc123",
            "chunk": "Hello world",
            "source_file": "doc.pdf",
            "type": "pdf",
            "chunk_index": 0,
            "engine": "pdfplumber",
            "hash": "deadbeef",
            "page": 1,
        },
        {
            "id": "def456",
            "chunk": "Second chunk",
            "source_file": "doc.pdf",
            "type": "pdf",
            "chunk_index": 1,
            "engine": "pdfplumber",
            "hash": "cafebabe",
            "page": 2,
        },
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_registered_df(entries: list, tmp_parquet: str) -> pd.DataFrame:
    """Run _do_write with a mocked DuckDB and return the DataFrame passed to register()."""
    mock_con = MagicMock()
    captured: dict = {}

    def capture_register(name: str, df: pd.DataFrame) -> None:
        captured[name] = df.copy()

    mock_con.register.side_effect = capture_register

    with patch("duckdb.connect", return_value=mock_con), patch("services.parquet_service.DUCKDB_FILE", "/fake/db"):
        ParquetService._do_write(entries, tmp_parquet)

    return captured.get("df")


# ---------------------------------------------------------------------------
# ensure_schema
# ---------------------------------------------------------------------------


def test_ensure_schema_creates_table(tmp_db):
    """ensure_schema creates the parquet_chunks table in DuckDB."""
    with patch("services.parquet_service.DUCKDB_FILE", tmp_db):
        ParquetService.ensure_schema()

    con = duckdb.connect(tmp_db)
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    con.close()

    assert "parquet_chunks" in tables


def test_ensure_schema_correct_columns(tmp_db):
    """parquet_chunks table has exactly the expected columns."""
    with patch("services.parquet_service.DUCKDB_FILE", tmp_db):
        ParquetService.ensure_schema()

    con = duckdb.connect(tmp_db)
    col_names = {c[0] for c in con.execute("DESCRIBE parquet_chunks").fetchall()}
    con.close()

    assert col_names == {"id", "chunk", "source_file", "type", "chunk_index", "engine", "hash", "page"}


def test_ensure_schema_id_is_primary_key(tmp_db):
    """id column is the primary key (duplicate insert raises)."""
    with patch("services.parquet_service.DUCKDB_FILE", tmp_db):
        ParquetService.ensure_schema()

    con = duckdb.connect(tmp_db)
    con.execute("INSERT INTO parquet_chunks VALUES ('dup', 'text', 'f.pdf', 'pdf', 0, 'e', 'h', 1)")
    with pytest.raises(Exception):
        con.execute("INSERT INTO parquet_chunks VALUES ('dup', 'other', 'f.pdf', 'pdf', 1, 'e', 'h', 2)")
    con.close()


def test_ensure_schema_is_idempotent(tmp_db):
    """Calling ensure_schema twice does not raise."""
    with patch("services.parquet_service.DUCKDB_FILE", tmp_db):
        ParquetService.ensure_schema()
        ParquetService.ensure_schema()


# ---------------------------------------------------------------------------
# write_to_parquet
# ---------------------------------------------------------------------------


def test_write_to_parquet_empty_entries_is_noop(tmp_parquet):
    """write_to_parquet with an empty list does not create any file."""
    ParquetService.write_to_parquet([], tmp_parquet)
    assert not os.path.exists(tmp_parquet)


def test_write_to_parquet_without_lock_calls_do_write(full_entries, tmp_parquet):
    """write_to_parquet without a lock delegates directly to _do_write."""
    with patch.object(ParquetService, "_do_write") as mock_write:
        ParquetService.write_to_parquet(full_entries, tmp_parquet)
        mock_write.assert_called_once_with(full_entries, tmp_parquet)


def test_write_to_parquet_with_lock_calls_do_write(full_entries, tmp_parquet):
    """write_to_parquet with a lock still calls _do_write with the same args."""
    lock = threading.Lock()
    with patch.object(ParquetService, "_do_write") as mock_write:
        ParquetService.write_to_parquet(full_entries, tmp_parquet, lock=lock)
        mock_write.assert_called_once_with(full_entries, tmp_parquet)


def test_write_to_parquet_lock_is_held_during_do_write(full_entries, tmp_parquet):
    """The lock must be acquired for the entire duration of _do_write."""
    lock = threading.Lock()
    lock_was_held = []

    def fake_do_write(entries, path):
        # If write_to_parquet holds the lock, a non-blocking acquire should fail
        acquired = lock.acquire(blocking=False)
        lock_was_held.append(not acquired)
        if acquired:
            lock.release()

    with patch.object(ParquetService, "_do_write", side_effect=fake_do_write):
        ParquetService.write_to_parquet(full_entries, tmp_parquet, lock=lock)

    assert lock_was_held == [True]


# ---------------------------------------------------------------------------
# _do_write: DataFrame construction
# ---------------------------------------------------------------------------


def test_do_write_columns_match_desired_order(full_entries, tmp_parquet):
    """_do_write produces a DataFrame with columns in the exact expected order."""
    df = _capture_registered_df(full_entries, tmp_parquet)
    expected = ["id", "chunk", "source_file", "type", "chunk_index", "engine", "hash", "page"]
    assert list(df.columns) == expected


def test_do_write_preserves_entry_values(full_entries, tmp_parquet):
    """_do_write preserves all values from the input entries."""
    df = _capture_registered_df(full_entries, tmp_parquet)
    assert df["id"].tolist() == ["abc123", "def456"]
    assert df["chunk"].tolist() == ["Hello world", "Second chunk"]
    assert df["source_file"].tolist() == ["doc.pdf", "doc.pdf"]
    assert df["page"].tolist() == [1, 2]


def test_do_write_missing_page_defaults_to_minus_one(tmp_parquet):
    """Missing 'page' column is filled with -1."""
    entries = [{"id": "x1", "chunk": "t", "source_file": "f.pdf", "type": "pdf", "chunk_index": 0, "engine": "e", "hash": "h"}]
    df = _capture_registered_df(entries, tmp_parquet)
    assert df["page"].iloc[0] == -1


def test_do_write_missing_non_page_columns_default_to_none(tmp_parquet):
    """Missing non-page columns are filled with None/NaN (not -1)."""
    entries = [{"id": "x2", "chunk": "t", "source_file": "f.pdf", "page": 3}]
    df = _capture_registered_df(entries, tmp_parquet)
    for col in ["type", "chunk_index", "engine", "hash"]:
        val = df[col].iloc[0]
        assert val is None or pd.isna(val), f"Expected None/NaN for '{col}', got {val!r}"


def test_do_write_extra_columns_are_dropped(tmp_parquet):
    """Columns not in the desired set are not included in the output DataFrame."""
    entries = [{"id": "x3", "chunk": "t", "source_file": "f.pdf", "type": "pdf", "chunk_index": 0, "engine": "e", "hash": "h", "page": 1, "extra_field": "should_be_gone"}]
    df = _capture_registered_df(entries, tmp_parquet)
    assert "extra_field" not in df.columns


# ---------------------------------------------------------------------------
# _do_write: DuckDB interaction
# ---------------------------------------------------------------------------


def test_do_write_copy_targets_correct_path(full_entries, tmp_parquet):
    """_do_write issues a COPY command pointing at the specified output path."""
    mock_con = MagicMock()
    with patch("duckdb.connect", return_value=mock_con), patch("services.parquet_service.DUCKDB_FILE", "/fake/db"):
        ParquetService._do_write(full_entries, tmp_parquet)

    execute_calls = [str(c) for c in mock_con.execute.call_args_list]
    assert any(tmp_parquet in call for call in execute_calls)


def test_do_write_closes_connection_on_success(full_entries, tmp_parquet):
    """_do_write closes the DuckDB connection after a successful write."""
    mock_con = MagicMock()
    with patch("duckdb.connect", return_value=mock_con), patch("services.parquet_service.DUCKDB_FILE", "/fake/db"):
        ParquetService._do_write(full_entries, tmp_parquet)

    mock_con.close.assert_called_once()


def test_do_write_swallows_connect_error(full_entries, tmp_parquet):
    """An exception raised by duckdb.connect is caught and not re-raised."""
    with patch("duckdb.connect", side_effect=RuntimeError("db exploded")), patch("services.parquet_service.DUCKDB_FILE", "/fake/db"):
        ParquetService._do_write(full_entries, tmp_parquet)  # must not raise


def test_do_write_swallows_execute_error(full_entries, tmp_parquet):
    """An exception raised by con.execute is caught and not re-raised."""
    mock_con = MagicMock()
    mock_con.execute.side_effect = RuntimeError("copy failed")
    with patch("duckdb.connect", return_value=mock_con), patch("services.parquet_service.DUCKDB_FILE", "/fake/db"):
        ParquetService._do_write(full_entries, tmp_parquet)  # must not raise


# ---------------------------------------------------------------------------
# Module-level aliases
# ---------------------------------------------------------------------------


def test_module_alias_init_schema():
    """init_schema is an alias for ParquetService.ensure_schema."""
    assert init_schema is ParquetService.ensure_schema


def test_module_alias_write_to_parquet():
    """write_to_parquet is an alias for ParquetService.write_to_parquet."""
    assert write_to_parquet is ParquetService.write_to_parquet
