import logging
from unittest.mock import patch

import pytest
from services.database import DatabaseService, execute, query
from services.parquet_service import cleanup_stale_staging, stage_chunks


@pytest.fixture
def db_file(tmp_path):
    """Create a temporary database file and patch settings."""
    path = str(tmp_path / "audit_test.duckdb")
    # Patch DUCKDB_FILE in settings to point to our temp DB
    with patch("services.database.settings.DUCKDB_FILE", path):
        DatabaseService.init_db()
        yield path

def test_cleanup_orphaned_chunks(db_file, caplog):
    """Verify that chunks with no corresponding lifecycle entry are purged."""
    caplog.set_level(logging.INFO)
    
    # 1. Stage some chunks for a file that doesn't exist in lifecycle
    chunks = [
        {"id": "c1", "source_file": "orphaned.md", "document_id": "doc1", "chunk": "text1"},
        {"id": "c2", "source_file": "orphaned.md", "document_id": "doc1", "chunk": "text2"}
    ]
    stage_chunks(chunks)
    
    # 2. Run cleanup
    cleanup_stale_staging()
    
    # 3. Verify they are gone
    res, _ = query("SELECT COUNT(*) FROM staged_chunks")
    assert res[0][0] == 0
    assert "Audit detected 2 orphaned chunks" in caplog.text
    assert "Successfully dropped 2 stale chunks" in caplog.text

def test_cleanup_non_consuming_chunks(db_file):
    """Verify that chunks for files NOT in 'CONSUMING' state are purged."""
    # 1. Add file to lifecycle as 'PREPROCESSING'
    execute("INSERT INTO ingestion_lifecycle (id, status, md_path) VALUES (?, ?, ?)", ("f1", "PREPROCESSING", "stale.md"))
    
    # 2. Stage chunks for it
    stage_chunks([{"id": "c3", "source_file": "stale.md", "document_id": "doc1", "chunk": "text3"}])
    
    # 3. Run cleanup
    cleanup_stale_staging()
    
    # 4. Verify purged
    res, _ = query("SELECT COUNT(*) FROM staged_chunks")
    assert res[0][0] == 0

def test_keep_active_consuming_chunks(db_file):
    """Verify that active 'CONSUMING' chunks within timeout are preserved."""
    # 1. Add file as 'CONSUMING'
    execute("INSERT INTO ingestion_lifecycle (id, status, md_path) VALUES (?, ?, ?)", ("f2", "CONSUMING", "active.md"))
    
    # 2. Stage chunks
    stage_chunks([{"id": "c4", "source_file": "active.md", "document_id": "doc1", "chunk": "text4"}])
    
    # 3. Run cleanup
    cleanup_stale_staging()
    
    # 4. Verify preserved
    res, _ = query("SELECT COUNT(*) FROM staged_chunks")
    assert res[0][0] == 1

def test_cleanup_expired_chunks(db_file):
    """Verify that chunks older than timeout are purged even if 'CONSUMING'."""
    # 1. Add file as 'CONSUMING'
    execute("INSERT INTO ingestion_lifecycle (id, status, md_path) VALUES (?, ?, ?)", ("f3", "CONSUMING", "expired.md"))
    
    # 2. Stage chunks
    stage_chunks([{"id": "c5", "source_file": "expired.md", "document_id": "doc1", "chunk": "text5"}])
    
    # 3. Manually backdate the timestamp in DuckDB
    execute("UPDATE staged_chunks SET timestamp = (CURRENT_TIMESTAMP - INTERVAL '10' HOUR)")
    
    # 4. Run cleanup with 6h timeout
    cleanup_stale_staging(timeout_hours=6)
    
    # 5. Verify purged
    res, _ = query("SELECT COUNT(*) FROM staged_chunks")
    assert res[0][0] == 0
