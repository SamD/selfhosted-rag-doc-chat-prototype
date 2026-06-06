from unittest.mock import patch

from services.job_service import STATUS_INGEST_FAILED, STATUS_INGESTING, STATUS_NEW, STATUS_PREPROCESSING, JobService


def test_reclaim_orphaned_jobs_resets_stuck_jobs(tmp_path):
    db_file = str(tmp_path / "test_reclaim.duckdb")
    with patch("config.settings.DUCKDB_FILE", db_file):
        JobService.ensure_schema(quiet=True, db_path=db_file)

        # Create a job and manually set it to PREPROCESSING with old timestamp
        job_id = JobService.create_job("/tmp/staging/test.pdf")
        assert job_id is not None

        JobService.transition_job(job_id, STATUS_PREPROCESSING)

        # Set the timestamp back by 2 hours so it's eligible for reclaim
        JobService._execute_with_retry(
            "UPDATE ingestion_lifecycle SET preprocessing_at = CURRENT_TIMESTAMP - INTERVAL '2' HOURS WHERE id = ?",
            (job_id,),
        )

        reclaimed = JobService.reclaim_orphaned_jobs(STATUS_PREPROCESSING, STATUS_NEW, timeout_hours=1)
        assert len(reclaimed) == 1
        assert reclaimed[0]["id"] == job_id

        # Verify it's now in NEW status
        res, cols = JobService._execute_with_retry(
            "SELECT status FROM ingestion_lifecycle WHERE id = ?", (job_id,), fetch=True
        )
        assert res[0] == STATUS_NEW


def test_reclaim_orphaned_jobs_skips_recent_jobs(tmp_path):
    db_file = str(tmp_path / "test_skip_recent.duckdb")
    with patch("config.settings.DUCKDB_FILE", db_file):
        JobService.ensure_schema(quiet=True, db_path=db_file)

        job_id = JobService.create_job("/tmp/staging/test.pdf")
        JobService.transition_job(job_id, STATUS_PREPROCESSING)

        # Recent job (not 2 hours old) should NOT be reclaimed
        reclaimed = JobService.reclaim_orphaned_jobs(STATUS_PREPROCESSING, STATUS_NEW, timeout_hours=1)
        assert len(reclaimed) == 0


def test_create_job_allows_reingestion_after_failure(tmp_path):
    db_file = str(tmp_path / "test_reingest.duckdb")
    with patch("config.settings.DUCKDB_FILE", db_file):
        JobService.ensure_schema(quiet=True, db_path=db_file)

        # First ingestion fails
        job1 = JobService.create_job("/tmp/staging/test.pdf")
        JobService.transition_job(job1, STATUS_INGEST_FAILED)

        # Second attempt should succeed (re-ingestion allowed)
        job2 = JobService.create_job("/tmp/staging/test.pdf")
        assert job2 is not None
        assert job2 != job1

        # Both records should exist
        res, cols = JobService._execute_with_retry(
            "SELECT status FROM ingestion_lifecycle WHERE original_filename = ? ORDER BY new_at",
            ("test.pdf",), fetch_all=True
        )
        assert len(res) == 2
        assert res[0][0] == STATUS_INGEST_FAILED
        assert res[1][0] == STATUS_NEW


def test_cleanup_failed_job_deletes_staged_chunks(tmp_path):
    db_file = str(tmp_path / "test_cleanup.duckdb")
    with patch("config.settings.DUCKDB_FILE", db_file):
        JobService.ensure_schema(quiet=True, db_path=db_file)

        # Insert a staged chunk
        JobService._execute_with_retry(
            "INSERT INTO staged_chunks (id, source_file, chunk, metadata) VALUES (?, ?, ?, ?)",
            ("chunk1", "test.pdf", "content", "{}"),
        )

        # Verify it exists
        res, _ = JobService._execute_with_retry(
            "SELECT COUNT(*) FROM staged_chunks WHERE source_file = ?", ("test.pdf",), fetch=True
        )
        assert res[0] == 1

        # Cleanup
        JobService.cleanup_failed_job("job_id", "test.pdf")

        # Verify it's gone
        res, _ = JobService._execute_with_retry(
            "SELECT COUNT(*) FROM staged_chunks WHERE source_file = ?", ("test.pdf",), fetch=True
        )
        assert res[0] == 0


def test_create_job_still_blocks_active_ingestion(tmp_path):
    db_file = str(tmp_path / "test_block_active.duckdb")
    with patch("config.settings.DUCKDB_FILE", db_file):
        JobService.ensure_schema(quiet=True, db_path=db_file)

        # Create a job and leave it in NEW (active)
        JobService.create_job("/tmp/staging/test.pdf")

        # Second attempt should be blocked
        job2 = JobService.create_job("/tmp/staging/test.pdf")
        assert job2 is None
