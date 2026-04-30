from unittest.mock import patch

from services.job_service import STATUS_COMPLETED, STATUS_NEW, STATUS_PROCESSING, JobService, get_job_status, init_job_db, is_file_processed, update_job_status


def test_job_service_lifecycle(tmp_path):
    db_file = str(tmp_path / "test_jobs.duckdb")

    with patch("config.settings.DUCKDB_FILE", db_file):
        # We pass the db_file explicitly to avoid using the real one from settings
        init_job_db(db_path=db_file)

        update_job_status("test.pdf", STATUS_PROCESSING, job_id="job1")
        job = get_job_status("test.pdf")
        assert job["status"] == STATUS_PROCESSING
        assert job["job_id"] == "job1"
        # is_file_processed now returns True for STATUS_PROCESSING to prevent re-ingestion
        assert is_file_processed("test.pdf") is True

        update_job_status("test.pdf", STATUS_COMPLETED)
        assert is_file_processed("test.pdf") is True

        update_job_status("error.pdf", "failed", error_message="something went wrong")
        job = get_job_status("error.pdf")
        assert job["status"] == "failed"
        assert job["error_message"] == "something went wrong"


def test_job_lifecycle_multi_type(tmp_path):
    """
    Verifies that the database correctly tracks multiple types (PDF and MP4).
    """
    db_file = str(tmp_path / "test_multi_type.duckdb")

    with patch("config.settings.DUCKDB_FILE", db_file):
        # Pass the db_file explicitly
        init_job_db(db_path=db_file)

        # 1. Create jobs for different types
        JobService.create_job("/tmp/staging/lecture.pdf")
        JobService.create_job("/tmp/staging/presentation.mp4")

        # 2. Verify both exist in the new lifecycle table
        res, cols = JobService._execute_with_retry("SELECT original_filename FROM ingestion_lifecycle WHERE status = ?", (STATUS_NEW,), fetch_all=True)

        filenames = [row[0] for row in res]
        assert "lecture.pdf" in filenames
        assert "presentation.mp4" in filenames
        assert len(filenames) == 2
