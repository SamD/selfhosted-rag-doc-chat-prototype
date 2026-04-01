from unittest.mock import patch

from services.job_service import STATUS_COMPLETED, STATUS_PROCESSING, get_job_status, init_job_db, is_file_processed, update_job_status


def test_job_service_lifecycle(tmp_path):
    db_file = str(tmp_path / "test_jobs.duckdb")

    with patch("services.job_service.DUCKDB_FILE", db_file):
        init_job_db()

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
