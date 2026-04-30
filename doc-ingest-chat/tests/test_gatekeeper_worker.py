import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

# Set environment variables for modules that use settings at import time
_test_temp_dir = tempfile.mkdtemp()
os.environ.setdefault("DEFAULT_DOC_INGEST_ROOT", _test_temp_dir)
os.environ.setdefault("EMBEDDING_MODEL_PATH", _test_temp_dir)
os.environ.setdefault("LLM_PATH", os.path.join(_test_temp_dir, "model.gguf"))
os.environ.setdefault("SUPERVISOR_LLM_PATH", os.environ["LLM_PATH"])

from workers.gatekeeper_worker import gatekeeper_process_job  # noqa: E402


class TestGateKeeperWorker(unittest.TestCase):
    def setUp(self):
        self.test_dir = _test_temp_dir
        self.staging_dir = os.path.join(self.test_dir, "staging")
        self.ingest_dir = os.path.join(self.test_dir, "ingestion")
        self.preprocessing_dir = os.path.join(self.test_dir, "preprocessing")
        self.failed_dir = os.path.join(self.test_dir, "failed")
        self.test_db = os.path.join(self.test_dir, "test.duckdb")
        
        os.makedirs(self.staging_dir, exist_ok=True)
        os.makedirs(self.ingest_dir, exist_ok=True)
        os.makedirs(self.preprocessing_dir, exist_ok=True)
        os.makedirs(self.failed_dir, exist_ok=True)

        # Patch settings
        self.settings_patcher = patch("config.settings.STAGING_DIR", self.staging_dir)
        self.settings_patcher2 = patch("config.settings.INGESTION_DIR", self.ingest_dir)
        self.settings_patcher3 = patch("config.settings.PREPROCESSING_DIR", self.preprocessing_dir)
        self.settings_patcher4 = patch("config.settings.FAILED_DIR", self.failed_dir)
        self.settings_patcher5 = patch("config.settings.DUCKDB_FILE", self.test_db)
        
        self.settings_patcher.start()
        self.settings_patcher2.start()
        self.settings_patcher3.start()
        self.settings_patcher4.start()
        self.settings_patcher5.start()

        # Initialize real DB schema for tests
        from services.database import DatabaseService
        DatabaseService.init_db(db_path=self.test_db)

    def tearDown(self):
        self.settings_patcher.stop()
        self.settings_patcher2.stop()
        self.settings_patcher3.stop()
        self.settings_patcher4.stop()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(_test_temp_dir)

    @patch("workers.gatekeeper_logic.get_handler_chain")
    @patch("workers.gatekeeper_logic.get_llm_and_grammar")
    def test_gatekeeper_process_job_success(self, mock_llm, mock_handler):
        # Create a dummy file in staging
        file_path = os.path.join(self.staging_dir, "test.pdf")
        with open(file_path, "w") as f:
            f.write("dummy content")

        # Create real record in test DB
        from services.job_service import JobService
        job_id = JobService.create_job(file_path)
        
        job = {
            "id": job_id,
            "pdf_path": file_path,
            "original_filename": "test.pdf",
            "trace_id": "test-trace"
        }

        with patch("workers.gatekeeper_worker.gatekeeper_extract_and_normalize") as mock_gen, \
             patch("services.job_service.JobService.transition_job"), \
             patch("workers.gatekeeper_worker.log_gatekeeper_result"):
            
            def fake_gen(jid, fpath, mpath):
                # Create the file so shutil.move succeeds
                with open(mpath, "w") as f:
                    f.write("content")
                return True, {"id": "123"}
            mock_gen.side_effect = fake_gen
            
            # Process the job
            success = gatekeeper_process_job(job)

        # Assertions
        self.assertTrue(success)
        # Check if file was moved to ingestion (success path)
        self.assertFalse(os.path.exists(file_path))
        self.assertTrue(os.path.exists(os.path.join(self.ingest_dir, "test.pdf")))

    def test_gatekeeper_process_job_failure(self):
        # Create a dummy file in staging
        file_path = os.path.join(self.staging_dir, "fail.pdf")
        with open(file_path, "w") as f:
            f.write("dummy content")

        # Create real record in test DB
        from services.job_service import JobService
        job_id = JobService.create_job(file_path)

        job = {
            "id": job_id,
            "pdf_path": file_path,
            "original_filename": "fail.pdf",
            "trace_id": "fail-trace"
        }

        with patch("workers.gatekeeper_worker.gatekeeper_extract_and_normalize") as mock_gen, \
             patch("services.job_service.JobService.transition_job"), \
             patch("workers.gatekeeper_worker.log_gatekeeper_result"):
            
            mock_gen.return_value = (False, None) # Normalization failure

            # Process the job
            success = gatekeeper_process_job(job)

        # Assertions
        self.assertFalse(success)
        # Check if file was moved to failed
        self.assertFalse(os.path.exists(file_path))
        self.assertTrue(os.path.exists(os.path.join(self.failed_dir, "fail.pdf")))


if __name__ == "__main__":
    unittest.main()
