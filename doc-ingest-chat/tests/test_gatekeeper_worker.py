import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Set environment variables for modules that use settings at import time
_test_temp_dir = tempfile.mkdtemp()
os.environ.setdefault("DEFAULT_DOC_INGEST_ROOT", _test_temp_dir)
os.environ.setdefault("EMBEDDING_MODEL_PATH", _test_temp_dir)
os.environ.setdefault("LLM_PATH", os.path.join(_test_temp_dir, "model.gguf"))
os.environ.setdefault("SUPERVISOR_LLM_PATH", os.environ["LLM_PATH"])

import workers.gatekeeper_logic as gatekeeper_logic  # noqa: E402
from workers.gatekeeper_worker import gatekeeper_process_file  # noqa: E402


class TestGateKeeperWorker(unittest.TestCase):
    def setUp(self):
        self.test_dir = _test_temp_dir
        self.staging_dir = os.path.join(self.test_dir, "staging")
        self.ingest_dir = os.path.join(self.test_dir, "ingestion")
        os.makedirs(self.staging_dir, exist_ok=True)
        os.makedirs(self.ingest_dir, exist_ok=True)

        # Patch settings
        self.settings_patcher = patch("config.settings.STAGING_DIR", self.staging_dir)
        self.settings_patcher2 = patch("config.settings.INGESTION_DIR", self.ingest_dir)
        self.settings_patcher3 = patch("config.settings.GATEKEEPER_FAILURE_DB", os.path.join(self.test_dir, "failures.db"))
        self.settings_patcher.start()
        self.settings_patcher2.start()
        self.settings_patcher3.start()

        # Reset global state in logic module
        gatekeeper_logic._MODEL = None
        gatekeeper_logic._GENERATOR = None

    def tearDown(self):
        self.settings_patcher.stop()
        self.settings_patcher2.stop()
        self.settings_patcher3.stop()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(_test_temp_dir)

    @patch("workers.gatekeeper_logic.Process")
    @patch("workers.gatekeeper_logic.Llama")
    @patch("workers.gatekeeper_logic.LlamaGrammar.from_string")
    @patch("workers.gatekeeper_logic.is_valid_pdf")
    @patch("workers.gatekeeper_logic.pdfplumber.open")
    @patch("workers.gatekeeper_logic.convert_from_path")
    @patch("workers.gatekeeper_logic.preprocess_image")
    @patch("workers.gatekeeper_logic.send_image_to_ocr")
    @patch("workers.gatekeeper_logic.is_bad_ocr")
    def test_gatekeeper_process_pdf_success(self, mock_is_bad_ocr, mock_send_image_to_ocr, mock_preprocess_image, mock_convert_from_path, mock_pdf_open, mock_is_valid_pdf, mock_grammar, mock_llama, mock_process):
        # Setup mocks
        mock_send_image_to_ocr.return_value = ("test_ocr_text", None, None, None, None, None)
        mock_preprocess_image.return_value = MagicMock()
        mock_convert_from_path.return_value = [MagicMock()]
        mock_is_valid_pdf.return_value = True
        mock_is_bad_ocr.return_value = False

        # Mock pdfplumber to return a fake PDF
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is a test PDF content with enough length to avoid OCR fallback."
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        # Create a dummy PDF in staging
        pdf_path = os.path.join(self.staging_dir, "test.pdf")
        with open(pdf_path, "w") as f:
            f.write("dummy pdf content")

        # We need to mock the response_dict behavior since gatekeeper_extract_and_normalize
        # reads from it.
        with patch("workers.gatekeeper_logic.Manager") as mock_manager:
            mock_m = MagicMock()
            mock_manager.return_value.__enter__.return_value = mock_m
            mock_response_dict = {0: "# Test Title\n\nThis is normalized content."}
            mock_m.dict.return_value = mock_response_dict
            mock_m.JoinableQueue.return_value = MagicMock()

            # Process the file
            success = gatekeeper_process_file(pdf_path)

        # Assertions
        self.assertTrue(success)
        # Check if file was moved to processed
        self.assertFalse(os.path.exists(pdf_path))
        self.assertTrue(os.path.exists(os.path.join(self.staging_dir, "processed", "test.pdf")))
        # Check if normalized md was saved to ingest folder
        ingest_files = [f for f in os.listdir(self.ingest_dir) if f.endswith(".md")]
        self.assertEqual(len(ingest_files), 1)

    @patch("workers.gatekeeper_logic.Process")
    @patch("workers.gatekeeper_logic.LlamaInferenceServer")
    @patch("workers.gatekeeper_logic.Llama")
    @patch("workers.gatekeeper_logic.LlamaGrammar.from_string")
    @patch("workers.gatekeeper_logic.is_valid_pdf")
    @patch("workers.gatekeeper_logic.pdfplumber.open")
    @patch("workers.gatekeeper_logic.convert_from_path")
    @patch("workers.gatekeeper_logic.preprocess_image")
    @patch("workers.gatekeeper_logic.send_image_to_ocr")
    @patch("workers.gatekeeper_logic.is_bad_ocr")
    def test_gatekeeper_process_pdf_failure_retry(self, mock_is_bad_ocr, mock_send_image_to_ocr, mock_preprocess_image, mock_convert_from_path, mock_pdf_open, mock_is_valid_pdf, mock_grammar, mock_llama_logic, mock_server_class, mock_process_logic):
        # Setup mocks to fail normalization
        mock_is_valid_pdf.return_value = True
        mock_is_bad_ocr.return_value = False

        # Mock pdf2image to return a fake PIL image
        mock_pil = MagicMock()
        mock_convert_from_path.return_value = [mock_pil]

        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is a test PDF content with enough length to avoid OCR fallback."
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        # Mock Manager to trigger failures
        with patch("workers.gatekeeper_logic.Manager") as mock_manager:
            mock_m = MagicMock()
            mock_manager.return_value.__enter__.return_value = mock_m
            mock_m.dict.return_value = {}  # Empty dict to potentially cause wait, but we'll patch process_chunk instead
            mock_m.JoinableQueue.return_value = MagicMock()

            with patch("workers.gatekeeper_logic.process_chunk") as mock_pc:
                mock_pc.side_effect = Exception("Processing failed")

                # Create a dummy PDF in staging
                pdf_path = os.path.join(self.staging_dir, "fail.pdf")
                with open(pdf_path, "w") as f:
                    f.write("dummy pdf content")

                # Process the file
                success = gatekeeper_process_file(pdf_path)

        # Assertions
        self.assertFalse(success)
        # Check if file was moved to failed
        self.assertFalse(os.path.exists(pdf_path))
        self.assertTrue(os.path.exists(os.path.join(self.staging_dir, "failed", "fail.pdf")))
        # Check if failure was logged (DuckDB file should exist)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "failures.db")))


if __name__ == "__main__":
    unittest.main()
