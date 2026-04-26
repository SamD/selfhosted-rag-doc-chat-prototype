import os
import unittest
from unittest.mock import MagicMock, patch

import duckdb
from config import settings
from workers.gatekeeper_logic import (
    assemble_metadata,
    gatekeeper_extract_and_normalize,
    get_slug,
    log_gatekeeper_result,
)


class TestGatekeeperLogic(unittest.TestCase):
    def setUp(self):
        self.test_db = "/tmp/test_gatekeeper_history.db"
        self.test_ingest = "/tmp/test_ingest"
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        if not os.path.exists(self.test_ingest):
            os.makedirs(self.test_ingest)

        # Override settings for test
        settings.GATEKEEPER_FAILURE_DB = self.test_db
        settings.DUCKDB_FILE = self.test_db  # Use same for test
        settings.INGESTION_DIR = self.test_ingest

    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        if os.path.exists(self.test_ingest):
            import shutil

            shutil.rmtree(self.test_ingest)

    @patch("workers.gatekeeper_logic.pdfplumber.open")
    @patch("workers.gatekeeper_logic.process_chunk")
    @patch("workers.gatekeeper_logic.get_llm_and_grammar")
    @patch("workers.gatekeeper_logic.is_valid_pdf", return_value=True)
    def test_streaming_flushes_periodic(self, mock_valid, mock_llm, mock_pc, mock_pdf_open):
        """Verify that chunks are flushed every 10 pages even if buffer is small."""
        # Setup fake PDF with 15 pages, each having 200 chars
        # Total = 3000 chars (not enough for 6000 chunk_size, but enough for flush)
        mock_pdf = MagicMock()
        pages = []
        for i in range(15):
            p = MagicMock()
            p.extract_text.return_value = "X" * 200
            pages.append(p)
        mock_pdf.pages = pages
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        # Run normalization
        # Should call process_chunk twice: once at page 10 and once at the end
        gatekeeper_extract_and_normalize("job-123", "test.pdf", "/tmp/test_ingest/test.md")

        # Assertion: process_chunk called at least twice
        self.assertGreaterEqual(mock_pc.call_count, 2)

    @patch("workers.gatekeeper_logic.convert_from_path")
    @patch("workers.gatekeeper_logic.pdfplumber.open")
    @patch("workers.gatekeeper_logic.process_chunk")
    @patch("workers.gatekeeper_logic.get_llm_and_grammar")
    @patch("workers.gatekeeper_logic.is_valid_pdf", return_value=True)
    def test_placeholder_for_failed_extraction(self, mock_valid, mock_llm, mock_pc, mock_pdf_open, mock_convert):
        """Verify that a placeholder is added when page extraction returns None."""
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None  # Failed extraction
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf
        mock_convert.return_value = []  # Simulate no images found either

        # Run normalization
        gatekeeper_extract_and_normalize("job-123", "test.pdf", "/tmp/test_ingest/test.md")

        # Verify that process_chunk was called with the placeholder
        args, _ = mock_pc.call_args
        raw_content = args[1]
        self.assertIn("[DOCUMENT PAGE 1 EXTRACTION FAILED OR PAGE IS EMPTY]", raw_content)

    def test_log_gatekeeper_result_success(self):
        slug = "test-document"
        metadata = {"id": "123", "tier": 3}

        log_gatekeeper_result(slug, "SUCCESS", metadata=metadata)

        con = duckdb.connect(self.test_db)
        res = con.execute("SELECT status FROM gatekeeper_history WHERE slug = ?", [slug]).fetchone()
        con.close()

        self.assertIsNotNone(res)
        self.assertEqual(res[0], "SUCCESS")

    def test_get_slug(self):
        title = "Constantine the Great: A History"
        slug = get_slug(title)
        self.assertTrue(slug.startswith("constantine-the-great-a-history-"))
        self.assertEqual(len(slug.split("-")[-1]), 8)  # Blake2b suffix

    def test_assemble_metadata(self):
        slug = "test-slug"
        meta = assemble_metadata("test.pdf", slug, 0, 10)
        self.assertEqual(meta["slug"], slug)
        self.assertEqual(meta["chunk_index"], 0)
        self.assertEqual(meta["total_chunks"], 10)
        self.assertEqual(meta["source_type"], "pdf_ocr_raw")


if __name__ == "__main__":
    unittest.main()
