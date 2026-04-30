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

        # Initialize schema for this test DB
        from services.database import DatabaseService

        DatabaseService.init_db(db_path=self.test_db)

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

    @patch("handlers.pdf_handler.pdfplumber.open")
    @patch("workers.gatekeeper_logic.process_chunk")
    @patch("workers.gatekeeper_logic.get_llm_and_grammar")
    @patch("utils.text_utils.is_valid_pdf", return_value=True)
    def test_streaming_flushes_periodic(self, mock_valid, mock_llm, mock_pc, mock_pdf_open):
        """Verify that chunks are flushed periodically based on batch size."""
        # Setup fake PDF with 15 pages, each having 200 chars
        mock_pdf = MagicMock()
        pages = []
        for i in range(15):
            p = MagicMock()
            p.extract_text.return_value = "X" * 200
            pages.append(p)
        mock_pdf.pages = pages
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        # Run normalization
        # GATEKEEPER_BATCH_SIZE is likely 5 or 10
        gatekeeper_extract_and_normalize("job-123", "test.pdf", "/tmp/test_ingest/test.md")

        # Assertion: process_chunk called for batches
        self.assertGreaterEqual(mock_pc.call_count, 2)

    @patch("handlers.pdf_handler.convert_from_path")
    @patch("handlers.pdf_handler.pdfplumber.open")
    @patch("workers.gatekeeper_logic.process_chunk")
    @patch("workers.gatekeeper_logic.get_llm_and_grammar")
    @patch("utils.text_utils.is_valid_pdf", return_value=True)
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

    @patch("workers.gatekeeper_logic.get_handler_chain")
    @patch("workers.gatekeeper_logic.process_chunk")
    @patch("workers.gatekeeper_logic.get_llm_and_grammar")
    @patch("shutil.move")
    def test_atomic_markdown_write_success(self, mock_move, mock_llm, mock_pc, mock_handler):
        """Verify that markdown is written to a .tmp file and moved on success."""
        mock_handler.return_value.handle.return_value = iter(["Segment 1"])
        md_path = os.path.join(self.test_ingest, "test.md")
        tmp_path = f"{md_path}.tmp"
        
        # Mock process_chunk to 'create' the tmp file
        def fake_pc(idx, content, f_path, slug, out_path, trace_id=None):
            with open(out_path, "w") as f:
                f.write("content")
            return {}
        mock_pc.side_effect = fake_pc

        success, _ = gatekeeper_extract_and_normalize("job-1", "test.txt", md_path)
        
        self.assertTrue(success)
        # Verify shutil.move was called to finalize the file
        mock_move.assert_called_once_with(tmp_path, md_path)

    @patch("workers.gatekeeper_logic.get_handler_chain")
    @patch("workers.gatekeeper_logic.get_llm_and_grammar")
    @patch("os.remove")
    def test_atomic_markdown_cleanup_on_failure(self, mock_remove, mock_llm, mock_handler):
        """Verify that partial .tmp files are cleaned up on failure."""
        # Simulate a crash during streaming
        def crash_iter():
            yield "Good"
            raise RuntimeError("Crashed!")
        mock_handler.return_value.handle.return_value = crash_iter()
        
        md_path = os.path.join(self.test_ingest, "test.md")
        tmp_path = f"{md_path}.tmp"
        
        # Create a dummy tmp file to simulate partial write
        with open(tmp_path, "w") as f:
            f.write("partial")

        success, _ = gatekeeper_extract_and_normalize("job-1", "test.txt", md_path)
        
        self.assertFalse(success)
        # Verify os.remove was called for the tmp file
        mock_remove.assert_any_call(tmp_path)


if __name__ == "__main__":
    unittest.main()
