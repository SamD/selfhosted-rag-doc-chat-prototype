import os
import unittest
from unittest.mock import MagicMock, patch

# Setup environment variables for testing
os.environ["INGEST_FOLDER"] = "/tmp/test_ingest"
os.environ["EMBEDDING_MODEL_PATH"] = "/tmp/test_models"
os.environ["LLM_PATH"] = "/tmp/test_models/model.gguf"
os.environ["SUPERVISOR_LLM_PATH"] = "/tmp/test_models/model.gguf"

from workers.gatekeeper_logic import assemble_metadata, get_slug, sliding_window_normalize


class TestSlidingWindowNormalization(unittest.TestCase):
    @patch("pdfplumber.open")
    def test_sliding_window_chunking(self, mock_pdf_open):
        # Mock PDF extraction
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "A" * 5000  # Enough for 2 chunks with 3000 size + 500 overlap
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        # 2. Run Normalization
        file_path = "test.pdf"
        # Use small chunk size to force multiple chunks easily
        chunks = sliding_window_normalize(file_path, chunk_size=3000, overlap=500)

        # 3. Assertions
        # Should have 2 chunks
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 3000)
        self.assertEqual(len(chunks[1]), 2502)  # Remaining 2000 + 500 overlap from prev + 2 newlines (from joining)

        print("\n✅ Sliding window chunking test passed!")

    def test_slug_generation(self):
        slug1 = get_slug("Test Document Name")
        slug2 = get_slug("Test Document Name")
        self.assertEqual(slug1, slug2)
        self.assertIn("test-document-name", slug1)
        print(f"✅ Slug: {slug1}")

    def test_metadata_assembly(self):
        meta = assemble_metadata("test.pdf", "test-slug", 0, 5)
        self.assertEqual(meta["chunk_index"], 0)
        self.assertEqual(meta["total_chunks"], 5)
        self.assertEqual(meta["source_type"], "pdf_ocr_raw")
        print("✅ Metadata assembly passed!")


if __name__ == "__main__":
    unittest.main()
