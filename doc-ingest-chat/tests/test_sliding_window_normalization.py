import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Setup environment variables for testing
os.environ["INGEST_FOLDER"] = "/tmp/test_ingest"
os.environ["EMBEDDING_MODEL_PATH"] = "/tmp/test_models"
os.environ["LLM_PATH"] = "/tmp/test_models/model.gguf"
os.environ["SUPERVISOR_LLM_PATH"] = "/tmp/test_models/model.gguf"

from workers.gatekeeper_logic import sliding_window_normalize, get_slug, assemble_metadata

class TestSlidingWindowNormalization(unittest.TestCase):

    @patch("workers.gatekeeper_logic.get_llm")
    @patch("workers.gatekeeper_logic.LlamaGrammar.from_string")
    @patch("pdfplumber.open")
    def test_sliding_window_flow(self, mock_pdf_open, mock_grammar_from_string, mock_get_llm):
        # 1. Setup Mocks
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # Mock streaming responses
        def mock_llm_side_effect(prompt, **kwargs):
            if "CHUNK 1" in prompt:
                return [{"choices": [{"text": "Chunk 2 content"}]}]
            return [{"choices": [{"text": "Chunk 1 content"}]}]
        
        mock_llm.side_effect = mock_llm_side_effect
        
        # Mock PDF extraction
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "A" * 5000 # Enough for 2 chunks with 4000 size
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        # 2. Run Normalization
        file_path = "test.pdf"
        # Use small chunk size to force multiple chunks easily
        tokens = list(sliding_window_normalize(file_path, chunk_size=3000, overlap=500))
        
        full_text = "".join(tokens)
        
        # 3. Assertions
        # Should have called LLM twice (for 2 chunks)
        self.assertEqual(mock_llm.call_count, 2)
        
        # First call should have grammar
        args1, kwargs1 = mock_llm.call_args_list[0]
        self.assertIsNotNone(kwargs1.get("grammar"))
        self.assertIn("ID:", args1[0]) # Metadata in prompt (actually anchor_header is in prompt)
        
        # Second call should NOT have grammar
        args2, kwargs2 = mock_llm.call_args_list[1]
        self.assertIsNone(kwargs2.get("grammar"))
        self.assertIn("CHUNK 1", args2[0]) # Our anchor header for chunk 2
        
        print("\n✅ Sliding window flow test passed!")

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
