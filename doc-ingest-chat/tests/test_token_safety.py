import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add doc-ingest-chat to sys.path
sys.path.insert(0, os.path.join(os.getcwd(), "doc-ingest-chat"))

from utils.ocr_utils import run_remote_ocr
from workers.gatekeeper_logic import process_chunk


class TestTokenSafety(unittest.TestCase):

    @patch('requests.post')
    def test_ocr_find_text_ignores_base64_leakage(self, mock_post):
        """Verify that the OCR parser ignores massive non-markdown strings (base64 leakage)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Payload simulating a base64 image in a generic key and actual markdown in the correct key
        mock_response.json.return_value = {
            "document": {
                "md": "# Valid Markdown Content",
                "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==" * 1000 # Massive dummy base64
            },
            "status": "success"
        }
        mock_post.return_value = mock_response

        # Dummy image
        np_image = np.zeros((10, 10, 3), dtype=np.uint8)
        
        text, engine, _ = run_remote_ocr(np_image, "test.pdf", 1, "http://fake-ocr")
        
        # Should ONLY find the content in the 'md' key
        self.assertEqual(text, "# Valid Markdown Content")
        self.assertEqual(engine, "remote_docling_serve")

    @patch('os.fsync')
    @patch('workers.gatekeeper_logic.get_llm_and_grammar')
    @patch('workers.gatekeeper_logic.is_bad_ocr')
    @patch('utils.text_utils.get_tokenizer')
    @patch('builtins.open', new_callable=MagicMock)
    def test_gatekeeper_enforces_context_limit(self, mock_open, mock_get_tokenizer, mock_bad_ocr, mock_get_llm_grammar, mock_fsync):
        mock_bad_ocr.return_value = True  # Force LLM path (skip quality bypass)
        """Verify that gatekeeper truncates batches that exceed the LLM context window."""
        
        # 1. Mock Tokenizer
        mock_tokenizer = MagicMock()
        # Simulate 1 token per word for simplicity
        mock_tokenizer.encode.side_effect = lambda x, **kwargs: [1] * len(x.split())
        mock_tokenizer.decode.side_effect = lambda x, **kwargs: " ".join(["word"] * len(x))
        mock_get_tokenizer.return_value = mock_tokenizer
        
        # 2. Mock LLM
        mock_llm = MagicMock()
        mock_get_llm_grammar.return_value = (mock_llm, "fake-grammar")
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Processed"}}]
        }
        
        # 3. Setup realistic context window for test
        with patch('config.settings.SUPERVISOR_N_CTX', 500):
            # CONTEXT_LIMIT will be 400
            # Send 1000 words (definitely over 400)
            massive_content = "word " * 1000
            
            process_chunk(0, massive_content, "test.pdf", "test-slug", "/tmp/test.md")
            
            # Verify the call to the LLM used truncated messages
            args, kwargs = mock_llm.create_chat_completion.call_args
            sent_msg = kwargs['messages'][0]['content']
            
            # The prompt length should now be around 230-250 words
            # (400 - 200 offset = 200 content + 30 instructions)
            self.assertTrue(len(sent_msg.split()) < 300)
            self.assertIn("word", sent_msg)

    def test_validate_chunk_char_length_guard(self):
        """Verify that validate_chunk force-splits when tokenizer under-counts but char length is excessive."""
        from processors.text_processor import validate_chunk
        
        # Tokenizer that under-counts: returns only 10 tokens for a massive string
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x, **kwargs: [1] * min(10, len(x))
        mock_tokenizer.decode.side_effect = lambda x, **kwargs: "safe chunk"
        
        # Create a text well over the character safety limit (MAX_TOKENS * 5)
        huge_text = "A" * (256 * 5 + 100)  # 1380 chars
        
        result = validate_chunk(huge_text, mock_tokenizer)
        
        # Should have split into multiple pieces
        self.assertGreater(len(result), 1, f"Expected split into multiple pieces, got {len(result)}")
        for piece in result:
            self.assertLess(len(piece), 256 * 5 + 50, f"Piece too large: {len(piece)} chars")

    def test_validate_chunk_passes_normal_text(self):
        """Verify that validate_chunk passes normal-sized text unchanged."""
        from processors.text_processor import validate_chunk
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1] * 100  # 100 tokens
        
        normal_text = "This is a perfectly normal chunk of text."
        result = validate_chunk(normal_text, mock_tokenizer)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], normal_text)

    def test_validate_chunk_splits_token_overflow(self):
        """Verify that validate_chunk splits text that exceeds token budget."""
        from processors.text_processor import validate_chunk
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x, **kwargs: [1] * (300 if kwargs.get('add_special_tokens') else 296)
        mock_tokenizer.decode.return_value = "split piece"
        
        long_text = "word " * 300
        result = validate_chunk(long_text, mock_tokenizer)
        
        self.assertGreater(len(result), 1, f"Expected split, got {len(result)} pieces")

    @patch('utils.text_utils.get_tokenizer')
    @patch('utils.consumer_utils.get_vectorstore')
    def test_store_chunks_in_db_rejects_oversized_at_embed_time(self, mock_get_vs, mock_get_tok):
        """Verify that store_chunks_in_db raises an error if a chunk exceeds token budget at embed time."""
        from utils.consumer_utils import store_chunks_in_db
        
        mock_db = MagicMock()
        mock_get_vs.return_value = mock_db
        
        # Mock tokenizer that reports excessive tokens
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1] * 500  # 500 tokens > MAX_TOKENS (256)
        mock_get_tok.return_value = mock_tok
        
        # Create a chunk with massive character count
        huge_chunk = {
            "chunk": "X" * 3000,
            "source_file": "test.pdf",
            "type": "pdf",
            "engine": "llamacpp",
            "hash": "abc123",
            "chunk_index": 0,
            "id": "DOC_TEST_001",
            "page": 1,
        }
        
        with self.assertRaises(RuntimeError, msg="Should reject chunk exceeding MAX_TOKENS at embed time"):
            store_chunks_in_db("test.pdf", [huge_chunk])

    @patch('utils.text_utils.get_tokenizer')
    @patch('utils.consumer_utils.get_vectorstore')
    def test_store_chunks_in_db_accepts_valid_chunks(self, mock_get_vs, mock_get_tok):
        """Verify that store_chunks_in_db accepts properly-sized chunks."""
        from utils.consumer_utils import store_chunks_in_db
        
        mock_db = MagicMock()
        mock_db.get_collection_count.return_value = 10
        mock_get_vs.return_value = mock_db
        
        # Mock tokenizer that reports few tokens
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1] * 10  # 10 tokens
        mock_get_tok.return_value = mock_tok
        
        valid_chunk = {
            "chunk": "passage: [DOC_TEST_001] A short valid chunk of text for embedding.",
            "source_file": "test.pdf",
            "type": "pdf",
            "engine": "llamacpp",
            "hash": "abc123",
            "chunk_index": 0,
            "id": "DOC_TEST_001",
            "page": 1,
        }
        
        # Should not raise
        result = store_chunks_in_db("test.pdf", [valid_chunk])
        self.assertGreaterEqual(result, 0)

if __name__ == "__main__":
    unittest.main()
