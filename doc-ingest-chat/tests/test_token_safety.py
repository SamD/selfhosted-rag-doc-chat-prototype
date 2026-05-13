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
    @patch('utils.text_utils.get_tokenizer')
    @patch('builtins.open', new_callable=MagicMock)
    def test_gatekeeper_enforces_context_limit(self, mock_open, mock_get_tokenizer, mock_get_llm_grammar, mock_fsync):
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
        with patch('config.settings.LLAMA_N_CTX', 500):
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

if __name__ == "__main__":
    unittest.main()
