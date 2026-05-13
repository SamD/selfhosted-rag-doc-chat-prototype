import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add doc-ingest-chat to sys.path
sys.path.insert(0, os.path.join(os.getcwd(), "doc-ingest-chat"))

from services.database import DatabaseService
from utils.ocr_utils import run_remote_ocr
from workers.whisperx_worker import RemoteWhisper


class TestIntegrationFidelity(unittest.TestCase):

    # --- 1. REMOTE OCR FIDELITY ---

    @patch('requests.post')
    def test_ocr_remote_payload_and_nesting(self, mock_post):
        """
        Verify that we send the plural 'files' field and can parse 
        deeply nested Docling responses while ignoring base64 garbage.
        """
        # REAL-WORLD NESTED PAYLOAD (with base64 noise)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "document": {
                "md": "# Title\nActual Content",
                "other_field": "iVBORw0KGgoAAAANSUhEUgA..." # Base64 noise
            },
            "status": "success",
            "metadata": {"pages": 1}
        }
        mock_post.return_value = mock_response

        np_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Execute the ACTUAL logic
        text, engine, _ = run_remote_ocr(np_image, "test.pdf", 1, "http://remote-ocr")
        
        # ASSERT: Logic must find the correct nested text and ignore the base64 noise
        self.assertEqual(text, "# Title\nActual Content")
        
        # ASSERT: Check that we used the plural 'files' field as the primary attempt
        args, kwargs = mock_post.call_args
        self.assertIn("files", kwargs["files"]) # Plural field check
        # Check that list of tuples was used for data to ensure correct serialization
        self.assertIsInstance(kwargs["data"], list) 
        self.assertIn(("ocr_lang", "en"), kwargs["data"])

    @patch('requests.post')
    def test_ocr_remote_422_fallback(self, mock_post):
        """Verify that we automatically fallback from 'files' to 'file' if the server rejects the plural."""
        # First call fails with 422, second succeeds with 200
        mock_422 = MagicMock(status_code=422)
        mock_200 = MagicMock(status_code=200)
        mock_200.json.return_value = {"md": "Succeeded on retry"}
        mock_post.side_effect = [mock_422, mock_200]

        np_image = np.zeros((100, 100, 3), dtype=np.uint8)
        text, _, _ = run_remote_ocr(np_image, "test.pdf", 1, "http://remote-ocr")
        
        self.assertEqual(text, "Succeeded on retry")
        self.assertEqual(mock_post.call_count, 2)
        
        # Check first attempt was plural, second was singular
        first_call = mock_post.call_args_list[0]
        second_call = mock_post.call_args_list[1]
        self.assertIn("files", first_call[1]["files"])
        self.assertIn("file", second_call[1]["files"])

    # --- 2. REMOTE WHISPER FIDELITY ---

    @patch('requests.post')
    def test_whisper_remote_format_fidelity(self, mock_post):
        """Verify that remote whisper sends the exact multi-part format expected by standard servers."""
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {"text": "Transcribed Speech"}
        mock_post.return_value = mock_response

        remote = RemoteWhisper("http://remote-whisper/inference")
        
        # Create a real temp file
        dummy_path = "/tmp/test_whisper_fidelity.wav"
        with open(dummy_path, "w") as f:
            f.write("audio data")
        
        try:
            result = remote.transcribe_file(dummy_path)
            self.assertEqual(result["segments"][0]["text"], "Transcribed Speech")
            
            # ASSERT: Multi-part structure check
            args, kwargs = mock_post.call_args
            self.assertIn("file", kwargs["files"])
            self.assertEqual(kwargs["data"]["response_format"], "json")
            self.assertEqual(kwargs["data"]["temperature"], "0.0")
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

    # --- 3. VECTOR DB CONNECTION ROBUSTNESS ---

    @patch('services.database.QdrantClient')
    def test_qdrant_client_init_fidelity(self, mock_qdrant):
        """Verify that Qdrant client uses high-fidelity settings (gRPC + Timeout)."""
        import services.database
        services.database._QDRANT_CLIENT_CACHE = None
        
        with patch('services.database.VECTOR_DB_URL', 'http://remote:6334'):
            with patch('services.database.VECTOR_DB_TIMEOUT', 45.0):
                with patch('services.database.VECTOR_DB_USE_GRPC', True):
                    DatabaseService.get_qdrant_client()
                    
                    mock_qdrant.assert_called_once_with(
                        url='http://remote:6334',
                        prefer_grpc=True,
                        timeout=45.0
                    )

if __name__ == "__main__":
    unittest.main()
