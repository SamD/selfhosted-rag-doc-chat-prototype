import os
import sys
import unittest
from unittest.mock import MagicMock, patch

os.environ["LLM_PATH"] = "/tmp/dummy.gguf"
os.environ["SUPERVISOR_LLM_ENDPOINTS"] = "/tmp/supervisor.gguf"
os.environ["EMBEDDING_ENDPOINTS"] = "/tmp/dummy_e5"
os.environ["DEFAULT_DOC_INGEST_ROOT"] = "/tmp/test_docs"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestStoreChunksInDb(unittest.TestCase):
    """Tests for store_chunks_in_db including error handling."""

    def setUp(self):
        self.chunks = [
            {"chunk": f"text {i}", "source_file": "test.pdf", "type": "pdf",
             "engine": "test", "hash": f"h{i}", "chunk_index": i, "id": f"id{i}", "page": 1}
            for i in range(10)
        ]

    @patch("utils.consumer_utils.get_vectorstore")
    @patch("utils.text_utils.get_tokenizer")
    @patch("utils.consumer_utils.chunked")
    def test_error_on_batch_failure(self, mock_chunked, mock_get_tok, mock_get_vs):
        """When a batch fails, the error is re-raised."""
        from utils.consumer_utils import store_chunks_in_db

        mock_db = MagicMock()
        mock_get_vs.return_value = mock_db

        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1] * 10
        mock_get_tok.return_value = mock_tok

        def fake_chunked(seq, size):
            if isinstance(seq, list):
                return [seq[:5], seq[5:]]
            return [seq]

        mock_chunked.side_effect = fake_chunked

        # First batch succeeds, second fails
        call_count = [0]

        def add_texts(texts, metadatas=None, ids=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return ["ok"] * len(texts)
            raise RuntimeError("Simulated Qdrant failure")

        mock_db.add_texts.side_effect = add_texts

        with self.assertRaises(RuntimeError):
            store_chunks_in_db("test.pdf", self.chunks)

    @patch("utils.consumer_utils.get_vectorstore")
    @patch("utils.text_utils.get_tokenizer")
    @patch("utils.consumer_utils.chunked")
    def test_no_rollback_when_all_batches_succeed(self, mock_chunked, mock_get_tok, mock_get_vs):
        """When all batches succeed, no error occurs."""
        from utils.consumer_utils import store_chunks_in_db

        mock_db = MagicMock()
        mock_get_vs.return_value = mock_db
        mock_db.add_texts.return_value = ["ok"]
        mock_db.get_collection_count.return_value = 10

        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1] * 10
        mock_get_tok.return_value = mock_tok

        def fake_chunked(seq, size):
            if isinstance(seq, list):
                return [seq[:5], seq[5:]]
            return [seq]
        mock_chunked.side_effect = fake_chunked

        result = store_chunks_in_db("test.pdf", self.chunks)
        self.assertEqual(result, 2)

    @patch("utils.consumer_utils.get_vectorstore")
    @patch("utils.text_utils.get_tokenizer")
    @patch("utils.consumer_utils.chunked")
    def test_error_is_raised_on_immediate_failure(self, mock_chunked, mock_get_tok, mock_get_vs):
        """If the first batch fails, the error is raised immediately."""
        from utils.consumer_utils import store_chunks_in_db

        mock_db = MagicMock()
        mock_get_vs.return_value = mock_db
        mock_db.add_texts.side_effect = RuntimeError("Batch failed")

        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1]
        mock_get_tok.return_value = mock_tok

        def fake_chunked(seq, size):
            if isinstance(seq, list):
                return [seq]
            return [seq]
        mock_chunked.side_effect = fake_chunked

        with self.assertRaises(RuntimeError):
            store_chunks_in_db("test.pdf", self.chunks[:1])


if __name__ == "__main__":
    unittest.main()
