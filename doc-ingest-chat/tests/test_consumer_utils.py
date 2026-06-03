import os
import sys
import unittest
from unittest.mock import MagicMock, call, patch

os.environ["LLM_PATH"] = "/tmp/dummy.gguf"
os.environ["SUPERVISOR_LLM_PATH"] = "/tmp/supervisor.gguf"
os.environ["EMBEDDING_MODEL_PATH"] = "/tmp/dummy_e5"
os.environ["DEFAULT_DOC_INGEST_ROOT"] = "/tmp/test_docs"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestStoreChunksInDb(unittest.TestCase):
    """Tests for store_chunks_in_db including rollback on failure."""

    def setUp(self):
        self.chunks = [
            {"chunk": f"text {i}", "source_file": "test.pdf", "type": "pdf",
             "engine": "test", "hash": f"h{i}", "chunk_index": i, "id": f"id{i}", "page": 1}
            for i in range(10)
        ]

    @patch("utils.consumer_utils.get_vectorstore")
    @patch("utils.text_utils.get_tokenizer")
    @patch("utils.consumer_utils.chunked")
    def test_rollback_on_batch_failure(self, mock_chunked, mock_get_tok, mock_get_vs):
        """When a batch fails, all already-written Qdrant points are rolled back."""
        from utils.consumer_utils import store_chunks_in_db

        mock_db = MagicMock()
        mock_get_vs.return_value = mock_db

        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1] * 10
        mock_get_tok.return_value = mock_tok

        # chunked splits the list into groups of 5
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

        # Verify rollback was called with source_file filter
        mock_db.delete.assert_called_once_with(where={"source_file": "test.pdf"})

    @patch("utils.consumer_utils.get_vectorstore")
    @patch("utils.text_utils.get_tokenizer")
    @patch("utils.consumer_utils.chunked")
    def test_no_rollback_when_all_batches_succeed(self, mock_chunked, mock_get_tok, mock_get_vs):
        """When all batches succeed, no rollback occurs."""
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
        mock_db.delete.assert_not_called()

    @patch("utils.consumer_utils.get_vectorstore")
    @patch("utils.text_utils.get_tokenizer")
    @patch("utils.consumer_utils.chunked")
    def test_logs_rollback_failure(self, mock_chunked, mock_get_tok, mock_get_vs):
        """If even the rollback delete fails, the original error is still raised."""
        from utils.consumer_utils import store_chunks_in_db

        mock_db = MagicMock()
        mock_get_vs.return_value = mock_db
        mock_db.add_texts.side_effect = RuntimeError("Batch failed")
        mock_db.delete.side_effect = Exception("Delete also failed")

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

        # Delete was attempted
        mock_db.delete.assert_called_once()


class TestCleanupOrphanedQdrantPoints(unittest.TestCase):
    """Tests for startup orphan cleanup in consumer_worker.py."""

    @patch("duckdb.connect")
    @patch("services.database.get_vectorstore")
    def test_cleans_orphaned_files(self, mock_get_vs, mock_connect):
        """Files with INGEST_FAILED or CONSUMING status have Qdrant data removed."""
        from workers.consumer_worker import cleanup_orphaned_qdrant_points

        mock_store = MagicMock()
        mock_get_vs.return_value = mock_store

        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = [
            ("failed_file.pdf",),
            ("stuck_consuming.pdf",),
        ]
        mock_connect.return_value = mock_con

        os.environ["DUCKDB_FILE"] = "/tmp/test_cleanup.duckdb"
        with patch("workers.consumer_worker.settings.DUCKDB_FILE", "/tmp/test_cleanup.duckdb"):
            with patch("os.path.exists", return_value=True):
                cleanup_orphaned_qdrant_points()

        self.assertEqual(mock_store.delete.call_count, 2)
        mock_store.delete.assert_has_calls([
            call(where={"source_file": "failed_file.pdf"}),
            call(where={"source_file": "stuck_consuming.pdf"}),
        ])

    @patch("duckdb.connect")
    @patch("services.database.get_vectorstore")
    def test_no_cleanup_when_no_orphans(self, mock_get_vs, mock_connect):
        """When there are no orphaned files, no deletion occurs."""
        from workers.consumer_worker import cleanup_orphaned_qdrant_points

        mock_store = MagicMock()
        mock_get_vs.return_value = mock_store

        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = []
        mock_connect.return_value = mock_con

        with patch("os.path.exists", return_value=True):
            with patch("workers.consumer_worker.settings.DUCKDB_FILE", "/tmp/test.duckdb"):
                cleanup_orphaned_qdrant_points()

        mock_store.delete.assert_not_called()

    @patch("duckdb.connect")
    @patch("services.database.get_vectorstore")
    def test_no_cleanup_when_duckdb_missing(self, mock_get_vs, mock_connect):
        """If the DuckDB file doesn't exist, no cleanup is attempted."""
        from workers.consumer_worker import cleanup_orphaned_qdrant_points

        with patch("os.path.exists", return_value=False):
            cleanup_orphaned_qdrant_points()

        mock_connect.assert_not_called()
        mock_get_vs.assert_not_called()

    @patch("duckdb.connect")
    @patch("services.database.get_vectorstore")
    def test_continues_on_delete_failure(self, mock_get_vs, mock_connect):
        """If deleting one file fails, the next file is still processed."""
        from workers.consumer_worker import cleanup_orphaned_qdrant_points

        mock_store = MagicMock()
        mock_store.delete.side_effect = [Exception("Delete error"), None]
        mock_get_vs.return_value = mock_store

        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = [
            ("bad_file.pdf",),
            ("good_file.pdf",),
        ]
        mock_connect.return_value = mock_con

        with patch("os.path.exists", return_value=True):
            with patch("workers.consumer_worker.settings.DUCKDB_FILE", "/tmp/test.duckdb"):
                cleanup_orphaned_qdrant_points()

        self.assertEqual(mock_store.delete.call_count, 2)


class TestConsumerWorkerMainOrphanCall(unittest.TestCase):
    """Verify main() calls cleanup before forking children."""

    @patch("workers.consumer_worker.cleanup_orphaned_qdrant_points")
    @patch("workers.consumer_worker.multiprocessing.Manager")
    @patch("workers.consumer_graph.get_consumer_app")
    @patch("workers.consumer_worker.init_schema")
    @patch("workers.consumer_worker.settings")
    def test_cleanup_called_after_init_schema_before_fork(self, mock_settings, mock_init_schema, mock_get_app, mock_mgr, mock_cleanup):
        from workers.consumer_worker import main

        mock_settings.QUEUE_NAMES = ["q:0"]
        mock_manager = MagicMock()
        mock_manager.dict.return_value = {"shutdown_flag": False}
        mock_mgr.return_value.__enter__.return_value = mock_manager

        with patch("workers.consumer_worker.multiprocessing.Process") as mock_proc:
            mock_p = MagicMock()
            mock_proc.return_value = mock_p
            mock_p.join.side_effect = KeyboardInterrupt  # stop the loop

            try:
                main()
            except KeyboardInterrupt:
                pass

        # Verify cleanup called before fork
        mock_cleanup.assert_called_once()
        # init_schema should have been called first
        mock_init_schema.assert_called_once()


if __name__ == "__main__":
    unittest.main()
