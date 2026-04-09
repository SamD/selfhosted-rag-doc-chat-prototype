import json
import os
import unittest

import duckdb
from config import settings
from workers.gatekeeper_logic import assemble_metadata, get_slug, log_gatekeeper_result


class TestGatekeeperLogic(unittest.TestCase):
    def setUp(self):
        self.test_db = "/tmp/test_gatekeeper_history.db"
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

        # Override settings for test
        settings.GATEKEEPER_FAILURE_DB = self.test_db

    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_log_gatekeeper_result_success(self):
        slug = "test-document"
        metadata = {"id": "123", "tier": 3}

        log_gatekeeper_result(slug, "SUCCESS", metadata=metadata)

        # Verify in DuckDB
        con = duckdb.connect(self.test_db)
        res = con.execute("SELECT status, metadata FROM gatekeeper_history WHERE slug = ?", [slug]).fetchone()
        con.close()

        self.assertIsNotNone(res)
        self.assertEqual(res[0], "SUCCESS")
        self.assertEqual(json.loads(res[1]), metadata)

    def test_log_gatekeeper_result_failure(self):
        slug = "fail-document"
        error_msg = "Something went wrong"

        log_gatekeeper_result(slug, "FAILURE", error_msg=error_msg)

        # Verify in DuckDB
        con = duckdb.connect(self.test_db)
        res = con.execute("SELECT status, error FROM gatekeeper_history WHERE slug = ?", [slug]).fetchone()
        con.close()

        self.assertIsNotNone(res)
        self.assertEqual(res[0], "FAILURE")
        self.assertEqual(res[1], error_msg)

    def test_get_slug_consistency(self):
        text = "This is a Test Document Name"
        slug1 = get_slug(text)
        slug2 = get_slug(text)

        self.assertEqual(slug1, slug2)
        self.assertIn("test-document-name", slug1)
        self.assertEqual(len(slug1.split("-")[-1]), 8)  # 4-byte hex = 8 chars

    def test_assemble_metadata(self):
        file_path = "test.pdf"
        slug = "test-slug"
        meta = assemble_metadata(file_path, slug, 0, 10)

        self.assertEqual(meta["slug"], slug)
        self.assertEqual(meta["chunk_index"], 0)
        self.assertEqual(meta["total_chunks"], 10)
        self.assertEqual(meta["source_type"], "pdf_ocr_raw")


if __name__ == "__main__":
    unittest.main()
