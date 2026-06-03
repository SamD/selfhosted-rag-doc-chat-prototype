import os
import unittest
from unittest.mock import patch

from shared.utils import (
    parse_endpoints,
    resolve_embedding_endpoint,
    resolve_supervisor_endpoint,
)


class TestParseEndpoints(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(parse_endpoints(""), [])

    def test_single_endpoint(self):
        result = parse_endpoints("http://192.168.30.81:11435/v1")
        self.assertEqual(result, ["http://192.168.30.81:11435/v1"])

    def test_multiple_endpoints(self):
        result = parse_endpoints("http://a:1/v1, http://b:2/v1")
        self.assertEqual(result, ["http://a:1/v1", "http://b:2/v1"])

    def test_strips_whitespace(self):
        result = parse_endpoints("  http://a:1/v1 ,  http://b:2/v1  ")
        self.assertEqual(result, ["http://a:1/v1", "http://b:2/v1"])

    def test_trailing_slash(self):
        result = parse_endpoints("http://a:1/v1/")
        self.assertEqual(result, ["http://a:1/v1"])

    def test_filters_empty(self):
        result = parse_endpoints("http://a:1/v1,,http://b:2/v1")
        self.assertEqual(result, ["http://a:1/v1", "http://b:2/v1"])


class TestResolveSupervisorEndpoint(unittest.TestCase):
    @patch.dict(os.environ, {"SUPERVISOR_LLM_ENDPOINTS": "http://single:11435/v1"})
    def test_single_endpoint_returns_it(self):
        result = resolve_supervisor_endpoint()
        self.assertEqual(result, "http://single:11435/v1")

    @patch.dict(os.environ, {"SUPERVISOR_LLM_ENDPOINTS": "http://single:11435"})
    def test_single_endpoint_adds_v1(self):
        result = resolve_supervisor_endpoint()
        self.assertEqual(result, "http://single:11435/v1")

    @patch.dict(os.environ, {"SUPERVISOR_LLM_ENDPOINTS": "", "SUPERVISOR_LLM_PATH": "http://remote:8080/v1"})
    def test_fallback_to_supervisor_path(self):
        result = resolve_supervisor_endpoint()
        self.assertEqual(result, "http://remote:8080/v1")

    @patch.dict(os.environ, {"SUPERVISOR_LLM_ENDPOINTS": "", "SUPERVISOR_LLM_PATH": "/local/model.gguf"})
    def test_fallback_to_local_path(self):
        result = resolve_supervisor_endpoint()
        self.assertEqual(result, "/local/model.gguf")


class TestResolveEmbeddingEndpoint(unittest.TestCase):
    @patch.dict(os.environ, {"EMBEDDING_ENDPOINTS": "http://single:11434/v1"})
    def test_single_endpoint_returns_it(self):
        result = resolve_embedding_endpoint()
        self.assertEqual(result, "http://single:11434/v1")

    @patch.dict(os.environ, {"EMBEDDING_ENDPOINTS": "", "EMBEDDING_MODEL_PATH": "http://remote:8080/v1"})
    def test_fallback_to_embedding_model_path(self):
        result = resolve_embedding_endpoint()
        self.assertEqual(result, "http://remote:8080/v1")

    @patch.dict(os.environ, {"EMBEDDING_ENDPOINTS": "", "EMBEDDING_MODEL_PATH": "/local/e5-model"})
    def test_local_model_returns_none(self):
        result = resolve_embedding_endpoint()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()