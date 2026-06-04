import os
import unittest
from unittest.mock import patch

from shared.utils import (
    EndpointDispatcher,
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

    @patch.dict(os.environ, {"SUPERVISOR_LLM_ENDPOINTS": "http://remote:8080/v1"})
    def test_fallback_to_supervisor_endpoints(self):
        result = resolve_supervisor_endpoint()
        self.assertEqual(result, "http://remote:8080/v1")

    @patch.dict(os.environ, {"SUPERVISOR_LLM_ENDPOINTS": "/local/model.gguf"})
    def test_fallback_to_local_path(self):
        result = resolve_supervisor_endpoint()
        self.assertEqual(result, "/local/model.gguf")


class TestResolveEmbeddingEndpoint(unittest.TestCase):
    @patch.dict(os.environ, {"EMBEDDING_ENDPOINTS": "http://single:11434/v1"})
    def test_single_endpoint_returns_it(self):
        result = resolve_embedding_endpoint()
        self.assertEqual(result, "http://single:11434/v1")

    @patch.dict(os.environ, {"EMBEDDING_ENDPOINTS": "http://remote:8080/v1"})
    def test_fallback_to_embedding_endpoints(self):
        result = resolve_embedding_endpoint()
        self.assertEqual(result, "http://remote:8080/v1")

    @patch.dict(os.environ, {"EMBEDDING_ENDPOINTS": "/local/e5-model"})
    def test_local_model_returns_none(self):
        result = resolve_embedding_endpoint()
        self.assertIsNone(result)


class TestEndpointDispatcher(unittest.TestCase):
    def test_empty_endpoints_raises(self):
        with self.assertRaises(ValueError):
            EndpointDispatcher([])

    def test_single_endpoint_pinned(self):
        d = EndpointDispatcher(["http://a:1"], interleave=False)
        seen = []

        def capture(url, label):
            seen.append((url, label))
            return f"{url}-{label}"

        results = d.dispatch(capture, [("x",), ("y",)])
        self.assertEqual(results, ["http://a:1-x", "http://a:1-y"])
        self.assertEqual(seen, [("http://a:1", "x"), ("http://a:1", "y")])

    def test_single_endpoint_interleaved(self):
        d = EndpointDispatcher(["http://a:1"], interleave=True)
        seen = []

        def capture(url, label):
            seen.append((url, label))
            return f"{url}-{label}"

        results = d.dispatch(capture, [("x",), ("y",)])
        self.assertEqual(results, ["http://a:1-x", "http://a:1-y"])
        self.assertEqual(seen, [("http://a:1", "x"), ("http://a:1", "y")])

    def test_pinned_all_to_same_endpoint(self):
        d = EndpointDispatcher(["http://a:1", "http://b:2"], interleave=False)
        seen = []

        def capture(url, label):
            seen.append(url)
            return label

        d.dispatch(capture, [("x",), ("y",), ("z",)])
        # All calls go to the first endpoint; subsequent dispatches rotate
        self.assertEqual(seen, ["http://a:1", "http://a:1", "http://a:1"])

    def test_interleaved_round_robins(self):
        d = EndpointDispatcher(["http://a:1", "http://b:2"], interleave=True)
        seen = []

        def capture(url, label):
            seen.append(url)
            return label

        d.dispatch(capture, [("x",), ("y",), ("z",), ("w",)])
        self.assertEqual(seen, ["http://a:1", "http://b:2", "http://a:1", "http://b:2"])

    def test_interleaved_single_item_no_threadpool(self):
        """A single item should not use ThreadPool."""
        d = EndpointDispatcher(["http://a:1", "http://b:2"], interleave=True)

        def fn(url, label):
            return url

        result = d.dispatch(fn, [("only",)])
        self.assertEqual(result, ["http://a:1"])

    def test_interleaved_result_order(self):
        """Results should return in the same order as args_list."""
        d = EndpointDispatcher(["http://a:1", "http://b:2"], interleave=True)

        def slow_fn(url, idx, delay):
            import time
            time.sleep(delay)
            return f"{url}-{idx}"

        # Dispatch with different delays to verify order preservation
        results = d.dispatch(slow_fn, [(1, 0.05), (2, 0.01), (3, 0.03)])
        self.assertEqual(len(results), 3)
        self.assertIn(results[0], ["http://a:1-1", "http://b:2-1"])
        self.assertIn(results[1], ["http://a:1-2", "http://b:2-2"])
        self.assertIn(results[2], ["http://a:1-3", "http://b:2-3"])

    def test_empty_args_list(self):
        d = EndpointDispatcher(["http://a:1"])
        result = d.dispatch(lambda url, label: url, [])
        self.assertEqual(result, [])

    def test_max_workers_clamped(self):
        """max_workers should be clamped to the number of items."""
        d = EndpointDispatcher(["http://a:1"], interleave=True, max_workers=10)
        seen = []

        def capture(url, label):
            seen.append(url)
            return label

        d.dispatch(capture, [("x",), ("y",)])
        self.assertEqual(len(seen), 2)

    def test_counter_persists_across_dispatches(self):
        """The round-robin counter persists across dispatch calls."""
        d = EndpointDispatcher(["http://a:1", "http://b:2"], interleave=False)

        def capture(url, label):
            return url

        r1 = d.dispatch(capture, [("x",)])
        self.assertEqual(r1, ["http://a:1"])
        r2 = d.dispatch(capture, [("y",)])
        self.assertEqual(r2, ["http://b:2"])
        r3 = d.dispatch(capture, [("z",)])
        self.assertEqual(r3, ["http://a:1"])

    def test_interleaved_counter_persists(self):
        d = EndpointDispatcher(["http://a:1", "http://b:2"], interleave=True)
        seen = []

        def capture(url, label):
            seen.append(url)
            return url

        d.dispatch(capture, [("x",), ("y",)])
        self.assertEqual(seen, ["http://a:1", "http://b:2"])
        d.dispatch(capture, [("z",)])
        self.assertEqual(seen, ["http://a:1", "http://b:2", "http://a:1"])

    def test_pinned_error_propagation(self):
        d = EndpointDispatcher(["http://a:1"], interleave=False)

        def failing(url, label):
            raise ValueError("test error")

        with self.assertRaises(RuntimeError) as ctx:
            d.dispatch(failing, [("x",)])
        self.assertIn("test error", str(ctx.exception))
        self.assertIn("http://a:1", str(ctx.exception))

    def test_interleaved_error_propagation(self):
        d = EndpointDispatcher(["http://a:1", "http://b:2"], interleave=True)

        def failing(url, label):
            raise ValueError("test error")

        with self.assertRaises(RuntimeError) as ctx:
            d.dispatch(failing, [("x",), ("y",)])
        self.assertIn("test error", str(ctx.exception))

    def test_job_label_in_pinned_error(self):
        d = EndpointDispatcher(["http://a:1"], interleave=False)

        def failing(url, label):
            raise ValueError("boom")

        with self.assertRaises(RuntimeError) as ctx:
            d.dispatch(failing, [("x",)], job_label="my-job")
        self.assertIn("my-job", str(ctx.exception))

    def test_job_label_in_interleaved_error(self):
        d = EndpointDispatcher(["http://a:1", "http://b:2"], interleave=True)

        def failing(url, label):
            raise ValueError("boom")

        with self.assertRaises(RuntimeError) as ctx:
            d.dispatch(failing, [("x",), ("y",)], job_label="my-job")
        self.assertIn("my-job", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()