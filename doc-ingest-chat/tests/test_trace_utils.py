import logging
import unittest

from utils.trace_utils import TraceLoggerAdapter, generate_trace_id, get_trace_id, set_trace_id, with_trace


class TestTraceUtils(unittest.TestCase):
    def setUp(self):
        set_trace_id(None)

    def test_generate_trace_id(self):
        tid = generate_trace_id()
        self.assertEqual(len(tid), 8)
        self.assertIsInstance(tid, str)

    def test_context_management(self):
        tid = "test-123"
        token = set_trace_id(tid)
        self.assertEqual(get_trace_id(), tid)
        
        # Test nesting/reset
        set_trace_id("nested")
        self.assertEqual(get_trace_id(), "nested")
        
        from utils.trace_utils import reset_trace_id
        # Reset to tid
        # We need the token for 'nested' to reset back to 'tid'
        # But we can test that resetting the first token returns to None
        reset_trace_id(token)
        self.assertIsNone(get_trace_id())

    def test_trace_logger_adapter(self):
        base_logger = logging.getLogger("test")
        adapter = TraceLoggerAdapter(base_logger, {})
        
        # Without trace_id
        set_trace_id(None)
        msg, _ = adapter.process("hello", {})
        self.assertEqual(msg, "hello")
        
        # With trace_id
        set_trace_id("abc")
        msg, _ = adapter.process("hello", {})
        self.assertEqual(msg, "[abc] hello")

    def test_with_trace_decorator(self):
        set_trace_id(None)
        @with_trace(trace_id_arg_name="tid")
        def traced_func(tid=None):
            return get_trace_id()
        
        self.assertEqual(traced_func(tid="dec-123"), "dec-123")
        self.assertIsNone(get_trace_id()) # Should reset after call

if __name__ == "__main__":
    unittest.main()
