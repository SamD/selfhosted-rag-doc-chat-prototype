#!/usr/bin/env python3
"""
Unit tests for Temporal WhisperX type correctness.

Directly checks that:
  1. All timeout values used in the workflow are timedelta instances
  2. The RetryPolicy is constructed with correct parameters
  3. execute_workflow receives TranscribeWorkflow (the class, not .run)
  4. execute_activity receives a valid, callable function reference
"""

# ============================================================================
# Module-level setup: augment conftest mocks BEFORE importing project modules
# ============================================================================

import asyncio
import inspect
import os
import sys
import types
from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Augment conftest mocks (same pattern as the integration test)
# ---------------------------------------------------------------------------

def _identity(f=None):
    """Identity decorator that returns the original function/class unchanged.

    Supports both ``@decorator`` and ``@decorator()`` usage.
    """
    if f is not None:
        return f  # used as @decorator → return original
    def wrapper(fn):
        return fn
    return wrapper  # used as @decorator() → return identity wrapper


import temporalio.workflow as _tw_mod  # noqa: E402 (already in sys.modules via conftest)
_tw_mod.defn = _identity
_tw_mod.run = _identity
_tw_mod.execute_activity = AsyncMock()
_tw_mod.execute_activity.return_value = None  # will be overridden per test


class _FakeRetryPolicy:
    """Stand-in for temporalio.common.RetryPolicy."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
        return f"_FakeRetryPolicy({attrs})"


if 'temporalio.common' not in sys.modules:
    _common_mod = types.ModuleType('temporalio.common')
    _common_mod.RetryPolicy = _FakeRetryPolicy
    sys.modules['temporalio.common'] = _common_mod

import temporalio as _temporalio_pkg
if not hasattr(_temporalio_pkg, 'common'):
    _temporalio_pkg.common = sys.modules['temporalio.common']


# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
from models.transcription_input import TranscriptionInput, TranscriptionResult  # noqa: E402
from temporal_worker.workflows import TranscribeWorkflow  # noqa: E402
from temporal_worker.activities import transcribe_media  # noqa: E402


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset the execute_activity mock before each test."""
    _tw_mod.execute_activity.reset_mock()
    _tw_mod.execute_activity.return_value = TranscriptionResult(
        segments=[{"text": "unit-test"}],
        source_file="unit.mp3",
        job_id="unit-job",
    )
    yield


@pytest.fixture(autouse=True)
def _minimal_env():
    """Set env vars needed for lazy settings resolution."""
    needed = {
        "EMBEDDING_ENDPOINTS": "/tmp/emb",
        "LLM_PATH": "/tmp/llm.gguf",
        "SUPERVISOR_LLM_ENDPOINTS": "/tmp/sup.gguf",
        "DEFAULT_DOC_INGEST_ROOT": "/tmp/ingest_root",
        "TEMPORAL_HOST": "localhost",
        "TEMPORAL_PORT": "7233",
        "TEMPORAL_SERVER_URL": "localhost:7233",
        "TEMPORAL_WHISPER_TASK_QUEUE": "whisperx",
        "USE_TEMPORAL_WHISPER": "true",
    }
    with patch.dict(os.environ, needed, clear=False):
        yield


@pytest.fixture(scope="session", autouse=True)
def _session_cleanup():
    yield
    sys.modules.pop('temporalio.common', None)


# ============================================================================
# Tests
# ============================================================================

class TestTemporalTypeCorrectness:
    """Four direct type-correctness checks on the Temporal WhisperX pipeline."""

    # ------------------------------------------------------------------
    # Check 1: All timeout values are timedelta instances
    # ------------------------------------------------------------------

    def test_all_timeout_values_are_timedelta(self):
        """Verify all timeout parameters in execute_activity are timedelta."""
        input_data = TranscriptionInput(file_path="/test/file.mp3")
        asyncio.run(TranscribeWorkflow().run(input_data))

        args, kwargs = _tw_mod.execute_activity.call_args

        # The workflow passes two timeouts
        for timeout_key in ("schedule_to_close_timeout", "start_to_close_timeout"):
            val = kwargs.get(timeout_key)
            assert val is not None, f"{timeout_key} is missing"
            assert isinstance(val, timedelta), \
                f"{timeout_key} should be timedelta, got {type(val).__name__}: {val!r}"

        # Also check execution_timeout from whisper_utils
        from utils.whisper_utils import send_media_to_whisperx_temporal
        source = inspect.getsource(send_media_to_whisperx_temporal)
        # Verify timedelta(minutes=90) is used
        assert "timedelta(minutes=90)" in source, \
            "send_media_to_whisperx_temporal must use timedelta(minutes=90)"

    # ------------------------------------------------------------------
    # Check 2: RetryPolicy is constructed correctly
    # ------------------------------------------------------------------

    def test_retry_policy_constructed_correctly(self):
        """Verify RetryPolicy has max_attempts=3, backoff=2.0, timedelta intervals."""
        input_data = TranscriptionInput(file_path="/test/file.mp3")
        asyncio.run(TranscribeWorkflow().run(input_data))

        args, kwargs = _tw_mod.execute_activity.call_args
        rp = kwargs.get("retry_policy")

        assert rp is not None, "retry_policy must be passed to execute_activity"

        # --- Constructor arguments ---
        assert rp.maximum_attempts == 3, \
            f"maximum_attempts expected 3, got {rp.maximum_attempts}"
        assert rp.backoff_coefficient == 2.0, \
            f"backoff_coefficient expected 2.0, got {rp.backoff_coefficient}"

        # --- Types ---
        assert isinstance(rp.initial_interval, timedelta), \
            f"initial_interval type: {type(rp.initial_interval)}"
        assert isinstance(rp.maximum_interval, timedelta), \
            f"maximum_interval type: {type(rp.maximum_interval)}"

        # --- Values ---
        assert rp.initial_interval == timedelta(seconds=5), \
            f"initial_interval expected 5s, got {rp.initial_interval}"
        assert rp.maximum_interval == timedelta(seconds=60), \
            f"maximum_interval expected 60s, got {rp.maximum_interval}"

    # ------------------------------------------------------------------
    # Check 3: execute_workflow receives TranscribeWorkflow (not .run)
    # ------------------------------------------------------------------

    def test_execute_workflow_receives_class_not_method(self):
        """Verify send_media_to_whisperx_temporal passes TranscribeWorkflow
        (the class) to execute_workflow, not TranscribeWorkflow.run."""
        from utils.whisper_utils import send_media_to_whisperx_temporal
        source = inspect.getsource(send_media_to_whisperx_temporal)

        # The class name should appear
        assert "TranscribeWorkflow" in source, \
            "Must reference the TranscribeWorkflow class"

        # Should NOT reference .run
        assert "TranscribeWorkflow.run" not in source, \
            "Must pass the class, not TranscribeWorkflow.run"

        # The call should use execute_workflow with TranscribeWorkflow as first arg
        # We look for the pattern: execute_workflow(\n    TranscribeWorkflow,\n
        import re
        match = re.search(
            r"execute_workflow\s*\(\s*TranscribeWorkflow\s*[,\)]",
            source,
        )
        assert match, \
            "execute_workflow must receive TranscribeWorkflow as first positional argument"

    # ------------------------------------------------------------------
    # Check 4: execute_activity gets a valid function reference
    # ------------------------------------------------------------------

    def test_execute_activity_receives_valid_function(self):
        """Verify execute_activity receives transcribe_media as a callable
        function (not a string name or a mock)."""
        input_data = TranscriptionInput(file_path="/test/file.mp3")
        asyncio.run(TranscribeWorkflow().run(input_data))

        args, kwargs = _tw_mod.execute_activity.call_args
        assert len(args) >= 1, "execute_activity needs at least one positional arg"

        activity_fn = args[0]

        # It should be the actual transcribe_media function object
        assert activity_fn is transcribe_media, \
            f"Expected transcribe_media function, got {activity_fn!r}"

        # It must be callable
        assert callable(activity_fn), "activity function must be callable"

        # It must be async (coroutine function)
        assert inspect.iscoroutinefunction(activity_fn), \
            "activity function must be an async coroutine function"

    # ------------------------------------------------------------------
    # Bonus: verify the TranscribeWorkflow class is actually a class
    # ------------------------------------------------------------------

    def test_transcribe_workflow_is_class(self):
        """Sanity check: TranscribeWorkflow is a class, not an instance."""
        assert inspect.isclass(TranscribeWorkflow), \
            "TranscribeWorkflow must be a class"

    # ------------------------------------------------------------------
    # Bonus: verify execute_activity has the right arity (activity + input)
    # ------------------------------------------------------------------

    def test_execute_activity_call_arity(self):
        """Verify execute_activity is called with the right positional args."""
        input_data = TranscriptionInput(file_path="/test/file.mp3")
        asyncio.run(TranscribeWorkflow().run(input_data))

        args, kwargs = _tw_mod.execute_activity.call_args
        # Positional: (activity_fn, input_data)
        assert len(args) >= 2, \
            f"Expected at least 2 positional args (activity, input), got {len(args)}"

        assert args[0] is transcribe_media
        assert isinstance(args[1], TranscriptionInput)
        assert args[1].file_path == "/test/file.mp3"
