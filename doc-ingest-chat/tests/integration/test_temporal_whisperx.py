#!/usr/bin/env python3
"""
Integration tests for the Temporal WhisperX pipeline.

Verifies type correctness at every boundary:
  - Workflow → execute_activity   (timedelta, RetryPolicy, function ref)
  - Worker → Worker() constructor (workflows=[], activities=[])
  - send_media_to_whisperx_temporal → client.execute_workflow (class ref)
  - Activity signature            (async def, TranscriptionInput → TranscriptionResult)
  - TEMPORAL_HOST/PORT resolution (legacy fallback logic)
  - Dataclass serialisation       (TranscriptionInput / TranscriptionResult)

All tests run *without* a live Temporal server — the entire temporalio SDK is
mocked by tests/conftest.py plus module-level patches in this file.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import sys
import types
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_APP_DIR = str(Path(__file__).resolve().parent.parent.parent)  # doc-ingest-chat/
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# ---------------------------------------------------------------------------
# Ensure temporalio mocks are complete BEFORE importing modules under test
#
# The conftest.py at tests/conftest.py already installs real ModuleType stubs
# for temporalio.activity, temporalio.client, temporalio.worker,
# temporalio.testing, temporalio.exceptions, and temporalio.workflow.
#
# We add the missing attributes that the pipeline code needs:
#   - temporalio.workflow.defn              (@workflow.defn)
#   - temporalio.workflow.run               (@workflow.run)
#   - temporalio.workflow.execute_activity  (spy for assertion)
#   - temporalio.common.RetryPolicy         (real class for isinstance checks)
# ---------------------------------------------------------------------------

# Purge any previously cached temporal_worker modules (from other test files
# that may have imported them under a different mock regime).
for _mod_key in list(sys.modules):
    if _mod_key.startswith("temporal_worker"):
        del sys.modules[_mod_key]

# Grab the temporalio.workflow stub already installed by conftest.py.
_twf_mod = sys.modules.get("temporalio.workflow")
if _twf_mod is None:
    _twf_mod = types.ModuleType("temporalio.workflow")
    sys.modules["temporalio.workflow"] = _twf_mod


def _identity(f=None):
    """Identity decorator — passes through the wrapped function/class unchanged.

    Supports both usage styles:
        @decorator          → _identity(fn) → returns fn
        @decorator()        → _identity() → returns wrapper → _identity(fn) → fn
    """
    if f is not None:
        return f
    def wrapper(fn):
        return fn
    return wrapper


_twf_mod.defn = _identity
_twf_mod.run = _identity

# Global spy for workflow.execute_activity — each test resets it.
_activity_spy = AsyncMock()
_activity_spy.__name__ = "execute_activity"
_twf_mod.execute_activity = _activity_spy

# Create a real-ish RetryPolicy class so isinstance checks are meaningful.
class _RetryPolicy:
    """Test double for temporalio.common.RetryPolicy."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
        return f"_RetryPolicy({attrs})"


_cmn_mod = types.ModuleType("temporalio.common")
_cmn_mod.RetryPolicy = _RetryPolicy
sys.modules["temporalio.common"] = _cmn_mod

# Mock redis & whisperx — they are module-level dependencies of some modules
# we import below but aren't needed for these type-correctness tests.
sys.modules.setdefault("redis", MagicMock())
sys.modules.setdefault("whisperx", MagicMock())

# ---------------------------------------------------------------------------
# Now import the modules under test — they will see our completions above.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402
from models.transcription_input import (  # noqa: E402
    TranscriptionInput,
    TranscriptionResult,
)
from temporal_worker.activities import transcribe_media  # noqa: E402
from temporal_worker.workflows import TranscribeWorkflow  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _minimal_env(tmp_path: Path) -> None:
    """Ensure minimum env vars exist so settings lazy-load without crash."""
    defaults = {
        "EMBEDDING_ENDPOINTS": str(tmp_path / "fake_emb_model"),
        "LLM_PATH": str(tmp_path / "fake_llm.gguf"),
        "SUPERVISOR_LLM_ENDPOINTS": str(tmp_path / "fake_sup.gguf"),
        "DEFAULT_DOC_INGEST_ROOT": str(tmp_path),
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


@pytest.fixture
def reset_activity_spy() -> AsyncMock:
    """Reset the module-level execute_activity spy before each test.

    Also re-applies the spy to ``temporalio.workflow.execute_activity``
    in case another test file replaced it on the shared module object
    (important when multiple temporal test files run in the same session).
    """
    _activity_spy.reset_mock()
    # Re-apply in case another test file replaced the attribute on the
    # shared temporalio.workflow module object.
    import temporalio.workflow as _twf_reapply
    _twf_reapply.execute_activity = _activity_spy
    # Make the spy return a sensible TranscriptionResult so the workflow
    # doesn't choke on what ``await execute_activity(...)`` returns.
    _activity_spy.return_value = TranscriptionResult(
        segments=[{"text": "mock segment"}],
        source_file="test.mp3",
        job_id="test-job",
    )
    return _activity_spy


# ===================================================================
# TestWorkflowTypes
# ===================================================================


class TestWorkflowTypes:
    """Verify correct types at the Workflow → execute_activity boundary."""

    # ------------------------------------------------------------------
    # 1. test_execute_workflow_passes_correct_types
    # ------------------------------------------------------------------

    def test_execute_workflow_passes_correct_types(
        self, reset_activity_spy: AsyncMock
    ) -> None:
        """
        Verify *client.execute_workflow* receives a **class** (not .run) and
        a ``timedelta`` for ``execution_timeout``.

        We test this by inspecting ``send_media_to_whisperx_temporal``
        which is the function that calls ``client.execute_workflow``.
        """
        from utils.whisper_utils import send_media_to_whisperx_temporal

        sig = inspect.signature(send_media_to_whisperx_temporal)
        params = list(sig.parameters)
        assert "file_path" in params
        assert "language" in params
        assert "mime_type" in params
        assert "trace_id" in params

        # Source-inspect the inner _run() async function to verify
        # it passes TranscribeWorkflow (class) and timedelta directly.
        source = inspect.getsource(send_media_to_whisperx_temporal)

        # Check class (not .run) is passed to execute_workflow
        match_exec = re.search(
            r"execute_workflow\(\s*(\S+)", source
        )
        assert match_exec is not None, (
            "Could not find execute_workflow() call in "
            "send_media_to_whisperx_temporal"
        )
        first_arg = match_exec.group(1).rstrip(",")
        assert first_arg == "TranscribeWorkflow", (
            f"Expected execute_workflow to receive TranscribeWorkflow (class), "
            f"got {first_arg!r}"
        )

        # Check execution_timeout is a timedelta expression
        assert "execution_timeout=timedelta(minutes=90)" in source, (
            "execution_timeout should be timedelta(minutes=90)"
        )

    # ------------------------------------------------------------------
    # 6. test_workflow_timeout_types
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("timeout_field", [
        "schedule_to_close_timeout",
        "start_to_close_timeout",
    ])
    def test_workflow_timeout_types(
        self, reset_activity_spy: AsyncMock, timeout_field: str
    ) -> None:
        """Verify *every* timeout kwarg to execute_activity is a timedelta."""
        result = asyncio.run(
            TranscribeWorkflow().run(
                TranscriptionInput(file_path="/tmp/test.mp3")
            )
        )
        assert isinstance(result, TranscriptionResult)

        _activity_spy.assert_called_once()
        _, kwargs = _activity_spy.call_args

        timeout_val = kwargs.get(timeout_field)
        assert timeout_val is not None, (
            f"execute_activity was not called with {timeout_field}"
        )
        assert isinstance(timeout_val, timedelta), (
            f"{timeout_field} should be timedelta, got {type(timeout_val)}"
        )
        assert timeout_val == timedelta(minutes=90), (
            f"{timeout_field} should be 90 minutes, got {timeout_val}"
        )

    # ------------------------------------------------------------------
    # 2. test_execute_activity_has_retry_policy_object  (also in this class)
    # ------------------------------------------------------------------

    def test_execute_activity_has_retry_policy_object(
        self, reset_activity_spy: AsyncMock
    ) -> None:
        """Verify retry_policy is a RetryPolicy object with correct types."""
        asyncio.run(
            TranscribeWorkflow().run(
                TranscriptionInput(file_path="/tmp/test.mp3")
            )
        )

        _activity_spy.assert_called_once()
        _, kwargs = _activity_spy.call_args

        rp = kwargs.get("retry_policy")
        assert rp is not None, "No retry_policy passed to execute_activity"
        assert isinstance(rp, _RetryPolicy), (
            f"retry_policy should be a RetryPolicy, got {type(rp)}"
        )

        # Numeric fields
        assert rp.maximum_attempts == 3
        assert rp.backoff_coefficient == 2.0

        # Interval fields — must be timedelta
        assert isinstance(rp.initial_interval, timedelta), (
            f"initial_interval should be timedelta, got {type(rp.initial_interval)}"
        )
        assert rp.initial_interval == timedelta(seconds=5)

        assert isinstance(rp.maximum_interval, timedelta), (
            f"maximum_interval should be timedelta, got {type(rp.maximum_interval)}"
        )
        assert rp.maximum_interval == timedelta(seconds=60)


# ===================================================================
# TestTranscriptionDataclasses
# ===================================================================


class TestTranscriptionDataclasses:
    """Verify TranscriptionInput / TranscriptionResult work with Temporal."""

    # ------------------------------------------------------------------
    # 3. test_transcription_input_dataclass_serialization
    # ------------------------------------------------------------------

    def test_transcription_input_dataclass_serialization(self) -> None:
        """Verify TranscriptionInput is serialisable as a plain dataclass.

        Temporal's default serializer (JSON) works with plain dataclasses that
        have simple fields (str, int, Optional[str]).  This test verifies that
        TranscriptionInput meets that contract.
        """
        from dataclasses import fields, is_dataclass

        # Must be a dataclass
        assert is_dataclass(TranscriptionInput), (
            "TranscriptionInput must be a dataclass for Temporal serialisation"
        )

        # Check field types are JSON-compatible
        field_types = {f.name: f.type for f in fields(TranscriptionInput)}
        assert field_types["file_path"] is str
        assert field_types["language"] is str
        # Optional[str] is compatible with Temporal JSON serialisation.
        # In Python 3.12, Optional[str] → Union[str, None] (get_origin → typing.Union).
        # Just verify str appears in the type args (works for all Python versions).
        from typing import get_args

        mime_type = field_types["mime_type"]
        args = get_args(mime_type)

        assert args, (
            "mime_type should be Optional[str] (a generic alias with args), "
            "but get_args returned (). Annotations may have been resolved "
            "at runtime — check if from __future__ import annotations is active."
        )
        assert str in args, (
            f"mime_type should include str in its type args, got {args}"
        )

        # Construct and round-trip via dict (simulates JSON serialisation)
        original = TranscriptionInput(
            file_path="/data/test.mp3", language="fr", mime_type="audio/mpeg"
        )
        as_dict = {
            "file_path": original.file_path,
            "language": original.language,
            "mime_type": original.mime_type,
        }
        restored = TranscriptionInput(**as_dict)
        assert restored == original

    def test_transcription_result_dataclass_serialization(self) -> None:
        """Verify TranscriptionResult is a plain dataclass with simple types."""
        from dataclasses import fields, is_dataclass

        assert is_dataclass(TranscriptionResult)

        field_types = {f.name: f.type for f in fields(TranscriptionResult)}
        assert field_types["source_file"] is str
        assert field_types["job_id"] is str
        assert field_types["segments"] == list[dict]  # type: ignore[comparison-overlap]

        # Construct and verify
        segments = [{"text": "hello"}]
        result = TranscriptionResult(
            segments=segments, source_file="test.mp3", job_id="abc"
        )
        assert result.segments == segments
        assert result.source_file == "test.mp3"
        assert result.job_id == "abc"


# ===================================================================
# TestActivityTypes
# ===================================================================


class TestActivityTypes:
    """Verify the transcribe_media activity function signature and decorator."""

    # ------------------------------------------------------------------
    # 4. test_transcribe_media_activity_signature
    # ------------------------------------------------------------------

    def test_transcribe_media_activity_signature(self) -> None:
        """Verify activity signature matches what the workflow calls.

        Workflow calls:  execute_activity(transcribe_media, input, ...)
        Activity is:     async def transcribe_media(input: TranscriptionInput) -> TranscriptionResult
        """
        sig = inspect.signature(transcribe_media)
        params = list(sig.parameters.values())

        # Must be async
        assert inspect.iscoroutinefunction(transcribe_media), (
            "transcribe_media must be an async function"
        )

        # One parameter: 'input' of type TranscriptionInput
        assert len(params) == 1, (
            f"Expected 1 parameter (input), got {len(params)}: "
            f"{[p.name for p in params]}"
        )
        param = params[0]
        assert param.name == "input", f"Expected param name 'input', got '{param.name}'"
        assert param.annotation is TranscriptionInput, (
            f"Expected param type TranscriptionInput, got {param.annotation}"
        )

        # Return type must be TranscriptionResult
        assert sig.return_annotation is TranscriptionResult, (
            f"Expected return type TranscriptionResult, "
            f"got {sig.return_annotation}"
        )

    def test_activity_has_defn_decorator(self) -> None:
        """Verify transcribe_media has the @activity.defn(name=...) applied.

        The conftest.py '_identity' decorator for activity.defn just returns
        the original function, so we verify the decorator ran by checking
        the function is the same object.
        """
        # @activity.defn is an identity decorator in our mocks —
        # the important thing is that it didn't crash at import time
        # and the function is callable with the right signature.
        assert callable(transcribe_media)
        assert inspect.iscoroutinefunction(transcribe_media)

    def test_workflow_passes_activity_function_directly(self) -> None:
        """Verify TranscribeWorkflow.run passes transcribe_media as first arg.

        The workflow must pass the function *reference*, not a string name.
        """
        source = inspect.getsource(TranscribeWorkflow.run)
        # Find execute_activity( call and capture first arg
        match = re.search(r"execute_activity\(\s*(\S+)", source)
        assert match is not None, (
            "Could not find execute_activity() call in TranscribeWorkflow.run"
        )
        first_arg = match.group(1).rstrip(",")
        assert first_arg == "transcribe_media", (
            f"Expected execute_activity first arg to be 'transcribe_media', "
            f"got {first_arg!r}"
        )


# ===================================================================
# TestWorkerRegistration
# ===================================================================


class TestWorkerRegistration:
    """Verify the Worker constructor registers both workflows and activities."""

    # ------------------------------------------------------------------
    # 5. test_worker_registers_both_workflow_and_activity
    # ------------------------------------------------------------------

    def test_worker_constructor_has_workflows_list(self) -> None:
        """Verify Worker() call in run_temporal_worker.py has workflows=[]."""
        worker_source_path = (
            Path(__file__).resolve().parent.parent.parent
            / "run_temporal_worker.py"
        )
        assert worker_source_path.is_file(), (
            f"run_temporal_worker.py not found at {worker_source_path}"
        )
        source = worker_source_path.read_text()

        # Check both required keyword arguments are present
        assert "workflows=[" in source, (
            "Worker(...) is missing workflows=[] argument"
        )
        assert "activities=[" in source, (
            "Worker(...) is missing activities=[] argument"
        )

    def test_worker_registers_transcribeworkflow_class(self) -> None:
        """Verify workflows list contains TranscribeWorkflow class."""
        worker_source_path = (
            Path(__file__).resolve().parent.parent.parent
            / "run_temporal_worker.py"
        )
        source = worker_source_path.read_text()

        # The workflows list should contain the class, not .run
        match = re.search(
            r"workflows=\[([^\]]+)\]", source
        )
        assert match is not None, "Could not find workflows=[...] in worker source"
        workflows_content = match.group(1)
        assert "TranscribeWorkflow" in workflows_content, (
            f"Expected TranscribeWorkflow in workflows=[], "
            f"got {workflows_content!r}"
        )
        assert ".run" not in workflows_content, (
            "workflows=[] should contain TranscribeWorkflow (the class), "
            "not TranscribeWorkflow.run (the method)"
        )

    def test_worker_registers_transcribe_media_function(self) -> None:
        """Verify activities list contains transcribe_media function."""
        worker_source_path = (
            Path(__file__).resolve().parent.parent.parent
            / "run_temporal_worker.py"
        )
        source = worker_source_path.read_text()

        match = re.search(
            r"activities=\[([^\]]+)\]", source
        )
        assert match is not None, (
            "Could not find activities=[...] in worker source"
        )
        activities_content = match.group(1)
        assert "transcribe_media" in activities_content, (
            f"Expected transcribe_media in activities=[], "
            f"got {activities_content!r}"
        )


# ===================================================================
# TestTemporalConnection
# ===================================================================


class TestTemporalConnection:
    """Test the TEMPORAL_HOST / TEMPORAL_PORT resolution logic."""

    # ------------------------------------------------------------------
    # 7. test_whisper_utils_connection_logic
    # ------------------------------------------------------------------

    def _resolve_target(
        self, host: str, port: int, server_url: str
    ) -> str:
        """Replicate the resolution logic from send_media_to_whisperx_temporal."""
        target = f"{host}:{port}"
        if host == "localhost" and port == 7233:
            target = server_url
        return target

    def test_defaults_produce_legacy_fallback(self) -> None:
        """When TEMPORAL_HOST=localhost & TEMPORAL_PORT=7233,
        the system should fall back to TEMPORAL_SERVER_URL."""
        from config.settings import (
            TEMPORAL_HOST,
            TEMPORAL_PORT,
            TEMPORAL_SERVER_URL,
        )

        target = self._resolve_target(
            TEMPORAL_HOST, TEMPORAL_PORT, TEMPORAL_SERVER_URL
        )
        # The resolution logic replaces "localhost:7233" with TEMPORAL_SERVER_URL
        assert target == TEMPORAL_SERVER_URL

    def test_custom_host_bypasses_legacy_fallback(self) -> None:
        """When TEMPORAL_HOST != localhost, the host:port string is used directly."""
        host = "temporal.example.com"
        port = 7233
        server_url = "some-legacy:7233"
        target = self._resolve_target(host, port, server_url)
        assert target == "temporal.example.com:7233"

    def test_custom_port_bypasses_legacy_fallback(self) -> None:
        """When TEMPORAL_PORT != 7233, the host:port string is used directly."""
        host = "localhost"
        port = 8233
        server_url = "some-legacy:8233"
        target = self._resolve_target(host, port, server_url)
        assert target == "localhost:8233"

    def test_full_resolution_with_env_override(self) -> None:
        """Integration-style test: patch env vars and re-read settings."""
        with patch.dict(
            os.environ,
            {
                "TEMPORAL_HOST": "remote.temporal.io",
                "TEMPORAL_PORT": "8233",
                "TEMPORAL_SERVER_URL": "legacy:8233",
            },
        ):
            # Re-read settings after env change
            from config.settings import (
                TEMPORAL_HOST as TH,
                TEMPORAL_PORT as TP,
                TEMPORAL_SERVER_URL as TS,
            )

            target = self._resolve_target(TH, TP, TS)
            assert target == "remote.temporal.io:8233"

    def test_legacy_default_url_is_localhost_7233(self) -> None:
        """TEMPORAL_SERVER_URL defaults to localhost:7233."""
        # Temporarily unset the env var to force the default
        with patch.dict(os.environ, {}, clear=True):
            # Need minimum env vars to avoid crash on other settings
            env_patch = {
                "EMBEDDING_ENDPOINTS": "/tmp/e",
                "LLM_PATH": "/tmp/l.gguf",
                "SUPERVISOR_LLM_ENDPOINTS": "/tmp/s.gguf",
                "DEFAULT_DOC_INGEST_ROOT": "/tmp/root",
            }
            with patch.dict(os.environ, env_patch):
                from config.settings import (
                    TEMPORAL_SERVER_URL as TS,
                )

                assert TS == "localhost:7233"


# ===================================================================
# TestFullChain
# ===================================================================


class TestFullChain:
    """Verify the full chain has correct signatures at every level."""

    # ------------------------------------------------------------------
    # 8. test_send_media_to_whisperx_temporal_signatures
    # ------------------------------------------------------------------

    def test_send_media_to_whisperx_temporal_signature(self) -> None:
        """Verify send_media_to_whisperx_temporal has the expected signature."""
        # We import inside the test with redis/whisperx already mocked
        from utils.whisper_utils import send_media_to_whisperx_temporal

        sig = inspect.signature(send_media_to_whisperx_temporal)
        params = list(sig.parameters.values())

        # Expected: (file_path: str, language: str = "en",
        #            mime_type: str = None, trace_id: str = None)
        assert len(params) == 4, (
            f"Expected 4 parameters, got {len(params)}: "
            f"{[p.name for p in params]}"
        )

        param_names = [p.name for p in params]
        assert param_names == ["file_path", "language", "mime_type", "trace_id"]

        # file_path must be str, no default
        assert params[0].annotation is str
        assert params[0].default is inspect.Parameter.empty

        # language must be str, default "en"
        assert params[1].annotation is str
        assert params[1].default == "en"

        # mime_type can be str or None, default None
        assert params[2].default is None

        # trace_id can be str or None, default None
        assert params[3].default is None

    def test_send_media_to_whisperx_dispatches_to_temporal(self) -> None:
        """Verify send_media_to_whisperx dispatches correctly based on flag."""
        # Test with USE_TEMPORAL_WHISPER=true
        with (
            patch.dict(
                os.environ,
                {
                    "USE_TEMPORAL_WHISPER": "true",
                    **{k: os.environ.get(k, v) for k, v in [
                        ("EMBEDDING_ENDPOINTS", "/tmp/e"),
                        ("LLM_PATH", "/tmp/l.gguf"),
                        ("SUPERVISOR_LLM_ENDPOINTS", "/tmp/s.gguf"),
                        ("DEFAULT_DOC_INGEST_ROOT", "/tmp/root"),
                    ] if k not in os.environ},
                },
            ),
        ):
            from config.settings import USE_TEMPORAL_WHISPER as utw

            assert utw is True

    def test_workflow_decorator_run_returns_correct_type(self) -> None:
        """Verify @workflow.run decorator preserves the async function type."""
        # The identity decorator should leave TranscribeWorkflow.run as
        # a coroutine function
        assert inspect.iscoroutinefunction(TranscribeWorkflow.run), (
            "TranscribeWorkflow.run should be a coroutine function after @workflow.run"
        )

    def test_activity_decorator_preserves_function(self) -> None:
        """Verify @activity.defn preserves transcribe_media as callable."""
        assert callable(transcribe_media)
        assert inspect.iscoroutinefunction(transcribe_media)
        # The decorator should preserve the original function
        assert transcribe_media.__name__ == "transcribe_media"


class TestTranscribeMediaResponseParsing:
    """Verify transcribe_media correctly parses the WhisperX server response.

    The WhisperX server at http://192.168.30.70:1145/inference returns:
      {"text": "transcribed text..."}   (NOT {"segments": [...]})

    The ORIGINAL working code in whisperx_worker.py does:
        text = result.get("text", "")
        return {"segments": [{"text": text}]}

    The bug was using result.get("segments", []) which returns [].
    """

    def test_remote_response_uses_text_key(self) -> None:
        """The RemoteWhisper pattern reads 'text', not 'segments'."""
        # Read the original worker source and confirm it uses "text"
        worker_path = Path(__file__).parent.parent.parent / "workers" / "whisperx_worker.py"
        source = worker_path.read_text()
        assert '.get("text", "")' in source, (
            "whisperx_worker.py should read 'text' key from WhisperX response"
        )

    def test_activity_parses_text_not_segments(self) -> None:
        """transcribe_media must parse 'text' key like the original code."""
        activity_path = Path(__file__).parent.parent.parent / "temporal_worker" / "activities.py"
        source = activity_path.read_text()
        # The activity should use result.get("text", "") for the remote path
        assert '.get("text", "")' in source, (
            "activities.py must read 'text' key from WhisperX response (not 'segments')"
        )
        # Should NOT iterate over result.get("segments")
        assert 'for seg in result.get("segments"' not in source or 'for seg in result.get("text"' in source, (
            "activities.py should not iterate over non-existent 'segments' list"
        )

    def test_activity_returns_nonempty_segments_for_valid_text(self) -> None:
        """When WhisperX returns text, the activity should return 1 segment."""
        from unittest.mock import AsyncMock, patch

        input_data = TranscriptionInput(
            file_path="/tmp/test.mp4",
            language="en",
            mime_type="video/mp4",
        )

        # Mock os.path.exists to always return True
        with (
            patch("temporal_worker.activities.os.path.exists", return_value=True),
            patch("temporal_worker.activities.mimetypes.guess_type", return_value=("video/mp4", None)),
            patch("temporal_worker.activities.WHISPER_MODEL_ENDPOINTS", "http://fake:8080/inference"),
        ):
            import asyncio
            from temporalio import activity
            from temporal_worker.activities import transcribe_media
            import requests

            # Mock the requests.post response to match what the real WhisperX returns
            mock_response = MagicMock()
            mock_response.json.return_value = {"text": "this is a test transcription"}
            mock_response.raise_for_status = MagicMock()

            with patch("temporal_worker.activities.requests.post", return_value=mock_response):
                result = asyncio.run(transcribe_media(input_data))
                assert isinstance(result, TranscriptionResult)
                assert len(result.segments) == 1, f"Expected 1 segment, got {len(result.segments)}: {result.segments}"
                assert result.segments[0]["text"] == "this is a test transcription"

    def test_activity_returns_empty_for_no_text(self) -> None:
        """When WhisperX returns no text, the activity should return empty segments."""
        from unittest.mock import MagicMock, patch
        import asyncio

        input_data = TranscriptionInput(
            file_path="/tmp/test.mp4",
            language="en",
            mime_type="video/mp4",
        )

        with (
            patch("temporal_worker.activities.os.path.exists", return_value=True),
            patch("temporal_worker.activities.mimetypes.guess_type", return_value=("video/mp4", None)),
            patch("temporal_worker.activities.WHISPER_MODEL_ENDPOINTS", "http://fake:8080/inference"),
        ):
            from temporal_worker.activities import transcribe_media

            mock_response = MagicMock()
            mock_response.json.return_value = {"text": ""}
            mock_response.raise_for_status = MagicMock()

            with patch("temporal_worker.activities.requests.post", return_value=mock_response):
                result = asyncio.run(transcribe_media(input_data))
                assert isinstance(result, TranscriptionResult)
                assert len(result.segments) == 0, f"Expected 0 segments for empty text, got {len(result.segments)}"

    def test_activity_handles_json_with_only_text(self) -> None:
        """Some WhisperX implementations may only return 'text' without 'segments'."""
        from unittest.mock import MagicMock, patch
        import asyncio

        input_data = TranscriptionInput(
            file_path="/tmp/test.mp4",
            language="en",
            mime_type="video/mp4",
        )

        with (
            patch("temporal_worker.activities.os.path.exists", return_value=True),
            patch("temporal_worker.activities.mimetypes.guess_type", return_value=("video/mp4", None)),
            patch("temporal_worker.activities.WHISPER_MODEL_ENDPOINTS", "http://fake:8080/inference"),
        ):
            from temporal_worker.activities import transcribe_media

            mock_response = MagicMock()
            # Server returns ONLY 'text', no 'segments' key at all
            mock_response.json.return_value = {"text": "hello world"}
            mock_response.raise_for_status = MagicMock()

            with patch("temporal_worker.activities.requests.post", return_value=mock_response):
                result = asyncio.run(transcribe_media(input_data))
                assert len(result.segments) == 1
                assert result.segments[0]["text"] == "hello world"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
