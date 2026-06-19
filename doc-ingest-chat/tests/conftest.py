"""Global test fixtures and mocks for temporal worker tests."""

import sys
import types
from unittest.mock import AsyncMock, MagicMock


def _setup_mocks():
    """Set up all mocks before any tests run.

    Must create real module objects (not MagicMock) so ``from temporalio.X
    import Y`` works correctly at module level in integration / e2e tests.
    """

    def _identity(name=None):
        """Identity decorator factory — passes through the wrapped function."""
        def wrapper(f):
            return f
        return wrapper

    # activity — needs defn as a real callable
    activity_mod = types.ModuleType('temporalio.activity')
    activity_mod.defn = _identity
    sys.modules['temporalio.activity'] = activity_mod

    # client — needs Client
    client_mod = types.ModuleType('temporalio.client')
    client_mod.Client = AsyncMock
    sys.modules['temporalio.client'] = client_mod

    # worker — needs Worker
    worker_mod = types.ModuleType('temporalio.worker')
    worker_mod.Worker = MagicMock
    sys.modules['temporalio.worker'] = worker_mod

    # testing — needs WorkflowEnvironment with start_local() async context manager
    async def _fake_start_local(*a, **kw):
        env = MagicMock()
        env.target_host = 'localhost:7233'
        async def _enter(e): return e
        async def _exit(e, *args): pass
        env.__aenter__ = _enter
        env.__aexit__ = _exit
        return env

    class _FakeWorkflowEnv:
        @staticmethod
        async def start_local(*a, **kw):
            return await _fake_start_local(*a, **kw)
    testing_mod = types.ModuleType('temporalio.testing')
    testing_mod.WorkflowEnvironment = _FakeWorkflowEnv
    sys.modules['temporalio.testing'] = testing_mod

    # exceptions
    exceptions_mod = types.ModuleType('temporalio.exceptions')
    exceptions_mod.ActivityError = Exception
    sys.modules['temporalio.exceptions'] = exceptions_mod

    # workflow — needs activity decorator + unsafe.imports_passed_through()
    workflow_mod = types.ModuleType('temporalio.workflow')
    workflow_mod.activity = _identity

    class _WorkflowUnsafe:
        @staticmethod
        def imports_passed_through():
            from contextlib import contextmanager

            @contextmanager
            def _inner():
                yield

            return _inner()

    workflow_mod.unsafe = _WorkflowUnsafe
    sys.modules['temporalio.workflow'] = workflow_mod

    # top-level temporalio
    temporalio_mod = types.ModuleType('temporalio')
    temporalio_mod.activity = activity_mod
    temporalio_mod.client = client_mod
    temporalio_mod.worker = worker_mod
    temporalio_mod.testing = testing_mod
    temporalio_mod.exceptions = exceptions_mod
    temporalio_mod.workflow = workflow_mod
    sys.modules['temporalio'] = temporalio_mod


def pytest_configure(config):
    _setup_mocks()


def pytest_unconfigure(config):
    for mod in ('temporalio', 'temporalio.activity', 'temporalio.client',
                'temporalio.worker', 'temporalio.testing', 'temporalio.exceptions',
                'temporalio.workflow', 'temporalio.workflow.unsafe', 'whisperx'):
        sys.modules.pop(mod, None)
