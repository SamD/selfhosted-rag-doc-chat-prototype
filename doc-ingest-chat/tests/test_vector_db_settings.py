import os
from unittest.mock import patch

import pytest
from config import settings


def load_test_env():
    """Helper to load the test.env file into os.environ."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, val = line.strip().split("=", 1)
                    os.environ[key] = val

def test_mandatory_variables_present():
    """
    Verifies that all 7 mandatory variables are correctly identified 
    and resolved from the environment.
    """
    load_test_env()
    
    # We mock os.path.exists to True so the _require_abs_path helper succeeds
    with patch("os.path.exists", return_value=True):
        assert settings.DEFAULT_DOC_INGEST_ROOT == os.environ["DEFAULT_DOC_INGEST_ROOT"]
        assert settings.INGEST_FOLDER == os.environ["INGEST_FOLDER"]
        assert settings.STAGING_FOLDER == os.environ["STAGING_FOLDER"]
        assert settings.EMBEDDING_MODEL_PATH == os.environ["EMBEDDING_MODEL_PATH"]
        assert settings.LLM_PATH == os.environ["LLM_PATH"]
        assert settings.SUPERVISOR_LLM_PATH == os.environ["SUPERVISOR_LLM_PATH"]
        assert settings.WHISPER_MODEL_PATH == os.environ["WHISPER_MODEL_PATH"]

def test_missing_mandatory_raises_system_exit():
    """
    Verifies that if a mandatory variable is missing, the settings 
    module correctly triggers an exit (via _require_abs_path).
    """
    # Clear a mandatory variable
    if "LLM_PATH" in os.environ:
        del os.environ["LLM_PATH"]
    
    with patch("os.path.exists", return_value=True):
        # Accessing the property should trigger the error
        with pytest.raises(SystemExit):
            _ = settings.LLM_PATH

def test_whisper_warning_only_mode():
    """
    Verifies that WHISPER_MODEL_PATH is optional and defaults 
    to NOT_SET rather than crashing.
    """
    if "WHISPER_MODEL_PATH" in os.environ:
        del os.environ["WHISPER_MODEL_PATH"]
    
    # It should not raise SystemExit
    val = settings.WHISPER_MODEL_PATH
    assert val == "NOT_SET"

def test_absolute_path_resolution():
    """
    Ensures that relative paths in environment are correctly 
    resolved to absolute paths.
    """
    with patch.dict(os.environ, {"DEFAULT_DOC_INGEST_ROOT": "./relative_docs"}):
        with patch("os.path.exists", return_value=True):
            # Should be an absolute path starting with /
            assert os.path.isabs(settings.DEFAULT_DOC_INGEST_ROOT)
