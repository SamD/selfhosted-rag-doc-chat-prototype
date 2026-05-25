#!/usr/bin/env python3
"""
Thin compatibility wrapper around shared/config.py.
All env-var resolution logic lives in the shared/ package so every component
in the monorepo uses the same canonical settings.

Import from here as you always have:
    from config.settings import LLM_PATH, CHUNK_SIZE, ...

The lazy __getattr__ machinery is in shared/config.py; this module re-exports
it so existing import paths keep working.
"""

import logging
import os
import sys

# Ensure shared/ is on sys.path. Two layouts are supported:
#   1) Dev:  shared/ at monorepo root  (3 dirnames up from config/)
#   2) Docker: shared/ volume-mounted at /app/shared (2 dirnames up from config/)
_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_repo_root = os.path.dirname(_app_dir)
for _candidate in (_app_dir, _repo_root):
    if os.path.isdir(os.path.join(_candidate, "shared")):
        sys.path.insert(0, _candidate)
        break

from shared.config import _SETTINGS  # noqa: E402

log = logging.getLogger("ingest.settings")


def __getattr__(name: str):
    if name in _SETTINGS:
        return _SETTINGS[name]()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def ensure_folders():
    """Ensure all lifecycle directories exist on disk."""
    for key in [
        "STAGING_DIR",
        "PREPROCESSING_DIR",
        "INGESTION_DIR",
        "CONSUMING_DIR",
        "SUCCESS_DIR",
        "FAILED_DIR",
        "DEBUG_IMAGE_DIR",
    ]:
        try:
            path = _SETTINGS[key]()
            if path and not path.startswith(("http", "https")):
                os.makedirs(path, exist_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    ensure_folders()
