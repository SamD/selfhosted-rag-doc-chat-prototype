#!/usr/bin/env python3
import os
import tempfile
from pathlib import Path

# Set minimal environment variables as early as possible (on import),
# so modules imported during test collection see them.
_base_dir = Path(tempfile.mkdtemp(prefix="test_env_"))
_root = _base_dir / "Docs"
_tmp_e5 = _base_dir / "e5-model"
_tmp_llama = _base_dir / "llama-model"

for _p in (_root, _tmp_e5, _tmp_llama):
    _p.mkdir(parents=True, exist_ok=True)

# Pre-create standard lifecycle folders
for _d in ["staging", "preprocessing", "ingestion", "consuming", "success", "failed"]:
    (_root / _d).mkdir(parents=True, exist_ok=True)

# Prefer developer's local model paths if available; fall back to temp dirs
_default_e5 = Path("/tmp/AI/e5-large-v2")
_default_llama = Path("/tmp/AI/llama-model")

_e5 = _default_e5 if _default_e5.exists() else _tmp_e5
_llama = _default_llama if _default_llama.exists() else _tmp_llama

os.environ.setdefault("DEFAULT_DOC_INGEST_ROOT", str(_root))
os.environ.setdefault("EMBEDDING_MODEL_PATH", str(_e5))
os.environ.setdefault("LLM_PATH", str(_llama))
os.environ.setdefault("SUPERVISOR_LLM_PATH", str(_llama))  # Use same dummy path for supervisor

# Explicit defaults used in tests
os.environ.setdefault("ALLOW_LATIN_EXTENDED", "true")
os.environ.setdefault("LATIN_SCRIPT_MIN_RATIO", "0.7")
