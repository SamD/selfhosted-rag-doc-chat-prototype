#!/usr/bin/env python3
import os
import tempfile
from pathlib import Path

# Set minimal environment variables as early as possible (on import),
# so modules imported during test collection see them.
_base_dir = Path(tempfile.mkdtemp(prefix="test_env_"))
_ingest = _base_dir / "ingest"
_chroma = _base_dir / "chroma"
_tmp_e5 = _base_dir / "e5-model"
_tmp_llama = _base_dir / "llama-model"

for _p in (_ingest, _chroma, _tmp_e5, _tmp_llama):
    _p.mkdir(parents=True, exist_ok=True)

# Prefer developer's local model paths if available; fall back to temp dirs
_default_e5 = Path("/home/samueldoyle/AI/e5-large-v2")
_default_llama = Path("/home/samueldoyle/AI/llama-model")

_e5 = _default_e5 if _default_e5.exists() else _tmp_e5
_llama = _default_llama if _default_llama.exists() else _tmp_llama

os.environ.setdefault("INGEST_FOLDER", str(_ingest))
os.environ.setdefault("CHROMA_DATA_DIR", str(_chroma))
os.environ.setdefault("E5_MODEL_PATH", str(_e5))
os.environ.setdefault("LLAMA_MODEL_PATH", str(_llama))

# Explicit defaults used in tests
os.environ.setdefault("ALLOW_LATIN_EXTENDED", "true")
os.environ.setdefault("LATIN_SCRIPT_MIN_RATIO", "0.7") 