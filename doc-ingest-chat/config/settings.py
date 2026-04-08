#!/usr/bin/env python3
"""
Configuration settings for the document ingestion system.

- When running standalone, default values are used for all environment variables.
- When running in Docker Compose, values from ingest-svc.env are used.
"""

import os
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Helper: Environment Getters
# ---------------------------------------------------------------------------


def _abs_path(key: str) -> str:
    """Return absolute path if environment variable is set, else None."""
    val = os.getenv(key)
    return os.path.abspath(val) if val else None


def _require_abs_path(key: str, default: str = None) -> str:
    """Return absolute path, requiring the env var to be set (unless default provided)."""
    val = os.getenv(key)
    if not val:
        if default:
            return os.path.abspath(default)
        raise ValueError(f"❌ CRITICAL ERROR: Environment variable '{key}' is NOT set. This is required for the system to function.")
    return os.path.abspath(val)


def _get_vector_db_port() -> int:
    """Returns the vector database port with dynamic defaults based on profile."""
    port = os.getenv("VECTOR_DB_PORT") or os.getenv("CHROMA_PORT")
    if port:
        return int(port)

    profile = os.getenv("VECTOR_DB_PROFILE", "qdrant").lower()
    return 6333 if profile == "qdrant" else 8000


# ---------------------------------------------------------------------------
# Settings Dictionary (Lazy Loaded)
# ---------------------------------------------------------------------------

_SETTINGS: dict[str, Callable[[], Any]] = {
    "LLAMA_USE_GPU": lambda: os.getenv("LLAMA_USE_GPU", "true").lower() == "true",
    "LLAMA_N_CTX": lambda: int(os.getenv("LLAMA_N_CTX", "8192")),
    "LLAMA_N_BATCH": lambda: int(os.getenv("LLAMA_N_BATCH", "512")),
    "STAGING_FOLDER": lambda: _require_abs_path("STAGING_FOLDER", "Docs/staging"),
    "GATEKEEPER_FAILURE_DB": lambda: _abs_path("GATEKEEPER_FAILURE_DB") or os.path.join(_require_abs_path("INGEST_FOLDER", "Docs"), "gatekeeper_failures.db"),
    "CHROMA_DATA_DIR": lambda: _abs_path("CHROMA_DATA_DIR") or os.path.join(_require_abs_path("INGEST_FOLDER", "Docs"), "chroma_db"),
    "QDRANT_DATA_DIR": lambda: _abs_path("QDRANT_DATA_DIR") or os.path.join(_require_abs_path("INGEST_FOLDER", "Docs"), "qdrant_data"),
    "VECTOR_DB_DATA_DIR": lambda: _abs_path("VECTOR_DB_DATA_DIR") or os.path.join(_require_abs_path("INGEST_FOLDER", "Docs"), "qdrant_data"),
    "EMBEDDING_MODEL_PATH": lambda: _require_abs_path("EMBEDDING_MODEL_PATH", "/home/samueldoyle/AI_LOCAL/e5-large-v2"),
    "LLM_PATH": lambda: _require_abs_path("LLM_PATH", "/home/samueldoyle/AI_LOCAL/Models/Phi/microsoft_Phi-4-mini-instruct-Q6_K.gguf"),
    "SUPERVISOR_LLM_PATH": lambda: _require_abs_path("SUPERVISOR_LLM_PATH", "/home/samueldoyle/AI_LOCAL/Models/Phi/microsoft_Phi-4-mini-instruct-Q6_K.gguf"),
    "FAILED_FILES": lambda: _abs_path("FAILED_FILES") or os.path.join(_require_abs_path("INGEST_FOLDER", "Docs"), "failed_files.txt"),
    "INGESTED_FILE": lambda: _abs_path("INGESTED_FILE") or os.path.join(_require_abs_path("INGEST_FOLDER", "Docs"), "ingested_files.txt"),
    "TRACK_FILE": lambda: _abs_path("TRACK_FILE") or os.path.join(_require_abs_path("INGEST_FOLDER", "Docs"), "ingested_files.txt"),
    "PARQUET_FILE": lambda: _abs_path("PARQUET_FILE") or os.path.join(_require_abs_path("INGEST_FOLDER", "Docs"), "chunks.parquet"),
    "DUCKDB_FILE": lambda: _abs_path("DUCKDB_FILE") or os.path.join(_require_abs_path("INGEST_FOLDER", "Docs"), "chunks.duckdb"),
    # Redis
    "REDIS_HOST": lambda: os.environ.get("REDIS_HOST") or os.getenv("REDIS_HOST", "localhost"),
    "REDIS_PORT": lambda: int(os.environ.get("REDIS_PORT") or os.getenv("REDIS_PORT", "6379")),
    "REDIS_OCR_JOB_QUEUE": lambda: os.getenv("REDIS_OCR_JOB_QUEUE", "ocr_processing_job"),
    "REDIS_INGEST_QUEUE": lambda: os.getenv("REDIS_INGEST_QUEUE", "chunk_ingest_queue"),
    "QUEUE_NAMES": lambda: os.getenv("QUEUE_NAMES", "chunk_ingest_queue:0,chunk_ingest_queue:1").split(","),
    "VECTOR_DB_PROFILE": lambda: os.getenv("VECTOR_DB_PROFILE", "qdrant").lower(),
    "USE_QDRANT": lambda: os.getenv("VECTOR_DB_PROFILE", "qdrant").lower() == "qdrant",
    "VECTOR_DB_HOST": lambda: os.getenv("VECTOR_DB_HOST", os.getenv("CHROMA_HOST", "vector-db")),
    "VECTOR_DB_PORT": _get_vector_db_port,
    "VECTOR_DB_COLLECTION": lambda: os.getenv("VECTOR_DB_COLLECTION", "vector_base_collection"),
    "MAX_CHROMA_BATCH_SIZE": lambda: int(os.getenv("MAX_CHROMA_BATCH_SIZE", "75")),
    "CHUNK_TIMEOUT": lambda: int(os.getenv("CHUNK_TIMEOUT", "300")),
    # Compatibility Aliases
    "CHROMA_HOST": lambda: os.getenv("VECTOR_DB_HOST", os.getenv("CHROMA_HOST", "vector-db")),
    "CHROMA_PORT": _get_vector_db_port,
    "CHROMA_COLLECTION": lambda: os.getenv("VECTOR_DB_COLLECTION", os.getenv("CHROMA_COLLECTION", "vector_base_collection")),
    # Processing
    "MAX_CHUNKS": lambda: int(os.getenv("MAX_CHUNKS", "5000")),
    "MAX_TOKENS": lambda: int(os.getenv("MAX_TOKENS", "512")),
    "CHUNK_SIZE": lambda: int(os.getenv("CHUNK_SIZE", "512")),
    "CHUNK_OVERLAP": lambda: int(os.getenv("CHUNK_OVERLAP", "50")),
    "ALLOW_LATIN_EXTENDED": lambda: os.getenv("ALLOW_LATIN_EXTENDED", "true").lower() == "true",
    "LATIN_SCRIPT_MIN_RATIO": lambda: float(os.getenv("LATIN_SCRIPT_MIN_RATIO", "0.7")),
    "INGEST_FOLDER": lambda: _require_abs_path("INGEST_FOLDER", "Docs"),
    "DEBUG_IMAGE_DIR": lambda: _abs_path("DEBUG_IMAGE_DIR") or os.path.join(_require_abs_path("INGEST_FOLDER", "Docs"), "ocr_debug"),
    "MAX_OCR_DIM": lambda: int(os.getenv("MAX_OCR_DIM", "3000")),
    "SUPPORTED_DOC_EXT": lambda: (".pdf", ".html", ".htm", ".txt", ".md"),
    "SUPPORTED_MEDIA_EXT": lambda: tuple(os.getenv("SUPPORTED_MEDIA_EXT", ".mp3,.wav,.m4a,.aac,.flac,.mp4,.mov,.mkv").split(",")),
    "ALL_SUPPORTED_EXT": lambda: (".pdf", ".html", ".htm", ".txt", ".md") + tuple(os.getenv("SUPPORTED_MEDIA_EXT", ".mp3,.wav,.m4a,.aac,.flac,.mp4,.mov,.mkv").split(",")),
    "DEVICE": lambda: os.getenv("DEVICE", "cuda"),
    "MEDIA_BATCH_SIZE": lambda: int(os.getenv("MEDIA_BATCH_SIZE", "8")),
    "COMPUTE_TYPE": lambda: os.getenv("COMPUTE_TYPE", "float16"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    if name in _SETTINGS:
        return _SETTINGS[name]()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Critical: Ensure debug directory exists
_debug_dir = os.getenv("DEBUG_IMAGE_DIR", "/tmp/ocr_debug")
if not os.path.exists(_debug_dir):
    try:
        os.makedirs(_debug_dir, exist_ok=True)
    except Exception:
        pass
