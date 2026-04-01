#!/usr/bin/env python3
"""
Configuration settings for the document ingestion system.

- When running standalone, default values are used for all environment variables.
- When running in Docker Compose, values from ingest-svc.env will override the defaults.
"""

import os
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# FORCE LOAD the env file from the root if it exists
if os.getenv("SKIP_LOAD_DOTENV", "false").lower() != "true":
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)


def _abs_path(key: str, default: Optional[str] = None) -> str:
    path = os.getenv(key, default)
    if not path:
        return default if default is not None else ""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def _require_abs_path(key: str) -> str:
    path = os.getenv(key)
    if not path:
        raise ValueError(f"❌ CRITICAL ERROR: Environment variable '{key}' is NOT set. This is required for the system to function.")
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


_default_vector_port = "6333" if os.getenv("VECTOR_DB_PROFILE", "qdrant").lower() == "qdrant" else "8000"

_SETTINGS = {
    "LLAMA_USE_GPU": lambda: os.getenv("LLAMA_USE_GPU", "true").lower() == "true",
    "ALLOW_LATIN_EXTENDED": lambda: os.getenv("ALLOW_LATIN_EXTENDED", "true").lower() == "true",
    "LATIN_SCRIPT_MIN_RATIO": lambda: float(os.getenv("LATIN_SCRIPT_MIN_RATIO", "0.7")),
    "INGEST_FOLDER": lambda: _require_abs_path("INGEST_FOLDER"),
    "CHROMA_DATA_DIR": lambda: _abs_path("CHROMA_DATA_DIR") or os.path.join(_require_abs_path("INGEST_FOLDER"), "chroma_db"),
    "QDRANT_DATA_DIR": lambda: _abs_path("QDRANT_DATA_DIR") or os.path.join(_require_abs_path("INGEST_FOLDER"), "qdrant_data"),
    "VECTOR_DB_DATA_DIR": lambda: _abs_path("VECTOR_DB_DATA_DIR") or os.path.join(_require_abs_path("INGEST_FOLDER"), "qdrant_data"),
    "EMBEDDING_MODEL_PATH": lambda: _require_abs_path("EMBEDDING_MODEL_PATH"),
    "LLM_PATH": lambda: _require_abs_path("LLM_PATH"),
    "SUPERVISOR_LLM_PATH": lambda: _require_abs_path("SUPERVISOR_LLM_PATH"),
    "FAILED_FILES": lambda: _abs_path("FAILED_FILES") or os.path.join(_require_abs_path("INGEST_FOLDER"), "failed_files.txt"),
    "INGESTED_FILE": lambda: _abs_path("INGESTED_FILE") or os.path.join(_require_abs_path("INGEST_FOLDER"), "ingested_files.txt"),
    "TRACK_FILE": lambda: _abs_path("TRACK_FILE") or os.path.join(_require_abs_path("INGEST_FOLDER"), "ingested_files.txt"),
    "PARQUET_FILE": lambda: _abs_path("PARQUET_FILE") or os.path.join(_require_abs_path("INGEST_FOLDER"), "chunks.parquet"),
    "DUCKDB_FILE": lambda: _abs_path("DUCKDB_FILE") or os.path.join(_require_abs_path("INGEST_FOLDER"), "chunks.duckdb"),
    # Redis
    "REDIS_HOST": lambda: os.getenv("REDIS_HOST", "redis"),
    "REDIS_PORT": lambda: int(os.getenv("REDIS_PORT", "6380")),
    "REDIS_OCR_JOB_QUEUE": lambda: os.getenv("REDIS_OCR_JOB_QUEUE", "ocr_processing_job"),
    "REDIS_INGEST_QUEUE": lambda: os.getenv("REDIS_INGEST_QUEUE", "chunk_ingest_queue"),
    "QUEUE_NAMES": lambda: os.getenv("QUEUE_NAMES", "chunk_ingest_queue:0,chunk_ingest_queue:1").split(","),
    "VECTOR_DB_PROFILE": lambda: os.getenv("VECTOR_DB_PROFILE", "qdrant").lower(),
    "USE_QDRANT": lambda: os.getenv("VECTOR_DB_PROFILE", "qdrant").lower() == "qdrant",
    "VECTOR_DB_HOST": lambda: os.getenv("VECTOR_DB_HOST", os.getenv("CHROMA_HOST", "vector-db")),
    "VECTOR_DB_PORT": lambda: int(os.getenv("VECTOR_DB_PORT", os.getenv("CHROMA_PORT", _default_vector_port))),
    "VECTOR_DB_COLLECTION": lambda: os.getenv("VECTOR_DB_COLLECTION", os.getenv("CHROMA_COLLECTION", "vector_base_collection")),
    "QDRANT_RETRIEVER_K": lambda: int(os.getenv("QDRANT_RETRIEVER_K", 10)),
    "QDRANT_DENSE_WEIGHT": lambda: float(os.getenv("QDRANT_DENSE_WEIGHT", 0.3)),
    "QDRANT_SPARSE_WEIGHT": lambda: float(os.getenv("QDRANT_SPARSE_WEIGHT", 0.7)),
    "CHROMA_HOST": lambda: os.getenv("CHROMA_HOST", os.getenv("VECTOR_DB_HOST", "vector-db")),
    "CHROMA_PORT": lambda: int(os.getenv("CHROMA_PORT", os.getenv("VECTOR_DB_PORT", _default_vector_port))),
    "CHROMA_COLLECTION": lambda: os.getenv("CHROMA_COLLECTION", os.getenv("VECTOR_DB_COLLECTION", "vector_base_collection")),
    "CHUNK_TIMEOUT": lambda: int(os.getenv("CHUNK_TIMEOUT", "300")),
    "MAX_CHUNKS": lambda: int(os.getenv("MAX_CHUNKS", "20000")),
    "MAX_CHROMA_BATCH_SIZE": lambda: int(os.getenv("MAX_CHROMA_BATCH_SIZE", "500")),
    "MAX_TOKENS": lambda: int(os.getenv("MAX_TOKENS", "480")),
    "DEBUG_IMAGE_DIR": lambda: _abs_path("DEBUG_IMAGE_DIR") or os.path.join(_require_abs_path("INGEST_FOLDER"), "ocr_debug"),
    "MAX_OCR_DIM": lambda: int(os.getenv("MAX_OCR_DIM", "3000")),
    "TESSERACT_LANGS": lambda: os.getenv("TESSERACT_LANGS", "eng+lat"),
    "TESSERACT_USE_SCRIPT_LATIN": lambda: os.getenv("TESSERACT_USE_SCRIPT_LATIN", "true").lower() == "true",
    "TESSERACT_PSM": lambda: int(os.getenv("TESSERACT_PSM", "6")),
    "TESSERACT_OEM": lambda: int(os.getenv("TESSERACT_OEM", "1")),
    "TESSDATA_PREFIX": lambda: os.getenv("TESSDATA_PREFIX", ""),
    "SUPPORTED_MEDIA_EXT": lambda: tuple(os.getenv("SUPPORTED_MEDIA_EXT", ".mp3,.wav,.m4a,.aac,.flac,.mp4,.mov,.mkv").split(",")),
    "ALL_SUPPORTED_EXT": lambda: (".pdf", ".html", ".htm") + tuple(os.getenv("SUPPORTED_MEDIA_EXT", ".mp3,.wav,.m4a,.aac,.flac,.mp4,.mov,.mkv").split(",")),
    "DEVICE": lambda: os.getenv("DEVICE", "cuda"),
    "MEDIA_BATCH_SIZE": lambda: int(os.getenv("MEDIA_BATCH_SIZE", "8")),
    "COMPUTE_TYPE": lambda: os.getenv("COMPUTE_TYPE", "float16"),
    "MAX_QUEUE_LENGTH": lambda: int(os.getenv("MAX_QUEUE_LENGTH", "500")),
    "POLL_INTERVAL": lambda: float(os.getenv("POLL_INTERVAL", "0.5")),
    "WAIT_WARN_THRESHOLD": lambda: float(os.getenv("WAIT_WARN_THRESHOLD", "10")),
    "MAX_CHROMA_BATCH_SIZE_LIMIT": lambda: int(os.getenv("MAX_CHROMA_BATCH_SIZE_LIMIT", "5461")),
    "USE_OLLAMA": lambda: os.getenv("USE_OLLAMA", "0") == "1",
    "OLLAMA_MODEL": lambda: os.getenv("OLLAMA_MODEL", "NeuralNet/openchat-3.6"),
    "OLLAMA_URL": lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"),
    "RETRIEVER_TOP_K": lambda: int(os.getenv("RETRIEVER_TOP_K", "20")),
    "LLAMA_N_CTX": lambda: int(os.getenv("LLAMA_N_CTX", "32768")),
    "LLAMA_N_GPU_LAYERS": lambda: int(os.getenv("LLAMA_N_GPU_LAYERS", "35")),
    "LLAMA_N_THREADS": lambda: int(os.getenv("LLAMA_N_THREADS", "24")),
    "LLAMA_N_BATCH": lambda: int(os.getenv("LLAMA_N_BATCH", "512")),
    "LLAMA_F16_KV": lambda: os.getenv("LLAMA_F16_KV", "True").lower() == "true",
    "LLAMA_TEMPERATURE": lambda: float(os.getenv("LLAMA_TEMPERATURE", "0.1")),
    "SUPERVISOR_TEMPERATURE": lambda: float(os.getenv("SUPERVISOR_TEMPERATURE", "0.1")),
    "SUPERVISOR_TOP_K": lambda: int(os.getenv("SUPERVISOR_TOP_K", "5")),
    "LLAMA_TOP_K": lambda: int(os.getenv("LLAMA_TOP_K", "25")),
    "LLAMA_TOP_P": lambda: float(os.getenv("LLAMA_TOP_P", "0.85")),
    "LLAMA_REPEAT_PENALTY": lambda: float(os.getenv("LLAMA_REPEAT_PENALTY", "1.2")),
    "LLAMA_MAX_TOKENS": lambda: int(os.getenv("LLAMA_MAX_TOKENS", "512")),
    "LLAMA_CHAT_FORMAT": lambda: os.getenv("LLAMA_CHAT_FORMAT", "chatml"),
    "LLAMA_VERBOSE": lambda: os.getenv("LLAMA_VERBOSE", "False").lower() == "true",
    "LLAMA_SEED": lambda: int(os.getenv("LLAMA_SEED", "42")),
    "METRICS_ENABLED": lambda: os.getenv("METRICS_ENABLED", "true").lower() == "true",
    "METRICS_LOG_FILE": lambda: _abs_path("METRICS_LOG_FILE") or os.path.join(_require_abs_path("INGEST_FOLDER"), "metrics.jsonl"),
    "METRICS_LOG_TO_STDOUT": lambda: os.getenv("METRICS_LOG_TO_STDOUT", "true").lower() == "true",
}


def __getattr__(name):
    if name in _SETTINGS:
        return _SETTINGS[name]()
    raise AttributeError(f"module {__name__} has no attribute {name}")


# Critical: Ensure debug directory exists
_debug_dir = os.getenv("DEBUG_IMAGE_DIR", "/tmp/ocr_debug")
if not os.path.exists(_debug_dir):
    try:
        os.makedirs(_debug_dir, exist_ok=True)
    except Exception:
        pass
