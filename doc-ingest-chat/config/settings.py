#!/usr/bin/env python3
"""
Configuration settings for the document ingestion system.
Implementing a database-driven state machine architecture.
"""

import logging
import os
import sys
from typing import Any, Callable

log = logging.getLogger("ingest.settings")

# ---------------------------------------------------------------------------
# Helper: Environment Getters
# ---------------------------------------------------------------------------


def _abs_path(key: str, default: str = None) -> str:
    """Return absolute path if environment variable is set, else default."""
    val = os.getenv(key)
    if not val:
        return default

    if val.startswith(("http://", "https://")):
        return val
    return os.path.abspath(val)


def _require_abs_path(key: str, default: str = None) -> str:
    """Return absolute path, requiring the env var to be set (unless default provided)."""
    val = os.getenv(key)
    if not val:
        if default:
            val = default
        else:
            log.error(f"❌ CRITICAL ERROR: Environment variable '{key}' is NOT set. This is required for the system to function.")
            sys.exit(1)
    if val.startswith(("http://", "https://")):
        return val
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
    # Boolean flag to enable/disable CUDA usage
    "LLAMA_USE_GPU": lambda: os.getenv("LLAMA_USE_GPU", "true").lower() == "true",
    # Context window size for the local Llama model
    "LLAMA_N_CTX": lambda: int(os.getenv("LLAMA_N_CTX", "8192")),
    # Number of tokens to process in a single batch
    "LLAMA_N_BATCH": lambda: int(os.getenv("LLAMA_N_BATCH", "512")),
    # Number of layers to offload to GPU (-1 for all, 0 for CPU)
    "LLAMA_N_GPU_LAYERS": lambda: int(os.getenv("LLAMA_N_GPU_LAYERS", "-1")),
    # Number of threads to use for generation (0 defaults to CPU core count)
    "LLAMA_N_THREADS": lambda: int(os.getenv("LLAMA_N_THREADS", "0")),
    # Random seed for deterministic generation
    "LLAMA_SEED": lambda: int(os.getenv("LLAMA_SEED", "42")),
    # Enable/disable verbose logging from llama-cpp
    "LLAMA_VERBOSE": lambda: os.getenv("LLAMA_VERBOSE", "false").lower() == "true",
    # Creativity setting for generation (0.0 for deterministic)
    "LLAMA_TEMPERATURE": lambda: float(os.getenv("LLAMA_TEMPERATURE", "0.1")),
    # Top-K sampling parameter
    "LLAMA_TOP_K": lambda: int(os.getenv("LLAMA_TOP_K", "40")),
    # Top-P (nucleus) sampling parameter
    "LLAMA_TOP_P": lambda: float(os.getenv("LLAMA_TOP_P", "0.95")),
    # Penalty for repeating the same tokens
    "LLAMA_REPEAT_PENALTY": lambda: float(os.getenv("LLAMA_REPEAT_PENALTY", "1.1")),
    # Maximum tokens the LLM can generate in one response
    "LLAMA_MAX_TOKENS": lambda: int(os.getenv("LLAMA_MAX_TOKENS", "8192")),
    # Timeout in seconds for remote llama-server API calls
    "LLAMA_REMOTE_TIMEOUT": lambda: float(os.getenv("LLAMA_REMOTE_TIMEOUT", "300.0")),
    # Format used for the prompt template (e.g. chatml, llama-3)
    "LLAMA_CHAT_FORMAT": lambda: os.getenv("LLAMA_CHAT_FORMAT", "chatml"),
    # Use 16-bit floats for the Key-Value cache
    "LLAMA_F16_KV": lambda: os.getenv("LLAMA_F16_KV", "true").lower() == "true",
    # --- LIFECYCLE FOLDERS ---
    # The parent directory for all ingestion stages. Defaults to local ./Docs for testing.
    "DEFAULT_DOC_INGEST_ROOT": lambda: _require_abs_path("DEFAULT_DOC_INGEST_ROOT", os.path.abspath("./Docs")),
    # [STAGE 1] Raw PDF input directory
    "STAGING_DIR": lambda: _abs_path("STAGING_DIR", os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "staging")),
    # [STAGE 2] PDFs currently being normalized by Gatekeeper
    "PREPROCESSING_DIR": lambda: _abs_path("PREPROCESSING_DIR", os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "preprocessing")),
    # [STAGE 3] Normalized MD waiting for Producer
    "INGESTION_DIR": lambda: _abs_path("INGESTION_DIR", os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "ingestion")),
    # [STAGE 4] Chunks currently being embedded/stored
    "CONSUMING_DIR": lambda: _abs_path("CONSUMING_DIR", os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "consuming")),
    # [STAGE 5] Final terminal location for success
    "SUCCESS_DIR": lambda: _abs_path("SUCCESS_DIR", os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "success")),
    # [STAGE 5] Final terminal location for failures
    "FAILED_DIR": lambda: _abs_path("FAILED_DIR", os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "failed")),
    # --- COMPATIBILITY ALIASES (Derived from Master Root) ---
    "INGEST_FOLDER": lambda: _SETTINGS["INGESTION_DIR"](),
    "STAGING_FOLDER": lambda: _SETTINGS["STAGING_DIR"](),
    # Path to the DuckDB database used for lifecycle and history tracking
    "GATEKEEPER_FAILURE_DB": lambda: _abs_path("GATEKEEPER_FAILURE_DB", os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "gatekeeper_history.db")),
    # [REQUIRED] Absolute path to the e5-large-v2 embedding model directory
    "EMBEDDING_MODEL_PATH": lambda: _require_abs_path("EMBEDDING_MODEL_PATH"),
    # [REQUIRED] Path to main Llama GGUF or remote llama-server URL
    "LLM_PATH": lambda: _require_abs_path("LLM_PATH"),
    # [REQUIRED] Path to supervisor Llama GGUF or remote llama-server URL
    "SUPERVISOR_LLM_PATH": lambda: _require_abs_path("SUPERVISOR_LLM_PATH"),
    # [OPTIONAL] Path to local Whisper models. If missing, media transcription is skipped.
    "WHISPER_MODEL_PATH": lambda: _abs_path("WHISPER_MODEL_PATH", "NOT_SET"),
    # Mandatory files required for offline WhisperX (CTranslate2 format)
    "WHISPER_REQUIRED_FILES": lambda: ["model.bin", "config.json", "vocabulary.txt"],
    # Model name required by OpenAI-compatible API (e.g. for Ollama routing)
    "SUPERVISOR_REMOTE_MODEL_NAME": lambda: os.getenv("SUPERVISOR_REMOTE_MODEL_NAME", "local-model"),
    # Creativity setting for the supervisor agent
    "SUPERVISOR_TEMPERATURE": lambda: float(os.getenv("SUPERVISOR_TEMPERATURE", "0.1")),
    # Top-K sampling for the supervisor agent
    "SUPERVISOR_TOP_K": lambda: int(os.getenv("SUPERVISOR_TOP_K", "40")),
    # Path to the Parquet archival file for all chunks
    "PARQUET_FILE": lambda: _abs_path("PARQUET_FILE", os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "chunks.parquet")),
    # Path to the relational DuckDB file storing chunk metadata
    "DUCKDB_FILE": lambda: _abs_path("DUCKDB_FILE", os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "chunks.duckdb")),
    # Use Ollama instead of local llama-cpp-python
    "USE_OLLAMA": lambda: os.getenv("USE_OLLAMA", "false").lower() == "true",
    # Base URL for the Ollama API
    "OLLAMA_URL": lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"),
    # Model name to use on the Ollama server
    "OLLAMA_MODEL": lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    # Hostname for the Redis message broker
    "REDIS_HOST": lambda: os.environ.get("REDIS_HOST") or os.getenv("REDIS_HOST", "localhost"),
    # Port for the Redis message broker
    "REDIS_PORT": lambda: int(os.environ.get("REDIS_PORT") or os.getenv("REDIS_PORT", "6379")),
    # Queue name used for offloading images to the OCR workers
    "REDIS_OCR_JOB_QUEUE": lambda: os.getenv("REDIS_OCR_JOB_QUEUE", "ocr_processing_job"),
    # Queue name used for offloading media to the WhisperX workers
    "REDIS_WHISPER_JOB_QUEUE": lambda: os.getenv("REDIS_WHISPER_JOB_QUEUE", "whisper_processing_job"),
    # Main queue used for sending semantic chunks to the consumer workers
    "REDIS_INGEST_QUEUE": lambda: os.getenv("REDIS_INGEST_QUEUE", "chunk_ingest_queue"),
    # List of Redis queues to monitor (supports partitioning)
    "QUEUE_NAMES": lambda: os.getenv("QUEUE_NAMES", "chunk_ingest_queue:0,chunk_ingest_queue:1").split(","),
    # Active vector database ("qdrant" or "chroma")
    "VECTOR_DB_PROFILE": lambda: os.getenv("VECTOR_DB_PROFILE", "qdrant").lower(),
    # Boolean helper for Qdrant mode
    "USE_QDRANT": lambda: os.getenv("VECTOR_DB_PROFILE", "qdrant").lower() == "qdrant",
    # Hostname for the vector database server
    "VECTOR_DB_HOST": lambda: os.getenv("VECTOR_DB_HOST", os.getenv("CHROMA_HOST", "vector-db")),
    # Port for the vector database server
    "VECTOR_DB_PORT": _get_vector_db_port,
    # Collection name used for storing vectors
    "VECTOR_DB_COLLECTION": lambda: os.getenv("VECTOR_DB_COLLECTION", "vector_base_collection"),
    # Number of chunks to process in a single embedding batch
    "MAX_CHROMA_BATCH_SIZE": lambda: int(os.getenv("MAX_CHROMA_BATCH_SIZE", "75")),
    # Time (seconds) before an incomplete chunk buffer is discarded
    "CHUNK_TIMEOUT": lambda: int(os.getenv("CHUNK_TIMEOUT", "300")),
    # Maximum chunks per file before discarding (safety limit)
    "MAX_CHUNKS": lambda: int(os.getenv("MAX_CHUNKS", "5000")),
    # [ALIAS] Compatibility with old Chroma-specific hostname
    "CHROMA_HOST": lambda: os.getenv("VECTOR_DB_HOST", os.getenv("CHROMA_HOST", "vector-db")),
    # [ALIAS] Compatibility with old Chroma-specific port
    "CHROMA_PORT": _get_vector_db_port,
    # [ALIAS] Compatibility with old Chroma-specific collection
    "CHROMA_COLLECTION": lambda: os.getenv("VECTOR_DB_COLLECTION", os.getenv("CHROMA_COLLECTION", "vector_base_collection")),
    # Strict limit for tokens stored in the Vector DB for RAG
    "MAX_TOKENS": lambda: int(os.getenv("MAX_TOKENS", "512")),
    # Target character size for initial splitting logic
    "CHUNK_SIZE": lambda: int(os.getenv("CHUNK_SIZE", "512")),
    # Number of characters to overlap between chunks
    "CHUNK_OVERLAP": lambda: int(os.getenv("CHUNK_OVERLAP", "50")),
    # Enable/disable mojibake and encoding fixes
    "ALLOW_LATIN_EXTENDED": lambda: os.getenv("ALLOW_LATIN_EXTENDED", "true").lower() == "true",
    # Minimum ratio of Latin characters required before triggering OCR fallback
    "LATIN_SCRIPT_MIN_RATIO": lambda: float(os.getenv("LATIN_SCRIPT_MIN_RATIO", "0.7")),
    # Directory where OCR failure images are stored for debugging
    "DEBUG_IMAGE_DIR": lambda: _abs_path("DEBUG_IMAGE_DIR", os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "ocr_debug")),
    # Maximum dimension for images before resizing to save memory during OCR
    "MAX_OCR_DIM": lambda: int(os.getenv("MAX_OCR_DIM", "3000")),
    # Extensions supported for direct text extraction
    "SUPPORTED_DOC_EXT": lambda: (".pdf", ".html", ".htm", ".txt", ".md"),
    # Extensions supported for media transcription
    "SUPPORTED_MEDIA_EXT": lambda: tuple(os.getenv("SUPPORTED_MEDIA_EXT", ".mp3,.wav,.m4a,.aac,.flac,.mp4,.mov,.mkv").split(",")),
    # List of ALL supported file types
    "ALL_SUPPORTED_EXT": lambda: (".pdf", ".html", ".htm", ".txt", ".md") + tuple(os.getenv("SUPPORTED_MEDIA_EXT", ".mp3,.wav,.m4a,.aac,.flac,.mp4,.mov,.mkv").split(",")),
    # Compute device for local models ("cuda" or "cpu")
    "DEVICE": lambda: os.getenv("DEVICE", "cuda"),
    # Batch size for Whisper media transcription
    "MEDIA_BATCH_SIZE": lambda: int(os.getenv("MEDIA_BATCH_SIZE", "8")),
    # Precision used for local model inference (e.g. float16, int8)
    "COMPUTE_TYPE": lambda: os.getenv("COMPUTE_TYPE", "float16"),
    # Number of documents to retrieve during RAG search
    "RETRIEVER_TOP_K": lambda: int(os.getenv("RETRIEVER_TOP_K", "4")),
    # Number of PDF pages to batch together for normalization
    "GATEKEEPER_BATCH_SIZE": lambda: int(os.getenv("GATEKEEPER_BATCH_SIZE", "5")),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    if name in _SETTINGS:
        return _SETTINGS[name]()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Ensure all lifecycle directories exist
def ensure_folders():
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
            # Handle potential permission errors in restricted environments (like tests)
            pass


if __name__ == "__main__":
    ensure_folders()
