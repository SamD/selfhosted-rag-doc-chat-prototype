#!/usr/bin/env python3
"""
Shared configuration system for the monorepo.
All components (doc-ingest-chat, mqtt_agent_hub, etc.) should import from here
or from their own thin compatibility wrappers that delegate to this module.

Settings are lazy-loaded: importing a name triggers os.getenv() with a
canonical default from shared.defaults.
"""

import logging
import os
import sys
from typing import Any, Callable

from shared.defaults import (
    DEFAULT_ALLOW_LATIN_EXTENDED,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_TIMEOUT,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_FORCE_MARKDOWN_LLM,
    DEFAULT_GATEKEEPER_BATCH_SIZE,
    DEFAULT_HA_INTERLEAVE,
    DEFAULT_LATIN_SCRIPT_MIN_RATIO,
    DEFAULT_LLAMA_CHAT_FORMAT,
    DEFAULT_LLAMA_F16_KV,
    DEFAULT_LLAMA_MAX_TOKENS,
    DEFAULT_LLAMA_N_BATCH,
    DEFAULT_LLAMA_N_CTX,
    DEFAULT_LLAMA_N_GPU_LAYERS,
    DEFAULT_LLAMA_N_THREADS,
    DEFAULT_LLAMA_REMOTE_TIMEOUT,
    DEFAULT_LLAMA_REPEAT_PENALTY,
    DEFAULT_LLAMA_SEED,
    DEFAULT_LLAMA_TEMPERATURE,
    DEFAULT_LLAMA_TOP_K,
    DEFAULT_LLAMA_TOP_P,
    DEFAULT_LLAMA_VERBOSE,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_MAX_OCR_DIM,
    DEFAULT_MAX_SESSION_TURNS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MEDIA_BATCH_SIZE,
    DEFAULT_METRICS_ENABLED,
    DEFAULT_METRICS_LOG_TO_STDOUT,
    DEFAULT_OCR_ENDPOINTS,
    DEFAULT_PDF_FORCE_OCR,
    DEFAULT_REDIS_HOST,
    DEFAULT_REDIS_INGEST_QUEUE,
    DEFAULT_REDIS_OCR_JOB_QUEUE,
    DEFAULT_REDIS_PORT,
    DEFAULT_REDIS_STAGING_QUEUE,
    DEFAULT_REDIS_WHISPER_JOB_QUEUE,
    DEFAULT_RETRIEVER_TOP_K,
    DEFAULT_SESSION_TTL_HOURS,
    DEFAULT_STUCK_JOB_TIMEOUT_HOURS,
    DEFAULT_SUPERVISOR_MAX_TOKENS,
    DEFAULT_SUPERVISOR_N_CTX,
    DEFAULT_SUPERVISOR_REMOTE_MODEL_NAME,
    DEFAULT_SUPERVISOR_TEMPERATURE,
    DEFAULT_SUPERVISOR_TOP_K,
    DEFAULT_SUPPORTED_MEDIA_EXT,
    DEFAULT_VECTOR_DB_BATCH_SIZE,
    DEFAULT_VECTOR_DB_COLLECTION,
    DEFAULT_VECTOR_DB_GRPC_PORT,
    DEFAULT_VECTOR_DB_HOST,
    DEFAULT_VECTOR_DB_PROFILE,
    DEFAULT_VECTOR_DB_TIMEOUT,
    DEFAULT_VECTOR_DB_USE_GRPC,
    SUPPORTED_DOC_EXT,
    WHISPER_REQUIRED_FILES_LIST,
)
from shared.env_names import (
    ENV_ALLOW_LATIN_EXTENDED,
    ENV_CHROMA_COLLECTION,
    ENV_CHROMA_HOST,
    ENV_CHROMA_PORT,
    ENV_CHUNK_OVERLAP,
    ENV_CHUNK_SIZE,
    ENV_CHUNK_TIMEOUT,
    ENV_COMPUTE_TYPE,
    ENV_CONSUMING_DIR,
    ENV_DEBUG_IMAGE_DIR,
    ENV_DEFAULT_DOC_INGEST_ROOT,
    ENV_DEVICE,
    ENV_DUCKDB_FILE,
    ENV_EMBEDDING_ENDPOINTS,
    ENV_FAILED_DIR,
    ENV_FAILED_FILES,
    ENV_FORCE_MARKDOWN_LLM,
    ENV_GATEKEEPER_BATCH_SIZE,
    ENV_GATEKEEPER_FAILURE_DB,
    ENV_HA_INTERLEAVE,
    ENV_INGESTED_FILE,
    ENV_INGESTION_DIR,
    ENV_LATIN_SCRIPT_MIN_RATIO,
    ENV_LLAMA_CHAT_FORMAT,
    ENV_LLAMA_F16_KV,
    ENV_LLAMA_MAX_TOKENS,
    ENV_LLAMA_N_BATCH,
    ENV_LLAMA_N_CTX,
    ENV_LLAMA_N_GPU_LAYERS,
    ENV_LLAMA_N_THREADS,
    ENV_LLAMA_REMOTE_TIMEOUT,
    ENV_LLAMA_REPEAT_PENALTY,
    ENV_LLAMA_SEED,
    ENV_LLAMA_TEMPERATURE,
    ENV_LLAMA_TOP_K,
    ENV_LLAMA_TOP_P,
    ENV_LLAMA_VERBOSE,
    ENV_LLM_PATH,
    ENV_MAX_CHROMA_BATCH_SIZE,
    ENV_MAX_CHUNKS,
    ENV_MAX_OCR_DIM,
    ENV_MAX_SESSION_TURNS,
    ENV_MAX_TOKENS,
    ENV_MEDIA_BATCH_SIZE,
    ENV_METRICS_ENABLED,
    ENV_METRICS_LOG_FILE,
    ENV_METRICS_LOG_TO_STDOUT,
    ENV_OCR_ENDPOINTS,
    ENV_PARQUET_FILE,
    ENV_PDF_FORCE_OCR,
    ENV_PREPROCESSING_DIR,
    ENV_QUEUE_NAMES,
    ENV_REDIS_HOST,
    ENV_REDIS_INGEST_QUEUE,
    ENV_REDIS_OCR_JOB_QUEUE,
    ENV_REDIS_PORT,
    ENV_REDIS_STAGING_QUEUE,
    ENV_REDIS_WHISPER_JOB_QUEUE,
    ENV_RETRIEVER_TOP_K,
    ENV_SESSION_TTL_HOURS,
    ENV_STAGING_DIR,
    ENV_STUCK_JOB_TIMEOUT_HOURS,
    ENV_SUCCESS_DIR,
    ENV_SUPERVISOR_LLM_ENDPOINTS,
    ENV_SUPERVISOR_MAX_TOKENS,
    ENV_SUPERVISOR_N_CTX,
    ENV_SUPERVISOR_REMOTE_MODEL_NAME,
    ENV_SUPERVISOR_TEMPERATURE,
    ENV_SUPERVISOR_TOP_K,
    ENV_SUPPORTED_MEDIA_EXT,
    ENV_VECTOR_DB_BATCH_SIZE,
    ENV_VECTOR_DB_COLLECTION,
    ENV_VECTOR_DB_GRPC_PORT,
    ENV_VECTOR_DB_HOST,
    ENV_VECTOR_DB_PORT,
    ENV_VECTOR_DB_PROFILE,
    ENV_VECTOR_DB_TIMEOUT,
    ENV_VECTOR_DB_URL,
    ENV_VECTOR_DB_USE_GRPC,
    ENV_WHISPER_MODEL_ENDPOINTS,
)

log = logging.getLogger("shared.config")

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
            log.error(
                f"❌ CRITICAL ERROR: Environment variable '{key}' is NOT set. "
                "This is required for the system to function."
            )
            sys.exit(1)
    if val.startswith(("http://", "https://")):
        return val
    return os.path.abspath(val)


def _get_vector_db_port() -> int:
    """Returns the vector database port with dynamic defaults based on profile."""
    port = os.getenv(ENV_VECTOR_DB_PORT) or os.getenv(ENV_CHROMA_PORT)
    if port:
        return int(port)

    profile = os.getenv(ENV_VECTOR_DB_PROFILE, DEFAULT_VECTOR_DB_PROFILE).lower()
    return 6333 if profile == "qdrant" else 8000


# ---------------------------------------------------------------------------
# Settings Dictionary (Lazy Loaded)
# ---------------------------------------------------------------------------

_SETTINGS: dict[str, Callable[[], Any]] = {
    # Context window size for the local Llama model
    "LLAMA_N_CTX": lambda: int(os.getenv(ENV_LLAMA_N_CTX, str(DEFAULT_LLAMA_N_CTX))),
    # Number of tokens to process in a single batch
    "LLAMA_N_BATCH": lambda: int(os.getenv(ENV_LLAMA_N_BATCH, str(DEFAULT_LLAMA_N_BATCH))),
    # Number of layers to offload to GPU (-1 for all, 0 for CPU)
    "LLAMA_N_GPU_LAYERS": lambda: int(os.getenv(ENV_LLAMA_N_GPU_LAYERS, str(DEFAULT_LLAMA_N_GPU_LAYERS))),
    # Number of threads to use for generation (0 defaults to CPU core count)
    "LLAMA_N_THREADS": lambda: int(os.getenv(ENV_LLAMA_N_THREADS, str(DEFAULT_LLAMA_N_THREADS))),
    # Random seed for deterministic generation
    "LLAMA_SEED": lambda: int(os.getenv(ENV_LLAMA_SEED, str(DEFAULT_LLAMA_SEED))),
    # Enable/disable verbose logging from llama-cpp
    "LLAMA_VERBOSE": lambda: os.getenv(ENV_LLAMA_VERBOSE, DEFAULT_LLAMA_VERBOSE).lower() == "true",
    # Creativity setting for generation (0.0 for deterministic)
    "LLAMA_TEMPERATURE": lambda: float(os.getenv(ENV_LLAMA_TEMPERATURE, str(DEFAULT_LLAMA_TEMPERATURE))),
    # Top-K sampling parameter
    "LLAMA_TOP_K": lambda: int(os.getenv(ENV_LLAMA_TOP_K, str(DEFAULT_LLAMA_TOP_K))),
    # Top-P (nucleus) sampling parameter
    "LLAMA_TOP_P": lambda: float(os.getenv(ENV_LLAMA_TOP_P, str(DEFAULT_LLAMA_TOP_P))),
    # Penalty for repeating the same tokens
    "LLAMA_REPEAT_PENALTY": lambda: float(os.getenv(ENV_LLAMA_REPEAT_PENALTY, str(DEFAULT_LLAMA_REPEAT_PENALTY))),
    # Maximum tokens the LLM can generate in one response
    "LLAMA_MAX_TOKENS": lambda: int(os.getenv(ENV_LLAMA_MAX_TOKENS, str(DEFAULT_LLAMA_MAX_TOKENS))),
    # Timeout in seconds for remote llama-server API calls
    "LLAMA_REMOTE_TIMEOUT": lambda: float(os.getenv(ENV_LLAMA_REMOTE_TIMEOUT, str(DEFAULT_LLAMA_REMOTE_TIMEOUT))),
    # Format used for the prompt template (e.g. chatml, llama-3)
    "LLAMA_CHAT_FORMAT": lambda: os.getenv(ENV_LLAMA_CHAT_FORMAT, DEFAULT_LLAMA_CHAT_FORMAT),
    # Use 16-bit floats for the Key-Value cache
    "LLAMA_F16_KV": lambda: os.getenv(ENV_LLAMA_F16_KV, DEFAULT_LLAMA_F16_KV).lower() == "true",
    # --- LIFECYCLE FOLDERS ---
    "DEFAULT_DOC_INGEST_ROOT": lambda: _require_abs_path(
        ENV_DEFAULT_DOC_INGEST_ROOT, os.path.abspath("./Docs")
    ),
    # [STAGE 1] Raw PDF input directory
    "STAGING_DIR": lambda: _abs_path(
        ENV_STAGING_DIR, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "staging")
    ),
    # [STAGE 2] PDFs currently being normalized by Gatekeeper
    "PREPROCESSING_DIR": lambda: _abs_path(
        ENV_PREPROCESSING_DIR, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "preprocessing")
    ),
    # [STAGE 3] Normalized MD waiting for Producer
    "INGESTION_DIR": lambda: _abs_path(
        ENV_INGESTION_DIR, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "ingestion")
    ),
    # [STAGE 4] Chunks currently being embedded/stored
    "CONSUMING_DIR": lambda: _abs_path(
        ENV_CONSUMING_DIR, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "consuming")
    ),
    # [STAGE 5] Final terminal location for success
    "SUCCESS_DIR": lambda: _abs_path(
        ENV_SUCCESS_DIR, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "success")
    ),
    # [STAGE 5] Final terminal location for failures
    "FAILED_DIR": lambda: _abs_path(
        ENV_FAILED_DIR, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "failed")
    ),
    # --- COMPATIBILITY ALIASES (Derived from Master Root) ---
    "INGEST_FOLDER": lambda: _SETTINGS["INGESTION_DIR"](),
    "STAGING_FOLDER": lambda: _SETTINGS["STAGING_DIR"](),
    # Path to the DuckDB database used for lifecycle and history tracking
    "GATEKEEPER_FAILURE_DB": lambda: _abs_path(
        ENV_GATEKEEPER_FAILURE_DB,
        os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "gatekeeper_history.db"),
    ),
    # [REQUIRED] Absolute path to the e5-large-v2 embedding model directory or remote URL(s)
    "EMBEDDING_ENDPOINTS": lambda: _require_abs_path(ENV_EMBEDDING_ENDPOINTS),
    # [REQUIRED] Path to main Llama GGUF or remote llama-server URL
    "LLM_PATH": lambda: _require_abs_path(ENV_LLM_PATH),
    # [REQUIRED] Path to supervisor Llama GGUF or remote llama-server URL(s)
    "SUPERVISOR_LLM_ENDPOINTS": lambda: _require_abs_path(ENV_SUPERVISOR_LLM_ENDPOINTS),
    # [OPTIONAL] Path to local Whisper models or remote URL.
    "WHISPER_MODEL_ENDPOINTS": lambda: _abs_path(ENV_WHISPER_MODEL_ENDPOINTS, "NOT_SET"),
    # Mandatory files required for offline WhisperX (CTranslate2 format)
    "WHISPER_REQUIRED_FILES": lambda: WHISPER_REQUIRED_FILES_LIST,
    # Model name required by OpenAI-compatible API (e.g. for Ollama routing)
    "SUPERVISOR_REMOTE_MODEL_NAME": lambda: os.getenv(ENV_SUPERVISOR_REMOTE_MODEL_NAME, DEFAULT_SUPERVISOR_REMOTE_MODEL_NAME),
    # Creativity setting for the supervisor agent
    "SUPERVISOR_TEMPERATURE": lambda: float(os.getenv(ENV_SUPERVISOR_TEMPERATURE, str(DEFAULT_SUPERVISOR_TEMPERATURE))),
    # Top-K sampling for the supervisor agent
    "SUPERVISOR_TOP_K": lambda: int(os.getenv(ENV_SUPERVISOR_TOP_K, str(DEFAULT_SUPERVISOR_TOP_K))),
    # Maximum tokens the supervisor LLM can generate in one response
    "SUPERVISOR_MAX_TOKENS": lambda: int(os.getenv(ENV_SUPERVISOR_MAX_TOKENS, str(DEFAULT_SUPERVISOR_MAX_TOKENS))),
    # Context window for the supervisor LLM (used for gatekeeper prompt truncation)
    "SUPERVISOR_N_CTX": lambda: int(os.getenv(ENV_SUPERVISOR_N_CTX, str(DEFAULT_SUPERVISOR_N_CTX))),
    # Path to the Parquet archival file for all chunks
    "PARQUET_FILE": lambda: _abs_path(
        ENV_PARQUET_FILE, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "chunks.parquet")
    ),
    # Path to the relational DuckDB file storing chunk metadata
    "DUCKDB_FILE": lambda: _abs_path(
        ENV_DUCKDB_FILE, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "chunks.duckdb")
    ),
    # Hostname for the Redis message broker
    "REDIS_HOST": lambda: os.environ.get(ENV_REDIS_HOST) or os.getenv(ENV_REDIS_HOST, DEFAULT_REDIS_HOST),
    # Port for the Redis message broker
    "REDIS_PORT": lambda: int(os.environ.get(ENV_REDIS_PORT) or os.getenv(ENV_REDIS_PORT, str(DEFAULT_REDIS_PORT))),
    # Queue name used for offloading images to the OCR workers
    "REDIS_OCR_JOB_QUEUE": lambda: os.getenv(ENV_REDIS_OCR_JOB_QUEUE, DEFAULT_REDIS_OCR_JOB_QUEUE),
    # Queue name used for offloading media to the WhisperX workers
    "REDIS_WHISPER_JOB_QUEUE": lambda: os.getenv(ENV_REDIS_WHISPER_JOB_QUEUE, DEFAULT_REDIS_WHISPER_JOB_QUEUE),
    # Main queue used for sending semantic chunks to the consumer workers
    "REDIS_INGEST_QUEUE": lambda: os.getenv(ENV_REDIS_INGEST_QUEUE, DEFAULT_REDIS_INGEST_QUEUE),
    # New dedicated queue for eager chunk staging into DuckDB
    "REDIS_STAGING_QUEUE": lambda: os.getenv(ENV_REDIS_STAGING_QUEUE, DEFAULT_REDIS_STAGING_QUEUE),
    # List of Redis queues to monitor (supports partitioning)
    "QUEUE_NAMES": lambda: os.getenv(
        ENV_QUEUE_NAMES, "chunk_ingest_queue:0,chunk_ingest_queue:1"
    ).split(","),
    # Active vector database ("qdrant" or "chroma")
    "VECTOR_DB_PROFILE": lambda: os.getenv(ENV_VECTOR_DB_PROFILE, DEFAULT_VECTOR_DB_PROFILE).lower(),
    # Boolean helper for Qdrant mode
    "USE_QDRANT": lambda: os.getenv(ENV_VECTOR_DB_PROFILE, DEFAULT_VECTOR_DB_PROFILE).lower() == "qdrant",
    # [OPTIONAL] Full URL for the vector database server (overrides host/port)
    "VECTOR_DB_URL": lambda: os.getenv(ENV_VECTOR_DB_URL),
    # Hostname for the vector database server
    "VECTOR_DB_HOST": lambda: os.getenv(ENV_VECTOR_DB_HOST, os.getenv(ENV_CHROMA_HOST, DEFAULT_VECTOR_DB_HOST)),
    # Port for the vector database server
    "VECTOR_DB_PORT": _get_vector_db_port,
    # [OPTIONAL] gRPC Port for the vector database server
    "VECTOR_DB_GRPC_PORT": lambda: int(os.getenv(ENV_VECTOR_DB_GRPC_PORT, str(DEFAULT_VECTOR_DB_GRPC_PORT))),
    # Whether to prefer gRPC for Qdrant operations
    "VECTOR_DB_USE_GRPC": lambda: os.getenv(ENV_VECTOR_DB_USE_GRPC, DEFAULT_VECTOR_DB_USE_GRPC).lower() == "true",
    # Collection name used for storing vectors
    "VECTOR_DB_COLLECTION": lambda: os.getenv(ENV_VECTOR_DB_COLLECTION, DEFAULT_VECTOR_DB_COLLECTION),
    # Number of chunks to process in a single embedding batch
    "VECTOR_DB_BATCH_SIZE": lambda: int(
        os.getenv(ENV_VECTOR_DB_BATCH_SIZE, os.getenv(ENV_MAX_CHROMA_BATCH_SIZE, str(DEFAULT_VECTOR_DB_BATCH_SIZE)))
    ),
    # Timeout (seconds) for vector database operations
    "VECTOR_DB_TIMEOUT": lambda: float(os.getenv(ENV_VECTOR_DB_TIMEOUT, str(DEFAULT_VECTOR_DB_TIMEOUT))),
    # [DEPRECATED] Use VECTOR_DB_BATCH_SIZE instead
    "MAX_CHROMA_BATCH_SIZE": lambda: int(
        os.getenv(ENV_VECTOR_DB_BATCH_SIZE, os.getenv(ENV_MAX_CHROMA_BATCH_SIZE, str(DEFAULT_VECTOR_DB_BATCH_SIZE)))
    ),
    # Time (seconds) before an incomplete chunk buffer is discarded
    "CHUNK_TIMEOUT": lambda: int(os.getenv(ENV_CHUNK_TIMEOUT, str(DEFAULT_CHUNK_TIMEOUT))),
    # Maximum chunks per file before discarding (safety limit)
    "MAX_CHUNKS": lambda: int(os.getenv(ENV_MAX_CHUNKS, str(DEFAULT_MAX_CHUNKS))),
    # [ALIAS] Compatibility with old Chroma-specific hostname
    "CHROMA_HOST": lambda: os.getenv(ENV_VECTOR_DB_HOST, os.getenv(ENV_CHROMA_HOST, DEFAULT_VECTOR_DB_HOST)),
    # [ALIAS] Compatibility with old Chroma-specific port
    "CHROMA_PORT": _get_vector_db_port,
    # [ALIAS] Compatibility with old Chroma-specific collection
    "CHROMA_COLLECTION": lambda: os.getenv(
        ENV_VECTOR_DB_COLLECTION, os.getenv(ENV_CHROMA_COLLECTION, DEFAULT_VECTOR_DB_COLLECTION)
    ),
    # Strict limit for tokens stored in the Vector DB for RAG
    "MAX_TOKENS": lambda: int(os.getenv(ENV_MAX_TOKENS, str(DEFAULT_MAX_TOKENS))),
    # Target character size for initial splitting logic
    "CHUNK_SIZE": lambda: int(os.getenv(ENV_CHUNK_SIZE, str(DEFAULT_CHUNK_SIZE))),
    # Number of characters to overlap between chunks
    "CHUNK_OVERLAP": lambda: int(os.getenv(ENV_CHUNK_OVERLAP, str(DEFAULT_CHUNK_OVERLAP))),
    # Enable/disable mojibake and encoding fixes
    "ALLOW_LATIN_EXTENDED": lambda: os.getenv(ENV_ALLOW_LATIN_EXTENDED, DEFAULT_ALLOW_LATIN_EXTENDED).lower() == "true",
    # Minimum ratio of Latin characters required before triggering OCR fallback
    "LATIN_SCRIPT_MIN_RATIO": lambda: float(os.getenv(ENV_LATIN_SCRIPT_MIN_RATIO, str(DEFAULT_LATIN_SCRIPT_MIN_RATIO))),
    # [OPTIONAL] Remote Docling endpoint or 'LOCAL'
    "OCR_ENDPOINTS": lambda: os.getenv(ENV_OCR_ENDPOINTS, DEFAULT_OCR_ENDPOINTS),
    # Directory where OCR failure images are stored for debugging
    "DEBUG_IMAGE_DIR": lambda: _abs_path(
        ENV_DEBUG_IMAGE_DIR, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "ocr_debug")
    ),
    # Maximum dimension for images before resizing to save memory during OCR
    "MAX_OCR_DIM": lambda: int(os.getenv(ENV_MAX_OCR_DIM, str(DEFAULT_MAX_OCR_DIM))),
    # Extensions supported for direct text extraction
    "SUPPORTED_DOC_EXT": lambda: SUPPORTED_DOC_EXT,
    # Extensions supported for media transcription
    "SUPPORTED_MEDIA_EXT": lambda: tuple(os.getenv(ENV_SUPPORTED_MEDIA_EXT, DEFAULT_SUPPORTED_MEDIA_EXT).split(",")),
    # List of ALL supported file types
    "ALL_SUPPORTED_EXT": lambda: SUPPORTED_DOC_EXT + tuple(
        os.getenv(ENV_SUPPORTED_MEDIA_EXT, DEFAULT_SUPPORTED_MEDIA_EXT).split(",")
    ),
    # Compute device for local models ("cuda" or "cpu")
    "DEVICE": lambda: os.getenv(ENV_DEVICE, DEFAULT_DEVICE),
    # Batch size for Whisper media transcription
    "MEDIA_BATCH_SIZE": lambda: int(os.getenv(ENV_MEDIA_BATCH_SIZE, str(DEFAULT_MEDIA_BATCH_SIZE))),
    # Precision used for local model inference (e.g. float16, int8)
    "COMPUTE_TYPE": lambda: os.getenv(ENV_COMPUTE_TYPE, DEFAULT_COMPUTE_TYPE),
    # Number of documents to retrieve during RAG search
    "RETRIEVER_TOP_K": lambda: int(os.getenv(ENV_RETRIEVER_TOP_K, str(DEFAULT_RETRIEVER_TOP_K))),
    # Number of PDF pages to batch together for normalization
    "GATEKEEPER_BATCH_SIZE": lambda: int(os.getenv(ENV_GATEKEEPER_BATCH_SIZE, str(DEFAULT_GATEKEEPER_BATCH_SIZE))),
    # Whether to force OCR for all PDFs (high fidelity but slower)
    "PDF_FORCE_OCR": lambda: os.getenv(ENV_PDF_FORCE_OCR, DEFAULT_PDF_FORCE_OCR).lower() in ("1", "true"),
    "FORCE_MARKDOWN_LLM": lambda: os.getenv(ENV_FORCE_MARKDOWN_LLM, DEFAULT_FORCE_MARKDOWN_LLM).lower() in ("1", "true"),
    # Enable HA batch interleaving: concurrent dispatch across multiple backends
    "HA_INTERLEAVE": lambda: os.getenv(ENV_HA_INTERLEAVE, DEFAULT_HA_INTERLEAVE).lower() in ("1", "true"),
    # Metrics reporting toggle
    "METRICS_ENABLED": lambda: os.getenv(ENV_METRICS_ENABLED, DEFAULT_METRICS_ENABLED).lower() == "true",
    # File path for metrics log output
    "METRICS_LOG_FILE": lambda: os.getenv(ENV_METRICS_LOG_FILE),
    # Emit metrics to stdout
    "METRICS_LOG_TO_STDOUT": lambda: os.getenv(ENV_METRICS_LOG_TO_STDOUT, DEFAULT_METRICS_LOG_TO_STDOUT).lower() == "true",
    # File path for tracking failed files
    "FAILED_FILES": lambda: _abs_path(
        ENV_FAILED_FILES, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "failed_files.txt")
    ),
    # File path for tracking ingested files
    "INGESTED_FILE": lambda: _abs_path(
        ENV_INGESTED_FILE, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "ingested_files.txt")
    ),
    # Maximum conversation turns per chat session (oldest dropped when exceeded)
    "MAX_SESSION_TURNS": lambda: int(os.getenv(ENV_MAX_SESSION_TURNS, str(DEFAULT_MAX_SESSION_TURNS))),
    # Session TTL in hours (inactive sessions are evicted from Redis)
    "SESSION_TTL_HOURS": lambda: int(os.getenv(ENV_SESSION_TTL_HOURS, str(DEFAULT_SESSION_TTL_HOURS))),
    # Hours before a job stuck in an intermediate state is eligible for reclaim
    "STUCK_JOB_TIMEOUT_HOURS": lambda: int(os.getenv(ENV_STUCK_JOB_TIMEOUT_HOURS, str(DEFAULT_STUCK_JOB_TIMEOUT_HOURS))),
}


def get_setting(name: str) -> Any:
    """Resolve a setting by name. Raises KeyError if unknown."""
    if name not in _SETTINGS:
        raise KeyError(f"Unknown setting: '{name}'")
    return _SETTINGS[name]()
