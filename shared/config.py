#!/usr/bin/env python3
"""
Shared configuration system for the monorepo.
All components (doc-ingest-chat, mqtt_agent_hub, etc.) should import from here.

Settings are lazy-loaded via the _SETTINGS dict: importing a name triggers
os.getenv() with a canonical default.
"""

import logging
import os
import sys
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Env var name constants
# ---------------------------------------------------------------------------

# MQTT Agent Hub
ENV_MQTT_BROKER_HOST = "MQTT_BROKER_HOST"
ENV_MQTT_BROKER_PORT = "MQTT_BROKER_PORT"
ENV_MQTT_HUB_TOKEN = "MQTT_HUB_TOKEN"
ENV_MQTT_WS_PORT = "MQTT_WS_PORT"
ENV_HUB_PORT = "HUB_PORT"
ENV_AGENT_VERSION = "AGENT_VERSION"
ENV_LLM_PATH = "LLM_PATH"
ENV_TELEMETRY_INTERVAL = "TELEMETRY_INTERVAL"

# LLM / llama-cpp
ENV_LLAMA_N_CTX = "LLAMA_N_CTX"
ENV_LLAMA_N_BATCH = "LLAMA_N_BATCH"
ENV_LLAMA_N_GPU_LAYERS = "LLAMA_N_GPU_LAYERS"
ENV_LLAMA_N_THREADS = "LLAMA_N_THREADS"
ENV_LLAMA_SEED = "LLAMA_SEED"
ENV_LLAMA_VERBOSE = "LLAMA_VERBOSE"
ENV_LLAMA_TEMPERATURE = "LLAMA_TEMPERATURE"
ENV_LLAMA_TOP_K = "LLAMA_TOP_K"
ENV_LLAMA_TOP_P = "LLAMA_TOP_P"
ENV_LLAMA_REPEAT_PENALTY = "LLAMA_REPEAT_PENALTY"
ENV_LLAMA_MAX_TOKENS = "LLAMA_MAX_TOKENS"
ENV_LLAMA_REMOTE_TIMEOUT = "LLAMA_REMOTE_TIMEOUT"
ENV_LLAMA_CHAT_FORMAT = "LLAMA_CHAT_FORMAT"
ENV_LLAMA_F16_KV = "LLAMA_F16_KV"

# Lifecycle folders
ENV_DEFAULT_DOC_INGEST_ROOT = "DEFAULT_DOC_INGEST_ROOT"
ENV_STAGING_DIR = "STAGING_DIR"
ENV_PREPROCESSING_DIR = "PREPROCESSING_DIR"
ENV_INGESTION_DIR = "INGESTION_DIR"
ENV_CONSUMING_DIR = "CONSUMING_DIR"
ENV_SUCCESS_DIR = "SUCCESS_DIR"
ENV_FAILED_DIR = "FAILED_DIR"

# Model paths / endpoints
ENV_EMBEDDING_ENDPOINTS = "EMBEDDING_ENDPOINTS"
ENV_SUPERVISOR_LLM_ENDPOINTS = "SUPERVISOR_LLM_ENDPOINTS"
ENV_WHISPER_MODEL_ENDPOINTS = "WHISPER_MODEL_ENDPOINTS"

# Supervisor LLM
ENV_SUPERVISOR_REMOTE_MODEL_NAME = "SUPERVISOR_REMOTE_MODEL_NAME"
ENV_SUPERVISOR_TEMPERATURE = "SUPERVISOR_TEMPERATURE"
ENV_SUPERVISOR_TOP_K = "SUPERVISOR_TOP_K"
ENV_SUPERVISOR_MAX_TOKENS = "SUPERVISOR_MAX_TOKENS"
ENV_SUPERVISOR_N_CTX = "SUPERVISOR_N_CTX"

# Storage
ENV_GATEKEEPER_FAILURE_DB = "GATEKEEPER_FAILURE_DB"
ENV_PARQUET_FILE = "PARQUET_FILE"
ENV_DUCKDB_FILE = "DUCKDB_FILE"

# Redis
ENV_REDIS_HOST = "REDIS_HOST"
ENV_REDIS_PORT = "REDIS_PORT"
ENV_REDIS_OCR_JOB_QUEUE = "REDIS_OCR_JOB_QUEUE"
ENV_REDIS_WHISPER_JOB_QUEUE = "REDIS_WHISPER_JOB_QUEUE"
ENV_REDIS_INGEST_QUEUE = "REDIS_INGEST_QUEUE"
ENV_REDIS_STAGING_QUEUE = "REDIS_STAGING_QUEUE"
ENV_QUEUE_NAMES = "QUEUE_NAMES"

# Vector DB
ENV_VECTOR_DB_PROFILE = "VECTOR_DB_PROFILE"
ENV_VECTOR_DB_URL = "VECTOR_DB_URL"
ENV_VECTOR_DB_HOST = "VECTOR_DB_HOST"
ENV_VECTOR_DB_PORT = "VECTOR_DB_PORT"
ENV_VECTOR_DB_GRPC_PORT = "VECTOR_DB_GRPC_PORT"
ENV_VECTOR_DB_USE_GRPC = "VECTOR_DB_USE_GRPC"
ENV_VECTOR_DB_COLLECTION = "VECTOR_DB_COLLECTION"
ENV_VECTOR_DB_BATCH_SIZE = "VECTOR_DB_BATCH_SIZE"
ENV_VECTOR_DB_TIMEOUT = "VECTOR_DB_TIMEOUT"

# Legacy Chroma aliases
ENV_CHROMA_HOST = "CHROMA_HOST"
ENV_CHROMA_PORT = "CHROMA_PORT"
ENV_CHROMA_COLLECTION = "CHROMA_COLLECTION"
ENV_MAX_CHROMA_BATCH_SIZE = "MAX_CHROMA_BATCH_SIZE"

# Chunking / ingestion
ENV_CHUNK_TIMEOUT = "CHUNK_TIMEOUT"
ENV_MAX_CHUNKS = "MAX_CHUNKS"
ENV_MAX_TOKENS = "MAX_TOKENS"
ENV_CHUNK_SIZE = "CHUNK_SIZE"
ENV_CHUNK_OVERLAP = "CHUNK_OVERLAP"

# Text / encoding
ENV_ALLOW_LATIN_EXTENDED = "ALLOW_LATIN_EXTENDED"
ENV_LATIN_SCRIPT_MIN_RATIO = "LATIN_SCRIPT_MIN_RATIO"

# OCR
ENV_OCR_ENDPOINTS = "OCR_ENDPOINTS"
ENV_DEBUG_IMAGE_DIR = "DEBUG_IMAGE_DIR"
ENV_MAX_OCR_DIM = "MAX_OCR_DIM"

# Load balancing
ENV_HA_INTERLEAVE = "HA_INTERLEAVE"

# File extensions
ENV_SUPPORTED_MEDIA_EXT = "SUPPORTED_MEDIA_EXT"

# Hardware / compute
ENV_DEVICE = "DEVICE"
ENV_COMPUTE_TYPE = "COMPUTE_TYPE"

# Batch sizes
ENV_EMBEDDING_BATCH_SIZE = "EMBEDDING_BATCH_SIZE"
ENV_MEDIA_BATCH_SIZE = "MEDIA_BATCH_SIZE"

# Retrieval
ENV_RETRIEVER_TOP_K = "RETRIEVER_TOP_K"

# Gatekeeper
ENV_GATEKEEPER_BATCH_SIZE = "GATEKEEPER_BATCH_SIZE"

# PDF processing
ENV_PDF_FORCE_OCR = "PDF_FORCE_OCR"
ENV_FORCE_MARKDOWN_LLM = "FORCE_MARKDOWN_LLM"

# Stuck job recovery
ENV_STUCK_JOB_TIMEOUT_HOURS = "STUCK_JOB_TIMEOUT_HOURS"

# Metrics
ENV_METRICS_ENABLED = "METRICS_ENABLED"
ENV_METRICS_LOG_FILE = "METRICS_LOG_FILE"
ENV_METRICS_LOG_TO_STDOUT = "METRICS_LOG_TO_STDOUT"

# File tracking
ENV_FAILED_FILES = "FAILED_FILES"
ENV_INGESTED_FILE = "INGESTED_FILE"

# Chat session
ENV_MAX_SESSION_TURNS = "MAX_SESSION_TURNS"
ENV_SESSION_TTL_HOURS = "SESSION_TTL_HOURS"

# Misc
ENV_HF_HUB_OFFLINE = "HF_HUB_OFFLINE"
ENV_HF_HOME = "HF_HOME"
ENV_API_BASE_URL = "API_BASE_URL"

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

# MQTT Agent Hub
DEFAULT_MQTT_BROKER_HOST = "localhost"
DEFAULT_MQTT_BROKER_PORT = 1883
DEFAULT_MQTT_WS_PORT = 9001
DEFAULT_HUB_PORT = 8100
DEFAULT_MQTT_HUB_TOKEN = "changeme"
DEFAULT_AGENT_VERSION = "NOT_SET"
AGENT_HEARTBEAT_TIMEOUT = 60
TELEMETRY_BUFFER_SIZE = 100
DEFAULT_SRE_INTERVAL = 300

# LLM / llama-cpp
DEFAULT_LLAMA_N_CTX = 8192
DEFAULT_LLAMA_N_BATCH = 512
DEFAULT_LLAMA_N_GPU_LAYERS = -1
DEFAULT_LLAMA_N_THREADS = 0
DEFAULT_LLAMA_SEED = 42
DEFAULT_LLAMA_VERBOSE = "false"
DEFAULT_LLAMA_TEMPERATURE = 0.1
DEFAULT_LLAMA_TOP_K = 40
DEFAULT_LLAMA_TOP_P = 0.95
DEFAULT_LLAMA_REPEAT_PENALTY = 1.1
DEFAULT_LLAMA_MAX_TOKENS = 8192
DEFAULT_LLAMA_REMOTE_TIMEOUT = 300.0
DEFAULT_LLAMA_CHAT_FORMAT = "chatml"
DEFAULT_LLAMA_F16_KV = "true"

# Supervisor LLM
DEFAULT_SUPERVISOR_REMOTE_MODEL_NAME = "local-model"
DEFAULT_SUPERVISOR_TEMPERATURE = 0.1
DEFAULT_SUPERVISOR_TOP_K = 40
DEFAULT_SUPERVISOR_MAX_TOKENS = 4096
DEFAULT_SUPERVISOR_N_CTX = 8192

# Redis
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_OCR_JOB_QUEUE = "ocr_processing_job"
DEFAULT_REDIS_WHISPER_JOB_QUEUE = "whisper_processing_job"
DEFAULT_REDIS_INGEST_QUEUE = "chunk_ingest_queue"
DEFAULT_REDIS_STAGING_QUEUE = "chunk_staging_queue"
DEFAULT_QUEUE_NAMES = "chunk_ingest_queue:0,chunk_ingest_queue:1"

# Vector DB
DEFAULT_VECTOR_DB_PROFILE = "qdrant"
DEFAULT_VECTOR_DB_HOST = "vector-db"
DEFAULT_VECTOR_DB_GRPC_PORT = 6334
DEFAULT_VECTOR_DB_USE_GRPC = "true"
DEFAULT_VECTOR_DB_COLLECTION = "vector_base_collection"
DEFAULT_VECTOR_DB_BATCH_SIZE = 20
DEFAULT_VECTOR_DB_TIMEOUT = 60.0

# Chunking / ingestion
DEFAULT_CHUNK_TIMEOUT = 300
DEFAULT_MAX_CHUNKS = 5000
DEFAULT_MAX_TOKENS = 256
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50

# Text / encoding
DEFAULT_ALLOW_LATIN_EXTENDED = "true"
DEFAULT_LATIN_SCRIPT_MIN_RATIO = 0.7

# OCR
DEFAULT_OCR_ENDPOINTS = "LOCAL"
DEFAULT_MAX_OCR_DIM = 3000

# File extensions
DEFAULT_SUPPORTED_MEDIA_EXT = ".mp3,.wav,.m4a,.aac,.flac,.mp4,.mov,.mkv"

# Hardware / compute
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"

# Batch sizes
DEFAULT_EMBEDDING_BATCH_SIZE = 5
DEFAULT_MEDIA_BATCH_SIZE = 8

# Stuck job recovery
DEFAULT_STUCK_JOB_TIMEOUT_HOURS = 1

# Retrieval
DEFAULT_RETRIEVER_TOP_K = 4

# Gatekeeper
DEFAULT_GATEKEEPER_BATCH_SIZE = 5

# PDF processing
DEFAULT_PDF_FORCE_OCR = "false"
DEFAULT_FORCE_MARKDOWN_LLM = "false"

# Metrics
DEFAULT_METRICS_ENABLED = "false"
DEFAULT_METRICS_LOG_TO_STDOUT = "false"

# Chat session
DEFAULT_MAX_SESSION_TURNS = 20
DEFAULT_SESSION_TTL_HOURS = 24

# Misc
DEFAULT_HF_HUB_OFFLINE = "0"
DEFAULT_API_BASE_URL = "http://localhost:8000"

# Load balancing
DEFAULT_HA_INTERLEAVE = "false"

# Hard-coded lists (not from env)
SUPPORTED_DOC_EXT = (".pdf", ".html", ".htm", ".txt", ".md")
WHISPER_REQUIRED_FILES_LIST = ["model.bin", "config.json", "vocabulary.txt"]

log = logging.getLogger("shared.config")

# ---------------------------------------------------------------------------
# Helper: Environment Getters
# ---------------------------------------------------------------------------


def _abs_path(key: str, default: str = None) -> str:
    val = os.getenv(key)
    if not val:
        return default
    if val.startswith(("http://", "https://")):
        return val
    return os.path.abspath(val)


def _require_abs_path(key: str, default: str = None) -> str:
    val = os.getenv(key)
    if not val:
        if default:
            val = default
        else:
            log.error(
                f"\u274c CRITICAL ERROR: Environment variable '{key}' is NOT set. "
                "This is required for the system to function."
            )
            sys.exit(1)
    if val.startswith(("http://", "https://")):
        return val
    return os.path.abspath(val)


def _get_vector_db_port() -> int:
    port = os.getenv(ENV_VECTOR_DB_PORT) or os.getenv(ENV_CHROMA_PORT)
    if port:
        return int(port)
    profile = os.getenv(ENV_VECTOR_DB_PROFILE, DEFAULT_VECTOR_DB_PROFILE).lower()
    return 6333 if profile == "qdrant" else 8000


# ---------------------------------------------------------------------------
# Settings Dictionary (Lazy Loaded)
# ---------------------------------------------------------------------------

_SETTINGS: dict[str, Callable[[], Any]] = {
    "LLAMA_N_CTX": lambda: int(os.getenv(ENV_LLAMA_N_CTX, str(DEFAULT_LLAMA_N_CTX))),
    "LLAMA_N_BATCH": lambda: int(os.getenv(ENV_LLAMA_N_BATCH, str(DEFAULT_LLAMA_N_BATCH))),
    "LLAMA_N_GPU_LAYERS": lambda: int(os.getenv(ENV_LLAMA_N_GPU_LAYERS, str(DEFAULT_LLAMA_N_GPU_LAYERS))),
    "LLAMA_N_THREADS": lambda: int(os.getenv(ENV_LLAMA_N_THREADS, str(DEFAULT_LLAMA_N_THREADS))),
    "LLAMA_SEED": lambda: int(os.getenv(ENV_LLAMA_SEED, str(DEFAULT_LLAMA_SEED))),
    "LLAMA_VERBOSE": lambda: os.getenv(ENV_LLAMA_VERBOSE, DEFAULT_LLAMA_VERBOSE).lower() == "true",
    "LLAMA_TEMPERATURE": lambda: float(os.getenv(ENV_LLAMA_TEMPERATURE, str(DEFAULT_LLAMA_TEMPERATURE))),
    "LLAMA_TOP_K": lambda: int(os.getenv(ENV_LLAMA_TOP_K, str(DEFAULT_LLAMA_TOP_K))),
    "LLAMA_TOP_P": lambda: float(os.getenv(ENV_LLAMA_TOP_P, str(DEFAULT_LLAMA_TOP_P))),
    "LLAMA_REPEAT_PENALTY": lambda: float(os.getenv(ENV_LLAMA_REPEAT_PENALTY, str(DEFAULT_LLAMA_REPEAT_PENALTY))),
    "LLAMA_MAX_TOKENS": lambda: int(os.getenv(ENV_LLAMA_MAX_TOKENS, str(DEFAULT_LLAMA_MAX_TOKENS))),
    "LLAMA_REMOTE_TIMEOUT": lambda: float(os.getenv(ENV_LLAMA_REMOTE_TIMEOUT, str(DEFAULT_LLAMA_REMOTE_TIMEOUT))),
    "LLAMA_CHAT_FORMAT": lambda: os.getenv(ENV_LLAMA_CHAT_FORMAT, DEFAULT_LLAMA_CHAT_FORMAT),
    "LLAMA_F16_KV": lambda: os.getenv(ENV_LLAMA_F16_KV, DEFAULT_LLAMA_F16_KV).lower() == "true",
    "DEFAULT_DOC_INGEST_ROOT": lambda: _require_abs_path(ENV_DEFAULT_DOC_INGEST_ROOT, os.path.abspath("./Docs")),
    "STAGING_DIR": lambda: _abs_path(ENV_STAGING_DIR, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "staging")),
    "PREPROCESSING_DIR": lambda: _abs_path(ENV_PREPROCESSING_DIR, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "preprocessing")),
    "INGESTION_DIR": lambda: _abs_path(ENV_INGESTION_DIR, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "ingestion")),
    "CONSUMING_DIR": lambda: _abs_path(ENV_CONSUMING_DIR, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "consuming")),
    "SUCCESS_DIR": lambda: _abs_path(ENV_SUCCESS_DIR, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "success")),
    "FAILED_DIR": lambda: _abs_path(ENV_FAILED_DIR, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "failed")),
    "INGEST_FOLDER": lambda: _SETTINGS["INGESTION_DIR"](),
    "STAGING_FOLDER": lambda: _SETTINGS["STAGING_DIR"](),
    "GATEKEEPER_FAILURE_DB": lambda: _abs_path(ENV_GATEKEEPER_FAILURE_DB, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "gatekeeper_history.db")),
    "EMBEDDING_ENDPOINTS": lambda: _require_abs_path(ENV_EMBEDDING_ENDPOINTS),
    "LLM_PATH": lambda: _require_abs_path(ENV_LLM_PATH),
    "SUPERVISOR_LLM_ENDPOINTS": lambda: _require_abs_path(ENV_SUPERVISOR_LLM_ENDPOINTS),
    "WHISPER_MODEL_ENDPOINTS": lambda: _abs_path(ENV_WHISPER_MODEL_ENDPOINTS, "NOT_SET"),
    "WHISPER_REQUIRED_FILES": lambda: WHISPER_REQUIRED_FILES_LIST,
    "SUPERVISOR_REMOTE_MODEL_NAME": lambda: os.getenv(ENV_SUPERVISOR_REMOTE_MODEL_NAME, DEFAULT_SUPERVISOR_REMOTE_MODEL_NAME),
    "SUPERVISOR_TEMPERATURE": lambda: float(os.getenv(ENV_SUPERVISOR_TEMPERATURE, str(DEFAULT_SUPERVISOR_TEMPERATURE))),
    "SUPERVISOR_TOP_K": lambda: int(os.getenv(ENV_SUPERVISOR_TOP_K, str(DEFAULT_SUPERVISOR_TOP_K))),
    "SUPERVISOR_MAX_TOKENS": lambda: int(os.getenv(ENV_SUPERVISOR_MAX_TOKENS, str(DEFAULT_SUPERVISOR_MAX_TOKENS))),
    "SUPERVISOR_N_CTX": lambda: int(os.getenv(ENV_SUPERVISOR_N_CTX, str(DEFAULT_SUPERVISOR_N_CTX))),
    "PARQUET_FILE": lambda: _abs_path(ENV_PARQUET_FILE, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "chunks.parquet")),
    "DUCKDB_FILE": lambda: _abs_path(ENV_DUCKDB_FILE, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "chunks.duckdb")),
    "REDIS_HOST": lambda: os.environ.get(ENV_REDIS_HOST) or os.getenv(ENV_REDIS_HOST, DEFAULT_REDIS_HOST),
    "REDIS_PORT": lambda: int(os.environ.get(ENV_REDIS_PORT) or os.getenv(ENV_REDIS_PORT, str(DEFAULT_REDIS_PORT))),
    "REDIS_OCR_JOB_QUEUE": lambda: os.getenv(ENV_REDIS_OCR_JOB_QUEUE, DEFAULT_REDIS_OCR_JOB_QUEUE),
    "REDIS_WHISPER_JOB_QUEUE": lambda: os.getenv(ENV_REDIS_WHISPER_JOB_QUEUE, DEFAULT_REDIS_WHISPER_JOB_QUEUE),
    "REDIS_INGEST_QUEUE": lambda: os.getenv(ENV_REDIS_INGEST_QUEUE, DEFAULT_REDIS_INGEST_QUEUE),
    "REDIS_STAGING_QUEUE": lambda: os.getenv(ENV_REDIS_STAGING_QUEUE, DEFAULT_REDIS_STAGING_QUEUE),
    "QUEUE_NAMES": lambda: os.getenv(ENV_QUEUE_NAMES, "chunk_ingest_queue:0,chunk_ingest_queue:1").split(","),
    "VECTOR_DB_PROFILE": lambda: os.getenv(ENV_VECTOR_DB_PROFILE, DEFAULT_VECTOR_DB_PROFILE).lower(),
    "USE_QDRANT": lambda: os.getenv(ENV_VECTOR_DB_PROFILE, DEFAULT_VECTOR_DB_PROFILE).lower() == "qdrant",
    "VECTOR_DB_URL": lambda: os.getenv(ENV_VECTOR_DB_URL),
    "VECTOR_DB_HOST": lambda: os.getenv(ENV_VECTOR_DB_HOST, os.getenv(ENV_CHROMA_HOST, DEFAULT_VECTOR_DB_HOST)),
    "VECTOR_DB_PORT": _get_vector_db_port,
    "VECTOR_DB_GRPC_PORT": lambda: int(os.getenv(ENV_VECTOR_DB_GRPC_PORT, str(DEFAULT_VECTOR_DB_GRPC_PORT))),
    "VECTOR_DB_USE_GRPC": lambda: os.getenv(ENV_VECTOR_DB_USE_GRPC, DEFAULT_VECTOR_DB_USE_GRPC).lower() == "true",
    "VECTOR_DB_COLLECTION": lambda: os.getenv(ENV_VECTOR_DB_COLLECTION, DEFAULT_VECTOR_DB_COLLECTION),
    "VECTOR_DB_BATCH_SIZE": lambda: int(os.getenv(ENV_VECTOR_DB_BATCH_SIZE, os.getenv(ENV_MAX_CHROMA_BATCH_SIZE, str(DEFAULT_VECTOR_DB_BATCH_SIZE)))),
    "VECTOR_DB_TIMEOUT": lambda: float(os.getenv(ENV_VECTOR_DB_TIMEOUT, str(DEFAULT_VECTOR_DB_TIMEOUT))),
    "MAX_CHROMA_BATCH_SIZE": lambda: int(os.getenv(ENV_VECTOR_DB_BATCH_SIZE, os.getenv(ENV_MAX_CHROMA_BATCH_SIZE, str(DEFAULT_VECTOR_DB_BATCH_SIZE)))),
    "CHUNK_TIMEOUT": lambda: int(os.getenv(ENV_CHUNK_TIMEOUT, str(DEFAULT_CHUNK_TIMEOUT))),
    "MAX_CHUNKS": lambda: int(os.getenv(ENV_MAX_CHUNKS, str(DEFAULT_MAX_CHUNKS))),
    "CHROMA_HOST": lambda: os.getenv(ENV_VECTOR_DB_HOST, os.getenv(ENV_CHROMA_HOST, DEFAULT_VECTOR_DB_HOST)),
    "CHROMA_PORT": _get_vector_db_port,
    "CHROMA_COLLECTION": lambda: os.getenv(ENV_VECTOR_DB_COLLECTION, os.getenv(ENV_CHROMA_COLLECTION, DEFAULT_VECTOR_DB_COLLECTION)),
    "MAX_TOKENS": lambda: int(os.getenv(ENV_MAX_TOKENS, str(DEFAULT_MAX_TOKENS))),
    "CHUNK_SIZE": lambda: int(os.getenv(ENV_CHUNK_SIZE, str(DEFAULT_CHUNK_SIZE))),
    "CHUNK_OVERLAP": lambda: int(os.getenv(ENV_CHUNK_OVERLAP, str(DEFAULT_CHUNK_OVERLAP))),
    "ALLOW_LATIN_EXTENDED": lambda: os.getenv(ENV_ALLOW_LATIN_EXTENDED, DEFAULT_ALLOW_LATIN_EXTENDED).lower() == "true",
    "LATIN_SCRIPT_MIN_RATIO": lambda: float(os.getenv(ENV_LATIN_SCRIPT_MIN_RATIO, str(DEFAULT_LATIN_SCRIPT_MIN_RATIO))),
    "OCR_ENDPOINTS": lambda: os.getenv(ENV_OCR_ENDPOINTS, DEFAULT_OCR_ENDPOINTS),
    "DEBUG_IMAGE_DIR": lambda: _abs_path(ENV_DEBUG_IMAGE_DIR, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "ocr_debug")),
    "MAX_OCR_DIM": lambda: int(os.getenv(ENV_MAX_OCR_DIM, str(DEFAULT_MAX_OCR_DIM))),
    "SUPPORTED_DOC_EXT": lambda: SUPPORTED_DOC_EXT,
    "SUPPORTED_MEDIA_EXT": lambda: tuple(os.getenv(ENV_SUPPORTED_MEDIA_EXT, DEFAULT_SUPPORTED_MEDIA_EXT).split(",")),
    "ALL_SUPPORTED_EXT": lambda: SUPPORTED_DOC_EXT + tuple(os.getenv(ENV_SUPPORTED_MEDIA_EXT, DEFAULT_SUPPORTED_MEDIA_EXT).split(",")),
    "DEVICE": lambda: os.getenv(ENV_DEVICE, DEFAULT_DEVICE),
    "EMBEDDING_BATCH_SIZE": lambda: int(os.getenv(ENV_EMBEDDING_BATCH_SIZE, str(DEFAULT_EMBEDDING_BATCH_SIZE))),
    "MEDIA_BATCH_SIZE": lambda: int(os.getenv(ENV_MEDIA_BATCH_SIZE, str(DEFAULT_MEDIA_BATCH_SIZE))),
    "COMPUTE_TYPE": lambda: os.getenv(ENV_COMPUTE_TYPE, DEFAULT_COMPUTE_TYPE),
    "RETRIEVER_TOP_K": lambda: int(os.getenv(ENV_RETRIEVER_TOP_K, str(DEFAULT_RETRIEVER_TOP_K))),
    "GATEKEEPER_BATCH_SIZE": lambda: int(os.getenv(ENV_GATEKEEPER_BATCH_SIZE, str(DEFAULT_GATEKEEPER_BATCH_SIZE))),
    "PDF_FORCE_OCR": lambda: os.getenv(ENV_PDF_FORCE_OCR, DEFAULT_PDF_FORCE_OCR).lower() in ("1", "true"),
    "FORCE_MARKDOWN_LLM": lambda: os.getenv(ENV_FORCE_MARKDOWN_LLM, DEFAULT_FORCE_MARKDOWN_LLM).lower() in ("1", "true"),
    "HA_INTERLEAVE": lambda: os.getenv(ENV_HA_INTERLEAVE, DEFAULT_HA_INTERLEAVE).lower() in ("1", "true"),
    "METRICS_ENABLED": lambda: os.getenv(ENV_METRICS_ENABLED, DEFAULT_METRICS_ENABLED).lower() == "true",
    "METRICS_LOG_FILE": lambda: os.getenv(ENV_METRICS_LOG_FILE),
    "METRICS_LOG_TO_STDOUT": lambda: os.getenv(ENV_METRICS_LOG_TO_STDOUT, DEFAULT_METRICS_LOG_TO_STDOUT).lower() == "true",
    "FAILED_FILES": lambda: _abs_path(ENV_FAILED_FILES, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "failed_files.txt")),
    "INGESTED_FILE": lambda: _abs_path(ENV_INGESTED_FILE, os.path.join(_SETTINGS["DEFAULT_DOC_INGEST_ROOT"](), "ingested_files.txt")),
    "MAX_SESSION_TURNS": lambda: int(os.getenv(ENV_MAX_SESSION_TURNS, str(DEFAULT_MAX_SESSION_TURNS))),
    "SESSION_TTL_HOURS": lambda: int(os.getenv(ENV_SESSION_TTL_HOURS, str(DEFAULT_SESSION_TTL_HOURS))),
    "STUCK_JOB_TIMEOUT_HOURS": lambda: int(os.getenv(ENV_STUCK_JOB_TIMEOUT_HOURS, str(DEFAULT_STUCK_JOB_TIMEOUT_HOURS))),
    "API_BASE_URL": lambda: os.getenv(ENV_API_BASE_URL, DEFAULT_API_BASE_URL),
    "HF_HOME": lambda: os.getenv(ENV_HF_HOME, "/usr/local/model_cache"),
    "HF_HUB_OFFLINE": lambda: os.getenv(ENV_HF_HUB_OFFLINE, DEFAULT_HF_HUB_OFFLINE),
}


def get_setting(name: str) -> Any:
    """Resolve a setting by name. Raises KeyError if unknown."""
    if name not in _SETTINGS:
        raise KeyError(f"Unknown setting: '{name}'")
    return _SETTINGS[name]()
