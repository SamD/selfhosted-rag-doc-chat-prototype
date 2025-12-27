#!/usr/bin/env python3
"""
Configuration settings for the document ingestion system.

- When running standalone, default values are used for all environment variables.
- When running in Docker Compose, values from ingest-svc.env will override the defaults.
"""
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _abs_path(path, base=PROJECT_ROOT):
    if not path:
        raise ValueError("Missing required path for ingestion.")
    if os.path.isabs(path):
        return path
    if not base:
        raise ValueError("Base directory must be provided for relative paths.")
    return os.path.join(base, path)

LLAMA_USE_GPU = os.getenv("LLAMA_USE_GPU", "true").lower() == "true"

# Text acceptance configuration
# When true, treat Latin extended characters (diacritics, ligatures like œ) as valid text
# in OCR quality checks. Can be overridden via environment variable.
ALLOW_LATIN_EXTENDED = os.getenv("ALLOW_LATIN_EXTENDED", "true").lower() == "true"
# Minimum fraction of letters that must be Latin to treat text as Latin script content
LATIN_SCRIPT_MIN_RATIO = float(os.getenv("LATIN_SCRIPT_MIN_RATIO", "0.7"))

# File Processing Configuration
INGEST_FOLDER = _abs_path(os.getenv("INGEST_FOLDER"))
CHROMA_DATA_DIR = _abs_path(os.getenv("CHROMA_DATA_DIR"))

# Model Paths
E5_MODEL_PATH = _abs_path(os.getenv("E5_MODEL_PATH"))
LLAMA_MODEL_PATH = _abs_path(os.getenv("LLAMA_MODEL_PATH"))

# File Paths
FAILED_FILES = _abs_path(os.getenv("FAILED_FILES", "failed_files.txt"))
INGESTED_FILE = _abs_path(os.getenv("INGESTED_FILE", "ingested_files.txt"))
TRACK_FILE = _abs_path(os.getenv("TRACK_FILE", "ingested_files.txt"))
PARQUET_FILE = _abs_path(os.getenv("PARQUET_FILE", "chunks.parquet"))
DUCKDB_FILE = _abs_path(os.getenv("DUCKDB_FILE", "chunks.duckdb"))


# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6380"))
REDIS_OCR_JOB_QUEUE = os.getenv("REDIS_OCR_JOB_QUEUE", "ocr_processing_job")
REDIS_INGEST_QUEUE = os.getenv("REDIS_INGEST_QUEUE", "chunk_ingest_queue")

# Queue Configuration
QUEUE_NAMES = os.getenv("QUEUE_NAMES", "chunk_ingest_queue:0,chunk_ingest_queue:1").split(",")

# ChromaDB Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "9001"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "chroma_collection")


CHUNK_TIMEOUT = int(os.getenv("CHUNK_TIMEOUT", "300"))  # seconds before we consider a buffer stale
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "20000"))
MAX_CHROMA_BATCH_SIZE = int(os.getenv("MAX_CHROMA_BATCH_SIZE", "75"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

# OCR Configuration
DEBUG_IMAGE_DIR = os.getenv("DEBUG_IMAGE_DIR", "/tmp/ocr_debug")
MAX_OCR_DIM = int(os.getenv("MAX_OCR_DIM", "3000"))
# Image.MAX_IMAGE_PIXELS = 500_000_000  # Set in utils when PIL is imported

# Media Processing Configuration
SUPPORTED_MEDIA_EXT = tuple(os.getenv("SUPPORTED_MEDIA_EXT", ".mp3,.wav,.m4a,.aac,.flac,.mp4,.mov,.mkv").split(","))
ALL_SUPPORTED_EXT = (".pdf", ".html", ".htm") + SUPPORTED_MEDIA_EXT

# Whisper Configuration
DEVICE = os.getenv("DEVICE", "cuda")
MEDIA_BATCH_SIZE = int(os.getenv("MEDIA_BATCH_SIZE", "8"))  # reduce if low on GPU mem
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")  # change to "int8" if low on GPU mem (may reduce accuracy)



# Queue Management
MAX_QUEUE_LENGTH = int(os.getenv("MAX_QUEUE_LENGTH", "25"))
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "0.5"))
WAIT_WARN_THRESHOLD = float(os.getenv("WAIT_WARN_THRESHOLD", "10"))
MAX_CHROMA_BATCH_SIZE_LIMIT = int(os.getenv("MAX_CHROMA_BATCH_SIZE_LIMIT", "5461"))

# PAD_RESERVE = 32  # Reserve space for padding/special tokens

# Ensure debug directory exists
os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True) 

# LLM and Chat Configuration
USE_OLLAMA = os.getenv("USE_OLLAMA", "0") == "1"
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "openchat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "NeuralNet/openchat-3.6")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


# Retrieve the top k most relevant chunks (based on vector similarity) from the database for a given query.
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "20"))

LLAMA_N_CTX = int(os.getenv("LLAMA_N_CTX", "32768"))
LLAMA_N_GPU_LAYERS = int(os.getenv("LLAMA_N_GPU_LAYERS", "35"))
LLAMA_N_THREADS = int(os.getenv("LLAMA_N_THREADS", "24"))
LLAMA_N_BATCH = int(os.getenv("LLAMA_N_BATCH", "512"))
LLAMA_F16_KV = os.getenv("LLAMA_F16_KV", "True").lower() == "true"
LLAMA_TEMPERATURE = float(os.getenv("LLAMA_TEMPERATURE", "0.3"))
# When generating each token, restrict sampling to the top k most likely next tokens (based on probability distribution).
LLAMA_TOP_K = int(os.getenv("LLAMA_TOP_K", "25"))
LLAMA_TOP_P = float(os.getenv("LLAMA_TOP_P", "0.85"))
LLAMA_REPEAT_PENALTY = float(os.getenv("LLAMA_REPEAT_PENALTY", "1.2"))
# LLAMA_MAX_TOKENS = int(os.getenv("LLAMA_MAX_TOKENS", "4096"))
LLAMA_MAX_TOKENS = int(os.getenv("LLAMA_MAX_TOKENS", "512"))
LLAMA_CHAT_FORMAT = os.getenv("LLAMA_CHAT_FORMAT", "chatml")
LLAMA_VERBOSE = os.getenv("LLAMA_VERBOSE", "False").lower() == "true"
LLAMA_SEED = int(os.getenv("LLAMA_SEED", "42"))

# Tesseract OCR configuration
TESSERACT_LANGS = os.getenv("TESSERACT_LANGS", "eng+lat")
TESSERACT_USE_SCRIPT_LATIN = os.getenv("TESSERACT_USE_SCRIPT_LATIN", "true").lower() == "true"
TESSERACT_PSM = int(os.getenv("TESSERACT_PSM", "6"))
TESSERACT_OEM = int(os.getenv("TESSERACT_OEM", "1"))
TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX", "")

# Metrics Configuration
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"
METRICS_LOG_FILE = os.getenv("METRICS_LOG_FILE", "metrics.jsonl")
METRICS_LOG_TO_STDOUT = os.getenv("METRICS_LOG_TO_STDOUT", "true").lower() == "true"

