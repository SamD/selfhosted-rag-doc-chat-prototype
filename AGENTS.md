# AGENTS.md - Self-Hosted RAG Pipeline

## Quick Start Commands

- **Backend API**: `uvicorn apimain:app --reload` (FastAPI backend)
- **Frontend**: `npm run dev` (Astro frontend, Node.js v22.12.0+)
- **Docker Compose**: `./run-compose.sh` (GPU/CPU profile, vector DB profile)
- **Chat System**: `./run-chat-system.sh` (backend/frontend startup)
- **Testing**: `pytest` (async support, conftest.py sets temp env vars)

## Architecture Overview

Self-Hosted RAG Pipeline with distributed ingestion workers, Redis queuing, dual vector storage (Qdrant/ChromaDB), DuckDB analytics, and Astro frontend. Multi-process architecture with producer/consumer/OCR workers.

## Required Local Models

Models MUST exist locally in these exact paths:
1. **E5 embedding**: `e5-large-v2` (directory with config.json or tokenizer_config.json)
2. **LLM**: `Phi-3.5-mini-instruct-Q6_K.gguf` (.gguf file)
3. **Supervisor LLM**: `Qwen2.5-1.5B-Instruct-GGUF` (.gguf file)

## Key Directories

- `doc-ingest-chat/` - Core ingestion pipeline (producer/consumer/ocr workers)
- `astro-frontend/` - Astro frontend (Node.js v22.12.0+)
- `docs/` - Documentation
- `project-docs/` - Architecture and production docs
- `tests/` - Test suite (pytest with async support)

## Environment Variables

Required env vars (defaults shown):
- `INGEST_FOLDER` - Source documents directory (default: `Docs`)
- `EMBEDDING_MODEL_PATH` - E5 model path (default: `/home/samueldoyle/AI_LOCAL/e5-large-v2`)
- `LLM_PATH` - Phi-3 LLM path (default: `/home/samueldoyle/AI_LOCAL/Models/Phi/microsoft_Phi-4-mini-instruct-Q6_K.gguf`)
- `SUPERVISOR_LLM_PATH` - Qwen2.5 LLM path (default: `/home/samueldoyle/AI_LOCAL/Models/Phi/microsoft_Phi-4-mini-instruct-Q6_K.gguf`)
- `VECTOR_DB_PROFILE` - `qdrant` or `chroma` (default: `qdrant`)
- `LLAMA_USE_GPU` - `true`/`false` (default: `true`)

## Docker Compose Profiles

`./run-compose.sh` supports profiles:
- `--profile gpu` - NVIDIA GPU acceleration
- `--profile cpu` - CPU-only mode
- `--profile qdrant` - Qdrant vector DB
- `--profile chroma` - Chroma vector DB

## Worker Scripts

All workers use `get_env_strategy().apply()` early:
- `run_producer.py` - Producer worker (ingests files)
- `run_consumer.py` - Consumer worker (processes chunks)
- `run_ocr_worker.py` - OCR worker (extracts text from images)

## FastAPI Backend

Entry point: `apimain.py` (use `uvicorn apimain:app --reload`)
- CORS middleware configured with `allow_origins=["*"]` (configure appropriately for production)
- Router: `api.endpoints.router`
- Root endpoint: `/` returns API status message

## LLM Interaction Patterns

Strict citation requirements in `prompts/chat_prompts.py`:
- **MUST** include exact citation tags from context (e.g., `[doc5]`, `[ref12]`, `[source:7]`)
- **MUST** include separate citation tag for every distinct factual sentence
- **MUST** trace each sentence to a specific passage
- **MUST NOT** use placeholders like `[source1]` — only actual tags found in context
- **MUST NOT** fabricate, infer, or summarize across sources
- **MUST NOT** use phrases like "however", "it is important to note", "some believe" unless verbatim in context
- **MUST NOT** mention "AI", "language model", "LLM", or own capabilities
- **MUST NOT** provide editorial commentary, opinions, recommendations, or tone modifications
- If no relevant info: respond with exactly: `"Not enough information in the provided sources."`
- Each factual sentence must be followed immediately by its source tag inline

## Environment Strategy

`config/env_strategy.py`:
- `GPUEnvConfig` - Sets `CUDA_VISIBLE_DEVICES="0"` and `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`
- `CPUEnvConfig` - Sets `CUDA_VISIBLE_DEVICES=""` and pops `PYTORCH_CUDA_ALLOC_CONF`
- `get_env_strategy()` checks `LLAMA_USE_GPU` env var (default: `true`)

## Settings

`config/settings.py` lazy-loaded settings:
- `STAGING_FOLDER` - Default: `Docs/staging`
- `VECTOR_DB_HOST` - Default: `vector-db` (or `CHROMA_HOST`)
- `VECTOR_DB_PORT` - Default: `6333` (Qdrant) or `8000` (Chroma)
- `VECTOR_DB_PROFILE` - Default: `qdrant`
- `USE_QDRANT` - Boolean based on profile
- `MAX_CHUNKS` - Default: `5000`
- `CHUNK_SIZE` - Default: `512`
- `CHUNK_OVERLAP` - Default: `50`
- `DEVICE` - Default: `cuda`
- `COMPUTE_TYPE` - Default: `float16`

## Testing

- `pytest` with async support
- `conftest.py` sets up minimal environment variables for test isolation
- `test_document_processor.py` exists but is empty

## CI/CD

- `.github/workflows/` - GitHub Actions workflows (Gemini triage, plan-execute, invoke, review, dispatch)
- `.github/commands/` - Custom GitHub CLI commands (gemini-triage, gemini-plan-execute, gemini-review, gemini-invoke, gemini-scheduled-triage)

## Development Quirks

- **Module imports**: Workers use `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` to ensure package imports work when running files directly
- **Environment strategy**: Must call `get_env_strategy().apply()` before any model loading or GPU operations
- **Model paths**: All model paths are absolute paths; relative paths are rejected unless default provided
- **Vector DB**: Dual support for Qdrant (default) and Chroma; profile selection via `VECTOR_DB_PROFILE` env var
- **Redis queues**: `REDIS_OCR_JOB_QUEUE` and `REDIS_INGEST_QUEUE` define queue names
- **CORS**: FastAPI middleware allows all origins (`["*"]`) — configure appropriately for production
- **No existing AI instruction files**: No `.cursorrules` or `.claude` files found

## File Extensions

Supported document types: `.pdf`, `.html`, `.htm`, `.txt`, `.md`
Supported media types: `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.mp4`, `.mov`, `.mkv`

## Critical Error Messages

- `❌ CRITICAL ERROR: Environment variable '{key}' is NOT set. This is required for the system to function.`