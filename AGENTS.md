# AGENTS.md - Self-Hosted RAG Pipeline

## Quick Start Commands

- **Backend API**: `uvicorn apimain:app --reload` (FastAPI backend)
- **Frontend**: `npm run dev` (Astro frontend, Node.js v22.12.0+)
- **Frontend Tests**: `npm test` + `bash test.sh` (both in `astro-frontend/` and `mqtt_agent_hub/astro-dashboard/`)
- **Docker Compose**: `./run-compose.sh` (GPU/CPU profile, vector DB profile)
- **Chat System**: `./run-chat-system.sh` (backend/frontend startup)
- **Testing**: `pytest` (async support, conftest.py sets temp env vars)
- **Linting**: `ruff check` (code quality)

## Architecture Overview

Self-Hosted RAG Pipeline with distributed ingestion workers, Redis queuing, dual vector storage (Qdrant/ChromaDB), DuckDB analytics, and Astro frontend. Multi-process architecture with producer/consumer/OCR workers.

## Content Handler Architecture (Chain of Responsibility)

The Gatekeeper worker delegates raw text extraction to specialized handlers based on file type:
- `PDFContentTypeHandler`: Handles `.pdf` files with OCR fallback. `MIME_TYPE = "application/pdf"`.
- `MP4ContentTypeHandler`: Handles `.mp4` files using WhisperX transcription. `MIME_TYPE = "video/mp4"`.
- `MP3ContentTypeHandler`: Handles `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac` files. Uses `get_mime_type()` to guess from extension.
- `TextContentTypeHandler`: Handles `.txt`, `.md`, `.html` files. Uses `get_mime_type()` to guess from extension.

All handlers inherit from `BaseContentTypeHandler` which provides:
- `MIME_TYPE` class var: Override in subclass for fixed MIME type, or leave `None` to guess from extension.
- `get_mime_type(file_path)`: Returns MIME type from class var or `mimetypes.guess_type()`.
- MIME type flows from handler through `send_media_to_whisperx()` to the whisperx worker.

## HAProxy Load Balancing

When multiple backend endpoints are configured via `*_ENDPOINTS` env vars, HAProxy containers start automatically and distribute requests across backends.

### Supported Services

| Service | Env Var (endpoints) | HAProxy Port | Stats Port | Auto-Override |
|---------|-------------------|-------------|------------|---------------|
| Supervisor LLM | `SUPERVISOR_LLM_ENDPOINTS` | 11437 | 8404 | `SUPERVISOR_LLM_PATH` -> `http://haproxy_supervisor:11437/v1` |
| Embeddings | `EMBEDDING_ENDPOINTS` | 11438 | 8405 | `EMBEDDING_MODEL_PATH` -> `http://haproxy_embd:11438/v1` |
| WhisperX | `WHISPER_ENDPOINTS` | 11439 | 8406 | `WHISPER_MODEL_PATH` -> `http://haproxy_whisper:11439/inference` |
| OCR | `OCR_ENDPOINTS` | 11440 | 8407 | `OCR_PATH` -> `http://haproxy_ocr:11440/v1/convert/file` |

### Behavior by Endpoint Count

- **0 (unset)**: HAProxy returns 503. `*_PATH` env var used directly (no haproxy dependency).
- **1**: Transparent proxy to that single backend.
- **2+**: `roundrobin` balancing with `httpclose` (no keep-alive pinning). Health checks on `/models` or `/health`.

### How It Works

1. `run-compose.sh` detects `*_ENDPOINTS` env vars and auto-sets the corresponding `*_PATH` to the haproxy container URL.
2. `infra/haproxy-entrypoint.sh` generates HAProxy config at container startup from `*_ENDPOINTS` env var.
3. Workers connect to haproxy as if it were a single backend. HAProxy distributes requests across all healthy backends.
4. Stats UI available at `http://localhost:<stats-port>/stats` for each service.

### Configuration

Endpoints are comma-separated URLs. Each URL can include the full path. HAProxy extracts `host:port` for routing.

```bash
export SUPERVISOR_LLM_ENDPOINTS=http://gpu0:11435/v1/chat/completions,http://gpu1:11436/v1/chat/completions
export EMBEDDING_ENDPOINTS=http://gpu0:11434/v1/embeddings,http://gpu1:11434/v1/embeddings
export WHISPER_ENDPOINTS=http://whisper0:1145/inference,http://whisper1:1145/inference
export OCR_ENDPOINTS=http://ocr0:5001/v1/convert/file,http://ocr1:5001/v1/convert/file
```

## Shared Utilities (`shared/utils.py`)

- `parse_endpoints(env_value)`: Parse comma-separated endpoint URLs from env vars.
- `_normalize_openai_url(url)`: Ensure URL ends with `/v1`.
- `resolve_supervisor_endpoint()`: Single-endpoint fallback from `SUPERVISOR_LLM_ENDPOINTS` or `SUPERVISOR_LLM_PATH`.
- `resolve_embedding_endpoint()`: Single-endpoint fallback from `EMBEDDING_ENDPOINTS` or `EMBEDDING_MODEL_PATH`.

## Required Local Models (or Remote Endpoints)

Models can exist locally in these paths OR be provided via HTTP(S) URLs:
1. **E5 embedding**: `e5-large-v2` (directory with config.json or remote URL)
2. **LLM**: `Phi-3.5-mini-instruct-Q6_K.gguf` (.gguf file or remote URL)
3. **Supervisor LLM**: `Qwen2.5-1.5B-Instruct-GGUF` (.gguf file or remote URL)
4. **Whisper**: Local models or remote URL
5. **OCR**: `docling-serve` remote URL or `LOCAL`

## Key Directories

- `doc-ingest-chat/` - Core ingestion pipeline (producer/consumer/ocr workers)
- `astro-frontend/` - Astro frontend (Node.js v22.12.0+)
- `docs/` - Documentation
- `project-docs/` - Architecture and production docs
- `tests/` - Test suite (pytest with async support)
- `infra/` - HAProxy entrypoint and config rendering scripts
- `shared/` - Shared utilities and config package

## Environment Variables

### Core Service Paths

- `DEFAULT_DOC_INGEST_ROOT` - Root directory for all lifecycle stages (required)
- `EMBEDDING_MODEL_PATH` - E5 model path or remote URL (required)
- `LLM_PATH` - Main chat LLM path or remote URL (required)
- `SUPERVISOR_LLM_PATH` - Normalization LLM path or remote URL (required)
- `WHISPER_MODEL_PATH` - Whisper model path or remote URL (default: `NOT_SET`)
- `OCR_PATH` - Remote docling-serve URL or `LOCAL` (default: `LOCAL`)

### Multi-Endpoint Load Balancing

- `SUPERVISOR_LLM_ENDPOINTS` - Comma-separated supervisor LLM backend URLs
- `EMBEDDING_ENDPOINTS` - Comma-separated embedding backend URLs
- `WHISPER_ENDPOINTS` - Comma-separated whisper backend URLs
- `OCR_ENDPOINTS` - Comma-separated OCR backend URLs

When set, `run-compose.sh` auto-overrides the corresponding `*_PATH` to point to the HAProxy container.

### Vector Database

- `VECTOR_DB_PROFILE` - `qdrant` or `chroma` (default: `qdrant`)
- `VECTOR_DB_URL` - Full URL for remote vector DB (e.g., `http://192.168.30.71:6333`)
- `VECTOR_DB_USE_GRPC` - Use gRPC for Qdrant (default: `true`)
- `VECTOR_DB_GRPC_PORT` - gRPC port (default: `6334`)

### GPU / Inference

- `LLAMA_USE_GPU` - `true`/`false` (default: `true`)
- `SUPERVISOR_REMOTE_MODEL_NAME` - Model name for remote supervisor (default: `local-model`)
- `LLAMA_REMOTE_TIMEOUT` - Timeout for remote LLM requests in seconds (default: `300`)

## Docker Compose Profiles

`./run-compose.sh` supports profiles:
- `--profile gpu` - NVIDIA GPU acceleration
- `--profile cpu` - CPU-only mode
- `--profile qdrant` - Qdrant vector DB
- `--profile chroma` - Chroma vector DB

HAProxy containers start automatically (no extra profile needed) when `*_ENDPOINTS` env vars are set.

## Worker Scripts

All workers use `get_env_strategy().apply()` early:
- `run_producer.py` - Producer worker (ingests files)
- `run_consumer.py` - Consumer worker (processes chunks)
- `run_ocr_worker.py` - OCR worker (extracts text from images)
- `run_gatekeeper.py` - Gatekeeper worker (extracts + normalizes via supervisor LLM)
- `run_whisperx_worker.py` - WhisperX worker (transcribes audio/video)

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
- **MUST NOT** use placeholders like `[source1]` - only actual tags found in context
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
- Tests require env vars: `EMBEDDING_MODEL_PATH`, `DEFAULT_DOC_INGEST_ROOT`, `LLM_PATH`, `SUPERVISOR_LLM_PATH`
- Run: `PYTHONPATH=doc-ingest-chat:shared .venv/bin/python -m pytest <test_file> -v`

## CI/CD

- `.github/workflows/` - GitHub Actions workflows (Gemini triage, plan-execute, invoke, review, dispatch)
- `.github/commands/` - Custom GitHub CLI commands (gemini-triage, gemini-plan-execute, gemini-review, gemini-invoke, gemini-scheduled-triage)

## Development Quirks

- **Module imports**: Workers use `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` to ensure package imports work when running files directly
- **Environment strategy**: Must call `get_env_strategy().apply()` before any model loading or GPU operations
- **Model paths**: All model paths are absolute paths; relative paths are rejected unless default provided
- **Vector DB**: Dual support for Qdrant (default) and Chroma; profile selection via `VECTOR_DB_PROFILE` env var
- **Redis queues**: `REDIS_OCR_JOB_QUEUE` and `REDIS_INGEST_QUEUE` define queue names
- **CORS**: FastAPI middleware allows all origins (`["*"]`) - configure appropriately for production
- **Docker build**: Uses `uv export --frozen` + `uv pip install` (not `uv pip install -r pyproject.toml`) to pin dependency versions from lock file
- **Warmup**: `warmup.py` skips Docling import when `OCR_PATH` is a remote URL
- **Whisper MIME type**: Content handlers determine MIME type, pass it through Redis to whisperx worker, which sends correct Content-Type to whisper server
- **Whisper server**: Must be started with `--convert` flag to handle non-WAV formats (MP4, MP3, etc.)

## File Extensions

Supported document types: `.pdf`, `.html`, `.htm`, `.txt`, `.md`
Supported media types: `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.mp4`, `.mov`, `.mkv`

## Critical Error Messages

- `CRITICAL ERROR: Environment variable '{key}' is NOT set. This is required for the system to function.`

## Logic and Relationship Flow

### Pipeline Orchestration (The "How")

The system operates as a distributed state machine using Redis as the message broker:

1.  **Ingestion Phase**: `producer_worker` -> `producer_graph` (adds files to `INGEST_QUEUE`).
2.  **Routing Phase**: `gatekeeper_worker` -> `gatekeeper_logic` (monitors `INGEST_QUEUE`, inspects file extensions, and dispatches to specific `handlers/` or `ocr_worker`).
3.  **Processing/Embedding Phase**: `consumer_worker` -> `consumer_graph` (monitors `CONSUME_QUEUE`, calls `handlers/` for text extraction, then uses `rag_service` and `database` services to perform chunking, embedding, and storage).
4.  **OCR Phase**: `ocr_worker` -> `ocr_graph` (specifically for heavy media/image-based text extraction from PDFs).
5.  **Media Phase**: `whisperx_worker` (dedicated container for transcribing media files via WhisperX).

### Service-Worker Mapping
- **`services/`**: Pure logic/data access (e.g., `rag_service` handles the logic of "what to do with text", `database` handles "how to talk to Qdrant").
- **`workers/`**: The "loopers" (e.g., `consumer_worker` is the actual process running in Docker that calls the services).
- **`handlers/`**: The "extractors" (e.g., `pdf_handler` provides the raw text/data needed by the service).

## Codebase Understanding (Full)

### Data Flow: Document Lifecycle

```
staging/ -> Gatekeeper (claims, normalizes via Supervisor LLM through HAProxy, writes .md) -> ingestion/
       -> Producer (chunks Markdown, assigns [DOC_XXXX] IDs via MurmurHash3) -> REDIS_STAGING_QUEUE
       -> Consumer (validates chunks, embeds via e5-large-v2 through HAProxy, upserts to Qdrant/Chroma, archives to Parquet)
       -> success/ (or failed/)
```

### All 6 Workers (Redis-Coordinated)

| Worker | Entry Point | Queue | Role |
|---|---|---|---|
| Gatekeeper | `run_gatekeeper.py` | Claims from DuckDB (`ingestion_lifecycle`) | Raw text extraction via handlers, LLM normalization to Markdown, pushes chunks to `REDIS_STAGING_QUEUE` |
| OCR | `run_ocr_worker.py` | `REDIS_OCR_JOB_QUEUE` | Processes image-based PDF pages via docling-serve (EasyOCR) |
| WhisperX | `run_whisperx_worker.py` | `REDIS_WHISPER_JOB_QUEUE` | Transcribes audio/video files (.mp3, .mp4, .mov, .mkv, .wav, etc.) |
| Producer | `run_producer.py` | Reads from `ingestion/` dir | Chunks normalized Markdown, injects `[DOC_XXXX]` IDs, sends to `REDIS_STAGING_QUEUE` |
| Consumer (x2) | `run_consumer.py` | `QUEUE_NAMES` (2 queues) | Validates/embeds/upserts chunks, moves files to success/failed |
| API | `apimain.py` | HTTP :8000 | FastAPI REST + static file serving |

### Full Directory Map

| Directory | Purpose |
|---|---|
| `doc-ingest-chat/` | Core pipeline - all workers, services, handlers, config, API, chat |
| `doc-ingest-chat/workers/` | Worker entry points + LangGraph state machines (`gatekeeper_worker.py`, `gatekeeper_logic.py`, `consumer_worker.py`, `consumer_graph.py`, `producer_worker.py`, `producer_graph.py`, `ocr_worker.py`, `ocr_graph.py`) |
| `doc-ingest-chat/handlers/` | Chain of Responsibility: `BaseContentTypeHandler` -> `PDFContentTypeHandler` -> `MP4ContentTypeHandler` -> `MP3ContentTypeHandler` -> `TextContentTypeHandler` (extraction order: PDF->MP4->MP3->Text) |
| `doc-ingest-chat/services/` | Business logic: `database.py` (Qdrant/Chroma/DuckDB), `redis_service.py`, `rag_service.py`, `job_service.py` (DuckDB lifecycle state machine), `parquet_service.py`, `dependencies.py` (FastAPI DI) |
| `doc-ingest-chat/config/` | `settings.py` (60+ lazy-evaluated env vars via `__getattr__`), `env_strategy.py` (GPU/CPU), `llama_strategy.py` |
| `doc-ingest-chat/api/` | `endpoints.py` - FastAPI router: `GET /api/v1/health`, `GET /api/v1/status`, `POST /api/v1/query` |
| `doc-ingest-chat/chat/` | `chroma_chat.py` - Core RAG: retrieves from vector DB, formats context with `SOURCE_ID`/`CITATION_TAG`, calls LLM, maps citations back to filenames |
| `doc-ingest-chat/processors/` | `text_processor.py` (Markdown header-aware hierarchical splitting + zero-loss sub-splitting), `document_processor.py` (HTML/media extraction) |
| `doc-ingest-chat/prompts/` | `chat_prompts.py` - strict citation-enforcing system prompt template |
| `doc-ingest-chat/models/` | `data_models.py` (ChunkEntry, OCRJob, FileEndMessage), `query.py` (QueryRequest/Response) |
| `doc-ingest-chat/utils/` | `llm_setup.py` (fork-safe singletons, RemoteLlama, RemoteEmbeddings), `trace_utils.py`, `ocr_utils.py`, `whisper_utils.py`, `text_utils.py`, `file_utils.py`, `metrics.py`, `logging_config.py` |
| `doc-ingest-chat/tests/` | 28 pytest files covering all major components |
| `doc-ingest-chat/sql/` | `schema.sql` - DuckDB schema for `ingestion_lifecycle` table |
| `astro-frontend/` | Astro v6 + Tailwind v4 SPA: `src/pages/index.astro` (single chat UI), `src/layouts/Layout.astro`, `src/styles/global.css` |
| `infra/` | HAProxy entrypoint script (`haproxy-entrypoint.sh`) and config renderer (`render-haproxy-cfg.sh`) |
| `shared/` | Shared utilities (`utils.py`, `config.py`, `defaults.py`, `env_names.py`) and tests |

### Key Technical Patterns

1. **Lazy Config** (`config/settings.py`): 60+ settings as lambda dict. `from config.settings import X` triggers `__getattr__("X")` which calls the lambda with `os.getenv()`. Settings like `_require_abs_path()` call `sys.exit(1)` if critical env vars are missing.

2. **Fork-Safe Singletons** (`utils/llm_setup.py`, `services/database.py`): Per-process caches (`_LLAMA_MODEL_CACHE`, `_QDRANT_CLIENT_CACHE`, etc.) with PID-checking via `_check_fork()` to reset after multiprocessing forks.

3. **Remote + Local Hybrid**: All model paths auto-detect format. `http(s)://` -> `RemoteLlama`/`RemoteEmbeddings` (OpenAI-compatible API). Local path -> `Llama()`/`HuggingFaceEmbeddings`. Same pattern for OCR and Whisper.

4. **Zero-Drop Chunking** (`processors/text_processor.py`): Hierarchical Markdown splitting (H1->H2->`INTERNAL_PAGE`->H3) + recursive character splitting at 85% safety budget. Oversized chunks are sub-split rather than truncated. Uses MurmurHash3 for deterministic `[DOC_XXXX]` IDs.

5. **Consumer Staging** (`workers/consumer_worker.py`): Chunks buffered in Redis, staged to DuckDB via Pandas batching, then atomically validated/embedded/upserted when `file_end` sentinel arrives. TTL sweep prevents stale staged data.

6. **DuckDB State Machine** (`services/job_service.py`): `ingestion_lifecycle` table tracks: `NEW -> PREPROCESSING -> PREPROCESSING_COMPLETE -> INGESTING -> CONSUMING -> INGEST_SUCCESS/FAILED`. Atomic claims via `UPDATE ... RETURNING *` with 20-retry exponential backoff on lock conflicts.

7. **Dual LLM Architecture**: Supervisor LLM (gatekeeper normalization) - loaded separately via `get_supervisor_llm()`, uses CPU-only (`n_gpu_layers=0`, 4K context, flash attention). Main LLM (RAG chat) - loaded via `get_chain_or_llama()`, configurable GPU layers, 8K context.

8. **Chain of Responsibility Gatekeeper** (`workers/gatekeeper_logic.py`): Extracts raw text via handler chain (pdf->mp4->mp3->text), batches content units (default 5 pages/segments), normalizes via Supervisor LLM to Markdown with `### [INTERNAL_PAGE_X]` anchors, then eagerly pushes chunks to `REDIS_STAGING_QUEUE`.

9. **HAProxy Load Balancing**: When `*_ENDPOINTS` env vars are set, `run-compose.sh` auto-creates HAProxy containers. Entrypoint script generates config from env vars at startup. `roundrobin` + `httpclose` for even distribution. Health checks on `/models` or `/health`.

10. **MIME Type Pipeline**: Content handlers declare `MIME_TYPE` class var or use `get_mime_type()`. MIME type flows: handler -> `send_media_to_whisperx()` -> Redis job -> whisperx worker -> `RemoteWhisper.transcribe_file()` -> whisper server.

### Chat/RAG Implementation (`chat/chroma_chat.py`)

1. Retrieve top-K chunks from vector DB (`query: {query}` prefix matches `passage:` prefix from ingest)
2. Deduplicate by content, extract `DOC_XXXX` anchor, assign sequential `[sourceN]` tags
3. Build context with `SOURCE_ID`/`CITATION_TAG`/`CONTENT` per chunk
4. Send unified user prompt (no system message for small models) enforcing citation rules
5. Post-process: map `[sourceN]` back to filenames via `replace_citation_labels()`

### All Worker Entry Points

| File | Run Command |
|---|---|
| `apimain.py` | `uvicorn apimain:app --reload` |
| `run_producer.py` | `python run_producer.py` |
| `run_gatekeeper.py` | `python run_gatekeeper.py` |
| `run_consumer.py` | `python run_consumer.py` |
| `run_ocr_worker.py` | `python run_ocr_worker.py` |
| `run_whisperx_worker.py` | `python run_whisperx_worker.py` |

### Critical Env Vars Quick Reference

- `DEFAULT_DOC_INGEST_ROOT` - Root for all lifecycle dirs (required)
- `EMBEDDING_MODEL_PATH` - e5-large-v2 path/URL (required)
- `LLM_PATH` - Main chat LLM path/URL (required)
- `SUPERVISOR_LLM_PATH` - Normalization LLM path/URL (required)
- `SUPERVISOR_LLM_ENDPOINTS` - Multi-endpoint supervisor LLM URLs (comma-separated)
- `EMBEDDING_ENDPOINTS` - Multi-endpoint embedding URLs (comma-separated)
- `WHISPER_ENDPOINTS` - Multi-endpoint whisper URLs (comma-separated)
- `OCR_ENDPOINTS` - Multi-endpoint OCR URLs (comma-separated)
- `VECTOR_DB_PROFILE` - `qdrant` or `chroma`
- `VECTOR_DB_URL` - Remote vector DB URL (bypasses host/port)
- `LLAMA_USE_GPU` - `true`/`false`
- `OCR_PATH` - `LOCAL` or docling-serve URL
- `WHISPER_MODEL_PATH` - Whisper model dir or URL

### Test Structure

28 test files under `doc-ingest-chat/tests/` covering: consumer/producer/OCR graphs, gatekeeper logic/worker, handler chain, database init/types, document processor, job service, parquet service, text processor, token budgeting/safety, vector DB config, sliding window normalization, media handlers, Whisper delegation, staging audit, startup basics, multiprocess locking, and more.

Additional tests: `shared/tests/test_utils.py` (parse_endpoints, resolve helpers), handler MIME type tests, whisper delegation with mime_type tests.
