# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **Redis-backed chat session management**: New `ChatSessionService` stores chat history server-side in Redis. API models replaced `chat_history` with `session_id` in request/response (BREAKING). Frontend generates UUID v4 session_id in localStorage. New env vars: `MAX_SESSION_TURNS` (default 20), `SESSION_TTL_HOURS` (default 24). 12 unit tests.
- **`SUPERVISOR_N_CTX` env var**: Separate context window setting for the gatekeeper supervisor LLM prompt truncation. Previously shared `LLAMA_N_CTX` with the main RAG LLM. Default: 8192.
- **Operational playbooks**: `infra/operations/day-1.md` (setup checklist) and `infra/operations/day-2.md` (symptom→diagnosis→fix runbook).
- **OpenSpec baseline**: 7 capability specs documenting current architecture (`openspec/specs/`). Project context and per-artifact rules in `openspec/config.yaml`.

### Changed
- **CPU profile removed**: Removed `--cpu` flag from `run-compose.sh`, `CPUEnvConfig` from `env_strategy.py`, `LLAMA_USE_GPU` env var, `USE_OLLAMA` env var, and `run-compose-cpu.sh`. Device hardcoded to `"cuda"` — non-GPU deployments use remote HTTP endpoints.
- **Ollama code path removed**: `USE_OLLAMA` branch removed from `chroma_chat.py` and `utils/llm_setup.py`. Users who want Ollama set `LLM_PATH` to the Ollama server URL (OpenAI-compatible).
- **Gatekeeper context limit**: Now uses `SUPERVISOR_N_CTX` instead of `LLAMA_N_CTX * 0.8`. Decouples supervisor prompt truncation from main LLM context.
- **Environment variable consolidation**: Renamed all single-endpoint service env vars to use the `*_ENDPOINTS` naming convention for consistency. `SUPERVISOR_LLM_PATH`→`SUPERVISOR_LLM_ENDPOINTS`, `EMBEDDING_MODEL_PATH`→`EMBEDDING_ENDPOINTS`, `WHISPER_MODEL_PATH`→`WHISPER_MODEL_ENDPOINTS`, `OCR_PATH`→`OCR_ENDPOINTS`. Each var now supports both single URLs and comma-separated multi-endpoint lists for HAProxy load balancing.
- **`run-compose.sh`**: HAProxy auto-override now saves original endpoints to `HAPROXY_*` vars so haproxy containers receive the raw endpoint list (not the overridden proxy URL).
- **`astro-frontend`: Replaced `@astrojs/tailwind` with `@tailwindcss/vite` and updated `astro` to `6.4.4`.
- **`astro-frontend`: API base URL now read from `PUBLIC_API_BASE_URL` env var instead of hardcoded. Defaults to `http://localhost:8000/api/v1`.
- **`doc-ingest-chat/utils/llm_setup.py`**: Restored `_LLAMA_MODEL_CACHE = Llama(**params)` for local GGUF files (was removed when `os.path.exists` check was added).

### Added
- **`daisyUI themes`**: Added explicit `themes:` config to `global.css` with all 12 themes — daisyUI 5 only ships dark/light by default.
- **`EndpointDispatcher` in `shared/utils.py`**: New class for dispatching batched LLM calls across multiple HA backends with two modes:
  - **Interleaved** (set `HA_INTERLEAVE=1`): Batches dispatched concurrently via `ThreadPoolExecutor`, each to a different backend in round-robin.
  - **Pinned** (default): All batches in a job go to one backend sequentially.
  - Control via env var `HA_INTERLEAVE` (default: `false`). Designed for benchmarking TPS differences.
- **`shared/env_names.py`** and **`shared/defaults.py`**: Added `ENV_HA_INTERLEAVE` and `DEFAULT_HA_INTERLEAVE`.
- **`shared/config.py`**: Added `HA_INTERLEAVE` setting.
- **`ingest-dockercompose.yaml`**: Added `HA_INTERLEAVE` and `HAPROXY_*` env vars to worker containers.
- **Tests for `EndpointDispatcher`**: 16 new tests in `shared/tests/test_utils.py` covering constructor validation, round-robin, pinned mode, result ordering, error propagation, job labels, counter persistence, and max workers clamping.

### Removed
- **Stale standalone scripts**: `run_full_ingestion.py`, `run_full_normalization.py`, `run_full_normalization_test.py`, `run-compose-cpu.sh` — old test scripts with hardcoded paths, superseded by the worker pipeline and `run-compose.sh`.

### Fixed
- **Chunk content loss**: Added regex in `text_processor.py` to separate `### [INTERNAL_PAGE_X]` headers from inline content before MarkdownHeaderTextSplitter processes them. Prevents the splitter from swallowing content into header metadata when the supervisor LLM places text on the same line as the header.
- **Local GGUF model never loaded**: Missing `_LLAMA_MODEL_CACHE = Llama(**params)` caused `get_chain_or_llama()` to return `None` for local GGUF paths, resulting in `'NoneType' object has no attribute 'create_chat_completion'`.
- **MP3 handler missing extensions**: Added `.m4a`, `.aac`, `.flac` to `MP3ContentTypeHandler.can_handle()`.
- **MP4 handler missing extensions**: Added `.mov`, `.mkv` to `MP4ContentTypeHandler.can_handle()`.
- **Frontend theme picker**: Changed `themeChange()` to `themeChange(false)` to avoid `DOMContentLoaded` race condition — Astro module scripts load after the event fires, so the listener never executes.
- **Frontend layout**: Replaced fixed `h-96` with `flex-1` + `min-h-0` so the chat area fills available viewport height.
- **Frontend chat bubbles**: Widened from `max-w-xs/md` to `max-w-lg/2xl` for better readability.
- **Docs state machine table**: Fixed `PREPROCESSING→PREPROCESSING_COMPLETE` directory move, `INGESTING→CONSUMING` worker (Producer not Consumer), and `CONSUMING→INGEST_SUCCESS` directory move.
- **Docs queue names**: Fixed references to stale `REDIS_STAGING_QUEUE`, `INGEST_QUEUE`, and `CONSUME_QUEUE` — actual queues are `QUEUE_NAMES` (`chunk_ingest_queue:0,1`).
- **Docs profile name**: Updated `--profile gpu` to `--profile cuda` throughout.
- **Docs paths**: Updated example paths to use `$DEFAULT_DOC_INGEST_ROOT` instead of hardcoded `/path/to/Docs/`.
- **Removed stale test_consumer_utils tests**: `TestCleanupOrphanedQdrantPoints`, `TestConsumerWorkerMainOrphanCall` — referenced non-existent `cleanup_orphaned_qdrant_points`.
- **Updated media handler tests**: Added `mime_type` to expected `send_media_to_whisperx` calls.
- **Updated `resolve_supervisor_endpoint`**: Only appends `/v1` to HTTP URLs, not local paths.
- **Updated `resolve_embedding_endpoint`**: Returns `None` for local paths (URLs only).
- **PlantUML diagram**: Updated node roles and descriptions, added coordinator host.

### Added
- **HAProxy Load Balancing**
  - **Multi-endpoint support**: `SUPERVISOR_LLM_ENDPOINTS`, `EMBEDDING_ENDPOINTS`, `WHISPER_MODEL_ENDPOINTS`, `OCR_ENDPOINTS` env vars accept comma-separated backend URLs.
  - **Automatic HAProxy containers**: `haproxy_supervisor`, `haproxy_embd`, `haproxy_whisper`, `haproxy_ocr` services in docker-compose start automatically with the `cuda` profile.
  - **Auto-override model paths**: `run-compose.sh` detects `*_ENDPOINTS` env vars and auto-sets `SUPERVISOR_LLM_PATH`, `EMBEDDING_MODEL_PATH`, `WHISPER_MODEL_ENDPOINTS`, `OCR_ENDPOINTS` to HAProxy container URLs.
  - **Round-robin balancing**: `balance roundrobin` with `option httpclose` ensures requests alternate across backends without keep-alive pinning.
  - **Health checks**: HAProxy checks `/models` (LLM/embedding) or `/health` (whisper/OCR) every 2s, marks backends down after 3 failures, up after 2 successes.
  - **Stats UI**: HAProxy stats available at `localhost:8404` (supervisor), `:8405` (embedding), `:8406` (whisper), `:8407` (OCR).
  - **Entrypoint script**: `infra/haproxy-entrypoint.sh` generates HAProxy config at container startup from `*_ENDPOINTS` env vars. Handles 0 (idle/503), 1 (transparent proxy), and 2+ (load-balanced) endpoints.
- **Shared utilities (`shared/utils.py`)**
  - `parse_endpoints()` — parses comma-separated endpoint URLs from env vars.
  - `resolve_supervisor_endpoint()` — single-endpoint fallback resolution.
  - `resolve_embedding_endpoint()` — single-endpoint fallback resolution.
- **Content handler MIME type support**
  - `BaseContentTypeHandler.MIME_TYPE` class var and `get_mime_type()` method.
  - `MP4ContentTypeHandler` declares `MIME_TYPE = "video/mp4"`.
  - `PDFContentTypeHandler` declares `MIME_TYPE = "application/pdf"`.
  - MIME type flows from content handler → `send_media_to_whisperx()` → Redis job → whisperx worker → `RemoteWhisper.transcribe_file()`.
- **Whisper.cpp server debugging section** in `docs/operations.md` with curl test commands and common issues.
- **HAProxy monitoring section** in `docs/operations.md` with stats URLs, log inspection, and traffic distribution verification.
- **Multi-endpoint configuration section** in `docs/quickstart.md`.
- **HAProxy architecture section** in `docs/overview.md` with component diagram and service table.

### Changed
- **`Dockerfile.worker`**: Uses `uv export --frozen` + `uv pip install` instead of `uv pip install -r pyproject.toml` to pin dependency versions from lock file. Added `OCR_ENDPOINTS` build arg to skip Docling warmup when OCR is remote.
- **`warmup.py`**: Skips Docling warmup when `OCR_ENDPOINTS` is a remote URL. Made tokenizer warmup non-fatal.
- **`whisperx_worker.py`**: `RemoteWhisper.transcribe_file()` accepts `mime_type` parameter from content handler instead of hardcoding `audio/wav`.
- **`whisper_utils.py`**: `send_media_to_whisperx()` accepts and passes `mime_type` parameter in Redis job.
- **`run-compose.sh`**: Added auto-detection of `*_ENDPOINTS` env vars with automatic `*_PATH` override to HAProxy container URLs.
- **`ingest-dockercompose.yaml`**: Added `SUPERVISOR_LLM_ENDPOINTS`, `EMBEDDING_ENDPOINTS`, `WHISPER_MODEL_ENDPOINTS`, `OCR_ENDPOINTS` to worker env vars. Added 4 HAProxy services. Added `OCR_ENDPOINTS` build arg.
- **`gatekeeper_logic.py`**: Restored `tmp_md_path` argument to `process_chunk` calls.
- **`base_handler.py`**: Added `MIME_TYPE` class var and `get_mime_type()` method.

### Fixed
- **Whisper transcription failing for non-WAV files**: `RemoteWhisper.transcribe_file()` was hardcoding `audio/wav` Content-Type for all files. Now uses correct MIME type from content handler (e.g., `video/mp4` for MP4).
- **`process_chunk` missing `md_path` argument**: Restored `tmp_md_path` to both call sites in `gatekeeper_extract_and_normalize`.
- **Docker build dependency conflict**: `transformers` updated to require `torch.float8_e8m0fnu` which didn't exist in installed torch. Fixed by using lock file (`uv export --frozen`) instead of `pyproject.toml` for dependency resolution.
- **Docling warmup failing when OCR is remote**: Warmup script now checks `OCR_ENDPOINTS` and skips Docling import when set to a remote URL.

### Added
- **Shared Configuration Package (`shared/`)**
  - **Canonical env var names**: Moved all 85+ environment variable name constants into `shared/env_names.py` — single source of truth for `MQTT_BROKER_HOST`, `LLM_PATH`, `VECTOR_DB_PROFILE`, etc.
  - **Canonical defaults**: Moved all 65+ default values into `shared/defaults.py` — ensures `doc-ingest-chat`, `mqtt_agent_hub`, and future components use identical fallback values.
  - **Shared settings module**: `shared/config.py` contains the full `_SETTINGS` lazy-loading dictionary with `get_setting()` API, usable by any component without duplicating config logic.
- **daisyUI Theme System (both frontends)**
  - **Astro v6 + Tailwind CSS v4 + daisyUI 5.5**: Both `astro-frontend/` and `mqtt_agent_hub/astro-dashboard/` now share the same theming stack.
  - **11-theme picker**: Dark (default), light, corporate, synthwave, cyberpunk, forest, dracula, night, nord, dim, sunset — selectable via dropdown, persisted in localStorage.
  - **`theme-change` integration**: FART-proof (Flash of inaccurate Theme) with `is:inline` script — theme applied before first paint.
  - **Semantic class migration**: Replaced hardcoded Tailwind color classes (`text-gray-900`, `bg-white`, `bg-blue-600`) with daisyUI semantic classes (`text-base-content`, `card bg-base-100`, `btn-primary`) so all components respond to theme changes.
- **Frontend Smoke Tests**
  - `npm test` in both frontends runs `astro check && astro build`.
  - `test.sh` shell scripts verify daisyUI CSS, theme-change JS, dark default, theme picker, and app-specific components appear in built output.
- **Dashboard Remote Broker Support**
  - `PUBLIC_MQTT_BROKER_HOST` env var overrides `window.location.hostname` when Mosquitto is on a separate host.
  - `PUBLIC_HUB_PORT`, `PUBLIC_MQTT_WS_PORT`, `PUBLIC_MQTT_USERNAME`, `PUBLIC_MQTT_PASSWORD` env vars for dashboard configuration.
  - **Remote Broker Deployment** section in README with docker-compose and standalone examples.

### Changed
- **Documentation reorganization**: Rewrote `README.md` as a clean landing page with table of contents, visible documentation navigation, and generic service-oriented hostnames (`<llm-host>`, `<embedding-host>`, `<vector-db-host>`, etc.). Moved detailed environment configuration, distributed deployment diagram, and hardware profile into a new `docs/quickstart.md`. Removed all LAN-specific hostnames and IP addresses from README. Moved cross-document navigation links from bottom to top of all docs for consistent discoverability.
- **`doc-ingest-chat/config/settings.py` refactored to thin wrapper**: All 83 settings, helper functions, and lazy-loading logic moved to `shared/config.py`. `config/settings.py` now imports `_SETTINGS` from shared and re-exports via `__getattr__` — all existing `from config.settings import X` imports continue working unchanged.
- **`config/llama_strategy.py` and `config/env_strategy.py`**: Updated to import env names and defaults from `shared/` instead of hardcoded strings. Added sys.path fix for Docker containers where `shared/` is at `/app/shared/`.
- **Dockerfile updates**: `Dockerfile.worker` and `Dockerfile.worker.inprogress` now copy `shared/` into `/app/shared/`.
- **`ingest-dockercompose.yaml`**: Added `../shared:/app/shared` volume mount to base service anchor so shared package is available in all worker containers.
- **`docker-compose.frontend.yaml`**: Added `--legacy-peer-deps` to npm install command for daisyUI compatibility.
- **MQTT dashboard docker-compose**: Passes `PUBLIC_*` env vars to `hub_dashboard` service.

### Fixed
- **Supervisor LLM timeout due to missing `max_tokens`**: Added `SUPERVISOR_MAX_TOKENS` (env var, default 4096) to cap supervisor LLM generation. Previously, `create_chat_completion` had no `max_tokens`, so the model could generate indefinitely if the stop token missed, causing HTTP timeouts.
- **Missing settings in `config/settings.py`**: Added `METRICS_ENABLED`, `METRICS_LOG_FILE`, `METRICS_LOG_TO_STDOUT`, `FAILED_FILES`, and `INGESTED_FILE` — previously imported at runtime but undefined.
- **Dashboard CSS not loaded at build time**: Added `<style is:global>` block importing `global.css` to dashboard `Layout.astro`.
- **Docker container `ModuleNotFoundError: No module named 'shared'`**: Fixed by copying `shared/` into Docker images and mounting it in docker-compose volumes.

### Added
- **Database-Driven Lifecycle State Machine**
  - **Atomic State Transitions**: Replaced filesystem-only monitoring with a DuckDB-backed `ingestion_lifecycle` registry. Files now move physically between stage directories (`staging/`, `preprocessing/`, `ingestion/`, `consuming/`, `success/`) only after successful atomic "Claim" and "Transition" database updates.
  - **Relational Resilience**: Implemented millisecond-precision tracking for every phase of a document's journey, including automatic error logging and worker identification.
- **Zero-Memory Persistent Staging**
  - **DuckDB Buffer Layer**: Refactored the Consumer to immediately persist incoming chunks to a `staged_chunks` table instead of holding them in RAM. This enables massive (1000+ page) document processing without OOM risk.
  - **High-Performance Pandas Batching**: Implemented `stage_chunks` using Pandas-driven batch inserts to resolve DuckDB lock contention and maximize ingestion throughput.
  - **Atomic Vector Persistence**: Documents are only upserted to Qdrant/ChromaDB as a single unit after a `file_end` sentinel is verified against the staged data, ensuring zero partial-visibility for RAG.
- **High-Fidelity Page Anchoring**
  - **Structural Tags**: Implemented injection of `### [INTERNAL_PAGE_X]` anchors into the Markdown normalization stream.
  - **Robust Anchor Extraction**: Refactored the Producer to scan all metadata levels and hierarchical headers to extract these anchors, ensuring 100% accurate page-level metadata for every chunk.
- **Hardened Token Budgeting**
  - **Conservative Safety Margin**: Lowered the hierarchical splitter budget to **450 tokens** to accommodate mandatory RAG prefixes and model special tokens.
   - **Zero-Drop Sub-Splitting**: Replaced the "Drop if Oversized" boolean validator with recursive character-based sub-splitting at a sliding window. Oversized chunks are split into valid-sized pieces rather than truncated, ensuring 100% content preservation.
  - **Special-Token Parity**: Synchronized length calculations between Producer and Consumer to use `add_special_tokens=True`, ensuring mathematical parity with the embedding model's actual requirements.
- **Chain of Responsibility Content Handlers**
  - **Modular Architecture**: Introduced `BaseContentTypeHandler` and specialized handlers (`PDF`, `Text`, `MP3`, `MP4`) using the Chain of Responsibility pattern for cleaner, more extensible document processing.
- **Multimedia Transcription Support**
  - **WhisperX Integration**: Added a dedicated `whisperx_worker` for high-performance audio and video transcription with alignment and timestamp support.
- **Enhanced Observability & Traceability**
  - **Distributed Trace IDs**: Implemented `trace_utils` to propagate unique `trace_id`s across all distributed workers, allowing end-to-end visibility of a document's processing journey.
- **Stream-Centric Processing**
  - **Chunk-Level Streaming**: Refactored the ingestion pipeline to process data as a continuous stream of chunks, optimizing memory usage and enabling massive document support.
- **Interactive RAG UI Enhancements**
  - **Clickable Document Citations**: Implemented a static file route (`/files`) in the FastAPI backend to serve ingested PDFs directly. 
  - **Markdown Link Mapping**: Refactored `chat_utils.py` and the AstroJS frontend to transform plain-text citations into interactive Markdown links pointing to the exact page of the original source.
  - **Unified User Prompt**: Re-architected RAG and Normalization prompts to use a single User-role message, significantly improving focus and reducing "Note:" hallucinations for smaller models (0.5B / 3.8B).
  - **Underscore-Aware Anchor Parsing**: Updated regex logic to correctly identify MurmurHash3 IDs containing underscores in the retrieval context.
- **Dependency & Frontend Modernization**
  - **Astro 6.1.9 Upgrade**: Major version jump for the frontend, bringing latest performance and stability improvements.
  - **Tailwind CSS 4.2.4**: Updated styling engine to the latest stable release.
  - **Universal uv Synchronization**: Fully locked all backend dependencies (288+ packages) to their latest stable versions, including major updates to `Docling (2.91.0)`, `FastAPI`, and `DuckDB`.

### Fixed
- **DuckDB Lock Contention**: Implemented a global **20-Retry Exponential Backoff** system for all database operations, resolving "Conflicting lock" crashes during high-concurrency ingestion.
- **Duplicate Document Pollution**: Implemented content-addressable ID generation using **MurmurHash3 (mmh3)** combining `DOC_ID` and `CHUNK_HASH`. This ensures that identical text within or across documents collides and overwrites instead of creating redundant vector points.
- **RAG History Leak**: Fixed a bug where user queries were being double-appended to the chat history, causing "Broken Record" LLM responses.
- **Metadata KeyError**: Refactored `store_chunks_in_db` to use safe `.get()` defaults, preventing Consumer crashes when encountering inconsistent metadata fields.
- **Indentation-Triggered Hallucinations**: Flattened the Gatekeeper prompt to zero-indentation to prevent models from interpreting whitespace as a "Compliance Report" requirement.
- **Frontend CSS Resolution**: Resolved a critical breakage where `global.css` was missing, preventing Tailwind 4 from initializing.
- **Frontend Docker Configuration**: Corrected `docker-compose.frontend.yaml` to point to the dedicated `Dockerfile.frontend` and fixed environment variable typos.
- **Persistent Staging Leak**: Implemented a startup **Audit & Cleanup** mechanism in `ParquetService` that automatically purges orphaned chunks from `staged_chunks` if the service restarts before a file is finalized.
- **Linting & Code Quality**: Fixed multiple `ruff` errors across `direct_integration_test.py` and `run_full_normalization.py` for better compliance and readability.

### Changed
- **Stateless Retyping Philosophy**: Re-locked the Gatekeeper into a pure pass-through mode that trusts server-side parameters (Temperature/Tokens) and focuses exclusively on high-density structural transcription.
- **Global mmh3 Standardization**: Standardized on MurmurHash3 across the entire pipeline for both document binary signatures and semantic chunk addressing.
- **Documentation Consolidation**: Merged 20 scattered documentation files into 4 structured docs: QuickStart (`README.md`), Architecture (`docs/overview.md`), Deep Dive (`docs/deep-dive.md`), and Operations (`docs/operations.md`). Removed stale/outdated planning documents. All images and assets now live under `docs/`.

---
... [Previous legacy changes maintained below] ...
