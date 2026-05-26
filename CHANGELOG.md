# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

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
