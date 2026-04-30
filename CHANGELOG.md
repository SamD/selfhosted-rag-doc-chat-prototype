# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

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
  - **Zero-Drop Truncation Policy**: Replaced the "Drop if Oversized" boolean validator with a "Hard Truncation" filter that forcibly caps chunks at 511 tokens, ensuring 100% data persistence.
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
- **Linting & Code Quality**: Fixed multiple `ruff` errors across `direct_integration_test.py` and `run_full_normalization.py` for better compliance and readability.

### Changed
- **Stateless Retyping Philosophy**: Re-locked the Gatekeeper into a pure pass-through mode that trusts server-side parameters (Temperature/Tokens) and focuses exclusively on high-density structural transcription.
- **Global mmh3 Standardization**: Standardized on MurmurHash3 across the entire pipeline for both document binary signatures and semantic chunk addressing.

---
... [Previous legacy changes maintained below] ...
