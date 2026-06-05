## Why

The self-hosted RAG pipeline has evolved organically across many iterations, accumulating significant complexity (6 workers, 2 vector DBs, 2 LLMs, 5+ content handlers, HAProxy load balancing). There is no canonical architectural specification that defines component boundaries, data contracts, and system invariants. Without this baseline spec, future changes risk regressions, unclear ownership, and inconsistent design decisions. This change creates the foundational spec documents that establish a single source of truth for the architecture.

## What Changes

- Create main spec files documenting all major system capabilities
- Document component boundaries, contracts, and data flows for each capability
- Establish the spec-driven development baseline for all future changes
- No code, configuration, or infrastructure changes — documentation only

## Capabilities

### New Capabilities

- `file-ingestion`: Document the full ingestion lifecycle from file drop in staging through DuckDB state machine transitions to final archival in Parquet. Covers: Gatekeeper normalization, Producer chunking with MurmurHash3 ID assignment, Consumer embedding/upsert, and the DuckDB lifecycle states (NEW → PREPROCESSING → PREPROCESSING_COMPLETE → INGESTING → CONSUMING → INGEST_SUCCESS/FAILED).
- `content-handlers`: Document the Chain of Responsibility pattern for multi-format text extraction. Covers: BaseContentTypeHandler, PDFContentTypeHandler (with OCR fallback), MP4ContentTypeHandler (WhisperX), MP3ContentTypeHandler, TextContentTypeHandler, MIME type flow, and handler ordering.
- `rag-chat`: Document the RAG query lifecycle: vector DB retrieval with asymmetric prefix matching (query:/passage:), context assembly with citation enforcement, LLM response generation with strict source tagging, and post-processing with filename mapping.
- `vector-database`: Document the dual vector store abstraction. Covers: Qdrant (gRPC + REST) and Chroma support via VECTOR_DB_PROFILE, VectorStoreWrapper interface, async embedding generation, and archive-to-Parquet workflow.
- `load-balancing`: Document the HAProxy-based multi-backend system. Covers: auto-detection of *_ENDPOINTS env vars, per-service HAProxy containers with roundrobin balancing, health checks, stats UI, and transparent proxy for single-endpoint mode.
- `frontend-ui`: Document the Astro v6 + Tailwind v4 + daisyUI chat frontend. Covers: single-page architecture, API routing through PUBLIC_API_BASE_URL, theme system, citation rendering, and frontend build/deployment.
- `infrastructure`: Document the deployment and operations infrastructure. Covers: Docker Compose profiles (cuda/cpu/qdrant/chroma), worker Dockerfiles, HAProxy containers, environment variable system, and startup/warmup sequencing.

### Modified Capabilities

<!-- No existing specs to modify — this is the initial baseline. -->

## Impact

- `openspec/specs/` directory: 7 new capability specs created
- `openspec/changes/document-current-architecture/`: proposal, design, tasks, and specs artifacts filled
- No code, configuration, or infrastructure changes
- AGENTS.md and existing docs/ remain as parallel references (not replaced)
