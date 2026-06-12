**[< Overview](overview.md) | [Quick Start](quickstart.md) | [Operations](operations.md)**

# Technical Deep Dive

This document covers design rationale, AI model usage, and production considerations for the Self-Hosted RAG system.

---

## Architecture Decisions

The ingestion pipeline is a custom distributed system built for production document processing. Rather than relying on framework abstractions, each component — worker pools, queue coordination, chunking logic, quality detection, backpressure handling, and atomicity guarantees — is explicitly implemented for full control and observability.

Vector store integration uses lightweight wrappers (`langchain_chroma.Chroma`, `langchain_qdrant.Qdrant`) with `langchain_huggingface.HuggingFaceEmbeddings` for embeddings. All pipeline logic is custom-built.

### Design Philosophy

**Chunking Strategy**: Token-aware splitting uses the embedding model's own tokenizer (e5-large-v2) for exact boundary calculation. Hierarchical Markdown headers (`H1 → H2 → INTERNAL_PAGE → H3`) preserve semantic structure. Oversized chunks are sub-split rather than truncated (zero-drop policy).

**Quality Assurance**: Multi-stage text quality detection catches gibberish, corruption, and encoding issues. The PDF handler attempts `pdfplumber` first, then falls back to the OCR worker (Docling/EasyOCR) for scanned or unreadable pages. Latin script ratio checking with configurable thresholds identifies OCR failures.

**Production Guarantees**: File-level atomicity via DuckDB staging — all chunks from a document commit to Qdrant together or not at all. NiFi-managed backpressure prevents memory overflow, with configurable queue thresholds and auto-backoff. Distributed worker pools with multiprocessing. Comprehensive error tracking with per-file and per-chunk diagnostics. End-to-end traceability via `trace_id` propagated through NiFi flowfile attributes.

**Transparency**: All parameters exposed via environment variables. Direct access to vector DB, Redis queues, and DuckDB for inspection. No hidden abstractions. Full logging with timestamped, trace-id-tagged, emoji-coded event types.

---

## The Markdown-First Approach

Converting raw text tokens directly into a vector database is a low-quality RAG strategy. By normalizing to Markdown first, the system transitions from syntactic chunking (split by character count) to semantic chunking (split by logical meaning).

### Noise Elimination

Raw PDF/OCR text contains "technical noise": running footers, page numbers, repeated book titles. The Supervisor LLM strips these artifacts during normalization. Every token in the vector DB is 100% content, ensuring top-k retrieval results are actually relevant.

### Contextual Healing

OCR produces artifacts like `C0nstantine` or broken line breaks. The Supervisor LLM uses its world knowledge during retyping to heal these errors — it recognizes the entity and fixes the spelling before it is indexed, ensuring search terms always find their targets.

### Structural Integrity

Markdown headers (`##`, `###`) allow the Producer to perform hierarchy-aware splitting. It preserves the connective tissue of information by splitting at paragraph breaks rather than mid-sentence.

### Semantic Density

Embedding models like e5-large-v2 look for the essence of a passage. Markdown headers provide a massive boost to the attention mechanism. A chunk starting with `## CHAPTER VIII: THE ANCESTRY OF MAN` immediately tells the embedding model what the following text is about, creating much tighter semantic clusters.

---

## Dual-LLM Architecture

The system uses two distinct LLMs. Both support local GGUF inference via llama-cpp or remote API endpoints (llama-server, Ollama, or any OpenAI-compatible endpoint). When running remotely, the system auto-detects the HTTP(S) URL and uses an OpenAI-compatible client — no llama-cpp loaded locally.

### Supervisor LLM (Normalization)

- **Env var**: `SUPERVISOR_LLM_ENDPOINTS`
- **When**: During ingestion — the Gatekeeper phase
- **Role**: Stateless "retyping" of raw extracted text into clean, structured Markdown
- **Local config**: CPU-only (`n_gpu_layers=0`), 4K context window, flash attention enabled — keeps VRAM free for the RAG LLM
- **Remote config**: Any OpenAI-compatible endpoint — the server handles its own context and GPU settings
- **Why separate**: Normalization requires clean structural output, not conversational reasoning. A smaller, focused model suffices and avoids VRAM contention with the RAG LLM.
- **Quality bypass**: Before calling the LLM, extracted text runs through `is_bad_ocr()` — a multi-heuristic quality check (gibberish detection, repetition ratio, word length analysis, visible corruption). Pages that pass are written directly as Markdown, skipping the LLM entirely. Only pages that fail the quality check consume supervisor LLM tokens. This is the three-tier approach: pdfplumber → OCR → LLM (last resort).

### RAG LLM (Chat)

- **Env var**: `LLM_PATH`
- **When**: During query — the generation phase
- **Role**: Conversational reasoning and grounded retrieval with strict citation enforcement
- **Local config**: Configurable GPU layers, 8K context window
- **Remote config**: Any OpenAI-compatible endpoint
- **Constraints**: Must include exact citation tags from context (`[doc5]`, `[ref12]`, `[source:7]`). Must not fabricate, infer, or editorialize. Each factual sentence must be followed immediately by its source tag inline.

### Embedding Model (e5-large-v2)

- **Env var**: `EMBEDDING_ENDPOINTS`
- **When**: Both ingestion (chunk embedding) and query (query embedding)
- **Local**: HuggingFace sentence-transformer loaded from a local directory
- **Remote**: OpenAI-compatible embeddings API (e.g., `http://host:11434/v1/embeddings`)
- **Behavior**: Deterministic — no stochasticity, no generation bias. Maps text to fixed-size vectors (512-token context window).
- **Prefix convention**: Chunks are prefixed with `passage: [DOC_XXXX]`, queries with `query:`, enabling asymmetric search in Qdrant.
- **Bias source**: Any retrieval "bias" comes from the embedding model's training data representation, not from generation behavior.

---

## Chunking Strategy

### Hierarchical Splitting

The Producer (`processors/text_processor.py`) splits normalized Markdown documents using a hierarchy:

```
H1 (#) → H2 (##) → INTERNAL_PAGE (###) → H3 (###)
```

This preserves the document's semantic structure. Each `### [INTERNAL_PAGE_X]` anchor (injected by the Gatekeeper) maps chunks back to their original page numbers for accurate citation.

### Token Budget

- **Limit**: 512 tokens per chunk (e5-large-v2 context window)
- **Safety budget**: 85% of `MAX_TOKENS`, minus prefix overhead
- **Overlap**: 50 tokens between consecutive chunks to prevent boundary information loss

### Zero-Drop Sub-Splitting

If a section still exceeds the token budget after hierarchical splitting, it is **sub-split** using a recursive character splitter with a sliding window. Chunks are never truncated — all content is preserved, split into valid-sized pieces.

### Deterministic IDs

Each chunk receives a MurmurHash3 content-addressable ID in the format `[DOC_XXXX]_xxxxxxxx`. This ensures:
- Re-ingesting a file cleanly overwrites existing chunks (no duplication)
- Chunk provenance is traceable back to the source document
- IDs are deterministic — same content always produces the same ID

---

## DuckDB State Machine

### Lifecycle Tracking

The `ingestion_lifecycle` table records every file's complete journey with dedicated timestamp columns:

```sql
new_at                     -- File first discovered
preprocessing_at           -- Gatekeeper claimed
preprocessing_complete_at  -- Normalization finished, .md written
ingesting_at               -- Producer claimed
consuming_at               -- Chunks enqueued, pending Consumer
finalized_at               -- Consumer finished (success or failed)
```

### Atomic Claims

Workers claim jobs using `UPDATE ... RETURNING *` to atomically advance the state. A 20-retry exponential backoff resolves lock contention across parallel worker processes.

### Staged Chunks (Write-Ahead Log)

The `staged_chunks` table buffers incoming chunks before Qdrant persistence:
- Consumer batches chunks using Pandas DataFrames for efficient bulk inserts
- On `file_end` sentinel: all staged chunks for a file are retrieved, embedded, and upserted to Qdrant
- After successful upsert: staged chunks are purged; main chunks committed to `parquet_chunks` for archival
- Stale staged data (TTL 6 hours, or files not in `CONSUMING` state) is cleaned up by periodic audit

---

## NiFi Middleware Layer

Apache NiFi operates as transparent middleware between every worker and Redis. Workers continue making standard Redis calls — they are unaware of NiFi's presence. NiFi provides data governance, provenance tracking, flow orchestration, and operational control without requiring a single line change in the worker code.

### Why NiFi

The ingestion pipeline's distributed, multi-worker architecture presents operational challenges:

- **Visibility gaps**: With direct Redis queues, there's no central view of data in flight. Is a job stuck? Which worker consumed it? Did the OCR worker finish?
- **Governance requirements**: For regulated environments, every data movement must be traceable. Who touched this chunk? When did it pass through preprocessing?
- **Backpressure complexity**: Managing flow control across 6 worker types with ad-hoc Redis calls is fragile. A slow consumer should not crash the producer.

NiFi solves these by sitting between workers and Redis, acting as a managed data highway:

- **Provenance**: Every flowfile (chunk, OCR job, whisper job) is tracked end-to-end. The original `trace_id` from the worker is preserved as a NiFi attribute, appearing in the provenance UI.
- **Backpressure**: NiFi enforces configurable queue limits. When `_input` queues fill, workers see Redis connection backpressure — no custom logic needed.
- **Operational visibility**: The NiFi UI shows every processor's status, queue depth, and flowfile throughput in real time. Stuck jobs are immediately visible.

### Bridge Processor Architecture

Each Redis queue pair is served by two Python processors running inside NiFi:

```
Worker → REDIS LPUSH → *_input → [RedisQueueConsumer] → [RedisQueueProducer] → LPUSH → *_output → REDIS BRPOP ← Worker
```

| Processor | Role |
|-----------|------|
| `RedisQueueConsumer` | Blocks on `BRPOP` from an `*_input` queue. On receiving a message, creates a NiFi flowfile with the JSON payload as content and extracts `trace_id`, `source_file`, and other metadata as flowfile attributes. |
| `RedisQueueProducer` | Receives flowfiles from the consumer (via NiFi's internal queues), extracts the content, and `LPUSH`es it to the corresponding `*_output` queue. Applies TTL to prevent stale data buildup. |

Processors are connected in pairs within NiFi process groups. A failed producer routes the flowfile to a failure relationship, where it can be inspected or retried.

### Queue Naming Convention

All queues use `_input` / `_output` suffix pairs:

| Worker Writes To | NiFi Reads From | NiFi Writes To | Worker Reads From |
|-------------------|-----------------|-----------------|-------------------|
| `ocr_job_input` | `ocr_job_input` | `ocr_job_output` | `ocr_job_output` |
| `whisper_job_input` | `whisper_job_input` | `whisper_job_output` | `whisper_job_output` |
| `chunk_ingest_input:N` | `chunk_ingest_input:N` | `chunk_ingest_output:N` | `chunk_ingest_output:N` |

The naming is configured via environment variables (`REDIS_OCR_JOB_QUEUE`, `REDIS_WHISPER_JOB_QUEUE`, `QUEUE_NAMES`) — the `_input`/`_output` suffix is appended by the bridge processors at flow creation time.

### Attribute Propagation

Every JSON payload sent by workers MUST include a `trace_id` field. The `RedisQueueConsumer` extracts this and sets it as a NiFi flowfile attribute:

```json
{"trace_id": "abc123", "source_file": "doc.pdf", "page": 5, "chunk_id": "DOC_0001_a1b2c3d4", "content": "..."}
```

In the NiFi provenance UI, this `trace_id` links every flowfile back to its origin worker and processing stage. The attribute is preserved through the entire pipeline — from Gatekeeper dispatch through OCR/Whisper, normalization, chunking, and final Consumer upsert.

### Idempotent Flow Setup

The flow is created programmatically via `nifi_bootstrap.py`, which runs as a one-shot Docker service on startup:

1. Waits for NiFi to become available (exponential backoff)
2. Checks if "RAG Pipeline" process group exists — skips creation if present
3. Creates process groups per queue pair (OCR, Whisper, Markdown, Consumer)
4. Instantiates `RedisQueueConsumer` + `RedisQueueProducer` processors per queue
5. Connects processors and starts the flow
6. Verifies all processors are running
7. Exits (one-shot, does not restart)

This ensures the system is self-deploying — no manual NiFi UI configuration required. On subsequent restarts, the bootstrap detects the existing flow and just starts any stopped processors.

---

## Production Roadmap

### What Would Change for Production

**High-priority additions:**
- Authentication, multi-tenancy, API keys, quota management
- Prometheus/Grafana monitoring and alerting
- Batch API with progress tracking and webhook callbacks
- Vector DB partitioning for 100K+ document collections
- Redis caching for frequently accessed chunks
- Usage-based cost tracking

**Infrastructure changes:**
- Clustered Redis (Sentinel or Cluster) for high availability
- Kubernetes deployment with horizontal pod autoscaling
- Separate compute tiers (CPU for text extraction, GPU for OCR/LLM)
- Object storage (S3/GCS) for document storage with CDN

**Operational improvements:**
- Dead Letter Queues with exponential backoff for failed chunks
- Circuit breakers for downstream service failures
- Rate limiting for embedding API and LLM
- A/B testing framework for chunking strategies
- Audit logging for compliance

### Current Limitations

- **Scale**: Designed for <50K chunks (~5K documents). Single-node deployment.
- **Updates**: No incremental updates — modifying a document requires re-ingestion.
- **Embedding model**: Changing models requires re-embedding the entire corpus.
- **Query complexity**: No cross-document reasoning, query rewriting, or result re-ranking.
- **Language support**: Optimized for English and Latin-script languages.
- **Admin UI**: Chat interface only; no admin dashboard for monitoring.
