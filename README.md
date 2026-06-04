# Self-Hosted RAG Pipeline

Drop PDFs, videos, or audio files into a folder and chat with them using an AI that cites its sources — all running on your own hardware with no internet connection required.

**What you can drop in:**

| Format | How it's processed |
|--------|-------------------|
| `.pdf` (scanned) | OCR via Docling/EasyOCR |
| `.pdf` (text layer) | Direct extraction via pdfplumber |
| `.mp4`, `.mov`, `.mkv` | WhisperX speech transcription |
| `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac` | WhisperX audio transcription |
| `.txt`, `.md`, `.html` | Direct read with charset detection |

**What you get out:** A chat interface where every answer sentence is traced back to its source with clickable citations linking to the original file and page.

Everything runs locally on dedicated LAN hosts — no cloud services, no external API calls, no data leaves your network.

**Technologies**: Python, llama-cpp, FastAPI, Redis, DuckDB, Qdrant/Chroma, WhisperX, Docling, Docker Compose, Astro, Tailwind CSS

![Self Hosted Rag Doc Pipeline](./docs/selfhosted-rag-doc-ingest.gif)

---

## Contents

- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [Documentation](#documentation)
- [Quick Start](#quick-start)

---

## What It Does

- **Multi-format ingestion**: Drop PDFs, HTML, Markdown, MP3, MP4, and WAV files into a staging folder for automatic processing.
- **Markdown-first normalization**: All raw text is normalized into clean, structured Markdown with page anchors before chunking. Noisy footers, page numbers, and OCR artifacts are stripped.
- **Zero-drop chunking**: Hierarchical splitting preserves document structure. Oversized chunks are sub-split rather than truncated. Deterministic IDs via MurmurHash3 prevent duplication on re-ingestion.
- **Semantic search**: Embeddings via e5 models, stored in Qdrant (or Chroma). Asymmetric search with `query:` and `passage:` prefixes for accurate retrieval.
- **RAG chat with citations**: Query your documents via a chat UI. Every answer sentence is traced back to its source with clickable citations linking to the original file and page.
- **Air-gapped by design**: Every service runs on dedicated LAN hosts with no internet access required. HuggingFace offline mode baked into all containers.

---

## How It Works

```
staging/ → Gatekeeper (extract + normalize to Markdown) → Producer (chunk + enqueue)
                                                          → Consumer (embed + store in Qdrant)
                                                          → success/
```

![Flow](./docs/arch.png)

1. **Gatekeeper**: A chain of responsibility routes each file to the correct handler. PDFs get pdfplumber extraction with automatic OCR fallback for scanned pages. Media files get WhisperX transcription. Raw text from all sources is normalized to clean Markdown by a supervisor LLM.

2. **Producer**: Normalized Markdown is split into 512-token chunks using hierarchical header boundaries. Each chunk gets a deterministic `[DOC_XXXX]` ID and is enqueued to Redis.

3. **Consumer**: Chunks are staged in DuckDB for safety, then embedded and batch-upserted to Qdrant atomically per file. Files move to `success/` on completion.

4. **Chat**: The FastAPI backend retrieves relevant chunks from Qdrant, formats context with citation tags, and streams an LLM response with clickable source links.

Ingestion and querying are fully concurrent. You can chat with already-processed documents while thousands more are being ingested.

---

## Documentation

| | | |
|---|---|---|
| **[Quick Start](docs/quickstart.md)** | Full environment configuration, service setup, and deployment diagram |
| **[Architecture](docs/overview.md)** | System flow, component map, state machine, and Redis queue architecture |
| **[Deep Dive](docs/deep-dive.md)** | Design rationale, dual-LLM architecture, chunking strategy, production roadmap |
| **[Operations](docs/operations.md)** | Debugging queries for DuckDB, Redis, and Qdrant. Metrics, schema evolution, war room scenarios |
| **[Edge Agent](docs/edge-agent.md)** | Deploy MQTT SRE agents on standalone Debian minipcs for telemetry and task execution |
| **[Changelog](CHANGELOG.md)** | Version history and feature tracking |

---

## Quick Start

### Prerequisites

- Docker v2.20+
- Node.js v22.12.0+ (frontend only)

### Configure

All services connect over HTTP. Run them on any hosts in your LAN and set these variables before starting the stack.

```bash
# --- Filesystem ---
# Root directory for the local ingestion pipeline. Lifecycle subdirectories
# (staging/, preprocessing/, ingestion/, consuming/, success/) are created here.
export DEFAULT_DOC_INGEST_ROOT=/path/to/docs

# --- Vector Database ---
# Qdrant (default) or Chroma.
export VECTOR_DB_PROFILE=qdrant
# Remote Qdrant host. The system connects over gRPC on port 6334 by default.
# To use the REST API on port 6333 instead, set VECTOR_DB_USE_GRPC=false and prefix the URL with http://.
export VECTOR_DB_URL=<vector-db-host>:6334
export VECTOR_DB_USE_GRPC=true

# --- LLMs ---
# Chat model that answers RAG queries. Remote http(s):// URL or local .gguf path.
export LLM_PATH=http://<llm-host>:11434/v1/chat/completions
# Gatekeeper model that normalizes raw extracted text to Markdown during ingestion.
# Often points to the same host as LLM_PATH but serves a distinct role.
export SUPERVISOR_LLM_PATH=http://<llm-host>:11434/v1/chat/completions

# --- Embedding ---
# Model that vectorizes document chunks during ingestion and user queries during chat.
# Remote http(s):// URL or local model directory path.
export EMBEDDING_MODEL_PATH=http://<embedding-host>:11434/v1/embeddings

# --- Audio / Video Transcription ---
# WhisperX host. Transcribes audio files (MP3, WAV, etc.) and extracts speech
# from video files (MP4, MOV, MKV) during ingestion.
export WHISPER_MODEL_PATH=http://<whisper-host>:1145/inference

# --- OCR Fallback ---
# docling-serve host. Used when pdfplumber cannot extract text from a page
# (scanned documents, image-heavy pages). Set to "LOCAL" to run Docling
# inside the container instead.
export OCR_PATH=http://<ocr-host>:5001/v1/convert/file
```

Full variable reference, local-file alternatives, and optional tuning flags are in [docs/quickstart.md](docs/quickstart.md).

### Multi-Endpoint Load Balancing

When you have multiple hosts running the same service, set `*_ENDPOINTS` env vars with comma-separated URLs. HAProxy starts automatically and distributes requests across all backends with health checks and failover.

```bash
export SUPERVISOR_LLM_ENDPOINTS=http://gpu0:11435/v1/chat/completions,http://gpu1:11436/v1/chat/completions
export EMBEDDING_ENDPOINTS=http://gpu0:11434/v1/embeddings,http://gpu1:11434/v1/embeddings
```

See [docs/quickstart.md](docs/quickstart.md#multi-endpoint-load-balancing) for full details.

### Launch

```bash
./doc-ingest-chat/run-compose.sh --build
```

### Use

Drop files into `$DEFAULT_DOC_INGEST_ROOT/staging/`. The system auto-detects and processes them. Open [http://localhost:4321](http://localhost:4321) to chat with your documents.

---
