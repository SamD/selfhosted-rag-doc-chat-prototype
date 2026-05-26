# Self-Hosted RAG Pipeline

A self-hosted retrieval augmented generation pipeline that ingests PDFs, videos, and audio files and serves them through a chat interface with clickable source citations. The system runs entirely on local hardware without external dependencies. Files are routed by type through a chain of content handlers. PDFs are processed with pdfplumber and automatic OCR fallback handles scanned pages. Media files are transcribed with WhisperX. Raw text formats are read directly. A supervisor LLM normalizes all extracted text to Markdown with page anchors. The producer splits the normalized content on header boundaries, generates embeddings, and stores the resulting vector chunks in Qdrant.

Compute workloads are separated into individual HTTP services so embedding generation, LLM inference, WhisperX transcription, and OCR can each run on different hosts. Redis-backed queues with backpressure coordinate ingestion across all services. Ingestion and querying run concurrently so already processed documents are available in chat while others continue ingesting. Hardware can range from an N100 minipc to a split inference setup with a Ryzen 7 7840HS and an RTX 3060 connected over OCuLink in one Proxmox LXC with the 780M iGPU in another. The chat UI retrieves relevant chunks from Qdrant and returns LLM responses with citations linking back to the source page.

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

- **Multi-format ingestion**: Drop PDFs, HTML, Markdown, MP3, MP4, WAV files into a staging folder. Each file type is routed to the correct handler automatically.
- **Markdown-first normalization**: All raw text is normalized into clean, structured Markdown with page anchors before chunking. Noisy footers, page numbers, and OCR artifacts are stripped.
- **Zero-drop chunking**: Hierarchical splitting preserves document structure. Oversized chunks are sub-split rather than truncated. Deterministic IDs via MurmurHash3 prevent duplication on re-ingestion.
- **Semantic search**: Embeddings via mxbai or e5 models, stored in Qdrant (or Chroma). Asymmetric search with `query:` and `passage:` prefixes for accurate retrieval.
- **RAG chat with citations**: Query your documents via a chat UI. Every answer sentence is traced back to its source with clickable citations linking to the original file and page.
- **Distributed, air-gapped**: Every service runs on dedicated LAN hosts. No internet dependency. HuggingFace offline mode baked into all containers.

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

Four environment variables are required. Each accepts either a local path (model loaded in-container) or a remote `http(s)://` URL pointing to a service on your LAN:

```bash
export DEFAULT_DOC_INGEST_ROOT=/path/to/docs
export LLM_PATH=http://<llm-host>:11434/v1/chat/completions
export SUPERVISOR_LLM_PATH=http://<llm-host>:11434/v1/chat/completions
export EMBEDDING_MODEL_PATH=http://<embedding-host>:11434/v1/embeddings
```

All service endpoints, optional variables, and local-file alternatives are documented in [docs/quickstart.md](docs/quickstart.md).

### Launch

```bash
./doc-ingest-chat/run-compose.sh --build
```

### Use

Drop files into `$DEFAULT_DOC_INGEST_ROOT/staging/`. The system auto-detects and processes them. Open [http://localhost:4321](http://localhost:4321) to chat with your documents.
