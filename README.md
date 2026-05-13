# Self-Hosted RAG Pipeline

**A transparent, air-gapped document processing system handling mixed-quality PDFs (scanned + digital), HTML, MP3/MP4 audio/video, and large document collections with automatic OCR fallback. Built for commodity hardware — minipcs, eGPU docks, and fully offline environments.**

![Self Hosted Rag Doc Pipeline](./docs/selfhosted-rag-doc-ingest.gif)

---

## Architecture at a Glance

```
staging/ → Gatekeeper (extract + normalize to Markdown) → Producer (chunk + enqueue)
                                                         → Consumer (embed + store in Qdrant)
                                                         → success/
```

![Flow](./docs/arch.png)

Every file is tracked through a DuckDB-backed state machine with atomic per-file handoffs, ensuring zero partial ingestions and crash-safe recovery.

---

## Quick Start

### 1. Prerequisites

**System dependencies** (baked into the Docker image, install on host if running workers locally):
- `ffmpeg` — audio extraction for WhisperX
- `poppler-utils` — PDF page rendering
- `libgl1` / `libglib2.0-0` — OpenCV/vision processing

**Docker** (v2.20+) — for the containerized stack
**NVIDIA Container Toolkit** — if using GPU acceleration
**Node.js v22.12.0+** — for frontend development

### 2. Get the Models

Download to a local directory with absolute paths:

| Model | Purpose | Source |
|-------|---------|--------|
| `e5-large-v2` | Embedding (512-token context) | [HuggingFace](https://huggingface.co/intfloat/e5-large-v2) |
| `Phi-4-mini-instruct-Q6_K.gguf` | RAG Chat LLM | [HuggingFace](https://huggingface.co/microsoft/Phi-4-mini-instruct-GGUF) |
| `Phi-4-mini-instruct-Q6_K.gguf` | Supervisor LLM (normalization) | Same GGUF or a smaller variant |
| `faster-whisper-large-v2` | Audio/video transcription | `faster-whisper` model directory |
| `docling-serve` (optional) | Remote OCR endpoint | [Docling Serve](https://github.com/DS4SD/docling-serve) |

All model paths support **local absolute paths** or **HTTP(S) remote endpoints**. The system auto-detects the mode.

### 3. Configure Environment

```bash
# REQUIRED — root directory for all lifecycle stages
export DEFAULT_DOC_INGEST_ROOT=/home/user/rag-docs

# Vector Database
export VECTOR_DB_PROFILE=qdrant
export VECTOR_DB_URL=http://192.168.30.68:6333

# Embedding model
export EMBEDDING_MODEL_PATH=/home/user/models/e5-large-v2

# Dual-LLM (both can be local GGUF or remote API URLs)
export LLM_PATH=/home/user/models/Phi-4-mini-instruct-Q6_K.gguf
export SUPERVISOR_LLM_PATH=/home/user/models/Phi-4-mini-instruct-Q6_K.gguf

# Whisper (local directory or remote URL)
export WHISPER_MODEL_PATH=/home/user/models/whisper

# OCR — LOCAL for inline Docling, or remote docling-serve URL
export OCR_PATH=LOCAL

# GPU control
export LLAMA_USE_GPU=true
```

**Remote API mode** (point at llama-server, Ollama, or any OpenAI-compatible endpoint):
```bash
export LLM_PATH=http://192.168.1.50:8080/v1
export SUPERVISOR_LLM_PATH=http://192.168.1.50:8080/v1
export EMBEDDING_MODEL_PATH=http://192.168.1.50:8080/v1
export OCR_PATH=http://192.168.1.50:5001/v1/convert/file
```

### 4. Launch

```bash
# GPU mode (default)
./doc-ingest-chat/run-compose.sh --build

# CPU-only mode
./doc-ingest-chat/run-compose.sh --build --profile cpu
```

The stack starts: Redis → Qdrant → Gatekeeper → Producer → OCR Worker → WhisperX Worker → Consumer → FastAPI backend.

### 5. Drop Files and Watch

Drop PDFs, MP3s, MP4s, HTML, or Markdown files into `$DEFAULT_DOC_INGEST_ROOT/staging/`. The system auto-detects them.

```bash
# Monitor normalization progress
docker logs -f gatekeeper_worker

# Monitor embedding progress
docker logs -f consumer_worker
```

### 6. Chat

Open [http://localhost:4321](http://localhost:4321). The UI queries the FastAPI backend (port 8000), which retrieves relevant chunks from Qdrant and streams an LLM response with clickable citations linking back to source documents.

---

## Air-Gapped / Offline Deployment

This system is designed to run without internet access:

- `HF_HUB_OFFLINE=1` is baked into the Docker image — HuggingFace libraries never phone home.
- Docling OCR models are cached during Docker **build-time warmup** — no runtime downloads.
- All model paths point to local filesystems or LAN endpoints only.
- Worker images are self-contained; no external API calls are made during ingestion or query.

---

## How Ingestion Works

### Step 1 — Gatekeeper: Extract & Normalize

The Gatekeeper claims files from `staging/`. A **Chain of Responsibility** routes each file to the correct handler:

| Handler | File Types | Extraction Method |
|---------|-----------|-------------------|
| `PDFContentTypeHandler` | `.pdf` | `pdfplumber` (fast text-layer check) → OCR fallback via Docling/EasyOCR for scanned pages |
| `MP4ContentTypeHandler` | `.mp4` | Delegates to WhisperX worker via Redis |
| `MP3ContentTypeHandler` | `.mp3`, `.wav` | Delegates to WhisperX worker via Redis |
| `TextContentTypeHandler` | `.txt`, `.md`, `.html` | Direct file read |

Raw text streams from handlers are batched and sent to the **Supervisor LLM** which normalizes the messy raw extraction into clean, structured Markdown with page anchors (`### [INTERNAL_PAGE_X]`). The result is a single `.md` file.

**State transition**: `NEW → PREPROCESSING → PREPROCESSING_COMPLETE`

### Step 2 — Producer: Chunk & Enqueue

The Producer claims the normalized Markdown. It performs hierarchical splitting (H1 → H2 → INTERNAL_PAGE → H3) with a 512-token budget per chunk. Oversized chunks are sub-split rather than truncated (zero-drop policy). Each chunk receives a deterministic `[DOC_XXXX]` ID via MurmurHash3. Chunks and a `file_end` sentinel are pushed to dedicated Redis queues (1:1 queue per consumer for file-level affinity).

**State transition**: `PREPROCESSING_COMPLETE → INGESTING → CONSUMING`

### Step 3 — Consumer: Embed & Persist

The Consumer buffers incoming chunks in DuckDB (acting as a write-ahead log). When the `file_end` sentinel arrives, all chunks for that file are retrieved, embedded via `e5-large-v2`, and upserted into Qdrant as a single atomic batch. Files are then moved to `success/` and chunks committed to Parquet for archival.

**State transition**: `CONSUMING → INGEST_SUCCESS`

---

## Hardware Profile

The system is designed for commodity hardware — the reference setup:

| Component | Specification |
|-----------|--------------|
| CPU | Intel i9-14900KF (24-core) — or any modern multi-core CPU |
| GPU | NVIDIA RTX 4070 12GB — or eGPU via OCuLink dock |
| RAM | 32 GB DDR5 |
| OS | Ubuntu 22.04+ / Python 3.11 |
| Storage | NVMe SSD recommended for document throughput |

Performance on this setup: 10-15 PDFs/min (mixed quality), ~400ms query latency (p50), Qdrant tested with 10K+ chunks at sub-second retrieval.

---

## Further Reading

- **[docs/overview.md](docs/overview.md)** — Full architecture, lifecycle flow, and component map
- **[docs/deep-dive.md](docs/deep-dive.md)** — Design rationale, dual-LLM architecture, chunking strategy, production roadmap
- **[docs/operations.md](docs/operations.md)** — Debugging, DuckDB queries, Redis/Qdrant inspection, metrics
- **[CHANGELOG.md](CHANGELOG.md)** — Version history and feature tracking
