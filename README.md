# Self-Hosted RAG Pipeline

**A distributed document processing and RAG chat system built for commodity hardware. Every component — LLMs, embeddings, Whisper transcription, OCR, vector database — runs as a standalone service on a separate minipc or container, communicating over the local network. Fully air-gapped, zero internet dependency.**

![Self Hosted Rag Doc Pipeline](./docs/selfhosted-rag-doc-ingest.gif)

---

## Architecture at a Glance

```
staging/ → Gatekeeper (extract + normalize to Markdown) → Producer (chunk + enqueue)
                                                         → Consumer (embed + store in Qdrant)
                                                         → success/
```

![Flow](./docs/arch.png)

All heavy models run as remote API endpoints. Workers connect to them over HTTP. Every file is tracked through a DuckDB-backed state machine with atomic per-file handoffs, ensuring zero partial ingestions and crash-safe recovery.

---

## Quick Start

### 1. Prerequisites

**System dependencies** (baked into the Docker image):
- `ffmpeg` — audio extraction for WhisperX
- `poppler-utils` — PDF page rendering
- `libgl1` / `libglib2.0-0` — OpenCV/vision processing

**Docker** (v2.20+) — for the containerized worker stack
**Node.js v22.12.0+** — for frontend development

### 2. Model Services (All Remote, Any LAN Host)

Every AI workload runs as an HTTP(S) service on a dedicated host. The system auto-detects local-vs-remote from the URL scheme.

| Service | Env Var | Remote Endpoint Format | Example Host |
|---------|---------|----------------------|-------------|
| RAG LLM | `LLM_PATH` | `http://<host>:<port>/v1/chat/completions` | lxc-nvidia (GPU) |
| Supervisor LLM | `SUPERVISOR_LLM_PATH` | `http://<host>:<port>/v1/chat/completions` | lxc-nvidia (GPU) |
| Embedding | `EMBEDDING_MODEL_PATH` | `http://<host>:<port>/v1/embeddings` | lxc-bee2 |
| WhisperX | `WHISPER_MODEL_PATH` | `http://<host>:<port>/inference` | lxc-amd |
| OCR (docling-serve) | `OCR_PATH` | `http://<host>:<port>/v1/convert/file` | lxc-bee3 |
| Vector DB (Qdrant) | `VECTOR_DB_URL` | `http://<host>:<port>` (gRPC or REST) | lxc-bee1 |

For local testing, any path can point to a local directory or GGUF file instead — the system detects the format automatically.

### 3. Configure Environment

Example configuration with all services on separate LAN hosts:

```bash
# REQUIRED — root directory for all lifecycle stages (staging, preprocessing, success, etc.)
export DEFAULT_DOC_INGEST_ROOT=/home/user/rag-docs

# Qdrant — gRPC on port 6334 (REST also available on 6333)
export VECTOR_DB_PROFILE=qdrant
export VECTOR_DB_URL=http://192.168.30.68:6334
export VECTOR_DB_USE_GRPC=true

# LLMs — llama-server or any OpenAI-compatible API on GPU host
export LLM_PATH=http://192.168.30.60:11434/v1/chat/completions
export SUPERVISOR_LLM_PATH=http://192.168.30.60:11434/v1/chat/completions

# Embedding — remote embeddings API
export EMBEDDING_MODEL_PATH=http://192.168.30.66:11434/v1/embeddings

# WhisperX — transcription service
export WHISPER_MODEL_PATH=http://192.168.30.70:1145/inference

# OCR — docling-serve (set to LOCAL for inline Docling)
export OCR_PATH=http://192.168.30.69:5001/v1/convert/file

# Force OCR on all PDF pages (skip pdfplumber)
export PDF_FORCE_OCR=true
```

Local-only alternatives (GGUF files / local directories):
```bash
export LLM_PATH=/home/user/models/Phi-4-mini-instruct-Q6_K.gguf
export EMBEDDING_MODEL_PATH=/home/user/models/e5-large-v2
export WHISPER_MODEL_PATH=/home/user/models/whisper
export OCR_PATH=LOCAL
```

### 4. Launch

```bash
./doc-ingest-chat/run-compose.sh --build
```

The stack starts: Redis → Gatekeeper → Producer → OCR Worker → WhisperX Worker → Consumer → FastAPI backend. Workers connect to remote model services over the LAN.

### 5. Drop Files and Watch

Drop PDFs, MP3s, MP4s, HTML, or Markdown files into `$DEFAULT_DOC_INGEST_ROOT/staging/`. The system auto-detects them.

```bash
docker logs -f gatekeeper_worker   # normalization progress
docker logs -f consumer_worker     # embedding and Qdrant upsert
```

### 6. Chat

Open [http://localhost:4321](http://localhost:4321). The UI queries the FastAPI backend (port 8000), which retrieves relevant chunks from Qdrant and streams an LLM response with clickable citations linking back to source documents.

---

## Distributed Deployment

Every heavy service runs on a dedicated minipc or container:

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  lxc-nvidia  │   │   lxc-bee2   │   │   lxc-bee1   │
│  (GPU host)  │   │  (Embedding) │   │   (Qdrant)   │
│              │   │              │   │              │
│ LLM + Sup.   │   │  e5-large-v2 │   │  gRPC:6334   │
│ :11434/v1    │   │  :11434/v1   │   │  REST:6333   │
└──────────────┘   └──────────────┘   └──────────────┘

┌──────────────┐   ┌──────────────┐
│   lxc-amd    │   │  lxc-bee3    │
│  (WhisperX)  │   │   (OCR)      │
│              │   │              │
│ :1145/inf.   │   │ docling-serve│
│              │   │ :5001/v1     │
└──────────────┘   └──────────────┘
```

The ingestion worker stack runs on the coordinator host — all heavy compute is offloaded to these services. This allows each minipc to be optimized for its specific workload (GPU for LLM, CPU+RAM for WhisperX, etc.).

### Air-Gapped / Offline by Design

- `HF_HUB_OFFLINE=1` is baked into all Docker images — HuggingFace libraries never phone home
- Docling OCR models are cached during Docker build-time warmup — no runtime downloads
- All service URLs are LAN addresses only
- Worker images are self-contained; no external API calls are made during ingestion or query
- The entire system operates on a private network with no internet access required

---

## Hardware Profile

Each minipc is specced for its workload:

| Host | Role | Key Hardware |
|------|------|-------------|
| lxc-nvidia | LLM inference (Phi-4-mini) | NVIDIA GPU via eGPU OCuLink dock |
| lxc-bee2 | Embedding (e5-large-v2) | Multi-core CPU, 16GB+ RAM |
| lxc-amd | WhisperX transcription | Multi-core CPU, 16GB+ RAM |
| lxc-bee3 | Docling OCR | Multi-core CPU, 8GB+ RAM |
| lxc-bee1 | Qdrant vector DB | NVMe SSD, 8GB+ RAM |
| Coordinator | Worker stack (ingestion) | Multi-core CPU, 16GB+ RAM |

All services scale horizontally — add more embedding workers behind a load balancer, or more LLM instances for higher query throughput.

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
