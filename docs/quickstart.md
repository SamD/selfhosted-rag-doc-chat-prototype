# Quick Start Guide

**[< README](../README.md) | [Architecture](overview.md) | [Deep Dive](deep-dive.md) | [Operations](operations.md)**

---

## Prerequisites

All system dependencies (`ffmpeg`, `poppler-utils`, `libgl1`, `libglib2.0-0`) are pre-installed in the Docker image — no host installation needed.

**Docker v2.20+** — required for the containerized worker stack
**Node.js v22.12.0+** — required only when running the Astro frontend dev server (`npm run dev` from `astro-frontend/`) outside of Docker

---

## Service Model

Every AI workload runs as an HTTP(S) service on a dedicated LAN host. Workers connect to these services as remote API clients. The system auto-detects local-vs-remote from the URL scheme:

- **Remote URL** (`http(s)://...`): Workers use OpenAI-compatible HTTP clients. No model loaded locally. The remote server handles its own GPU, context, and batching settings.
- **Local path** (`/home/user/models/...`): The model is loaded directly in the worker container via llama-cpp or HuggingFace.

You can mix and match — some services remote, some local.

---

## Local-Only Deployment

To run everything locally without remote services, use local model paths instead of URLs:

```bash
export DEFAULT_DOC_INGEST_ROOT=/home/user/rag-docs
export LLM_PATH=/home/user/models/Phi-4-mini-instruct-Q6_K.gguf
export SUPERVISOR_LLM_PATH=/home/user/models/Phi-4-mini-instruct-Q6_K.gguf
export EMBEDDING_MODEL_PATH=/home/user/models/e5-large-v2
export WHISPER_MODEL_PATH=/home/user/models/whisper
export OCR_PATH=LOCAL
export VECTOR_DB_PROFILE=qdrant
export LLAMA_USE_GPU=true
```

---

## Filesystem Requirement

| Env Var | Role | Example |
|---------|------|---------|
| `DEFAULT_DOC_INGEST_ROOT` | Local directory for all lifecycle stages (staging, preprocessing, ingestion, consuming, success, failed) | `export DEFAULT_DOC_INGEST_ROOT=/home/user/rag-docs` |

---

## Mandatory Service Endpoints

These three must be set. Each accepts a remote URL or a local path.

| Env Var | Service Role | Example (Remote) | Example (Local) |
|---------|-------------|-----------------|-----------------|
| `LLM_PATH` | **Chat LLM** — the inference model used for RAG queries. Runs on a GPU host via llama-server or any OpenAI-compatible API. | `http://<llm-host>:11434/v1/chat/completions` | `/models/Phi-4-mini-instruct-Q6_K.gguf` |
| `SUPERVISOR_LLM_PATH` | **Gatekeeper LLM** — the normalization model used by the gatekeeper worker to convert raw extracted text into clean Markdown during ingestion. Often the same host as the chat LLM. May point to the same model as `LLM_PATH`. | `http://<llm-host>:11434/v1/chat/completions` | `/models/Phi-4-mini-instruct-Q6_K.gguf` |
| `EMBEDDING_MODEL_PATH` | **Embedding model** — vectorizes chunks during ingestion and queries during chat. | `http://<embedding-host>:11434/v1/embeddings` | `/models/e5-large-v2` |

---

## Optional Service Endpoints

The remaining services have defaults and only need configuration when using remote hosts.

| Env Var | Service Role | Default | Example (Remote) |
|---------|-------------|---------|-----------------|
| `VECTOR_DB_URL` | **Vector DB** — Qdrant (gRPC on port 6334, REST on 6333) or Chroma. Overrides `VECTOR_DB_HOST` and `VECTOR_DB_PORT`. | `vector-db` (Docker service name) | `http://<vector-db-host>:6334` |
| `VECTOR_DB_PROFILE` | Vector database selection | `qdrant` | `qdrant` or `chroma` |
| `VECTOR_DB_USE_GRPC` | Use gRPC for Qdrant | `true` | `true` or `false` |
| `WHISPER_MODEL_PATH` | **WhisperX** — transcribes MP3, MP4, WAV, MOV, MKV files. When `NOT_SET` (default), audio and video files are skipped during ingestion. Set to a URL or local path to enable transcription. | `NOT_SET` | `http://<whisper-host>:1145/inference` |
| `OCR_PATH` | **OCR** — docling-serve for PDF OCR fallback when pdfplumber cannot extract text. | `LOCAL` | `http://<ocr-host>:5001/v1/convert/file` |
| `PDF_FORCE_OCR` | Skip pdfplumber and use OCR for all PDF pages | `false` | `true` |
| `LLAMA_USE_GPU` | Enable GPU acceleration for locally-loaded models | `true` | `true` or `false` |

---

## Worker Stack

The ingestion workers run on the coordinator host. They handle document lifecycle state transitions and coordinate with the remote services.

| Worker | Entry Point | Role |
|--------|------------|------|
| **Gatekeeper** | `run_gatekeeper.py` | Claims files from staging, extracts raw text via content handlers, normalizes to Markdown via the supervisor LLM, writes `.md` output |
| **Producer** | `run_producer.py` | Claims normalized Markdown, splits into chunks with `[DOC_XXXX]` IDs, enqueues to Redis consumer queues |
| **Consumer** | `run_consumer.py` | Buffers chunks in DuckDB, embeds via the embedding service, batch-upserts to Qdrant, archives to Parquet |
| **OCR Worker** | `run_ocr_worker.py` | Processes image-based PDF pages via docling-serve |
| **WhisperX Worker** | `run_whisperx_worker.py` | Transcribes audio/video files |

---

## Distributed Deployment

Every heavy service runs on a dedicated host or container. The worker stack runs on the coordinator.

```
+-------------------+   +-------------------+   +-------------------+
|    <llm-host>     |   |  <embedding-host> |   |  <vector-db-host> |
|    (GPU host)     |   |                   |   |                   |
|                   |   |  embedding model  |   |  Qdrant           |
| chat LLM +        |   |  :11434/v1        |   |  gRPC:6334        |
| gatekeeper LLM    |   |                   |   |  REST:6333        |
| :11434/v1         |   |                   |   |                   |
+-------------------+   +-------------------+   +-------------------+

+-------------------+   +-------------------+
|   <whisper-host>  |   |    <ocr-host>     |
|                   |   |                   |
|  WhisperX         |   |  docling-serve    |
|  :1145/inference  |   |  :5001/v1         |
+-------------------+   +-------------------+

              +-------------------------------+
              |       Coordinator Host        |
              |                               |
              |  gatekeeper + producer +      |
              |  consumer + OCR worker +      |
              |  WhisperX worker              |
              +-------------------------------+
```

The coordinator connects to all services over HTTP. Each remote host can be optimized for its specific workload (GPU for LLM, CPU+RAM for WhisperX, NVMe for Qdrant).

---

## Offline Operation

- `HF_HUB_OFFLINE=1` is baked into all Docker images — HuggingFace libraries never phone home
- Docling OCR models are cached during Docker build-time warmup — no runtime downloads
- All service URLs are LAN addresses only
- Worker images are self-contained; no external API calls during ingestion or query
- The entire system operates on a private network with no internet access required

---

## Hardware Profile

Reference specs for a coordinator host running the worker stack:

| Component | Specification |
|-----------|--------------|
| CPU | Intel i9-14900KF (24-core) — or any modern multi-core CPU |
| GPU | NVIDIA RTX 4070 12GB — or eGPU via OCuLink dock |
| RAM | 32 GB DDR5 |
| OS | Ubuntu 22.04+ / Python 3.11 |
| Storage | NVMe SSD recommended for document throughput |

Performance on this setup: 10-15 PDFs/min (mixed quality), ~400ms query latency (p50), Qdrant tested with 10K+ chunks at sub-second retrieval.

---

## Launch

```bash
./doc-ingest-chat/run-compose.sh --build
```

The compose stack starts: Redis, Gatekeeper, Producer, Consumer (2x), and the FastAPI backend. OCR and WhisperX workers start alongside and connect to their configured backends (remote HTTP endpoint or local processing, depending on `OCR_PATH` and `WHISPER_MODEL_PATH`).

Docker Compose supports profiles:
- `--profile gpu` (default) — NVIDIA GPU acceleration
- `--profile cpu` — CPU-only mode
- `--profile qdrant` or `--profile chroma` — vector database selection

---

## Usage

1. Drop files into `$DEFAULT_DOC_INGEST_ROOT/staging/`. Supported formats: `.pdf`, `.html`, `.htm`, `.txt`, `.md`, `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.mp4`, `.mov`, `.mkv`

2. Monitor progress:

```bash
docker logs -f gatekeeper_worker   # normalization progress
docker logs -f consumer_worker     # embedding and Qdrant upsert
```

3. Open [http://localhost:4321](http://localhost:4321) to chat with your documents.

Ingestion and querying are fully concurrent. You can start chatting with already-processed documents immediately while the system continues ingesting the rest.
