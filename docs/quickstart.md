# Quick Start Guide

**[< README](../README.md) | [Architecture](overview.md) | [Deep Dive](deep-dive.md) | [Operations](operations.md)**

---

## Prerequisites

All system dependencies (`ffmpeg`, `poppler-utils`, `libgl1`, `libglib2.0-0`) are pre-installed in the Docker image — no host installation needed.

| Requirement | Purpose |
|-------------|---------|
| **Docker v2.20+** | Containerized worker stack |
| **Node.js v22.12.0+** | Astro frontend dev server only (`npm run dev` from `astro-frontend/`) |

---

## Service Model

Every AI workload runs as an HTTP(S) service on a dedicated LAN host. Workers connect to these services as remote API clients. The system auto-detects local-vs-remote from the URL scheme:

| Mode | Example | Behavior |
|------|---------|----------|
| **Remote URL** (`http(s)://...`) | `http://llm-host:11434/v1/chat/completions` | Workers use OpenAI-compatible HTTP clients. No model loaded locally. The remote server handles its own GPU, context, and batching settings. |
| **Local path** (`/home/user/models/...`) | `/models/Phi-4-mini-instruct-Q6_K.gguf` | Model is loaded directly in the worker container via llama-cpp or HuggingFace. |

You can mix and match — some services remote, some local.

---

## Local-Only Deployment

To run everything locally without remote services, use local model paths instead of URLs:

```bash
export DEFAULT_DOC_INGEST_ROOT=/home/user/rag-docs
export LLM_PATH=/home/user/models/Phi-4-mini-instruct-Q6_K.gguf
export SUPERVISOR_LLM_ENDPOINTS=/home/user/models/Phi-4-mini-instruct-Q6_K.gguf
export EMBEDDING_ENDPOINTS=/home/user/models/e5-large-v2
export WHISPER_MODEL_ENDPOINTS=/home/user/models/whisper
export OCR_ENDPOINTS=LOCAL
export VECTOR_DB_PROFILE=qdrant
```


---

## Filesystem Requirement

| Env Var | Role | Example |
|---------|------|---------|
| `DEFAULT_DOC_INGEST_ROOT` | Local directory for all lifecycle stages (staging, preprocessing, ingestion, consuming, success, failed) | `/home/user/rag-docs` |

All subdirectories (`staging/`, `preprocessing/`, `ingestion/`, `consuming/`, `success/`) are created automatically.

---

## Mandatory Service Endpoints

These three must be set. Each accepts a remote URL or a local path.

| Env Var | Service Role | Remote Example | Local Example |
|---------|-------------|----------------|---------------|
| `LLM_PATH` | **Chat LLM** — inference model for RAG queries. Runs on a GPU host via llama-server or any OpenAI-compatible API. | `http://<llm-host>:11434/v1/chat/completions` | `/models/Phi-4-mini-instruct-Q6_K.gguf` |
| `SUPERVISOR_LLM_ENDPOINTS` | **Gatekeeper LLM** — normalization model used by the gatekeeper worker to convert raw extracted text into clean Markdown during ingestion. Often the same host as the chat LLM, may point to the same model. Supports comma-separated URLs for HAProxy load balancing. | `http://<llm-host>:11434/v1/chat/completions` | `/models/Phi-4-mini-instruct-Q6_K.gguf` |
| `EMBEDDING_ENDPOINTS` | **Embedding model** — vectorizes chunks during ingestion and queries during chat. Supports comma-separated URLs for HAProxy load balancing. | `http://<embedding-host>:11434/v1/embeddings` | `/models/e5-large-v2` |

---

## All Settings Are Overridable

Every env var in `ingest-svc.env` and `shared/defaults.py` can be overridden at runtime by exporting it before starting the stack. This includes LLM context sizes, batch sizes, timeouts, and compute settings. For example:

```bash
export SUPERVISOR_N_CTX=65536      # Override supervisor context window
export GATEKEEPER_BATCH_SIZE=15     # More pages per normalization call
export RETRIEVER_TOP_K=10          # More chunks for RAG context
```

The full reference is in `shared/env_names.py` (all variable names) and `shared/defaults.py` (all default values).

## Optional Service Endpoints

The remaining services have defaults and only need configuration when using remote hosts.

| Env Var | Service Role | Default | Remote Example |
|---------|-------------|---------|----------------|
| `VECTOR_DB_URL` | **Vector DB** — Qdrant (gRPC on port 6334, REST on 6333) or Chroma. Overrides `VECTOR_DB_HOST` and `VECTOR_DB_PORT`. | `vector-db` (Docker service name) | `http://<vector-db-host>:6334` |
| `VECTOR_DB_PROFILE` | Vector database selection | `qdrant` | `qdrant` or `chroma` |
| `VECTOR_DB_USE_GRPC` | Use gRPC for Qdrant | `true` | `true` or `false` |
| `WHISPER_MODEL_ENDPOINTS` | **WhisperX** — transcribes MP3, MP4, WAV, MOV, MKV files. When `NOT_SET`, audio/video files are skipped during ingestion. Set to a URL or local path to enable transcription. | `NOT_SET` | `http://<whisper-host>:1145/inference` |
| `OCR_ENDPOINTS` | **OCR** — docling-serve for PDF OCR fallback when pdfplumber cannot extract text. | `LOCAL` | `http://<ocr-host>:5001/v1/convert/file` |
| `PDF_FORCE_OCR` | Skip pdfplumber and use OCR for all PDF pages | `false` | `true` or `false` |


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

Every heavy service runs on a dedicated host or container. The worker stack runs on the coordinator. When multiple backends exist for a service, HAProxy load-balances across them.

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
              |  HAProxy (supervisor :11437)  |
              |  HAProxy (embedding :11438)   |
              |  HAProxy (whisper   :11439)   |
              |  HAProxy (OCR       :11440)   |
              |                               |
              |  gatekeeper + producer +      |
              |  consumer + OCR worker +      |
              |  WhisperX worker              |
              +-------------------------------+
```

The coordinator connects to all services over HTTP via HAProxy. Each remote host can be optimized for its specific workload (GPU for LLM, CPU+RAM for WhisperX, NVMe for Qdrant). When only one backend exists for a service, HAProxy proxies transparently. When multiple backends exist, HAProxy round-robins across them with health checks.

---

## Multi-Endpoint Load Balancing

When you have multiple hosts running the same service, set `*_ENDPOINTS` env vars with comma-separated URLs. HAProxy starts automatically and distributes requests across all backends with health checks and failover.

```bash
# Two GPU hosts running supervisor LLM
export SUPERVISOR_LLM_ENDPOINTS=http://gpu0:11435/v1/chat/completions,http://gpu1:11436/v1/chat/completions

# Two hosts running embedding model
export EMBEDDING_ENDPOINTS=http://gpu0:11434/v1/embeddings,http://gpu1:11434/v1/embeddings

# Two hosts running WhisperX
export WHISPER_MODEL_ENDPOINTS=http://whisper0:1145/inference,http://whisper1:1145/inference

# Two hosts running OCR
export OCR_ENDPOINTS=http://ocr0:5001/v1/convert/file,http://ocr1:5001/v1/convert/file
```

When `*_ENDPOINTS` is set, `run-compose.sh` auto-overrides the corresponding `*_PATH` to point to the HAProxy container. No manual path configuration needed.

### HAProxy Stats

Monitor load balancing at:
- Supervisor: `http://localhost:8404/stats`
- Embedding: `http://localhost:8405/stats`
- Whisper: `http://localhost:8406/stats`
- OCR: `http://localhost:8407/stats`

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

Performance on this setup: **10–15 PDFs/min** (mixed quality), **~400ms query latency** (p50), Qdrant tested with 10K+ chunks at sub-second retrieval.

---

## Launch

```bash
./doc-ingest-chat/run-compose.sh --build
```

The compose stack starts: Redis, Gatekeeper, Producer, Consumer (2x), and the FastAPI backend. OCR and WhisperX workers start when the `cuda` profile is active and connect to their configured backends (remote HTTP endpoint or local processing, depending on `OCR_ENDPOINTS` and `WHISPER_MODEL_ENDPOINTS`).

Docker Compose supports profiles:
- `--profile cuda` (default) — NVIDIA GPU acceleration
- `--profile cpu` — CPU-only mode
- `--profile qdrant` or `--profile chroma` — vector database selection

---

## Usage

1. Drop files into `$DEFAULT_DOC_INGEST_ROOT/staging/`. Supported formats: `.pdf`, `.html`, `.htm`, `.txt`, `.md`, `.mp3`, `.wav`, `.mp4`

2. Monitor progress:
   ```bash
   docker logs -f gatekeeper_worker   # normalization progress
   docker logs -f consumer_worker     # embedding and Qdrant upsert
   ```

3. Open [http://localhost:4321](http://localhost:4321) to chat with your documents.

Ingestion and querying are fully concurrent. You can start chatting with already-processed documents immediately while the system continues ingesting the rest.
