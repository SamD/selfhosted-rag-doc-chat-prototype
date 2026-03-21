# GEMINI.md - Project Context & Instructions

## Project Overview
This project is a **Scalable RAG (Retrieval-Augmented Generation) Pipeline** designed for distributed document ingestion and chat. It handles mixed-quality PDFs (scanned + digital), HTML, and large document collections with automatic quality detection and OCR fallback.

### Key Technologies
- **Backend (Python)**: FastAPI, Redis (Queue/Coordination), Qdrant/ChromaDB (Vector DBs), Llama-cpp-python (LLM Inference), Transformers (Tokenization), DuckDB/Parquet (Storage/Analytics).
- **Frontend (Astro)**: Astro, TypeScript, Tailwind CSS.
- **OCR/Processing**: Tesseract OCR, PDFPlumber.
- **Infrastructure**: Docker Compose, Redis (port 6380).

### Architecture
1. **Producer Worker**: Extracts text from documents, detects quality, and enqueues chunks into Redis.
2. **OCR Worker**: Handles OCR fallback for scanned pages using Tesseract.
3. **Consumer Worker**: Embeds chunks and stores them in the vector database (Qdrant or ChromaDB).
4. **API Service**: Provides endpoints for chat and status, performing RAG by retrieving context from the vector DB and generating responses via a local LLM.
5. **Astro Frontend**: A web-based chat interface.

---

## Building and Running

### Prerequisites
- **Models**:
  - Embedding Model: `e5-large-v2` (must be downloaded and pointed to by `EMBEDDING_MODEL_PATH`).
  - LLM: GGUF format (e.g., Llama-3.1-8B), pointed to by `LLM_PATH`.
- **Environment Variables**: Defined in `doc-ingest-chat/ingest-svc.env` or exported in shell.

### Launching the System
The primary way to run the system is via the provided shell scripts in `doc-ingest-chat/`:

```bash
# Start the full stack (Default: GPU/CUDA + Qdrant)
./doc-ingest-chat/run-compose.sh

# Start in CPU-only mode
./doc-ingest-chat/run-compose-cpu.sh
```

### Key Ports
- **FastAPI Backend**: `http://localhost:8000`
- **Astro Frontend**: `http://localhost:4321`
- **Redis**: `6380`
- **Qdrant**: `9002` (HTTP) / `9003` (gRPC)
- **ChromaDB**: `9001`

---

## Development Conventions

### Python Backend (`doc-ingest-chat/`)
- **Configuration**: Managed in `config/settings.py` via environment variables. Use `ingest-svc.env` for overrides.
- **Worker Pattern**: Distributed workers communicate via Redis. Check `workers/` for implementation details.
- **Logging**: Uses custom logging with emojis and structured metrics.
- **Database**: DuckDB is used as a local persistent store (`chunks.duckdb`), with Parquet as an archival format.

### Frontend (`astro-frontend/`)
- **Framework**: Astro 5.x.
- **Styling**: Tailwind CSS 4.x.
- **Scripts**:
  - `npm run dev`: Start development server.
  - `npm run build`: Build for production.

### General
- **Docker**: The system is highly containerized. Use `docker-compose` profiles (`cuda`, `cpu`, `qdrant`, `chroma`, `with-frontend`) to selectively start services.
- **Testing**: Python tests are located in `doc-ingest-chat/tests/` and the root `tests/` directory. Run via `pytest`.

---

## Key Files & Directories
- `doc-ingest-chat/apimain.py`: Entry point for the FastAPI server.
- `doc-ingest-chat/run_producer.py`, `run_consumer.py`, `run_ocr_worker.py`: Worker entry points.
- `doc-ingest-chat/ingest-dockercompose.yaml`: Main Docker Compose configuration.
- `doc-ingest-chat/config/settings.py`: Centralized configuration logic.
- `astro-frontend/src/pages/index.astro`: Main chat UI.
- `quarkus/`: Experimental/Alternative Java-based implementation (incomplete).
