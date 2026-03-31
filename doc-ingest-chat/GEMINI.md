# GEMINI.md - doc-ingest-chat Context

## Project Overview
`doc-ingest-chat` is a **Scalable RAG (Retrieval-Augmented Generation) Pipeline** designed for distributed document ingestion, quality-aware OCR, and chat capabilities. It handles diverse document types (PDF, HTML) and media (Audio/Video via Whisper) with an emphasis on high-performance local inference.

### Core Stack
- **Backend**: FastAPI (Python 3.10+)
- **Coordination**: Redis (Queue management for workers)
- **Vector DBs**: Qdrant (preferred) or ChromaDB
- **Inference**: `llama-cpp-python` (GGUF) or Ollama
- **OCR**: Tesseract OCR (with custom quality detection)
- **Transcription**: Faster-Whisper for audio/video processing
- **Persistence**: DuckDB (local metadata) and Parquet (archival storage)
- **Infrastructure**: Docker Compose with GPU/CPU and Vector DB profiles

### Architecture
1.  **Producer Worker (`run_producer.py`)**: Monitors `INGEST_FOLDER`, extracts text/images, detects quality, and enqueues chunks into Redis.
2.  **OCR Worker (`run_ocr_worker.py`)**: Specialized worker for performing Tesseract OCR on scanned pages or poor-quality text detected by the producer.
3.  **Consumer Worker (`run_consumer.py`)**: Embeds text chunks using a local embedding model (e.g., `e5-large-v2`) and stores them in the selected vector database.
4.  **API Service (`apimain.py`)**: FastAPI server providing endpoints for chat, status, and RAG-based query resolution.

---

## Building and Running

### Prerequisites
- **Python**: 3.10+ (recommend `venv` or `conda`)
- **Docker**: With NVIDIA Container Toolkit if using GPU.
- **Models**:
    - `EMBEDDING_MODEL_PATH`: Directory containing `e5-large-v2` or similar.
    - `LLM_PATH`: Path to a `.gguf` file for Llama-cpp-python.

### Key Commands
- **Launch via Docker (GPU + Qdrant)**:
  ```bash
  ./run-compose.sh
  ```
- **Launch via Docker (CPU-only)**:
  ```bash
  ./run-compose-cpu.sh
  ```
- **Local API Development**:
  ```bash
  uvicorn apimain:app --reload
  ```
- **Run Individual Workers**:
  ```bash
  python run_producer.py
  python run_ocr_worker.py
  python run_consumer.py
  ```

### Ports
- **FastAPI Backend**: `8000`
- **Astro Frontend**: `4321` (if running with `--profile with-frontend`)
- **Redis**: `6380`
- **Qdrant**: `9002` (HTTP) / `9003` (gRPC)
- **ChromaDB**: `9001`

---

## Development Conventions

### Python Backend
- **Configuration**: Managed in `config/settings.py` using `python-dotenv`. Overrides are applied via `ingest-svc.env`.
- **Environment Strategy**: Uses `config/env_strategy.py` to handle runtime environment setup.
- **Worker Logic**: Distributed workers communicate via Redis queues defined in `config/settings.py` (`ocr_processing_job`, `chunk_ingest_queue`).
- **Data Modeling**: Pydantic models in `models/data_models.py` for structured data.
- **Logging**: Uses structured logging with emoji indicators for different stages (Producer, OCR, Consumer).
- **Service Layer**: Core logic (DB, RAG, Parquet) is encapsulated in `services/`.

### Testing
Tests are located in `tests/` and use `pytest`.
```bash
pytest
```
Key tests include `test_consumer_worker.py`, `test_database_service.py`, and `test_document_processor.py`.

### Project Structure
- `api/`: API router and endpoint definitions.
- `chat/`: Core RAG and chat logic.
- `config/`: Centralized settings and environment strategies.
- `models/`: Pydantic and data schemas.
- `processors/`: Document, text, and media processing.
- `services/`: Interfaces for Vector DBs, Redis, and storage.
- `workers/`: Background worker implementation.
- `utils/`: Common helpers (metrics, file I/O, logging).
