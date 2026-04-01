# GEMINI.md - Project Context & Instructions

## Project Overview
This project is a **Scalable RAG (Retrieval-Augmented Generation) Pipeline** designed for distributed document ingestion and chat. It handles mixed-quality PDFs (scanned + digital), HTML, and large document collections with automatic quality detection and OCR fallback.

## Agent Instructions
- **Subagent Delegation**: 
    - Use `@researcher` for searching latest documentation for example (Astro 5.x, FastAPI, redis, qdrant, chromadb) as wella as Python and HuggingFace AI and other supporting libraries
    - Use `@analyzer` for deep code analysis or refactoring tasks.
- **Execution**: When suggesting shell commands, prioritize the scripts in `doc-ingest-chat/`. Always include `PYTHONPATH=doc-ingest-chat` for test execution.

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
