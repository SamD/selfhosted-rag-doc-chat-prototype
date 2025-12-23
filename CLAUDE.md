# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a self-hosted RAG (Retrieval-Augmented Generation) system for chatting with PDF and HTML documents using local LLMs. The project was built as a learning exercise emphasizing transparency over abstraction - it intentionally avoids orchestration frameworks like LangChain to provide direct control over each pipeline component.

### Key Technologies
- **Embedding Model**: e5-large-v2 (HuggingFace)
- **LLM**: Meta-Llama-3.1-8B-Instruct via llama-cpp (GGUF format)
- **Vector Store**: ChromaDB
- **Message Queue**: Redis
- **Metadata Store**: DuckDB + Parquet
- **OCR**: Tesseract/EasyOCR for scanned PDFs
- **Frontend**: Astro + Tailwind CSS

## Architecture

The system is divided into three main phases:

### 1. Ingestion Pipeline
A distributed document processing pipeline with three worker types:
- **Producer** (`doc-ingest-chat/workers/producer_worker.py`): Scans `INGEST_FOLDER`, extracts and tokenizes text from PDFs/HTML using pdfplumber. Chunks text using token-aware splitting based on `e5-large-v2` tokenizer. Pushes chunks to Redis queues with file-level atomicity.
- **Consumer** (`doc-ingest-chat/workers/consumer_worker.py`): Pulls chunks from Redis queues, batches them (controlled by `MAX_CHROMA_BATCH_SIZE` and `MAX_CHROMA_BATCH_SIZE_LIMIT`), generates embeddings via e5-large-v2, and stores vectors in ChromaDB and metadata in DuckDB.
- **OCR Worker** (`doc-ingest-chat/workers/ocr_worker.py`): Handles fallback OCR processing for scanned/image-based PDF pages that pdfplumber cannot parse.

Redis queues (`chunk_ingest_queue:0-3` and `ocr_processing_job`) coordinate workers with backpressure and retry logic. The consumer maintains file-level transaction guarantees - all chunks from a file are committed atomically.

### 2. Retrieval
At query time, the system embeds the user query using e5-large-v2 and retrieves top-k semantically similar chunks from ChromaDB using cosine similarity (k controlled by `RETRIEVER_TOP_K`).

### 3. Generation
Retrieved chunks are formatted as context and passed to the local llama-cpp LLM which generates a conversational response. The LLM is NOT used during ingestion or retrieval - only for final answer generation.

## Development Commands

### Environment Setup
Required environment variables must be set before running (see `doc-ingest-chat/run-compose.sh` for examples):
```bash
export LLAMA_MODEL_PATH=/absolute/path/to/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf
export E5_MODEL_PATH=/absolute/path/to/e5-large-v2
export INGEST_FOLDER=/absolute/path/to/your/docs
export CHROMA_DATA_DIR=${INGEST_FOLDER}/chroma_db  # Optional, defaults to INGEST_FOLDER
```

All other configuration is in `doc-ingest-chat/ingest-svc.env`.

### Running the System

**Start full stack with GPU support:**
```bash
./doc-ingest-chat/run-compose.sh
```

**Start with CPU-only mode (slower):**
```bash
./doc-ingest-chat/run-compose-cpu.sh
```

The script validates model paths and launches Docker Compose with profiles (cuda/cpu and with-frontend).

**Services will be available at:**
- Ingestion API: `http://localhost:8000`
- Frontend: `http://localhost:4321`
- ChromaDB: `http://localhost:9001`
- Redis: `localhost:6380`

### Frontend Development
```bash
cd astro-frontend
npm run dev        # Start dev server with hot reload
npm run build      # Build for production
npm run preview    # Preview production build
```

### Testing

**Run all tests:**
```bash
pytest
```

**Run specific test file:**
```bash
pytest tests/test_document_processor.py
pytest tests/test_producer_worker.py
pytest tests/test_text_utils.py
```

**Run with verbose output:**
```bash
pytest -v
```

The test environment is configured in `conftest.py` which sets up temporary directories and points to local model paths if available.

### Linting

**Run ruff checks:**
```bash
ruff check .
```

**Run ruff with auto-fix:**
```bash
ruff check --fix .
```

Configuration is in `ruff.toml` (line length: 270, selects: E, F, I).

### Debugging

**Inspect Redis queues:**
```bash
redis-cli -p 6380
> LRANGE chunk_ingest_queue:0 0 -1
> LRANGE chunk_ingest_queue:1 0 -1
> LRANGE ocr_processing_job 0 -1
```

**Query DuckDB for ingested chunks:**
```bash
duckdb /path/to/chunks.duckdb
> SELECT source_file, COUNT(*) FROM chunks GROUP BY source_file;
> SELECT * FROM chunks WHERE text ILIKE '%search_term%';
> SELECT engine, COUNT(*) FROM chunks GROUP BY engine;
```

**Check Docker logs:**
```bash
docker logs consumer_worker_gpu
docker logs producer_worker
docker logs ocr_worker
```

## Configuration Notes

### Performance Tuning
Key environment variables in `ingest-svc.env`:
- `RETRIEVER_TOP_K`: Number of chunks to retrieve (default: 5 for demo, 20 for accuracy)
- `LLAMA_MAX_TOKENS`: Max tokens LLM can generate (256 for fast demo, 4096 for detailed)
- `LLAMA_N_GPU_LAYERS`: GPU layer offloading (35 for RTX 4070)
- `MAX_CHROMA_BATCH_SIZE`: Chunks per batch sent to ChromaDB (75)
- `MAX_CHROMA_BATCH_SIZE_LIMIT`: Max total tokens per batch (5461)

### Latin Script Support
Recent addition (see branch `feature/add-latin-support-ruff-checks`):
- `ALLOW_LATIN_EXTENDED=true`: Enable Latin extended character processing
- `LATIN_SCRIPT_MIN_RATIO=0.7`: Minimum ratio of Latin chars for text acceptance

## Key Implementation Details

- **Token-aware chunking**: Uses `transformers.AutoTokenizer` for the embedding model to ensure chunks respect token boundaries
- **File-level atomicity**: Consumer commits all chunks from a file in a transaction or none at all
- **Dual storage**: DuckDB as primary queryable store, Parquet for append-only archival
- **OCR fallback**: Automatically triggered when pdfplumber fails to extract text from a page
- **No framework abstraction**: Direct use of Redis, HuggingFace, ChromaDB, llama-cpp for full transparency
- **Conversation history**: Frontend maintains session-based chat history for context-aware follow-ups

## Output Artifacts

After ingestion, the following files appear in `INGEST_FOLDER`:
- `chunks.duckdb`: Primary queryable database of all chunks with metadata
- `chunks.parquet`: Append-only archive regenerated from DuckDB
- `ingested_files.txt`: List of successfully processed files
- `failed_files.txt`: List of files that failed processing
- `producer_failed_chunks.json`: Diagnostic log of chunks that failed in producer
- `consumer_failed_chunks.json`: Diagnostic log of chunks that failed in consumer
- `chroma_db/`: ChromaDB vector store directory
