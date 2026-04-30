# Project Map: Self-Hosted RAG Pipeline

## 🏗️ Architecture Overview
This project is a distributed, multi-stage document ingestion and RAG (Retrieval-Augmented Generation) pipeline. It uses a producer-consumer pattern managed via Redis queues.

## 🛠️ Core Components

### 1. Ingestion Pipeline (`doc-ingest-chat/`)
The pipeline is divided into specialized workers that handle specific file types and processing stages:
- **Producer**: Watches source directories and injects new files into the pipeline.
- **Gatekeeper**: Orchestrates the workflow, deciding which handler should process a file.
- **Handlers**: Specialized logic for `.pdf`, `.mp4`, `.mp3`, `.txt`, `.md`, and `.html`.
- **OCR Worker**: Specialized worker for extracting text from images/scanned PDFs.
- **WhisperX Worker**: Specialized worker for transcribing media files (GPU accelerated).
- **Consumer**: Processes text chunks, generates embeddings, and stores them in a vector database (Qdrant/Chroma).

### 2. Services & Infrastructure
- **FastAPI Backend**: Provides a REST API for interacting with the pipeline.
- **Redis**: Acts as the message broker (queues) between workers.
- **Vector DB**: Qdrant (default) or Chroma for storing and searching embeddings.
- **Frontend (Astro)**: A web interface for document chat and monitoring.

## 📂 Directory Structure (Key Areas)
- `doc-ingest-chat/workers/`: Contains the logic for `producer`, `consumer`, `gatekeeper`, and `ocr`.
- `doc-ingest-chat/handlers/`: Contains the `Chain of Responsibility` handlers (PDF, MP4, etc.).
- `doc-ingest-chat/services/`: Core business logic (Database, Job management, RAG logic).
- `doc-ingest-chat/config/`: Environment and strategy settings (GPU/CPU, LLM/Embedding paths).
- `doc-ingest-chat/api/`: API route definitions.
- `astro-frontend/`: The web-based user interface.

## ⚙️ Deployment
- **Docker Compose**: Managed via `run-compose.sh` with support for different profiles (GPU/CPU, Qdrant/Chroma).
- **Environment Strategy**: Uses `env_strategy.py` to handle CUDA/GPU visibility and memory allocation.
