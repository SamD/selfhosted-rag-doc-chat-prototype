# Project Memory Map

## Architecture Overview
Distributed, self-hosted RAG pipeline using Redis as a message broker.

## Key Components
- **doc-ingest-chat/**: Core backend (FastAPI, Workers, Handlers, Services).
- **astro-frontend/**: Astro-based web UI.
- **Orchestration**: Docker Compose (supports GPU/CPU/VectorDB profiles).

## Workflow
1. **Ingestion**: `producer_worker` $\rightarrow$ `INGEST_QUEUE`.
2. **Routing**: `gatekeeper_worker` $\rightarrow$ `handlers/` or `ocr_worker`.
3. **Processing**: `consumer_worker` $\rightarrow$ `rag_service` $\rightarrow$ `Vector DB`.
4. **Querying**: FastAPI $\rightarrow$ Vector DB $\rightarrow$ Local LLM.
