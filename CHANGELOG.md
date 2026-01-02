# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added - Qdrant Vector Database Support

#### Major Features
- **Qdrant as Default Vector Store**: Qdrant is now the default vector database, replacing ChromaDB. Qdrant offers improved performance and scalability for production workloads.
- **Dual Vector Store Support**: The system now supports both Qdrant and ChromaDB through a unified interface, allowing users to choose their preferred vector database via the `VECTOR_DB_PROFILE` environment variable.
- **VectorStoreWrapper Abstraction**: Introduced `VectorStoreWrapper` class in `doc-ingest-chat/services/database.py` that provides a unified interface for both ChromaDB and Qdrant operations:
  - Transparent ID conversion for Qdrant (string IDs to UUIDs)
  - Unified `add_texts()`, `delete()`, `as_retriever()`, and `get_collection_count()` methods
  - Automatic metadata filter translation between ChromaDB and Qdrant formats
- **Flexible Docker Compose Profiles**: Updated `ingest-dockercompose.yaml` with profile-based deployment supporting:
  - `cuda-qdrant` / `cpu-qdrant` - GPU/CPU mode with Qdrant
  - `cuda-chroma` / `cpu-chroma` - GPU/CPU mode with ChromaDB
  - Both vector databases use the same `vector-db` network alias for seamless switching

#### Configuration Changes
- **New Environment Variables**:
  - `VECTOR_DB_PROFILE`: Set to "qdrant" (default) or "chroma" to choose vector database
  - `VECTOR_DB_HOST`: Unified host setting (defaults to "vector-db")
  - `VECTOR_DB_PORT`: Defaults to 6333 for Qdrant, 8000 for ChromaDB
  - `VECTOR_DB_COLLECTION`: Unified collection name setting
  - `VECTOR_DB_DATA_DIR`: Unified data directory path
- **Renamed Environment Variables** (backward compatible):
  - `E5_MODEL_PATH` → `EMBEDDING_MODEL_PATH`
  - `LLAMA_MODEL_PATH` → `LLM_PATH`
- **Backward Compatibility**: Old `CHROMA_*` variables still work as aliases

#### Infrastructure
- **Qdrant Service**: Added Qdrant container configuration with:
  - HTTP port 6333, gRPC port 6334
  - Persistent storage volume
  - Debug logging and telemetry disabled
  - Same network alias as ChromaDB for transparent switching
- **Service Dependencies**: Updated all workers (producer, consumer, OCR) and API services to properly depend on the selected vector database service

#### Testing
- **Comprehensive Test Suite**: Added 4 new test files with extensive coverage:
  - `test_database_service.py`: Tests for VectorStoreWrapper and database service methods
  - `test_vector_db_config.py`: Tests for vector database configuration and initialization
  - `test_vector_db_settings.py`: Tests for environment variable handling and settings
  - `test_vectorstore_wrapper.py`: Tests for wrapper class functionality

#### Documentation
- **Updated README**: Corrected environment variable names in setup instructions
- **Updated CLAUDE.md**: Added documentation for Qdrant support and configuration options

### Changed
- **Database Service Refactoring**: Split `get_db()` into `get_chromadb()` and `get_qdrant()` with a common `_get_embeddings()` helper
- **Worker Updates**: All workers (producer, consumer, OCR) updated to use the new vector database abstraction
- **Docker Compose**: Restructured to support multiple deployment profiles with different vector database backends

### Technical Details
- Qdrant client uses HTTP API via `langchain-qdrant` integration
- Vector similarity search uses the same embedding model (e5-large-v2) regardless of vector database choice
- Collection management handled automatically with fallback creation if collection doesn't exist
- UUID generation for Qdrant IDs uses UUID5 (SHA-1 based) for deterministic ID mapping from string IDs

### Migration Notes
For users upgrading from ChromaDB-only version:
1. Set `VECTOR_DB_PROFILE=chroma` in your environment to continue using ChromaDB
2. To migrate to Qdrant, set `VECTOR_DB_PROFILE=qdrant` and re-run ingestion (vector embeddings will be regenerated)
3. Update environment variables to use new names (`EMBEDDING_MODEL_PATH`, `LLM_PATH`) though old names still work