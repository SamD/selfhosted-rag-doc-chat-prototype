## ADDED Requirements

### Requirement: Dual vector store support

The system SHALL support both Qdrant and Chroma as vector database backends. The active profile SHALL be selected via the VECTOR_DB_PROFILE environment variable (default: "qdrant"). The system SHALL expose a unified VectorStoreWrapper interface that abstracts over the underlying vector store implementation.

#### Scenario: Qdrant profile selected
- **WHEN** VECTOR_DB_PROFILE is "qdrant" or unset
- **THEN** the system SHALL use QdrantVectorStore and QdrantClient for all vector operations

#### Scenario: Chroma profile selected
- **WHEN** VECTOR_DB_PROFILE is "chroma"
- **THEN** the system SHALL use langchain_chroma.Chroma for all vector operations

### Requirement: Qdrant client configuration

When using Qdrant, the system SHALL support both gRPC and REST connections. The connection URL, gRPC usage, and gRPC port SHALL be configurable via VECTOR_DB_URL, VECTOR_DB_USE_GRPC, and VECTOR_DB_GRPC_PORT environment variables. If VECTOR_DB_URL is not set, the system SHALL fall back to VECTOR_DB_HOST and VECTOR_DB_PORT.

#### Scenario: Qdrant gRPC connection
- **WHEN** VECTOR_DB_USE_GRPC is "true"
- **THEN** the Qdrant client SHALL connect using gRPC on the configured gRPC port

#### Scenario: Qdrant URL override
- **WHEN** VECTOR_DB_URL is set to a full URL
- **THEN** the system SHALL use that URL directly instead of constructing from VECTOR_DB_HOST and VECTOR_DB_PORT

### Requirement: Embedding generation

The system SHALL use the e5-large-v2 embedding model. The model path SHALL be configured via EMBEDDING_ENDPOINTS. The system SHALL support both local HuggingFaceEmbeddings (from local model files) and remote OpenAI-compatible endpoints. Embeddings SHALL be cached per-process using fork-safe singleton caching.

#### Scenario: Local embedding model
- **WHEN** EMBEDDING_ENDPOINTS points to a local directory path
- **THEN** the system SHALL load HuggingFaceEmbeddings from that path

#### Scenario: Remote embedding endpoint
- **WHEN** EMBEDDING_ENDPOINTS is an HTTP(S) URL
- **THEN** the system SHALL use RemoteEmbeddings to call the OpenAI-compatible API

#### Scenario: Fork-safe singleton caching
- **WHEN** a new process is forked
- **THEN** the embedding singleton cache SHALL be re-initialized (checked via PID comparison)

### Requirement: Parquet archival

The system SHALL archive all ingested chunks to Parquet files after successful vector DB upsert. Each Parquet file SHALL contain the chunk content, metadata, and embedding. The archival SHALL be managed by ParquetService.

#### Scenario: Successful chunk archival
- **WHEN** chunks are successfully upserted to the vector DB
- **THEN** the chunks and their metadata SHALL be written to a Parquet file for long-term archival

### Requirement: Consumer chunk staging

The Consumer SHALL stage incoming chunks in DuckDB via Pandas batch insertion as they arrive. Staged chunks SHALL be associated with their file_end sentinel. On sentinel receipt, all staged chunks for that file SHALL be retrieved for batch processing. A TTL sweep SHALL prevent stale staged data from accumulating.

#### Scenario: Chunk staged on arrival
- **WHEN** a chunk message is popped from a consumer queue
- **THEN** it SHALL be inserted into the DuckDB staging table

#### Scenario: Staged chunks retrieved on sentinel
- **WHEN** a file_end sentinel is received
- **THEN** all staged chunks for that file SHALL be retrieved for embedding and storage

#### Scenario: TTL sweep for stale data
- **WHEN** staged data exceeds the configured TTL without a sentinel
- **THEN** the system SHALL clean up the stale entries
