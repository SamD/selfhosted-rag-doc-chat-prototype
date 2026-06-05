## ADDED Requirements

### Requirement: File discovery and lifecycle registration

The system SHALL watch the staging directory for new files and register them in the DuckDB ingestion_lifecycle table with a NEW status. Each file SHALL be assigned a unique UUID job ID and a trace ID. Duplicate filenames SHALL be rejected unless the previous job is in INGEST_FAILED state.

#### Scenario: New file discovered in staging
- **WHEN** a file is placed in the staging directory
- **THEN** the system creates an ingestion_lifecycle record with status NEW, a UUID, a trace ID, and the original filename

#### Scenario: Duplicate file rejection
- **WHEN** a file with the same name already exists in ingestion_lifecycle with a non-failed status
- **THEN** the system SHALL return None and not create a duplicate record

### Requirement: Gatekeeper normalization

The system SHALL claim jobs in NEW status and transition them to PREPROCESSING. The Gatekeeper SHALL extract raw text from the file using the Chain of Responsibility handler system, batch the content, normalize it via the Supervisor LLM into clean Markdown with ### [INTERNAL_PAGE_X] anchors, and write the output to the ingestion directory. On success, status SHALL transition to PREPROCESSING_COMPLETE. On failure, status SHALL transition to INGEST_FAILED.

#### Scenario: Gatekeeper claims a NEW job
- **WHEN** a job exists in NEW status
- **THEN** the Gatekeeper atomically claims it via UPDATE ... RETURNING * and transitions to PREPROCESSING

#### Scenario: Successful normalization
- **WHEN** the Supervisor LLM successfully normalizes all content batches into Markdown
- **THEN** the system writes a .md file to the ingestion directory and transitions to PREPROCESSING_COMPLETE

#### Scenario: Normalization failure
- **WHEN** the Supervisor LLM fails or any content handler throws an exception
- **THEN** the system transitions to INGEST_FAILED with an error log

### Requirement: Producer chunking and enqueuing

The system SHALL claim jobs in PREPROCESSING_COMPLETE status and transition them to INGESTING. The Producer SHALL read the normalized Markdown file, perform hierarchical splitting (H1 -> H2 -> INTERNAL_PAGE -> H3 -> recursive character split at 85% safety budget), assign deterministic DOC_XXXX IDs via MurmurHash3, and enqueue chunks to the Redis consumer queues. After all chunks are enqueued, a file_end sentinel SHALL be sent. Status SHALL transition to CONSUMING.

#### Scenario: Producer processes a completed Markdown file
- **WHEN** a job is in PREPROCESSING_COMPLETE status
- **THEN** the Producer claims it, transitions to INGESTING, reads the .md file, and generates hierarchical chunks

#### Scenario: Chunk ID generation
- **WHEN** a chunk is created
- **THEN** it SHALL receive a deterministic ID using MurmurHash3 of the chunk content, formatted as DOC_<8-char-hex>

#### Scenario: Oversized chunk sub-splitting
- **WHEN** a chunk exceeds MAX_TOKENS after hierarchical splitting
- **THEN** it SHALL be sub-split into smaller chunks using a sliding window, with zero content loss

#### Scenario: Sentinel enqueue
- **WHEN** all chunks for a file have been enqueued
- **THEN** a file_end sentinel message SHALL be enqueued to signal the Consumer to finalize

### Requirement: Consumer embedding and storage

The system SHALL listen on partitioned Redis consumer queues. Chunks SHALL be staged in DuckDB as they arrive. When a file_end sentinel is received, the Consumer SHALL validate all staged chunks (sub-splitting any oversized chunks), embed them via the e5-large-v2 model, upsert to the configured vector database (Qdrant or Chroma), archive to Parquet, and move the processed file to the success directory. Status SHALL transition to INGEST_SUCCESS.

#### Scenario: Consumer processes a file_end sentinel
- **WHEN** a file_end sentinel is popped from a consumer queue
- **THEN** the Consumer retrieves all staged chunks for that file, embeds them, upserts to vector DB, and archives to Parquet

#### Scenario: Chunk validation
- **WHEN** a chunk exceeds MAX_TOKENS
- **THEN** it SHALL be sub-split into valid chunks rather than truncated

#### Scenario: Successful ingestion finalization
- **WHEN** all chunks are embedded and stored
- **THEN** the file is moved to the success directory and status transitions to INGEST_SUCCESS

### Requirement: DuckDB lifecycle state machine

The system SHALL maintain a state machine in DuckDB with the following valid transitions: NEW -> PREPROCESSING -> PREPROCESSING_COMPLETE -> INGESTING -> CONSUMING -> INGEST_SUCCESS/INGEST_FAILED. State transitions SHALL be atomic using UPDATE ... RETURNING *. Lock contention SHALL be handled with exponential backoff up to 20 retries.

#### Scenario: Atomic claim prevents double-processing
- **WHEN** two workers attempt to claim the same job simultaneously
- **THEN** only one worker SHALL receive the job due to the atomic UPDATE ... RETURNING * pattern

#### Scenario: Lock contention retry
- **WHEN** DuckDB returns a lock error
- **THEN** the system SHALL retry with exponential backoff (base_delay=0.2s, max_retries=20)
