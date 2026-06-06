## MODIFIED Requirements

### Requirement: Consumer embedding and storage

The system SHALL listen on partitioned Redis consumer queues. Chunks SHALL be staged in DuckDB as they arrive. When a file_end sentinel is received, the Consumer SHALL validate all staged chunks (sub-splitting any oversized chunks), embed them via the e5-large-v2 model, upsert to the configured vector database (Qdrant or Chroma), archive to Parquet, and move the processed file to the success directory. Status SHALL transition to INGEST_SUCCESS.

If the consumer graph fails (returns False from run_consumer_graph), the consumer worker SHALL transition the job to INGEST_FAILED, move the file to the failed directory, and log the error. The return value of run_consumer_graph() SHALL NOT be ignored.

#### Scenario: Consumer processes a file_end sentinel
- **WHEN** a file_end sentinel is popped from a consumer queue
- **THEN** the Consumer retrieves all staged chunks for that file, embeds them, upserts to vector DB, and archives to Parquet

#### Scenario: Chunk validation
- **WHEN** a chunk exceeds MAX_TOKENS
- **THEN** it SHALL be sub-split into valid chunks rather than truncated

#### Scenario: Successful ingestion finalization
- **WHEN** all chunks are embedded and stored
- **THEN** the file is moved to the success directory and status transitions to INGEST_SUCCESS

#### Scenario: Consumer graph failure
- **WHEN** run_consumer_graph() returns False
- **THEN** the consumer worker SHALL transition the job to INGEST_FAILED and move the file to the failed directory
