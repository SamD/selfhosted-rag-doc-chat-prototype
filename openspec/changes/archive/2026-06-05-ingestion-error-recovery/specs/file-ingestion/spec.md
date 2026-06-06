## MODIFIED Requirements

### Requirement: Gatekeeper normalization

The system SHALL claim jobs in NEW status and transition them to PREPROCESSING. The Gatekeeper SHALL extract raw text from the file using the Chain of Responsibility handler system, batch the content, normalize it via the Supervisor LLM into clean Markdown with ### [INTERNAL_PAGE_X] anchors, and write the output to the ingestion directory. On success, status SHALL transition to PREPROCESSING_COMPLETE. On failure, status SHALL transition to INGEST_FAILED.

Before calling the Supervisor LLM, the system SHALL run `is_bad_ocr()` on the extracted text using multi-heuristic quality checks (gibberish, visible corruption, low tokens, repetition ratio, abnormal word lengths). If the quality check passes, the text SHALL be written directly as Markdown without LLM normalization. The output format (metadata anchor + content) SHALL be identical regardless of which path is taken.

If `FORCE_MARKDOWN_LLM` is set to `true` or `1`, the quality check SHALL be skipped and all pages SHALL be sent to the Supervisor LLM. This flag accepts both `true`/`false` and `1`/`0` for consistency with `HA_INTERLEAVE`.

#### Scenario: Gatekeeper claims a NEW job
- **WHEN** a job exists in NEW status
- **THEN** the Gatekeeper atomically claims it via UPDATE ... RETURNING * and transitions to PREPROCESSING

#### Scenario: Successful normalization
- **WHEN** the Supervisor LLM successfully normalizes all content batches into Markdown
- **THEN** the system writes a .md file to the ingestion directory and transitions to PREPROCESSING_COMPLETE

#### Scenario: Normalization failure
- **WHEN** the Supervisor LLM fails or any content handler throws an exception
- **THEN** the system transitions to INGEST_FAILED with an error log

#### Scenario: Quality check bypass
- **WHEN** a batch of extracted text passes `is_bad_ocr()` quality check
- **THEN** the text SHALL be written directly as Markdown with the standard metadata anchor, and the Supervisor LLM SHALL NOT be called

#### Scenario: Quality check failure routes to LLM
- **WHEN** a batch of extracted text fails `is_bad_ocr()` quality check
- **THEN** the text SHALL be sent to the Supervisor LLM for normalization as before

#### Scenario: FORCE_MARKDOWN_LLM bypasses quality check
- **WHEN** `FORCE_MARKDOWN_LLM` is set to `true` or `1`
- **THEN** the quality check SHALL be skipped and all pages SHALL be sent to the Supervisor LLM

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
