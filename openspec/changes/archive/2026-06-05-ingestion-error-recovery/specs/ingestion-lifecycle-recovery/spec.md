## ADDED Requirements

### Requirement: Manual reclaim of orphaned jobs

A `reclaim_orphaned_jobs()` method SHALL be available on `JobService` to reset jobs stuck in an intermediate state to a previous valid state. It SHALL NOT be auto-called on worker startup — reclaim is manual or trigger-based only. A job SHALL be considered orphaned if it has been in its current state longer than STUCK_JOB_TIMEOUT_HOURS (default: 1). Mapping: PREPROCESSING → NEW, INGESTING → PREPROCESSING_COMPLETE, CONSUMING → INGESTING.

#### Scenario: Manual reclaim via method call
- **WHEN** `JobService.reclaim_orphaned_jobs('PREPROCESSING', 'NEW')` is called manually
- **THEN** jobs stuck in PREPROCESSING for longer than STUCK_JOB_TIMEOUT_HOURS SHALL be reset to NEW

#### Scenario: Atomic reclaim prevents double-processing
- **WHEN** two callers attempt to reclaim the same orphaned job simultaneously
- **THEN** only one SHALL succeed due to the atomic UPDATE ... RETURNING * pattern

### Requirement: Stuck job timeout

The system SHALL expose a STUCK_JOB_TIMEOUT_HOURS environment variable (default: 1) that controls the threshold for considering a job orphaned. Jobs that have been in an intermediate state for longer than this threshold SHALL be eligible for reclaim on worker startup.

#### Scenario: Configurable timeout
- **WHEN** STUCK_JOB_TIMEOUT_HOURS is set to 2
- **THEN** jobs must be stuck for more than 2 hours before they are eligible for reclaim

### Requirement: Cleanup on INGEST_FAILED

When a job transitions to INGEST_FAILED, the system SHALL purge any corresponding entries from Redis consumer queues and delete any staged chunks from DuckDB for that file. This prevents orphaned chunk data from accumulating when a job fails mid-ingestion.

#### Scenario: Redis queue purge on failure
- **WHEN** a job transitions to INGEST_FAILED
- **THEN** any chunks in Redis consumer queues for that source_file SHALL be purged

#### Scenario: DuckDB staged chunks purge on failure
- **WHEN** a job transitions to INGEST_FAILED
- **THEN** any staged_chunks entries for that source_file SHALL be deleted

### Requirement: Re-ingestion after failure

Files in INGEST_FAILED state SHALL be eligible for re-ingestion. The create_job() duplicate check SHALL permit re-ingestion of previously failed files. The old INGEST_FAILED record SHALL remain in DuckDB for audit purposes. A new record SHALL be created with a fresh UUID and trace_id for the re-ingest attempt.

#### Scenario: Re-ingestion creates new record
- **WHEN** a previously failed file is returned to the staging directory
- **THEN** the system SHALL create a new ingestion_lifecycle record with status NEW and a fresh UUID, while preserving the old INGEST_FAILED record
