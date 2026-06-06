## Why

When a worker process crashes (kill -9, segfault, container OOM), the DuckDB lifecycle state machine leaves jobs stuck in intermediate states (PREPROCESSING, INGESTING, CONSUMING) indefinitely. No worker reclaims these orphaned jobs on restart. Chunks can leak into Redis queues. Failed files in INGEST_FAILED cannot be re-ingested without manual DuckDB intervention. The consumer worker silently ignores graph failures. Without automated recovery, every crash creates a manual cleanup burden and risks data loss.

## What Changes

- **NEW**: Startup reclaim — each worker scans for jobs stuck in its claimed intermediate state on startup, where the worker_id no longer corresponds to a live process, and resets them to the previous valid state
- **NEW**: Stuck job timeout — jobs that remain in PREPROCESSING, INGESTING, or CONSUMING beyond a configurable timeout are automatically reset to their previous state (e.g., INGESTING → PREPROCESSING_COMPLETE)
- **FIXED**: Consumer worker checks the return value of `run_consumer_graph()` and transitions to INGEST_FAILED on failure
- **FIXED**: Redis queue entries for a failed file are cleaned up on transition to INGEST_FAILED
- **FIXED**: DuckDB staged_chunks for a failed file are purged on transition to INGEST_FAILED
- **REMOVED**: Duplicate filename rejection for INGEST_FAILED files — files in FAILED_DIR can be rediscovered by moving them back to staging (the INGEST_FAILED record stays in DuckDB, but a new ingestion creates a new record)

## Capabilities

### New Capabilities
- `ingestion-lifecycle-recovery`: Automated recovery from worker crashes. Covers: startup reclaim of orphaned jobs, configurable timeout for stuck intermediate states, and cleanup of orphaned Redis/DuckDB state on failure.

### Modified Capabilities
- `file-ingestion`: The "Consumer embedding and storage" requirement changes — consumer worker must check graph return value and transition to INGEST_FAILED on failure. The file lifecycle requirement changes — INGEST_FAILED no longer blocks re-ingestion.

## Impact

- `doc-ingest-chat/services/job_service.py`: New `reclaim_orphaned_jobs()` method. New `reset_stuck_jobs()` method. Modify `create_job()` to allow re-ingestion after INGEST_FAILED.
- `doc-ingest-chat/workers/consumer_worker.py`: Check return value of `run_consumer_graph()`, transition to INGEST_FAILED on failure.
- `doc-ingest-chat/workers/gatekeeper_worker.py`: Call `reclaim_orphaned_jobs('PREPROCESSING')` on startup.
- `doc-ingest-chat/workers/producer_worker.py`: Call `reclaim_orphaned_jobs('INGESTING')` on startup.
- `doc-ingest-chat/services/database.py`: Add cleanup call for orphaned staged_chunks and Redis queue entries on transition to INGEST_FAILED.
- `doc-ingest-chat/services/redis_service.py`: New method to purge queue entries for a given source_file.
- `shared/env_names.py`, `shared/defaults.py`, `shared/config.py`: New env var `STUCK_JOB_TIMEOUT_HOURS` (default: 1).
- `infra/operations/day-2.md`: New runbook entry for stuck job recovery.
- `openspec/specs/file-ingestion/spec.md`: MODIFIED — consumer error handling and re-ingestion requirements.
