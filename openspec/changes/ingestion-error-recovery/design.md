## Context

The ingestion pipeline uses a DuckDB-backed state machine with atomic `UPDATE ... RETURNING *` claims. Six states: `NEW → PREPROCESSING → PREPROCESSING_COMPLETE → INGESTING → CONSUMING → INGEST_SUCCESS/INGEST_FAILED`. Each state transition is recorded with a timestamp and `worker_id` (OS PID). When a worker crashes, the job stays in whatever intermediate state it was in — no mechanism exists to reclaim or timeout these orphaned jobs.

Redis queues (`chunk_ingest_queue:0,1`) have no TTL. If the Producer pushes chunks and then crashes before sending the `file_end` sentinel, those chunks sit in Redis forever. The Consumer worker ignores the return value of `run_consumer_graph()`, silently swallowing graph failures.

## Goals / Non-Goals

**Goals:**
- Workers scan for orphaned jobs on startup and reset them to the previous valid state
- Configurable timeout resets jobs stuck in intermediate states for too long
- Consumer worker checks graph return value and transitions to INGEST_FAILED
- Redis queue entries and DuckDB staged_chunks are cleaned up on INGEST_FAILED
- Files in FAILED_DIR can be re-ingested (human moves back to staging)
- Only workers that claimed a job clean it up (not cross-worker)

**Non-Goals:**
- No cross-worker recovery — a Gatekeeper crash won't be recovered by the Producer
- No automatic move from FAILED_DIR → staging — human decides when to retry
- No changes to the existing state machine transitions or claim mechanism
- No Qdrant/Chroma point cleanup (duplicate vectors are handled by MurmurHash3 dedup)

## Decisions

1. **Worker_id PID is unreliable for cross-host recovery**: In Docker, PIDs are recycled. A newly restarted worker could have the same PID as a crashed one from a previous run. Mitigation: use a combination of `worker_id` + `preprocessing_at`/`ingesting_at` timestamp — only reclaim jobs that have been stuck for more than `STUCK_JOB_TIMEOUT_HOURS`.

2. **Reclaim at startup, not on a timer**: A periodic reaper adds complexity (scheduling, race conditions). Startup reclaim is simpler: each worker reclaims its relevant stuck state before entering the main loop. This covers the Docker `restart: unless-stopped` case.

3. **Reclaim target state depends on the stuck state**: `PREPROCESSING` → reset to `NEW` (file is still in staging). `INGESTING` → reset to `PREPROCESSING_COMPLETE` (file is in ingestion/ with .md). `CONSUMING` → reset to `INGESTING` (file is in consuming/, needs re-chunking). This avoids re-doing work unnecessarily.

4. **Cleanup on INGEST_FAILED**: When any worker transitions to `INGEST_FAILED`, it should also purge Redis queue entries for that file and delete DuckDB staged_chunks. This is added to `transition_job()` or called explicitly after transition.

5. **Re-ingestion after INGEST_FAILED**: The `create_job()` duplicate check currently blocks INGEST_FAILED files. Change it to allow re-ingestion — the old INGEST_FAILED record stays for audit purposes, a new record is created for the re-ingest attempt.

## Risks / Trade-offs

- **Race condition on restart**: If two workers start simultaneously and both try to reclaim the same orphaned job, the atomic `UPDATE ... RETURNING *` prevents double-claims. Only one will succeed.
- **Data loss if reclaim resets too aggressively**: A job that was legitimately in an intermediate state (e.g., a slow normalization) could be reset if `STUCK_JOB_TIMEOUT_HOURS` is too low. Default: 1 hour, configurable.
- **Redis queue purge is best-effort**: We can purge by iterating the queue, but for large queues this is expensive. Alternative: track a dedup set or use a marker. For now, purge by scanning — it's only done on failure, which should be rare.
