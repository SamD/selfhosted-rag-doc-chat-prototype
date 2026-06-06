## 1. Job Service — Reclaim and Cleanup

- [x] 1.1 Add `reclaim_orphaned_jobs(current_status, target_status, timeout_hours)` to `job_service.py`
- [x] 1.2 Modify `create_job()` — allow re-ingestion after INGEST_FAILED
- [x] 1.3 Add `cleanup_failed_job()` called from `transition_job()` on INGEST_FAILED

## 2. Worker Startup Reclaim

- [x] 2.1 Update `gatekeeper_worker.py` — call `reclaim_orphaned_jobs('PREPROCESSING', 'NEW')` before main loop
- [ ] 2.2 **REVERTED**: Startup reclaim causes unintended re-processing of files left in lifecycle directories. Needs a safer trigger (e.g., only on non-zero exit, not every restart).
- [ ] 2.3 **REVERTED**: Same issue — startup reclaim is too aggressive.

## 3. Consumer Error Handling

- [x] 3.1 Update `consumer_worker.py` — capture return value of `run_consumer_graph()`, transition to INGEST_FAILED on failure

## 4. Redis Queue Cleanup

- [x] 4.1 Add `purge_queue_entries(source_file, queue_names)` to `redis_service.py`
- [x] 4.2 Wired purge into `cleanup_failed_job()` called from `transition_job()` on INGEST_FAILED

## 5. Configuration

- [x] 5.1 Add `ENV_STUCK_JOB_TIMEOUT_HOURS` to `shared/env_names.py`
- [x] 5.2 Add `DEFAULT_STUCK_JOB_TIMEOUT_HOURS = 1` to `shared/defaults.py`
- [x] 5.3 Add `STUCK_JOB_TIMEOUT_HOURS` setting to `shared/config.py` `_SETTINGS` dict

## 6. Verification

- [x] 6.1 Run `ruff check --fix` on all changed files
- [x] 6.2 Run full pytest suite — 138/138 (133 original + 5 new)
- [x] 6.3 Write unit tests for `reclaim_orphaned_jobs` — 2 tests
- [x] 6.4 Write unit tests for consumer graph failure handling — covered by cleanup test
- [x] 6.5 Write unit tests for re-ingestion after INGEST_FAILED — 2 tests

## 7. Quality Bypass and Flags

- [x] 7.1 Add quality check bypass in `process_chunk()` — `is_bad_ocr()` runs before LLM call, skips LLM for clean text
- [x] 7.2 Add `is_repetitive()` and `has_abnormal_word_lengths()` heuristics to `is_bad_ocr()`
- [x] 7.3 Add `FORCE_MARKDOWN_LLM` env var (default: false) — when true, bypasses quality check and always routes to LLM
- [x] 7.4 Make `PDF_FORCE_OCR` and `FORCE_MARKDOWN_LLM` accept both `true`/`1` and `false`/`0` for consistency with `HA_INTERLEAVE`
- [x] 7.5 Add `FORCE_MARKDOWN_LLM` to `ingest-svc.env`
- [x] 7.6 Add 4 unit tests for FORCE_MARKDOWN_LLM (2) and PDF_FORCE_OCR (2)
- [x] 7.7 Update test_gatekeeper_enforces_context_limit to mock is_bad_ocr

## 8. Documentation

- [x] 8.1 Update `infra/operations/day-2.md` — add runbook entry for stuck job recovery
- [x] 8.2 Update `docs/overview.md`, `docs/quickstart.md`, `docs/deep-dive.md`, `AGENTS.md` — supervisor LLM now has quality bypass
- [x] 8.3 Update `CHANGELOG.md`