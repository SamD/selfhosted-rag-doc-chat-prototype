## 1. Spike: Temporal Server + Activity Wrapper

- [x] 1.1 Add `temporalio>=1.10.0` to project dependencies (pyproject.toml or requirements.txt)
- [x] 1.2 Create `docker-compose.yml` service definition for `temporalite` (in-memory Temporal server, port 7233)
- [x] 1.3 Verify `temporalite start` works and `localhost:7233/health` returns 200
- [x] 1.4 Create `doc-ingest-chat/models/transcription_input.py` with `TranscriptionInput` and `TranscriptionResult` dataclasses
- [x] 1.5 Create `doc-ingest-chat/temporal_worker/__init__.py` package
- [x] 1.6 Create `doc-ingest-chat/temporal_worker/activities.py` — implement `transcribe_media()` Activity that wraps existing `RemoteWhisper.transcribe_file()` logic
- [x] 1.7 Create `run_temporal_worker.py` entry point that starts a Temporal async Worker on the `whisperx` task queue
- [x] 1.8 Write a simple test script (`test_spike.py`) that submits a job to the Temporal client, runs an Activity against a real 10-second MP3 file, and validates output
- [x] 1.9 Measure latency: run same file through Redis path vs Temporal path; record overhead (target <50ms)
- [x] 1.10 Write spike summary: results, pain points, go/no-go recommendation

## 2. Infrastructure: Docker Compose Services

- [x] 2.1 Add `temporal-server` service to `docker-compose.yml` with embedded storage, port 7233, profile `[temporal]`
- [x] 2.2 Add `temporal-web` service to `docker-compose.yml`, port 8233, profile `[temporal]`, depends on temporal-server
- [x] 2.3 Update `run-compose.sh` to pass `--profile temporal` when `USE_TEMPORAL_WHISPER=true`
- [x] 2.4 Add health checks for both services in compose file
- [x] 2.5 Verify `./run-compose.sh --profile temporal` starts all services and health checks pass

## 3. Implementation: Temporal Worker

- [x] 3.1 Implement `TranscriptionInput` dataclass with fields: `file_path (str)`, `language (str, default="en")`, `mime_type (str | None)`
- [x] 3.2 Implement `TranscriptionResult` dataclass with fields: `segments (list[dict])`, `source_file (str)`, `job_id (str)`
- [x] 3.3 Implement `transcribe_media()` Activity in `activities.py` with proper retry policy via `@activity.defn` decorator
- [x] 3.4 Add retry policy configuration: `maximum_attempts=3`, `initial_interval=5s`, `backoff_coefficient=2.0`, `maximum_interval=60s`
- [x] 3.5 Set Activity timeout to 90 minutes, Workflow timeout to 4 hours
- [x] 3.6 Handle `FileNotFoundError` as non-retryable (raise immediately without retry)
- [x] 3.7 Handle `requests.ConnectionError` / `TimeoutError` as retryable
- [x] 3.8 Implement graceful shutdown in `run_temporal_worker.py` (handle SIGTERM, complete in-flight Activity)
- [x] 3.9 Add structured logging: each Activity invocation logs `trace_id`, `file_path`, `mime_type`, `attempt_number`
- [x] 3.10 Ensure fork-safe singleton compatibility (PID checking not needed for single-process async worker, but document the assumption)

## 4. Implementation: Sender Dispatch

- [x] 4.1 Add `USE_TEMPORAL_WHISPER` env var to `shared/env_names.py`
- [x] 4.2 Add default value `"false"` to `shared/defaults.py`
- [x] 4.3 Add lazy-evaluated setting in `config/settings.py`: `USE_TEMPORAL_WHISPER = lambda: os.getenv("USE_TEMPORAL_WHISPER", "false").lower() == "true"`
- [x] 4.4 Implement `send_media_to_whisperx_temporal()` function in `whisper_utils.py` that:
  - Creates a Temporal Client connected to `TEMPORAL_SERVER_URL`
  - Executes the `transcribe_media` workflow synchronously
  - Yields segment text from the returned `TranscriptionResult`
  - Raises `RuntimeError` on transcription failure (matching existing error contract)
- [x] 4.5 Modify `send_media_to_whisperx()` to check `USE_TEMPORAL_WHISPER` and dispatch to either Temporal or Redis path
- [x] 4.6 Verify backward compatibility: when flag is `false`, exact same behavior as before (same reply_key format, same segment yield order)

## 5. Testing

- [x] 5.1 Create `tests/unit/test_temporal_activities.py`: unit test `transcribe_media()` with mock `RemoteWhisper`
  - Test success path (returns expected segments)
  - Test `FileNotFoundError` is raised immediately (non-retryable)
  - Test retry on `ConnectionError` (verifies retry count via mock)
- [x] 5.2 Create `tests/integration/test_temporal_worker.py`: integration tests using `temporalite` dev server in pytest
  - Start temporalite as pytest fixture
  - Submit workflow, wait for completion, verify result
  - Simulate worker crash mid-Activity, verify Activity restarts on recovery
- [x] 5.3 Create `tests/e2e/test_transcription_latency.py`: measure overhead of Temporal path vs Redis path with real 10-second MP3
  - Run same file through both paths
  - Assert Temporal latency overhead <50ms
- [x] 5.4 Run full test suite: `PYTHONPATH=doc-ingest-chat:shared .venv/bin/python -m pytest tests/ -v`
- [x] 5.5 Run linting: `ruff check doc-ingest-chat/temporal_worker/ doc-ingest-chat/models/transcription_input.py`

## 6. Operations Documentation

- [x] 6.1 Update `infra/operations/day-1.md`: add Temporal deployment section (services, ports, startup order)
- [x] 6.2 Update `infra/operations/day-2.md`: add Temporal troubleshooting runbook (service down, stuck workflows, retry inspection via CLI/Web UI)
- [x] 6.3 Update `AGENTS.md`: add Temporal worker run command, new env vars, new profiles
- [x] 6.4 Update `CHANGELOG.md` under `[Unreleased]` → Added: Temporal whisperx activity wrapper, durability, dual-run mode

## 7. Final Verification

- [x] 7.1 Run `ruff check --fix` across all changed files
- [x] 7.2 Run full pytest suite: pass
- [x] 7.3 Verify `./run-compose.sh --profile temporal` starts and health checks pass
- [x] 7.4 Verify Temporal Web UI accessible at configured port
- [x] 7.5 Submit a transcription job via Temporal path and confirm result appears correctly in downstream pipeline
- [x] 7.6 Flip feature flag to `false`, verify Redis path still works (no regression)