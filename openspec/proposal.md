## Why

The ingestion pipeline's WhisperX worker (audio/video transcription) processes jobs from Redis queues with zero retry semantics, no crash recovery, and no observability into in-progress work. When a worker container dies mid-transcription (OOM, network blip, deployment), the file is silently lost or requires manual rediscovery via staging directory audit. Transcription is also the longest-running step in the pipeline (minutes per file), making failures both costly and opaque — operators cannot see what is running, stuck, or failed without SSH-ing into containers.

Wrapping workers as Temporal Activities gives us durability, retry, and workflow visibility with minimal code change: the existing `RemoteWhisper.transcribe_file()` logic stays untouched inside an Activity wrapper. This is the highest-ROI starting point because the WhisperX worker has no DuckDB coupling, no LangGraph dependency, and a clean single-Activity boundary.

## What Changes

- **New Temporal server service** added to Docker Compose (`temporal-server` + `temporal-web` UI)
- **WhisperX worker restructured**: Redis `brpop` polling loop replaced by Temporal Task Queue; existing `run_whisperx_worker.py` refactored into:
  - `activities.py`: `transcribe_media(Activity)` wrapping `RemoteWhisper.transcribe_file()` with retry policy (3 attempts, exponential backoff)
  - `temporal_worker.py`: Async Activity Worker entry point (`run_temporal_worker.py`)
  - Backward-compatible `send_media_to_whisperx_temporal()` generator wrapper in `whisper_utils.py` with `USE_TEMPORAL_WHISPER` feature flag
- **Dual-run mode**: Both Redis and Temporal paths run simultaneously during transition; feature flag controls dispatch
- **Feature flag**: `USE_TEMPORAL_WHISPER` env var (default: `false`) gates Temporal path without changing Docker compose
- **Environment variables**: `TEMPORAL_SERVER_URL` (default: `http://localhost:7233`), `TEMPORAL_WHISPER_TASK_QUEUE` (default: `whisperx`)
- **Operations docs updated**: `infra/operations/day-1.md` and `day-2.md` include Temporal deployment and troubleshooting

## Capabilities

### New Capabilities

- `whisperx-temporal`: Wraps the WhisperX transcription worker as a Temporal Activity with durable retry, crash recovery, and workflow observability; introduces Temporal server and web UI as infrastructure services; supports dual-run mode with existing Redis path via feature flag.

### Modified Capabilities

- `ingestion-lifecycle-recovery`: Temporal Activity idempotency adds a new durability mechanism alongside DuckDB state machine transitions; retries are now bounded (3 attempts) instead of silent loss.
- `infrastructure`: Docker Compose gains `temporal-server` and `temporal-web` services; `run-compose.sh` starts them automatically when `USE_TEMPORAL_WHISPER=true`.

## Impact

**Code changes:**
- `doc-ingest-chat/workers/whisperx_worker.py` — extract polling loop, preserve handler chain
- `doc-ingest-chat/utils/whisper_utils.py` — add `send_media_to_whisperx_temporal()` wrapper
- New: `doc-ingest-chat/temporal_worker/activities.py`, `worker.py`
- New: `doc-ingest-chat/models/transcription_input.py` (dataclasses for Activity I/O)
- `run-whisperx-worker.service` or Dockerfile reference updated
- `doc-ingest-chat/config/settings.py` — 2 new env var entries

**Infrastructure:**
- Docker Compose: `temporal-server` container, `temporal-web` container
- `infra/haproxy-entrypoint.sh` — no change (Temporal server is single-node, no LB needed)
- `infra/operations/day-1.md` — add Temporal deployment steps
- `infra/operations/day-2.md` — add Temporal troubleshooting runbook section

**Dependencies:**
- New Python package: `temporalio>=1.10.0` (added to pyproject.toml / requirements.txt)
- No breaking API changes — feature-flagged dual-run preserves backward compatibility

**Testing:**
- Unit tests with mock Temporal client (Activity execution)
- Integration test using `temporalite` (dev Temporal server) in pytest
- E2E test: real MP3 file through full Temporal path, measure latency overhead vs Redis (< 50ms expected)

## Cross-cutting

**Error handling patterns:**
- Non-retryable errors (`FileNotFoundError`, `ConfigurationError`) → immediate Activity failure
- Retryable errors (server timeout, network error) → exponential backoff, max 3 attempts
- Activity timeout: 90 minutes (longest known transcription: ~10-minute video at 6x speed)
- Workflow timeout: 4 hours (safety net for chained workflows)

**Logging/observability:**
- Each Activity invocation logs `trace_id`, `file_path`, `mime_type`, `attempt_number`
- Temporal Web UI provides real-time workflow status, history, and error inspection
- Metrics exposed via Temporal's built-in Prometheus metrics endpoint (port 9090)

**Documentation conventions:**
- All new modules use `logger = logging.getLogger("ingest.temporal")`
- Google-style docstrings on all public methods
- Operations runbook updated with Temporal sections

**Testing strategy:**
- Unit: Mock `temporalio.client.Client`, assert Activity return value matches expected schema
- Integration: `temporalite` dev server in CI, full activity execution path
- Performance: latency comparison Redis vs Temporal overhead (target: < 50ms)