# WhisperX Temporal

## Purpose

The WhisperX Temporal capability provides durable, observable transcription of audio and video files by running WhisperX transcription as Temporal Activities. This replaces the previous Redis-based fire-and-forget model with Temporal's built-in retry, timeout, and observability features. When enabled via the `USE_TEMPORAL_WHISPER` feature flag, transcription jobs are dispatched to a Temporal workflow instead of the Redis queue, providing automatic retries, visibility into execution history, and protection against worker crashes.

## Requirements

### Requirement: WhisperX worker transcribes media as Temporal Activity

The system SHALL expose a `transcribe_media()` Activity function that accepts a `TranscriptionInput` dataclass and returns a `TranscriptionResult` dataclass. The Activity SHALL execute the transcription logic (either via `RemoteWhisper.transcribe_file()` or local whisperx model) within a bounded retry policy.

#### Scenario: Successful transcription via Activity
- **WHEN** `transcribe_media(TranscriptionInput(file_path="/path/to/audio.mp3", language="en", mime_type="audio/mpeg"))` is invoked as a Temporal Activity
- **THEN** the Activity SHALL return `TranscriptionResult(segments=[{"text": "..."}], source_file="audio.mp3")` containing all transcription segments

#### Scenario: Transcription with missing file raises non-retryable error
- **WHEN** the `file_path` in `TranscriptionInput` does not exist on disk
- **THEN** the Activity SHALL raise `FileNotFoundError` immediately (not retried)

#### Scenario: Transcription with transient server error retries
- **WHEN** `RemoteWhisper.transcribe_file()` raises `requests.ConnectionError`
- **THEN** the Activity SHALL retry up to 3 times with exponential backoff before failing

#### Scenario: Transcription timeout exceeds activity timeout
- **WHEN** a transcription takes longer than 90 minutes
- **THEN** the Temporal workflow SHALL mark the Activity as timed out and trigger its default retry policy

### Requirement: Activity worker runs as independent Docker service

The system SHALL run a separate `whisperx-temporal-worker` Docker container that starts a Temporal async Activity Worker. This worker SHALL register the `transcribe_media` Activity and poll the `whisperx` task queue for incoming jobs.

#### Scenario: Activity worker starts successfully
- **WHEN** `run_temporal_worker.py` is executed with `TEMPORAL_SERVER_URL=http://temporal-server:7233` and `TEMPORAL_WHISPER_TASK_QUEUE=whisperx`
- **THEN** the Activity Worker SHALL connect to the Temporal server, register the `transcribe_media` Activity, and begin polling the task queue

#### Scenario: Activity worker handles graceful shutdown
- **WHEN** SIGTERM is sent to the `whisperx-temporal-worker` container
- **THEN** the worker SHALL stop accepting new tasks, complete any in-flight Activity, and exit cleanly within 30 seconds

### Requirement: Sender dispatches to Temporal or Redis based on feature flag

The system SHALL provide a `send_media_to_whisperx()` function in `whisper_utils.py` that checks the `USE_TEMPORAL_WHISPER` environment variable and dispatches the transcription job to either the Temporal client or the existing Redis queue path.

#### Scenario: Feature flag off uses Redis path
- **WHEN** `USE_TEMPORAL_WHISPER` is not set or is `"false"`
- **THEN** `send_media_to_whisperx()` SHALL execute the existing Redis-based flow (lpush job, blpop reply key)

#### Scenario: Feature flag on uses Temporal path
- **WHEN** `USE_TEMPORAL_WHISPER` is `"true"`
- **THEN** `send_media_to_whisperx()` SHALL create a Temporal workflow handle for `transcribe_media`, send the `TranscriptionInput`, and yield segments from the Activity return value

#### Scenario: Temporal connection failure falls back gracefully
- **WHEN** `USE_TEMPORAL_WHISPER=true` but the Temporal server is unreachable
- **THEN** `send_media_to_whisperx()` SHALL log an error and raise `ConnectionError` (the caller can handle fallback or surface the error)

### Requirement: Temporal server runs as Docker Compose service

The system SHALL include `temporal-server` and `temporal-web` services in `docker-compose.yml`, started via the `temporal` profile. The Temporal server SHALL use embedded storage for local development and support PostgreSQL for production.

#### Scenario: Temporal services start with profile
- **WHEN** `run-compose.sh --profile temporal` is executed
- **THEN** both `temporal-server` (port 7233) and `temporal-web` (port 8233) containers SHALL start and be reachable

#### Scenario: Temporal server health check passes
- **WHEN** the `temporal-server` container is running
- **THEN** a HTTP GET to `http://localhost:7233/health` SHALL return HTTP 200 with `{"status": "ok"}`

### Requirement: Transcription results are observable in Temporal Web UI

The system SHALL expose workflow execution status, Activity attempt counts, error messages, and timing data through the Temporal Web UI at the configured port.

#### Scenario: Workflow appears in Web UI
- **WHEN** a transcription job is dispatched via Temporal
- **THEN** the workflow SHALL appear in the Temporal Web UI with status "RUNNING" during execution and "COMPLETED" or "FAILED" after termination

#### Scenario: Activity errors are visible in history
- **WHEN** a transcribe_media Activity fails and retries
- **THEN** the Temporal Web UI SHALL show each retry attempt with its error message and duration in the workflow history

### Requirement: Manual reclaim of orphaned jobs

A `reclaim_orphaned_jobs()` method SHALL be available on `JobService` to reset jobs stuck in an intermediate state to a previous valid state. Additionally, when `USE_TEMPORAL_WHISPER=true`, the system SHALL treat an Activity that has been running longer than the activity timeout (90 minutes) as eligible for reclamation via Temporal's built-in retry mechanism, independent of `STUCK_JOB_TIMEOUT_HOURS`. A job SHALL be considered orphaned if it has been in its current state longer than STUCK_JOB_TIMEOUT_HOURS (default: 1). Mapping: PREPROCESSING → NEW, INGESTING → PREPROCESSING_COMPLETE, CONSUMING → INGESTING.

#### Scenario: Manual reclaim via method call
- **WHEN** `JobService.reclaim_orphaned_jobs('PREPROCESSING', 'NEW')` is called manually
- **THEN** jobs stuck in PREPROCESSING for longer than STUCK_JOB_TIMEOUT_HOURS SHALL be reset to NEW

#### Scenario: Atomic reclaim prevents double-processing
- **WHEN** two callers attempt to reclaim the same orphaned job simultaneously
- **THEN** only one SHALL succeed due to the atomic UPDATE ... RETURNING * pattern

#### Scenario: Temporal Activity timeout triggers automatic retry
- **WHEN** `USE_TEMPORAL_WHISPER=true` and a transcription Activity exceeds 90 minutes
- **THEN** Temporal SHALL automatically mark the Activity as timed out and schedule a retry (up to 3 attempts), without requiring DuckDB state reclamation