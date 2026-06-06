## Why

Several workers read environment variables directly via `os.getenv()` with inline defaults instead of using the shared lazy-loaded settings from `shared/config.py`. This means defaults can drift (as happened with `MEDIA_BATCH_SIZE` defaulting to 16 in the worker while `shared/defaults.py` had 8), and settings are resolved at module import time instead of lazily. The whisperx worker is the worst offender with 6 direct `os.getenv` calls.

## What Changes

- **MODIFIED**: `doc-ingest-chat/workers/whisperx_worker.py` — replace all 6 `os.getenv()` calls with imports from `config.settings`
- **MODIFIED**: `doc-ingest-chat/workers/ocr_worker.py` — replace 2 debug `os.getenv()` calls with shared settings
- No breaking changes — all existing env var names remain the same

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `infrastructure`: The environment variable configuration requirement is strengthened — all workers must load settings through the shared system, not via direct `os.getenv()` calls.

## Impact

- `doc-ingest-chat/workers/whisperx_worker.py`: Replace `REDIS_HOST`, `REDIS_PORT`, `REDIS_WHISPER_JOB_QUEUE`, `DEVICE`, `COMPUTE_TYPE`, `BATCH_SIZE` with imports from `config.settings`
- `doc-ingest-chat/workers/ocr_worker.py`: Replace debug print `os.getenv` calls with shared settings
