## Why

Multiple workers and utility modules read environment variables directly via `os.getenv()` with inline defaults, bypassing the shared lazy-loaded settings system (`shared/config.py` → `config/settings.py`). This causes default drift (e.g., `MEDIA_BATCH_SIZE` defaulted to 16 in whisperx_worker while `shared/defaults.py` had 8), inconsistent configuration resolution, and settings that are resolved at module import time instead of lazily. Additionally, `shared/config.py` may contain entries that are no longer consumed, and may be missing entries for env vars that workers read directly.

## What Changes

- Audit all workers, handlers, utils, and services for direct `os.getenv()` / `os.environ.get()` calls
- Replace each direct call with an import from `config.settings` (env var priority, shared default fallback)
- Consolidate `shared/defaults.py` and `shared/env_names.py` INTO `shared/config.py` — `config.py` is the single entry point. The other two become implementation details consumed only by `config.py`, not imported directly by workers.
- Move any env var entries that workers read directly but aren't in the shared system into `config.py`
- Remove dead entries from all three shared files
- No module outside `shared/` shall import from `shared.defaults` or `shared.env_names` directly — only through `shared.config` or `config.settings`

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `infrastructure`: The "Environment variable configuration" requirement changes — all workers MUST load settings through `config.settings`, not via direct `os.getenv()` calls. The shared settings files must be kept in sync with each other and pruned of dead entries.

## Impact

- `doc-ingest-chat/workers/whisperx_worker.py`: 6 direct `os.getenv()` calls → imports from `config.settings`
- `doc-ingest-chat/utils/ocr_utils.py`: 2 direct `os.getenv()` calls → imports from `config.settings`
- `doc-ingest-chat/utils/chat_utils.py`: 2 direct `os.getenv()` calls → imports from `config.settings`
- `doc-ingest-chat/utils/text_utils.py`: 1 direct `os.getenv()` call → import from `config.settings`
- `doc-ingest-chat/utils/warmup.py`: 1 direct `os.environ.get()` call → import from `config.settings`
- `doc-ingest-chat/workers/gatekeeper_logic.py`: 1 direct `os.environ.get("HAPROXY_SUPERVISOR_ENDPOINTS")` call
- `shared/env_names.py`: Consolidated INTO `shared/config.py`. No module outside `shared/` may import from this directly.
- `shared/defaults.py`: Consolidated INTO `shared/config.py`. No module outside `shared/` may import from this directly.
- `shared/config.py`: Becomes the single source of truth. Workers import from `config.settings` which delegates here.
- `doc-ingest-chat/config/llama_strategy.py`: Remove direct `shared.defaults` and `shared.env_names` imports — must use `config.settings` instead.
