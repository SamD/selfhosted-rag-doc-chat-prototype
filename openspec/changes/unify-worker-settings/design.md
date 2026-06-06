## Context

The shared settings system (`shared/config.py` → `config/settings.py`) provides lazy-loaded environment variables with canonical defaults from `shared/defaults.py`. Most workers already use this system via `from config.settings import SETTING_NAME`. However, `whisperx_worker.py` predates the shared settings system and reads env vars directly, creating a maintenance hazard where defaults can diverge.

## Goals / Non-Goals

**Goals:**
- All workers load settings through `config.settings`
- No functional changes to any setting values
- All existing env var names remain valid

**Non-Goals:**
- No changes to workers that already use shared settings
- No changes to the settings system itself
- No changes to env var names or semantics

## Decisions

1. **Import at module level**: Replace each `os.getenv("KEY", default)` with `from config.settings import KEY`. The lazy-loading machinery resolves the value on first access, same as other workers.

2. **Preserve module-level constants**: The whisperx worker assigns settings to module-level constants (`BATCH_SIZE`, `DEVICE`, etc.) for use throughout the file. Keep this pattern, just load the values from shared settings instead of os.getenv.

## Risks / Trade-offs

- **Import order**: `config.settings` must be imported after the sys.path fix at the top of the worker. Verify the existing pattern used by other workers and replicate it.
