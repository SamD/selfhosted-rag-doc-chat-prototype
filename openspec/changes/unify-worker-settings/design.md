## Context

The shared settings system (`shared/config.py`) provides lazy-loaded environment variables with canonical defaults from `shared/defaults.py`. Workers and utilities should load settings via `from config.settings import NAME`, which triggers `_SETTINGS[NAME]()` — a lambda that calls `os.getenv(ENV_NAME, DEFAULT_VALUE)` at access time. This ensures env var priority over defaults and lazy resolution.

An audit found 15+ direct `os.getenv()` calls across 6+ modules that bypass this system. Some settings in `shared/config.py` may be dead (no longer consumed). Some env vars used directly may not have entries in the shared system at all.

## Goals / Non-Goals

**Goals:**
- All modules load configuration through `config.settings`
- Missing env vars added to the shared system
- Dead entries removed from the shared system
- `env_names.py`, `defaults.py`, `config.py` in sync
- No functional changes to any setting values or env var names

**Non-Goals:**
- No changes to the lazy-loading mechanism itself
- No renames of existing env vars
- No changes to the `config/settings.py` re-export layer

## Decisions

1. **Audit first, then fix**: List all direct `os.getenv()` calls and all `_SETTINGS` entries. Cross-reference to find what needs adding and what needs removing.

2. **Replace, don't wrap**: Each `os.getenv("KEY", default)` becomes `from config.settings import KEY`. The shared system's lazy lambda resolves the value. No inline overrides needed — the lambda already applies `os.getenv(ENV_NAME, DEFAULT_VALUE)`.

3. **Consolidation of shared files**: `shared/env_names.py` and `shared/defaults.py` are implementation details consumed only by `shared/config.py`. No module outside `shared/` shall import from them directly. After consolidation, `config.settings` (the re-export layer) is the only import path for all workers.

4. **`llama_strategy.py` needs cleanup**: It currently imports `DEFAULT_LLAMA_USE_GPU` from `shared.defaults` and `ENV_LLAMA_USE_GPU` from `shared.env_names` directly. These must be replaced with imports from `config.settings`.

5. **`HAPROXY_SUPERVISOR_ENDPOINTS` is special**: This env var is set dynamically by `run-compose.sh` and read by `gatekeeper_logic.py`. It follows a different pattern — set by the compose script, not by user config. Leave as direct `os.environ.get()` unless a strong case exists to move it.

6. **Dead entry detection**: An entry in `_SETTINGS` is dead if no module imports it via `from config.settings import X` AND no module references it via `settings.X`. Grep the entire `doc-ingest-chat/` tree to confirm.

## Risks / Trade-offs

- **Import order**: `config.settings` must be imported after the sys.path fix in each worker. Follow the existing pattern from other workers.
- **Backward compatibility**: All existing env var names remain valid. Only the import mechanism changes. No defaults change.
