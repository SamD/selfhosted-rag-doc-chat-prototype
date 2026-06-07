## 1. Consolidate Shared Files

- [x] 1.1 Audit complete — only `llama_strategy.py` imports from `shared.env_names` directly (no imports from `shared.defaults` outside `shared/`)
- [x] 1.2 Audit complete — 3 missing _SETTINGS entries (HF_HOME, HF_HUB_OFFLINE, API_BASE_URL)
- [x] 1.3 Verified: env_names and defaults exist for all doc-ingest settings. 3 _SETTINGS entries added.

## 2. Llama Strategy Cleanup

- [x] 2.1 `llama_strategy.py` — removed direct `shared.env_names` import, uses `config.settings` values directly (no redundant os.getenv wrapper)
- [x] 2.2 Verified: no modules outside `shared/` import from `shared.defaults` or `shared.env_names` directly

## 3. WhisperX Worker

- [x] 3.1 Replaced 7 direct `os.getenv()` calls with imports from `config.settings`

## 4. OCR Utils

- [x] 4.1 Replaced `os.getenv("DEVICE", "cpu")` and `os.getenv("HF_HOME", ...)` with `config.settings`

## 5. Chat Utils

- [x] 5.1 Replaced `os.getenv("API_BASE_URL", ...)` with import from `config.settings`

## 6. Text Utils

- [x] 6.1 Replaced `os.getenv("HF_HUB_OFFLINE", "0")` with import from `config.settings`

## 7. Warmup

- [x] 7.1 Reverted — warmup runs during Docker build where config.settings isn't available. Kept `os.environ.get("OCR_ENDPOINTS", "LOCAL")` as-is.

## 8. Shared Config Sync and Cleanup

- [x] 8.1 Added `ENV_API_BASE_URL`, `ENV_HF_HOME`, `ENV_HF_HUB_OFFLINE` to `shared/env_names.py` imports in `config.py`
- [x] 8.2 Added `DEFAULT_API_BASE_URL`, `DEFAULT_HF_HUB_OFFLINE` to `shared/defaults.py` imports in `config.py`
- [x] 8.3 Added `API_BASE_URL`, `HF_HOME`, `HF_HUB_OFFLINE` to `_SETTINGS` dict
- [x] 8.4 No dead entries found — all _SETTINGS keys are either imported or referenced via `from config import settings`

## 9. Verification

- [x] 9.1 `ruff check` — clean
- [x] 9.2 Full pytest suite — 142/142 passed
