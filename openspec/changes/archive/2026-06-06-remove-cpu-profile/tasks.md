## 1. Env Vars and Config

- [x] 1.1 Remove `ENV_LLAMA_USE_GPU`, `ENV_USE_OLLAMA`, `ENV_OLLAMA_URL`, `ENV_OLLAMA_MODEL` from `shared/env_names.py`
- [x] 1.2 Remove `DEFAULT_LLAMA_USE_GPU`, `DEFAULT_USE_OLLAMA`, `DEFAULT_OLLAMA_URL`, `DEFAULT_OLLAMA_MODEL` from `shared/defaults.py`
- [x] 1.3 Remove `LLAMA_USE_GPU`, `USE_OLLAMA`, `OLLAMA_URL`, `OLLAMA_MODEL` entries from `shared/config.py` `_SETTINGS` dict
- [x] 1.4 Clean up all Ollama and LLAMA_USE_GPU imports from `shared/config.py`
- [x] 1.5 Update `doc-ingest-chat/config/settings.py` — no references found, no changes needed

## 2. Environment Strategy

- [x] 2.1 Replace `doc-ingest-chat/config/env_strategy.py` — remove `CPUEnvConfig`, remove `LLAMA_USE_GPU` import/check, make `get_env_strategy()` return `GPUEnvConfig` unconditionally
- [x] 2.2 Update `doc-ingest-chat/config/llama_strategy.py` — remove `ENV_LLAMA_USE_GPU` conditional (line 63)

## 3. Docker Compose and Scripts

- [x] 3.1 Remove `--cpu` flag handling from `run-compose.sh` (lines 44-48)
- [x] 3.2 Remove `GPU_CPU_PROFILE` variable from `run-compose.sh`, simplify profile logic (line 77-78)
- [ ] 3.3 Delete `run-compose-cpu.sh` entirely — **BLOCKER: rm blocked, user must delete manually**
- [x] 3.4 Remove `LLAMA_USE_GPU=true` from `ingest-dockercompose.yaml` `x-gpu-params` environment (line 34)
- [x] 3.5 Remove `USE_OLLAMA=0` from `ingest-svc.env` (line 61)

## 4. Device Selection and Ollama Removal

- [x] 4.1 In `doc-ingest-chat/services/database.py`: replace `device = "cuda" if LLAMA_USE_GPU else "cpu"` with hardcoded `device = "cuda"`
- [x] 4.2 In `doc-ingest-chat/services/database_qdrant_sparse_testing.py`: same device hardcode
- [x] 4.3 In `doc-ingest-chat/utils/llm_setup.py`: remove `USE_OLLAMA`, `OLLAMA_URL`, `OLLAMA_MODEL` imports and `ChatOllama` code path
- [x] 4.4 In `doc-ingest-chat/chat/chroma_chat.py`: remove `USE_OLLAMA` import and the Ollama branch, keep only the custom path

## 5. Documentation and Specs

- [x] 5.1 Update `infra/operations/day-1.md` — remove `LLAMA_USE_GPU` env var reference and CPU profile option
- [x] 5.2 Update `docs/quickstart.md` — removed `LLAMA_USE_GPU` references
- [x] 5.3 Update `docs/overview.md` — updated `LLAMA_USE_GPU` reference
- [x] 5.4 Update `AGENTS.md` — removed `LLAMA_USE_GPU`, updated `CPUEnvConfig` references

## 6. Verification

- [x] 6.1 Run `ruff check` on all changed files — clean
- [x] 6.2 Run full pytest suite — 133/133 passed
- [x] 6.3 Verify `run-compose.sh` no longer accepts `--cpu` flag
- [ ] 6.4 Delete `run-compose-cpu.sh` — **BLOCKER: rm blocked by shell rules, delete manually**
