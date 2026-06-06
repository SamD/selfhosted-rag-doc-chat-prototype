## Why

The project maintains dual CPU/GPU code paths across Docker Compose profiles (--profile cpu vs cuda), environment strategy (CPUEnvConfig vs GPUEnvConfig), and the LLAMA_USE_GPU environment variable. In practice, every deployment uses either remote HTTP endpoints (llama.cpp server, OpenAI-compatible API) or local GPU inference. The CPU-only path is untested, unused, and adds maintenance overhead across 4+ files. Removing it simplifies the deployment model: GPU for local inference, remote HTTP for everything else.

## What Changes

- **BREAKING**: Remove `--cpu` flag from `run-compose.sh` — only GPU profile remains
- **BREAKING**: Remove `CPUEnvConfig` from `env_strategy.py` — `get_env_strategy()` always returns `GPUEnvConfig`
- **REMOVED**: `LLAMA_USE_GPU` environment variable — GPU is always assumed; users who cannot use GPU use remote HTTP endpoints
- **REMOVED**: `cpu` profile logic from `run-compose.sh` profile selection
- **MODIFIED**: Docker worker images — strip CPU-only fallback paths, assume CUDA availability
- **MODIFIED**: Operational playbooks — remove CPU deployment scenarios from day-1.md

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `infrastructure`: The "Docker Compose deployment" requirement changes — CPU profile and `--cpu` flag removed. The "GPU environment strategy" requirement changes — `CPUEnvConfig` removed, `LLAMA_USE_GPU` removed, GPU always assumed.

## Impact

- `doc-ingest-chat/config/env_strategy.py`: Remove `CPUEnvConfig` class, simplify `get_env_strategy()` to return `GPUEnvConfig` unconditionally
- `doc-ingest-chat/run-compose.sh`: Remove `--cpu` flag handling (lines 44-48), remove `GPU_CPU_PROFILE` variable, simplify profile logic
- `doc-ingest-chat/run-compose-cpu.sh`: **DELETE** — entire script is CPU-specific
- `doc-ingest-chat/ingest-dockercompose.yaml`: Remove `LLAMA_USE_GPU=true` from `x-gpu-params` environment (line 34)
- `doc-ingest-chat/ingest-svc.env`: Remove `USE_OLLAMA=0` (line 61)
- `shared/env_names.py`: Remove `ENV_LLAMA_USE_GPU`, `ENV_USE_OLLAMA`, `ENV_OLLAMA_URL`, `ENV_OLLAMA_MODEL`
- `shared/defaults.py`: Remove `DEFAULT_LLAMA_USE_GPU`, `DEFAULT_USE_OLLAMA`, `DEFAULT_OLLAMA_URL`, `DEFAULT_OLLAMA_MODEL`
- `shared/config.py`: Remove all LLAMA_USE_GPU and Ollama settings
- `doc-ingest-chat/config/settings.py`: Remove `LLAMA_USE_GPU` import references
- `doc-ingest-chat/config/llama_strategy.py`: Remove `ENV_LLAMA_USE_GPU` check (line 63)
- `doc-ingest-chat/services/database.py`: Remove `LLAMA_USE_GPU` usage for device selection (hardcode `"cuda"`)
- `doc-ingest-chat/services/database_qdrant_sparse_testing.py`: Remove `LLAMA_USE_GPU` usage
- `doc-ingest-chat/chat/chroma_chat.py`: Remove `USE_OLLAMA` code path
- `doc-ingest-chat/utils/llm_setup.py`: Remove `USE_OLLAMA`, `OLLAMA_URL`, `OLLAMA_MODEL` imports and conditional
- `docs/overview.md`, `docs/quickstart.md`, `AGENTS.md`: Remove or update LLAMA_USE_GPU references
- `infra/operations/day-1.md`: Remove CPU profile option from deployment checklist
- `openspec/specs/infrastructure/spec.md`: Remove CPU scenario from Docker Compose deployment, update GPU strategy requirement
