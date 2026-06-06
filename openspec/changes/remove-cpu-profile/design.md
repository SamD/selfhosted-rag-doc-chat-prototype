## Context

The project currently has CPU vs GPU duality across multiple layers. The `run-compose.sh` script accepts `--cpu` or defaults to `--profile cuda`. The `env_strategy.py` module has `CPUEnvConfig` and `GPUEnvConfig` classes. The `LLAMA_USE_GPU` env var controls device selection in `database.py` and `llm_setup.py`. The compose file has `cuda` and `cuda-qdrant` profiles but no `cpu` profile — `--profile cpu` simply omits GPU services rather than providing CPU equivalents. This duality is untested, unused, and complicates every new change that touches deployment or device configuration.

The practical deployment model is: GPU for local inference (llama.cpp with CUDA), remote HTTP endpoints for everything else (llama-server, OpenAI-compatible APIs, docling-serve, whisper-server). CPU-only local inference is not a supported scenario.

## Goals / Non-Goals

**Goals:**
- Remove `--cpu` flag from `run-compose.sh`
- Remove `CPUEnvConfig` from `env_strategy.py`
- Remove `LLAMA_USE_GPU` env var — GPU is assumed; non-GPU users use remote HTTP
- Remove `USE_OLLAMA` env var — off by default, unused, adds complexity
- Simplify device selection in `database.py` and `llm_setup.py` to hardcode `"cuda"`
- Update infrastructure spec and operational playbooks

**Non-Goals:**
- No changes to remote HTTP endpoint support — that remains the primary non-GPU path
- No changes to vector DB profiles (Qdrant/Chroma duality stays)
- No changes to HAProxy load balancing
- No architectural changes to workers or pipeline logic

## Decisions

1. **Remove, don't deprecate**: CPU code paths are dead code. Remove them outright rather than adding deprecation warnings. If CPU local inference is ever needed, it can be reintroduced as a new change with proper testing.

2. **`USE_OLLAMA` goes too**: It's off by default (`"false"`), has its own code path in `chroma_chat.py` using `ConversationalRetrievalChain`, and adds complexity. Users who want Ollama can set `LLM_PATH` to their Ollama server URL since it's OpenAI-compatible.

3. **Device hardcoded to `"cuda"`**: In `database.py` and `llm_setup.py`, the `device` variable derived from `LLAMA_USE_GPU` becomes a hardcoded `"cuda"`. Users without a GPU use remote HTTP endpoints which don't use this device setting.

## Risks / Trade-offs

- **Breaking change for CPU users**: Anyone running local CPU inference with GGUF models will break. Mitigation: this is the intent — they should switch to a remote endpoint (even locally hosted) or reintroduce CPU support as a formal change.
- **Ollama users**: Anyone using `USE_OLLAMA=true` loses that path. Mitigation: Ollama exposes an OpenAI-compatible API, so `LLM_PATH=http://localhost:11434/v1` works as a remote endpoint.
