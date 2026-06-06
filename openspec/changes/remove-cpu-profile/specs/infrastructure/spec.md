## MODIFIED Requirements

### Requirement: Docker Compose deployment

The system SHALL be deployable via Docker Compose with profile support. Supported profiles SHALL include: cuda (NVIDIA GPU acceleration), qdrant (Qdrant vector DB), and chroma (Chroma vector DB). The cpu profile SHALL be removed — users without GPU access SHALL use remote HTTP endpoints. Multiple profiles SHALL be composable (e.g., --profile cuda --profile qdrant).

#### Scenario: GPU deployment
- **WHEN** deployed with --profile cuda
- **THEN** containers SHALL have NVIDIA GPU access and CUDA acceleration enabled

#### Scenario: Vector DB profile selection
- **WHEN** deployed with --profile qdrant
- **THEN** Qdrant SHALL be started as the vector database

## REMOVED Requirements

### Requirement: GPU environment strategy

**Reason**: Replaced by hardcoded GPU assumption. The `LLAMA_USE_GPU` env var, `CPUEnvConfig`, and `get_env_strategy()` dispatch are removed. GPU is always assumed; users without GPU access use remote HTTP endpoints.

**Migration**: Remove `LLAMA_USE_GPU` from environment configuration. Remove `env_strategy.py` module entirely or simplify to unconditional GPU config. Device selection in `database.py` and `llm_setup.py` hardcodes `"cuda"`.
