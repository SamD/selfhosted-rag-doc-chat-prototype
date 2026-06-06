# Infrastructure

## Purpose

The infrastructure capability manages deployment, containerization, environment configuration, and startup sequencing. It defines Docker Compose profiles for GPU and vector DB selection, worker Docker images, the 60+ environment variable configuration system, and the warmup sequence that initializes models and services.

## Requirements

### Requirement: Docker Compose deployment

The system SHALL be deployable via Docker Compose with profile support. Supported profiles SHALL include: cuda (NVIDIA GPU acceleration), qdrant (Qdrant vector DB), and chroma (Chroma vector DB). The cpu profile SHALL be removed — users without GPU access SHALL use remote HTTP endpoints. Multiple profiles SHALL be composable (e.g., --profile cuda --profile qdrant).

#### Scenario: GPU deployment
- **WHEN** deployed with --profile cuda
- **THEN** containers SHALL have NVIDIA GPU access and CUDA acceleration enabled

#### Scenario: Vector DB profile selection
- **WHEN** deployed with --profile qdrant
- **THEN** Qdrant SHALL be started as the vector database

### Requirement: Worker Docker images

The system SHALL provide separate Docker images for standard workers (Dockerfile.worker) and the WhisperX worker (Dockerfile.whisperx). Builds SHALL use uv export --frozen and pip install to pin dependency versions from the lock file.

#### Scenario: Standard worker build
- **WHEN** Dockerfile.worker is built
- **THEN** dependencies SHALL be installed via uv export --frozen and pip install

#### Scenario: WhisperX worker build
- **WHEN** Dockerfile.whisperx is built
- **THEN** the WhisperX-specific dependencies and model SHALL be included

### Requirement: Environment variable configuration

The system SHALL support 60+ environment variables for configuration. Critical required variables SHALL include: DEFAULT_DOC_INGEST_ROOT, EMBEDDING_ENDPOINTS, LLM_PATH, SUPERVISOR_LLM_ENDPOINTS. Missing required variables SHALL cause a CRITICAL ERROR message and sys.exit(1). Variables SHALL be loaded lazily on first access.

#### Scenario: Missing required variable
- **WHEN** a required environment variable is not set
- **THEN** the system SHALL log "CRITICAL ERROR: Environment variable '{key}' is NOT set." and exit

#### Scenario: Lazy loading
- **WHEN** a setting is accessed
- **THEN** it SHALL be resolved from the environment at access time, not at import time

### Requirement: Startup and warmup sequencing

The system SHALL run a warmup sequence on startup. The warmup SHALL perform model loading and initialization. When OCR_ENDPOINTS is a remote URL, the warmup SHALL skip importing heavy Docling dependencies.

#### Scenario: Remote OCR warmup
- **WHEN** OCR_ENDPOINTS is a remote URL
- **THEN** the warmup SHALL skip importing Docling to reduce startup time and memory

#### Scenario: GPU environment strategy
- **WHEN** the system starts
- **THEN** the system SHALL set CUDA_VISIBLE_DEVICES="0" and PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" (GPU is always assumed; non-GPU deployments use remote HTTP endpoints)

### Requirement: HAProxy container lifecycle

HAProxy containers SHALL start automatically when *_ENDPOINTS variables are set, without requiring a separate Docker Compose profile. The run-compose.sh script SHALL detect endpoint variables and add HAProxy service definitions accordingly.

#### Scenario: Automatic HAProxy container
- **WHEN** run-compose.sh detects *_ENDPOINTS variables
- **THEN** HAProxy containers SHALL be included in the Docker Compose startup for those services
