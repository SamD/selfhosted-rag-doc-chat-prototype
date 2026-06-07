## MODIFIED Requirements

### Requirement: Environment variable configuration

The system SHALL support 60+ environment variables for configuration. Critical required variables SHALL include: DEFAULT_DOC_INGEST_ROOT, EMBEDDING_ENDPOINTS, LLM_PATH, SUPERVISOR_LLM_ENDPOINTS. Missing required variables SHALL cause a CRITICAL ERROR message and sys.exit(1). Variables SHALL be loaded lazily on first access.

All workers, handlers, utils, and services SHALL load their configuration through `from config.settings import <NAME>`, which resolves the value lazily with env var priority over the canonical default. Direct `os.getenv()` calls with inline defaults SHALL NOT be used — they bypass the shared system and allow defaults to drift.

The three shared files (`shared/env_names.py`, `shared/defaults.py`, `shared/config.py`) SHALL be kept in sync. Dead entries (env var constants or settings that no module consumes) SHALL be removed. Missing entries for env vars that modules read directly SHALL be added.

#### Scenario: Missing required variable
- **WHEN** a required environment variable is not set
- **THEN** the system SHALL log "CRITICAL ERROR: Environment variable '{key}' is NOT set." and exit

#### Scenario: Lazy loading
- **WHEN** a setting is accessed
- **THEN** it SHALL be resolved from the environment at access time, not at import time

#### Scenario: Worker loads setting via shared config
- **WHEN** a worker needs a configuration value
- **THEN** it SHALL import it from `config.settings` rather than calling `os.getenv()` directly
