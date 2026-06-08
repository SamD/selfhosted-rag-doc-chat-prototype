## ADDED Requirements

### Requirement: Programmatic flow creation
The system SHALL provide a Python script (`nifi/setup_flow.py`) that creates the complete RAG pipeline flow in NiFi via the NiFi REST API. The script SHALL create a process group, Redis controller service, source/sink processor pairs for each queue, and connections between them.

#### Scenario: Initial flow creation
- **WHEN** the script runs against a fresh NiFi instance with no existing RAG pipeline flow
- **THEN** it SHALL create a process group named "RAG Pipeline" containing source/sink processor pairs for each queue defined in `QUEUE_NAMES`, `REDIS_OCR_JOB_QUEUE`, and `REDIS_WHISPER_JOB_QUEUE`

#### Scenario: Idempotent execution
- **WHEN** the script runs and the "RAG Pipeline" process group already exists
- **THEN** it SHALL skip creation and log `"Flow already exists, skipping creation"`

#### Scenario: NiFi unavailable
- **WHEN** the script cannot connect to the NiFi REST API
- **THEN** it SHALL retry with exponential backoff (max 5 attempts, starting at 2 seconds) and exit with code 1 if all retries fail

### Requirement: Redis controller service configuration
The setup script SHALL create a `RedisConnectionPoolService` controller service within the process group, configured with `REDIS_HOST` and `REDIS_PORT` from environment variables.

#### Scenario: Controller service creation
- **WHEN** the setup script creates the flow
- **THEN** it SHALL create a `RedisConnectionPoolService` named "Redis Pool" with host and port from `REDIS_HOST` and `REDIS_PORT` env vars, and enable it

#### Scenario: Controller service reuse
- **WHEN** a `RedisConnectionPoolService` named "Redis Pool" already exists in the process group
- **THEN** the script SHALL reuse the existing service without creating a duplicate

### Requirement: Connection configuration with backpressure
The setup script SHALL create connections between source and sink processors with configurable backpressure thresholds. Default thresholds SHALL be 10,000 FlowFiles or 1GB total size.

#### Scenario: Connection backpressure
- **WHEN** a connection between a source and sink processor reaches 10,000 queued FlowFiles
- **THEN** NiFi SHALL apply backpressure, stopping the source processor from generating new FlowFiles until the queue drains below the threshold

#### Scenario: Connection success routing
- **WHEN** a `RedisSourceProcessor` routes a FlowFile to its `success` relationship
- **THEN** the FlowFile SHALL be transferred to the connected `RedisSinkProcessor` via the NiFi connection

### Requirement: Flow auto-start
The setup script SHALL start all processors in the process group after successful creation and configuration.

#### Scenario: Auto-start after creation
- **WHEN** the setup script completes flow creation
- **THEN** it SHALL start all processors in the "RAG Pipeline" process group

#### Scenario: Start failure handling
- **WHEN** a processor fails to start (e.g., invalid configuration)
- **THEN** the script SHALL log the error with the processor name and validation errors, and exit with code 1

### Requirement: Environment-driven queue discovery
The setup script SHALL read queue names from environment variables (`QUEUE_NAMES`, `REDIS_OCR_JOB_QUEUE`, `REDIS_WHISPER_JOB_QUEUE`) and create source/sink pairs for each.

#### Scenario: Dynamic queue creation
- **WHEN** `QUEUE_NAMES=chunk_ingest_queue:0,chunk_ingest_queue:1,chunk_ingest_queue:2`
- **THEN** the script SHALL create 3 source/sink pairs: one for each partitioned queue, with source reading from `{name}_input` and sink writing to `{name}_output`

#### Scenario: Missing queue configuration
- **WHEN** `QUEUE_NAMES` is not set
- **THEN** the script SHALL use the default value `"chunk_ingest_queue:0,chunk_ingest_queue:1"` from `shared/defaults.py`

### Requirement: Docker-compose integration
The system SHALL add a NiFi service to `ingest-dockercompose.yaml` that starts only when `MESSAGING_IMPL=nifi`. The service SHALL mount `nifi/setup_flow.py` and execute it after NiFi is ready.

#### Scenario: NiFi container startup
- **WHEN** `MESSAGING_IMPL=nifi` and docker-compose is started
- **THEN** the NiFi container SHALL start, wait for the NiFi REST API to be available, and execute `setup_flow.py`

#### Scenario: NIFI_ENDPOINT override
- **WHEN** `NIFI_ENDPOINT=http://nifi-core01:8080` is set
- **THEN** docker-compose SHALL NOT start a local NiFi container, and `setup_flow.py` SHALL connect to the remote endpoint

#### Scenario: Redis dependency
- **WHEN** the NiFi container starts
- **THEN** it SHALL depend on the Redis container (same as other services)

### Requirement: Flow health monitoring
The setup script SHALL verify flow health after startup by checking that all processors are running and no bulletins (errors) are present.

#### Scenario: Healthy flow verification
- **WHEN** the setup script completes
- **THEN** it SHALL query the NiFi REST API for processor statuses and log `"All processors running"` if all are in RUNNING state

#### Scenario: Unhealthy processor detection
- **WHEN** any processor is in STOPPED or INVALID state after startup
- **THEN** the script SHALL log the processor name, state, and any validation errors at ERROR level
