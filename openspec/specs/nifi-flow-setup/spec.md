# NiFi Flow Setup

## Purpose

Provide automated, programmatic setup of the RAG pipeline flow in a NiFi instance, creating processor pairs for each Redis queue with proper configuration, backpressure, and health verification.

## Requirements

### Requirement: Programmatic flow creation via NiPyAPI
The system SHALL provide a Python script (`nifi/setup_flow.py`) that creates the complete RAG pipeline flow in a remote NiFi instance using NiPyAPI 1.x (for NiFi 2.0 support). The script SHALL connect to `NIFI_ENDPOINT` with SSL verification controlled by `NIFI_SSL_VERIFY` (default: `false` for self-signed certs) and authenticate via `NIFI_USERNAME`/`NIFI_PASSWORD` using single-user basic auth. The script SHALL create a process group, `RedisQueueConsumer`/`RedisQueueProducer` processor pairs (one per queue), and connections between them.

#### Scenario: Initial flow creation
- **WHEN** the script runs against a fresh NiFi instance with no existing RAG pipeline flow
- **THEN** it SHALL create a process group named "RAG Pipeline" containing source/sink processor pairs for each queue defined in `QUEUE_NAMES`, `REDIS_OCR_JOB_QUEUE`, and `REDIS_WHISPER_JOB_QUEUE`

#### Scenario: Idempotent execution
- **WHEN** the script runs and the "RAG Pipeline" process group already exists
- **THEN** it SHALL skip creation and log `"Flow already exists, skipping creation"`

#### Scenario: NiFi unavailable
- **WHEN** the script cannot connect to NiFi
- **THEN** it SHALL retry with exponential backoff (max 5 attempts, starting at 2 seconds) and exit with code 1 if all retries fail

#### Scenario: Self-signed certificate
- **WHEN** `NIFI_SSL_VERIFY=false` (default) and NiFi uses a self-signed cert
- **THEN** the script SHALL set `nipyapi.config.nifi_config.verify_ssl = False`

#### Scenario: Authentication
- **WHEN** `NIFI_USERNAME` and `NIFI_PASSWORD` are set
- **THEN** the script SHALL set `nipyapi.config.nifi_config.username` and `nipyapi.config.nifi_config.password`, then call `nipyapi.utils.set_endpoint(NIFI_ENDPOINT, ssl=True, login=True)` for single-user basic auth

#### Scenario: NiFi endpoint URL format
- **WHEN** `NIFI_ENDPOINT` is set
- **THEN** the URL SHALL include the `/nifi-api` suffix (e.g., `https://nifi-host:8443/nifi-api`)

#### Scenario: Registry configuration
- **WHEN** connecting to NiFi
- **THEN** the script SHALL also configure `nipyapi.config.registry_config.host` for Registry access. For simple setups using HTTP Registry (no auth), set `ssl=False, login=False`. For HTTPS Registry, configure auth similarly to NiFi.

### Requirement: NiPyAPI 1.x dependency
The system SHALL use `nipyapi>=1.0.0` (1.x series) for NiFi 2.0 compatibility. The 0.x series SHALL NOT be used as it does not support NiFi 2.0.

#### Scenario: NiPyAPI version
- **WHEN** the system imports nipyapi
- **THEN** it SHALL use version 1.x or higher

### Requirement: Redis connection configuration
The setup script SHALL configure processors with Redis connection details (`REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`) as processor properties. Processors SHALL use the native Python `redis` library directly, NOT NiFi's `RedisConnectionPoolService` controller service.

#### Scenario: Processor Redis configuration
- **WHEN** the setup script creates consumer and producer processors
- **THEN** it SHALL set processor properties `Redis Host`, `Redis Port`, and `Redis DB` from `REDIS_HOST`, `REDIS_PORT`, and `REDIS_DB` environment variables

#### Scenario: No controller service
- **WHEN** the setup script creates the flow
- **THEN** it SHALL NOT create a `RedisConnectionPoolService` controller service (processors use native redis library)

### Requirement: One processor per queue
The setup script SHALL create one `RedisQueueConsumer` and one `RedisQueueProducer` per Redis queue. Each processor instance SHALL be configured with a single `Redis List Key` property.

#### Scenario: Consumer processor configuration
- **WHEN** the setup script creates a consumer for queue `ocr_processing_job`
- **THEN** it SHALL create a `RedisQueueConsumer` with `Redis List Key` set to `ocr_processing_job_input`

#### Scenario: Producer processor configuration
- **WHEN** the setup script creates a producer for queue `ocr_processing_job`
- **THEN** it SHALL create a `RedisQueueProducer` with `Redis List Key` set to `ocr_processing_job_output`

#### Scenario: Processor naming
- **WHEN** processors are created on the NiFi canvas
- **THEN** they SHALL be named descriptively (e.g., `RedisConsumer - ocr_processing_job_input`, `RedisProducer - ocr_processing_job_output`)

### Requirement: Connection configuration with backpressure
The setup script SHALL create connections between consumer and producer processors with configurable backpressure thresholds. Default thresholds SHALL be 10,000 FlowFiles or 1GB total size.

#### Scenario: Connection backpressure
- **WHEN** a connection between a consumer and producer processor reaches 10,000 queued FlowFiles
- **THEN** NiFi SHALL apply backpressure, stopping the consumer processor from generating new FlowFiles until the queue drains below the threshold

#### Scenario: Connection success routing
- **WHEN** a `RedisQueueConsumer` routes a FlowFile to its `success` relationship
- **THEN** the FlowFile SHALL be transferred to the connected `RedisQueueProducer` via the NiFi connection

### Requirement: Flow auto-start
The setup script SHALL start all processors in the process group after successful creation and configuration.

#### Scenario: Auto-start after creation
- **WHEN** the setup script completes flow creation
- **THEN** it SHALL start all processors in the "RAG Pipeline" process group

#### Scenario: Start failure handling
- **WHEN** a processor fails to start (e.g., invalid configuration)
- **THEN** the script SHALL log the error with the processor name and validation errors, and exit with code 1

### Requirement: Environment-driven queue discovery
The setup script SHALL read queue names from environment variables (`QUEUE_NAMES`, `REDIS_OCR_JOB_QUEUE`, `REDIS_WHISPER_JOB_QUEUE`) and create consumer/producer pairs for each.

#### Scenario: Dynamic queue creation
- **WHEN** `QUEUE_NAMES=chunk_ingest_queue:0,chunk_ingest_queue:1,chunk_ingest_queue:2`
- **THEN** the script SHALL create 3 consumer/producer pairs: one for each partitioned queue, with consumer reading from `{name}_input` and producer writing to `{name}_output`

#### Scenario: Missing queue configuration
- **WHEN** `QUEUE_NAMES` is not set
- **THEN** the script SHALL use the default value `"chunk_ingest_queue:0,chunk_ingest_queue:1"` from `shared/defaults.py`

### Requirement: Remote NiFi connection
The setup script SHALL connect to a remote NiFi instance via `NIFI_ENDPOINT` environment variable. No local NiFi container is started.

#### Scenario: NIFI_ENDPOINT configuration
- **WHEN** `NIFI_ENDPOINT=https://nifi.example.com:8443/nifi-api` is set
- **THEN** the script SHALL set `nipyapi.config.nifi_config.host` to that URL (must include `/nifi-api` suffix)

#### Scenario: Missing NIFI_ENDPOINT
- **WHEN** `NIFI_ENDPOINT` is not set
- **THEN** the script SHALL exit with code 1 and log `"NIFI_ENDPOINT environment variable is required"`

### Requirement: Flow health monitoring
The setup script SHALL verify flow health after startup by checking that all processors are running and no bulletins (errors) are present.

#### Scenario: Healthy flow verification
- **WHEN** the setup script completes
- **THEN** it SHALL query NiFi for processor statuses and log `"All processors running"` if all are in RUNNING state

#### Scenario: Unhealthy processor detection
- **WHEN** any processor is in STOPPED or INVALID state after startup
- **THEN** the script SHALL log the processor name, state, and any validation errors at ERROR level
