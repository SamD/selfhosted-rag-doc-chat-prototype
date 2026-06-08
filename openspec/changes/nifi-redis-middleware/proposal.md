## Why

The current Redis-based messaging works but lacks flow orchestration, provenance tracking, and built-in backpressure management. Apache NiFi provides these capabilities natively, but a full migration is risky. This change introduces NiFi as a middleware layer between workers and Redis, establishing the foundation for incremental migration while keeping existing workers unchanged.

## What Changes

- **Strategy pattern for messaging**: Introduce `MessageQueue` abstraction with `RedisQueue` and `NifiQueue` implementations, selected via `MESSAGING_IMPL` environment variable
- **Queue name resolution**: When `MESSAGING_IMPL=nifi`, queue names are transformed with `_input`/`_output` suffixes to avoid infinite loops (workers push to `{queue}_input`, NiFi pops and pushes to `{queue}_output`, consumers pop from `{queue}_output`)
- **NiFi Python processors**: Two new processors in `nifi/python/extensions/`:
  - `RedisSourceProcessor`: BRPOP/BLPOP from Redis, emit as FlowFile with attributes
  - `RedisSinkProcessor`: LPUSH/RPUSH FlowFile content to Redis
- **Programmatic NiFi flow setup**: Python script (`nifi/setup_flow.py`) that builds the flow via NiFi REST API at container startup, avoiding manual XML blob management
- **Docker-compose integration**: NiFi container added to `ingest-dockercompose.yaml` with `NIFI_ENDPOINT` override support (similar to HAProxy pattern)
- **Configuration**: New `MESSAGING_IMPL` and `NIFI_ENDPOINT` settings in `shared/config.py`

## Capabilities

### New Capabilities
- `messaging-strategy`: Abstract factory pattern for message queue implementations (Redis vs NiFi), including queue name resolution and client initialization
- `nifi-bridge-processors`: NiFi Python processors that bridge between NiFi flows and Redis queues (bidirectional push/pop operations)
- `nifi-flow-setup`: Programmatic NiFi flow configuration via REST API, including process groups, connections, backpressure settings, and controller services

### Modified Capabilities
<!-- No existing capabilities are being modified at the spec level. Workers continue to use the same queue.push/pop interface; only the underlying implementation changes based on MESSAGING_IMPL. -->

## Impact

**Code changes**:
- `shared/config.py`: Add `MESSAGING_IMPL`, `NIFI_ENDPOINT` settings
- `shared/env_names.py`: Add `ENV_MESSAGING_IMPL`, `ENV_NIFI_ENDPOINT` constants
- `shared/defaults.py`: Add `DEFAULT_MESSAGING_IMPL = "redis"`
- `doc-ingest-chat/services/`: New `messaging/` module with `MessageQueue` ABC, `RedisQueue`, `NifiQueue` implementations
- `doc-ingest-chat/workers/`: Update all queue operations to use `messaging.push()`/`messaging.pop()` with resolved queue names
- `doc-ingest-chat/utils/ocr_utils.py`, `whisper_utils.py`, `producer_utils.py`: Replace direct Redis calls with messaging facade
- `nifi/python/extensions/`: Add `RedisSourceProcessor.py`, `RedisSinkProcessor.py`
- `nifi/setup_flow.py`: New script for programmatic flow deployment
- `doc-ingest-chat/ingest-dockercompose.yaml`: Add NiFi service, update worker environment variables

**Dependencies**:
- NiFi 2.x container (apache/nifi:2.x)
- NiFi RedisConnectionPoolService controller service
- Python `requests` library (already present) for NiFi REST API calls

**Systems**:
- Docker-compose profile: NiFi container starts when `MESSAGING_IMPL=nifi`
- Remote deployment: `NIFI_ENDPOINT` allows connecting to external NiFi instance
- Backward compatibility: `MESSAGING_IMPL=redis` (default) maintains current behavior

## Cross-cutting

**Error handling**:
- `NifiQueue` implementation includes retry logic for REST API failures (exponential backoff, max 3 attempts)
- Graceful degradation: if NiFi is unavailable, log error and raise exception (no silent fallback to Redis)
- NiFi processors handle Redis connection failures by routing to `failure` relationship with error attributes

**Logging/observability**:
- All messaging operations log queue name, operation type, and data size (not content)
- NiFi processors set FlowFile attributes: `queue.name`, `operation`, `timestamp`, `trace_id`
- Provenance events automatically captured by NiFi for all FlowFile transformations

**Documentation**:
- Update `AGENTS.md` with `MESSAGING_IMPL` configuration examples
- Add `infra/operations/day-1.md` section for NiFi setup (local vs remote)
- Add `infra/operations/day-2.md` troubleshooting for NiFi connection issues

**Testing strategy**:
- Unit tests for `MessageQueue` implementations (mock Redis and NiFi REST API)
- Integration test: verify FlowFile round-trip through NiFi (push → NiFi → Redis → pop)
- Smoke test: `MESSAGING_IMPL=redis` maintains current behavior (no regression)
- NiFi processor tests: verify Redis operations via mocked `RedisConnectionPoolService`
