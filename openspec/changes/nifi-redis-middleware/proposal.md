## Why

The current Redis-based messaging works but lacks flow orchestration, provenance tracking, and built-in backpressure management. Apache NiFi provides these capabilities natively, but a full migration is risky. This change introduces NiFi as a transparent middleware layer ("NiFi sandwich") between Redis queues, establishing the foundation for incremental migration while keeping existing workers completely unchanged.

## What Changes

- **NiFi sandwich pattern**: Workers continue using direct Redis LPUSH/BRPOP. NiFi sits between `_input` and `_output` queues, transparently moving messages without any worker code changes.
- **Queue topology**: For each queue (e.g., `ocr_processing_job`), workers push to `ocr_processing_job_input`, NiFi pops and pushes to `ocr_processing_job_output`, consumers pop from `ocr_processing_job_output`.
- **NiFi Python processors**: Two new processors in `nifi/python/extensions/`:
  - `RedisQueueConsumer`: BRPOP/BLPOP/LPOP from Redis `_input` queues using native `redis` library, emit as FlowFile with attributes (one processor per queue)
  - `RedisQueueProducer`: LPUSH/RPUSH FlowFile content to Redis `_output` queues using native `redis` library (one processor per queue)
- **Programmatic NiFi flow setup**: Python script (`nifi/setup_flow.py`) that builds the flow via NiPyAPI against a remote NiFi instance, avoiding manual XML blob management
- **Remote NiFi deployment**: NiFi runs as an external service (not containerized locally). Setup scripts connect via `NIFI_ENDPOINT` (must include `/nifi-api` suffix, e.g., `https://nifi-host:8443/nifi-api`). Self-signed certificate support via `NIFI_SSL_VERIFY=false` (default). Registry configured via `REGISTRY_ENDPOINT`.
- **Configuration**: New `NIFI_ENDPOINT`, `NIFI_SSL_VERIFY`, `NIFI_USERNAME`, `NIFI_PASSWORD`, and `REGISTRY_ENDPOINT` settings in `shared/config.py`

## Capabilities

### New Capabilities
- `nifi-bridge-processors`: NiFi Python processors (`RedisQueueConsumer`, `RedisQueueProducer`) that bridge between Redis `_input` and `_output` queues. One processor instance per queue.
- `nifi-flow-setup`: Programmatic NiFi flow configuration via NiPyAPI against remote NiFi, including process groups, connections, backpressure settings

### Modified Capabilities
<!-- No existing capabilities are being modified. Workers continue using direct Redis calls unchanged. NiFi is inserted as transparent middleware. -->

## Impact

**Code changes**:
- `shared/config.py`: Add `NIFI_ENDPOINT`, `NIFI_SSL_VERIFY`, `NIFI_USERNAME`, `NIFI_PASSWORD`, `REGISTRY_ENDPOINT` settings
- `shared/env_names.py`: Add `ENV_NIFI_ENDPOINT`, `ENV_NIFI_SSL_VERIFY`, `ENV_NIFI_USERNAME`, `ENV_NIFI_PASSWORD`, `ENV_REGISTRY_ENDPOINT` constants
- `shared/defaults.py`: Add `DEFAULT_NIFI_SSL_VERIFY = "false"`
- `nifi/python/extensions/`: Add `RedisQueueConsumer.py`, `RedisQueueProducer.py`
- `nifi/setup_flow.py`: New script for programmatic flow deployment against remote NiFi using NiPyAPI
- `nifi/nifi_client.py`: NiPyAPI wrapper with SSL verify=False and basic auth support for self-signed certs

**No changes to**:
- `doc-ingest-chat/workers/` — workers continue using direct Redis calls
- `doc-ingest-chat/utils/` — no messaging abstraction layer
- `doc-ingest-chat/services/redis_service.py` — remains unchanged

**Dependencies**:
- NiFi 2.x (remote instance, already deployed)
- `nipyapi>=1.0.0` (NiPyAPI 1.x for NiFi 2.0 support)
- `redis>=5.0.0` (native Python redis library for NiFi processors)

**Systems**:
- Remote deployment: `NIFI_ENDPOINT` points to external NiFi instance
- No local NiFi container — NiFi is managed externally
- Zero worker changes — NiFi is transparent middleware
- Backward compatibility: without NiFi running, workers continue using direct Redis (no `_input`/`_output` queues)

## Cross-cutting

**Error handling**:
- NiFi processors handle Redis connection failures by routing to `failure` relationship with error attributes
- Setup script includes retry logic for NiFi REST API failures (exponential backoff, max 3 attempts)
- SSL connection failures logged with descriptive message suggesting `NIFI_SSL_VERIFY=false` for self-signed certs
- Authentication failures logged with hint to check `NIFI_USERNAME`/`NIFI_PASSWORD`

**Logging/observability**:
- NiFi processors set FlowFile attributes: `redis.queue`, `redis.pop.time`, `timestamp`, `trace_id`
- Provenance events automatically captured by NiFi for all FlowFile transformations
- NiFi UI provides real-time flow monitoring and queue depth visualization

**Documentation**:
- Update `AGENTS.md` with `NIFI_ENDPOINT` configuration examples
- Add `infra/operations/day-1.md` section for remote NiFi setup and `NIFI_ENDPOINT` usage
- Add `infra/operations/day-2.md` troubleshooting for NiFi connection issues (including SSL/cert errors)

**Testing strategy**:
- Integration test: verify FlowFile round-trip through NiFi (push to `_input` → NiFi → pop from `_output`)
- NiFi processor tests: verify Redis operations via mocked Redis client
- Smoke test: verify workers continue working without NiFi running (direct Redis mode)
- Correlation test: verify `reply_key` in payload is preserved through NiFi sandwich
