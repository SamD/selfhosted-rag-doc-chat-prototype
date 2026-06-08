## 1. Configuration Setup

- [ ] 1.1 Add `ENV_MESSAGING_IMPL` and `ENV_NIFI_ENDPOINT` constants to `shared/env_names.py`
- [ ] 1.2 Add `DEFAULT_MESSAGING_IMPL = "redis"` to `shared/defaults.py`
- [ ] 1.3 Add `MESSAGING_IMPL` and `NIFI_ENDPOINT` settings to `shared/config.py` `_SETTINGS` dict with lazy-loaded lambdas
- [ ] 1.4 Verify settings load correctly with `python -c "from shared.config import MESSAGING_IMPL, NIFI_ENDPOINT; print(MESSAGING_IMPL, NIFI_ENDPOINT)"`

## 2. Messaging Abstraction

- [ ] 2.1 Create `doc-ingest-chat/services/messaging/` directory structure with `__init__.py`
- [ ] 2.2 Implement `MessageQueue` ABC in `services/messaging/base.py` with abstract methods: `push`, `pop`, `push_reply`, `wait_reply`, `resolve_queue_name`, `queue_length`, `purge`, `blocking_push_with_backpressure`
- [ ] 2.3 Implement `RedisQueue` in `services/messaging/redis_queue.py` that wraps direct Redis operations (LPUSH/BRPOP/RPUSH/BLPOP)
- [ ] 2.4 Implement `NifiQueue` in `services/messaging/nifi_queue.py` with queue name suffix transformation (`_input`/`_output`) and validation
- [ ] 2.5 Implement `get_messaging()` factory function in `services/messaging/__init__.py` with fork-safe PID checking and caching
- [ ] 2.6 Implement consolidated `get_redis_client()` in `services/messaging/redis_client.py` with fork-safe singleton pattern
- [ ] 2.7 Add logging to all messaging operations (queue name, operation type, data size at DEBUG level)
- [ ] 2.8 Write unit tests for `MessageQueue` interface enforcement in `tests/test_messaging_base.py`
- [ ] 2.9 Write unit tests for `RedisQueue` operations (push, pop, reply, backpressure) in `tests/test_messaging_redis.py`
- [ ] 2.10 Write unit tests for `NifiQueue` queue name resolution and validation in `tests/test_messaging_nifi.py`
- [ ] 2.11 Write unit tests for `get_messaging()` factory (default, nifi, unknown, fork safety) in `tests/test_messaging_factory.py`

## 3. NiFi Bridge Processors

- [ ] 3.1 Create `nifi/python/extensions/RedisSourceProcessor.py` with `FlowFileTransform` base class
- [ ] 3.2 Implement `RedisSourceProcessor.Details` with name, version, description, tags, and `dependencies=['redis>=5.0.0']`
- [ ] 3.3 Implement `RedisSourceProcessor` PropertyDescriptors: `Redis Connection Pool`, `Queue Name` (with expression language support), `Pop Operation` (BRPOP/BLPOP), `Timeout Seconds`
- [ ] 3.4 Implement `RedisSourceProcessor.transform()` method: BRPOP/BLPOP from Redis, create FlowFile with content and attributes (`redis.queue`, `redis.pop.time`), route to `success` relationship
- [ ] 3.5 Handle empty queue (timeout) by yielding processor for 1 second and producing no FlowFile
- [ ] 3.6 Handle Redis connection failure by logging error, yielding for 5 seconds, and routing no FlowFile
- [ ] 3.7 Create `nifi/python/extensions/RedisSinkProcessor.py` with `FlowFileTransform` base class
- [ ] 3.8 Implement `RedisSinkProcessor.Details` with name, version, description, tags, and `dependencies=['redis>=5.0.0']`
- [ ] 3.9 Implement `RedisSinkProcessor` PropertyDescriptors: `Redis Connection Pool`, `Queue Name` (with expression language support), `Push Operation` (LPUSH/RPUSH), `TTL Seconds` (optional)
- [ ] 3.10 Implement `RedisSinkProcessor.transform()` method: read FlowFile content, LPUSH/RPUSH to Redis, apply EXPIRE if TTL set, route to `success` relationship
- [ ] 3.11 Handle Redis connection failure by routing FlowFile to `failure` relationship with `redis.error` attribute
- [ ] 3.12 Implement batch push optimization (pipeline multiple FlowFiles in single Redis operation)
- [ ] 3.13 Add logging to both processors (INFO for success, ERROR for failures with traceback)
- [ ] 3.14 Write unit tests for `RedisSourceProcessor` (successful pop, empty queue, connection failure) in `nifi/tests/test_redis_source_processor.py`
- [ ] 3.15 Write unit tests for `RedisSinkProcessor` (successful push, connection failure, batch push, TTL) in `nifi/tests/test_redis_sink_processor.py`

## 4. NiFi Flow Setup Script

- [ ] 4.1 Create `nifi/setup_flow.py` script with NiFi REST API client functions
- [ ] 4.2 Implement `wait_for_nifi(url, max_retries=5)` function with exponential backoff (2s, 4s, 8s, 16s, 32s)
- [ ] 4.3 Implement `create_process_group(nifi_url, name)` function that creates "RAG Pipeline" process group
- [ ] 4.4 Implement `check_flow_exists(nifi_url, name)` function for idempotent execution
- [ ] 4.5 Implement `create_redis_controller_service(nifi_url, process_group_id, redis_host, redis_port)` function that creates and enables `RedisConnectionPoolService`
- [ ] 4.6 Implement `create_source_processor(nifi_url, process_group_id, queue_name, redis_service_id)` function that creates `RedisSourceProcessor` configured for `{queue_name}_input`
- [ ] 4.7 Implement `create_sink_processor(nifi_url, process_group_id, queue_name, redis_service_id)` function that creates `RedisSinkProcessor` configured for `{queue_name}_output`
- [ ] 4.8 Implement `create_connection(nifi_url, process_group_id, source_id, sink_id)` function that creates connection with backpressure (10,000 FlowFiles or 1GB)
- [ ] 4.9 Implement `start_process_group(nifi_url, process_group_id)` function that starts all processors
- [ ] 4.10 Implement `verify_flow_health(nifi_url, process_group_id)` function that checks all processors are RUNNING and logs status
- [ ] 4.11 Implement `main()` function that orchestrates: wait_for_nifi → check_flow_exists → create_process_group → create_redis_controller_service → for each queue: create_source/sink/connection → start_process_group → verify_flow_health
- [ ] 4.12 Read queue names from environment variables (`QUEUE_NAMES`, `REDIS_OCR_JOB_QUEUE`, `REDIS_WHISPER_JOB_QUEUE`) with defaults from `shared/defaults.py`
- [ ] 4.13 Add error handling for NiFi REST API failures with descriptive error messages
- [ ] 4.14 Add logging throughout (INFO for progress, ERROR for failures)
- [ ] 4.15 Write integration test for flow setup script (mock NiFi REST API) in `nifi/tests/test_setup_flow.py`

## 5. Docker-Compose Integration

- [ ] 5.1 Add NiFi service to `doc-ingest-chat/ingest-dockercompose.yaml` with `apache/nifi:2.2.0` image
- [ ] 5.2 Configure NiFi service with environment variables: `NIFI_WEB_HTTP_PORT=8080`, `REDIS_HOST`, `REDIS_PORT`, `QUEUE_NAMES`, `REDIS_OCR_JOB_QUEUE`, `REDIS_WHISPER_JOB_QUEUE`
- [ ] 5.3 Mount `nifi/setup_flow.py` into NiFi container at `/opt/nifi/nifi-current/scripts/setup_flow.py`
- [ ] 5.4 Add entrypoint script that waits for NiFi REST API and executes `setup_flow.py`
- [ ] 5.5 Configure NiFi service to depend on Redis service
- [ ] 5.6 Add conditional startup: NiFi container only starts when `MESSAGING_IMPL=nifi` (use Docker Compose profiles or environment variable check)
- [ ] 5.7 Add `NIFI_ENDPOINT` environment variable to worker services (passed through but not used in Phase 1)
- [ ] 5.8 Test docker-compose startup with `MESSAGING_IMPL=redis` (NiFi should not start)
- [ ] 5.9 Test docker-compose startup with `MESSAGING_IMPL=nifi` (NiFi should start and create flow)

## 6. Worker Integration

- [ ] 6.1 Update `doc-ingest-chat/utils/ocr_utils.py` to use `get_messaging().push()` and `get_messaging().wait_reply()` instead of direct Redis calls
- [ ] 6.2 Update `doc-ingest-chat/utils/whisper_utils.py` to use `get_messaging().push()` and `get_messaging().wait_reply()` instead of direct Redis calls
- [ ] 6.3 Update `doc-ingest-chat/utils/producer_utils.py` to use `get_messaging().push()` and `get_messaging().blocking_push_with_backpressure()` instead of direct Redis calls
- [ ] 6.4 Update `doc-ingest-chat/workers/ocr_worker.py` to use `get_messaging().pop()` instead of direct Redis BRPOP
- [ ] 6.5 Update `doc-ingest-chat/workers/whisperx_worker.py` to use `get_messaging().pop()` and `get_messaging().push_reply()` instead of direct Redis calls
- [ ] 6.6 Update `doc-ingest-chat/workers/consumer_worker.py` to use `get_messaging().pop()` instead of direct Redis BLPOP
- [ ] 6.7 Update `doc-ingest-chat/workers/producer_graph.py` to use `get_messaging().push()` with resolved queue names
- [ ] 6.8 Update `doc-ingest-chat/services/redis_service.py` to delegate to `get_messaging()` for queue operations (maintain backward compatibility for `purge_queue_entries`)
- [ ] 6.9 Update `doc-ingest-chat/services/chat_session_service.py` to use consolidated `get_redis_client()` from `services/messaging/redis_client.py`
- [ ] 6.10 Remove duplicate `get_redis_client()` functions from `ocr_utils.py`, `whisper_utils.py`, `producer_utils.py`
- [ ] 6.11 Run full test suite: `PYTHONPATH=doc-ingest-chat:shared .venv/bin/python -m pytest doc-ingest-chat/tests/ -v`
- [ ] 6.12 Verify no regressions with `MESSAGING_IMPL=redis` (default behavior unchanged)

## 7. Integration Testing

- [ ] 7.1 Write integration test: verify FlowFile round-trip through NiFi (push to `_input` queue → NiFi pops → NiFi pushes to `_output` queue → pop from `_output` queue) in `tests/test_nifi_integration.py`
- [ ] 7.2 Write smoke test: verify `MESSAGING_IMPL=redis` maintains current behavior (no regression) in `tests/test_messaging_smoke_redis.py`
- [ ] 7.3 Test OCR job flow with `MESSAGING_IMPL=nifi` (manual test with docker-compose)
- [ ] 7.4 Test Whisper job flow with `MESSAGING_IMPL=nifi` (manual test with docker-compose)
- [ ] 7.5 Test chunk ingest flow with `MESSAGING_IMPL=nifi` (manual test with docker-compose)
- [ ] 7.6 Verify NiFi provenance events are captured for all FlowFile transformations (check NiFi UI or REST API)

## 8. Documentation

- [ ] 8.1 Update `AGENTS.md` with `MESSAGING_IMPL` configuration examples and queue name resolution explanation
- [ ] 8.2 Add section to `infra/operations/day-1.md` for NiFi setup (local vs remote, `NIFI_ENDPOINT` usage)
- [ ] 8.3 Add troubleshooting section to `infra/operations/day-2.md` for NiFi connection issues (NiFi unavailable, flow not created, processors stopped)
- [ ] 8.4 Update `docs/overview.md` to mention NiFi middleware as optional layer
- [ ] 8.5 Update `docs/quickstart.md` with `MESSAGING_IMPL` configuration
- [ ] 8.6 Update `CHANGELOG.md` under `[Unreleased]` → `Added` section with: messaging strategy abstraction, NiFi bridge processors, NiFi flow setup script, docker-compose NiFi integration
- [ ] 8.7 Add README to `nifi/` directory explaining the processors, flow setup script, and how to extend

## 9. Code Quality

- [ ] 9.1 Run `ruff check --fix` on all modified files
- [ ] 9.2 Run `ruff check` to verify no linting errors remain
- [ ] 9.3 Run full test suite: `PYTHONPATH=doc-ingest-chat:shared .venv/bin/python -m pytest doc-ingest-chat/tests/ -v`
- [ ] 9.4 Verify all tests pass (142+ tests)
- [ ] 9.5 Run `npm run build` in `astro-frontend/` to verify no frontend regressions
- [ ] 9.6 Review all new code for proper logging (logger name, level, message format)
- [ ] 9.7 Review all new code for proper error handling (graceful degradation, exception context)
- [ ] 9.8 Review all new code for fork safety (PID checking on singletons)
