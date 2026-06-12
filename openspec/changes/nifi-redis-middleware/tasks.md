## 1. Configuration Setup

- [x] 1.1 Add `ENV_NIFI_ENDPOINT`, `ENV_NIFI_SSL_VERIFY`, `ENV_NIFI_USERNAME`, `ENV_NIFI_PASSWORD`, `ENV_REGISTRY_ENDPOINT` constants to `shared/env_names.py`
- [x] 1.2 Add `DEFAULT_NIFI_SSL_VERIFY = "false"` to `shared/defaults.py`
- [x] 1.3 Add `NIFI_ENDPOINT`, `NIFI_SSL_VERIFY`, `NIFI_USERNAME`, `NIFI_PASSWORD`, `REGISTRY_ENDPOINT` settings to `shared/config.py` `_SETTINGS` dict with lazy-loaded lambdas
- [x] 1.4 Add `nipyapi>=1.0.0` to `pyproject.toml` dependencies (NiPyAPI 1.x for NiFi 2.0 support)
- [x] 1.5 Verify settings load correctly with `python -c "from shared.config import NIFI_ENDPOINT, NIFI_SSL_VERIFY, NIFI_USERNAME, NIFI_PASSWORD, REGISTRY_ENDPOINT; print(NIFI_ENDPOINT)"`

## 2. NiFi Bridge Processors

- [x] 2.1 Create `nifi/python/extensions/RedisQueueConsumer.py` with `FlowFileSource` base class (following NiFi 2.x Python processor API)
- [x] 2.2 Implement `RedisQueueConsumer.ProcessorDetails` with name, version, description, tags, and `dependencies=['redis>=5.0.0']`
- [x] 2.3 Implement `RedisQueueConsumer` PropertyDescriptors: `Redis Host`, `Redis Port`, `Redis DB`, `Redis List Key` (with expression language support), `Pop Operation` (BRPOP/BLPOP/LPOP), `Timeout Seconds`
- [x] 2.4 Implement `RedisQueueConsumer.onScheduled()` method to create `redis.ConnectionPool` from properties
- [x] 2.5 Implement `RedisQueueConsumer.create()` method: BRPOP/BLPOP/LPOP from Redis `_input` queue, create FlowFile with content and attributes (`redis.queue`, `redis.pop.time`), route to `success` relationship
- [x] 2.6 Handle empty queue (timeout) by returning `None` so NiFi yields and stops burning CPU
- [x] 2.7 Handle Redis connection failure by logging error and returning `None`
- [x] 2.8 Create `nifi/python/extensions/RedisQueueProducer.py` with `FlowFileTransform` base class
- [x] 2.9 Implement `RedisQueueProducer.ProcessorDetails` with name, version, description, tags, and `dependencies=['redis>=5.0.0']`
- [x] 2.10 Implement `RedisQueueProducer` PropertyDescriptors: `Redis Host`, `Redis Port`, `Redis DB`, `Redis List Key` (with expression language support), `Push Operation` (LPUSH/RPUSH), `TTL Seconds` (optional)
- [x] 2.11 Implement `RedisQueueProducer.onScheduled()` method to create `redis.ConnectionPool` from properties
- [x] 2.12 Implement `RedisQueueProducer.transform()` method: read FlowFile content, LPUSH/RPUSH to Redis `_output` queue, apply EXPIRE if TTL set, route to `success` relationship
- [x] 2.13 Handle Redis connection failure by routing FlowFile to `failure` relationship with `redis.error` attribute
- [x] 2.14 Implement batch push optimization (pipeline multiple FlowFiles in single Redis operation)
- [x] 2.15 Add logging to both processors (INFO for success, ERROR for failures with traceback)
- [x] 2.16 Write unit tests for `RedisQueueConsumer` (successful pop, empty queue, connection failure) in `nifi/tests/test_redis_queue_consumer.py`
- [x] 2.17 Write unit tests for `RedisQueueProducer` (successful push, connection failure, batch push, TTL) in `nifi/tests/test_redis_queue_producer.py`

## 3. NiFi Client (NiPyAPI Wrapper)

- [x] 3.1 Create `nifi/nifi_client.py` with `NifiClient` class that wraps NiPyAPI configuration
- [x] 3.2 Implement `NifiClient.__init__(base_url, username, password, ssl_verify, registry_url=None)` that:
  - Sets `nipyapi.config.nifi_config.host = base_url` (must include `/nifi-api` suffix)
  - Sets `nipyapi.config.nifi_config.verify_ssl = ssl_verify`
  - Sets `nipyapi.config.nifi_config.username = username` and `password = password`
  - Configures Registry if `registry_url` provided (HTTP = no auth, HTTPS = same credentials)
  - Calls `nipyapi.utils.set_endpoint(base_url, ssl=True, login=True)`
- [x] 3.3 Implement `wait_for_nifi(max_retries=5)` method with exponential backoff (2s, 4s, 8s, 16s, 32s) that calls `nipyapi.canvas.get_root_pg_id()`
- [x] 3.4 Implement `create_process_group(name)` method using NiPyAPI flow methods
- [x] 3.5 Implement `check_flow_exists(name)` method for idempotent execution by querying existing process groups
- [x] 3.6 Implement `create_consumer_processor(process_group_id, queue_name, redis_host, redis_port, redis_db)` method that creates `RedisQueueConsumer` with Redis connection properties
- [x] 3.7 Implement `create_producer_processor(process_group_id, queue_name, redis_host, redis_port, redis_db)` method that creates `RedisQueueProducer` with Redis connection properties
- [x] 3.8 Implement `create_connection(process_group_id, source_id, sink_id)` method that creates connection with backpressure (10,000 FlowFiles or 1GB)
- [x] 3.9 Implement `start_process_group(process_group_id)` method that starts all processors
- [x] 3.10 Implement `verify_flow_health(process_group_id)` method that checks all processors are RUNNING
- [x] 3.11 Add error handling for all NiPyAPI calls with descriptive error messages (connection refused, SSL errors, auth failures)
- [x] 3.12 Add logging throughout (INFO for progress, ERROR for failures, DEBUG for API details)
- [x] 3.13 Write unit tests for `NifiClient` (successful requests, SSL errors, auth failures, retries) in `nifi/tests/test_nifi_client.py`

## 4. NiFi Flow Setup Script

- [x] 4.1 Create `nifi/setup_flow.py` script that uses `NifiClient` to orchestrate flow creation
- [x] 4.2 Implement `main()` function that reads `NIFI_ENDPOINT`, `NIFI_SSL_VERIFY`, `NIFI_USERNAME`, `NIFI_PASSWORD`, `REGISTRY_ENDPOINT` from environment, instantiates `NifiClient`, and calls `wait_for_nifi()`
- [x] 4.3 Implement flow creation orchestration: `check_flow_exists()` → `create_process_group()` → for each queue: `create_consumer_processor()` + `create_producer_processor()` + `create_connection()` → `start_process_group()` → `verify_flow_health()`
- [x] 4.4 Read queue names from environment variables (`QUEUE_NAMES`, `REDIS_OCR_JOB_QUEUE`, `REDIS_WHISPER_JOB_QUEUE`) with defaults from `shared/defaults.py`
- [x] 4.5 Read Redis connection details from environment variables (`REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`) and pass to processor creation methods
- [x] 4.6 Add error handling for missing environment variables with descriptive error messages
- [x] 4.7 Add exit code 1 on failure, exit code 0 on success
- [x] 4.8 Write integration test for flow setup script (mock `NifiClient`) in `nifi/tests/test_setup_flow.py`

## 5. Worker Queue Name Migration

- [x] 5.1 Update `doc-ingest-chat/utils/ocr_utils.py` to push to `ocr_processing_job_input` instead of `ocr_processing_job`
- [x] 5.2 Update `doc-ingest-chat/utils/whisper_utils.py` to push to `whisper_processing_job_input` instead of `whisper_processing_job`
- [x] 5.3 Update `doc-ingest-chat/utils/producer_utils.py` to push to `{queue_name}_input` instead of `{queue_name}`
- [x] 5.4 Update `doc-ingest-chat/workers/ocr_worker.py` to pop from `ocr_processing_job_output` instead of `ocr_processing_job`
- [x] 5.5 Update `doc-ingest-chat/workers/whisperx_worker.py` to pop from `whisper_processing_job_output` instead of `whisper_processing_job`
- [x] 5.6 Update `doc-ingest-chat/workers/consumer_worker.py` to pop from `{queue_name}_output` instead of `{queue_name}`
- [x] 5.7 Update `doc-ingest-chat/workers/producer_graph.py` to push to `{queue_name}_input` instead of `{queue_name}`
- [x] 5.8 Run full test suite: `PYTHONPATH=doc-ingest-chat:shared .venv/bin/python -m pytest doc-ingest-chat/tests/ -v`
- [x] 5.9 Verify no regressions with queue name changes

## 6. Integration Testing

- [x] 6.1 Write integration test: verify FlowFile round-trip through NiFi (push to `_input` queue → NiFi pops → NiFi pushes to `_output` queue → pop from `_output` queue) in `tests/test_nifi_integration.py`

## 7. Documentation

- [x] 7.1 Update `AGENTS.md` with `NIFI_ENDPOINT`, `NIFI_USERNAME`, `NIFI_PASSWORD`, `NIFI_SSL_VERIFY` configuration examples
- [x] 7.2 Add section to `infra/operations/day-1.md` for remote NiFi setup and `NIFI_ENDPOINT` usage
- [x] 7.3 Add troubleshooting section to `infra/operations/day-2.md` for NiFi connection issues (including SSL/cert errors, auth failures)
- [x] 7.4 Update `docs/overview.md` to mention NiFi sandwich middleware
- [x] 7.5 Update `docs/quickstart.md` with NiFi configuration
- [x] 7.6 Update `CHANGELOG.md` under `[Unreleased]` → `Added` section with: NiFi bridge processors, NiPyAPI client, NiFi flow setup script
- [x] 7.7 Add README to `nifi/` directory explaining the processors, NiPyAPI client, flow setup script, and how to extend

## 8. Code Quality

- [x] 8.1 Run `ruff check --fix` on all modified files
- [x] 8.2 Run `ruff check` to verify no linting errors remain
- [x] 8.3 Run full test suite: `PYTHONPATH=doc-ingest-chat:shared .venv/bin/python -m pytest doc-ingest-chat/tests/ -v`
- [x] 8.4 Verify all tests pass (142+ tests)
- [x] 8.5 Run `npm run build` in `astro-frontend/` to verify no frontend regressions
- [x] 8.6 Review all new code for proper logging (logger name, level, message format)
- [x] 8.7 Review all new code for proper error handling (graceful degradation, exception context)
