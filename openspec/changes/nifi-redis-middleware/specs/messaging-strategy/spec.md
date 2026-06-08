## ADDED Requirements

### Requirement: MessageQueue abstract interface
The system SHALL provide a `MessageQueue` abstract base class defining the contract for all messaging implementations. The interface SHALL include: `push(queue_name, data)`, `pop(queue_name, timeout)`, `push_reply(reply_key, data, expire)`, `wait_reply(reply_key, timeout)`, `resolve_queue_name(base_name, role)`, `queue_length(queue_name)`, `purge(queue_name, filter_fn)`, and `blocking_push_with_backpressure(queue_name, entries, max_length)`.

#### Scenario: Interface contract enforcement
- **WHEN** a new messaging implementation is created
- **THEN** it MUST implement all abstract methods defined in `MessageQueue` or Python SHALL raise `TypeError` at instantiation

### Requirement: RedisQueue implementation
The system SHALL provide a `RedisQueue` implementation that communicates directly with Redis using LPUSH/BRPOP/RPUSH/BLPOP commands. This implementation SHALL be functionally identical to the current direct Redis usage.

#### Scenario: Push to queue
- **WHEN** `RedisQueue.push("ocr_processing_job", {"job_id": "abc"})` is called
- **THEN** the data SHALL be JSON-serialized and LPUSH'd to the Redis list `ocr_processing_job`

#### Scenario: Pop from queue with blocking
- **WHEN** `RedisQueue.pop("ocr_processing_job", timeout=5)` is called and the queue is empty
- **THEN** the call SHALL block for up to 5 seconds using BRPOP and return `None` if no data arrives

#### Scenario: Queue name resolution (no transformation)
- **WHEN** `RedisQueue.resolve_queue_name("ocr_processing_job", "producer")` is called
- **THEN** it SHALL return `"ocr_processing_job"` unchanged

### Requirement: NifiQueue implementation
The system SHALL provide a `NifiQueue` implementation that transforms queue names with `_input`/`_output` suffixes and delegates actual Redis operations to the same Redis client. The NiFi middleware layer sits between the `_input` and `_output` queues.

#### Scenario: Producer queue name resolution
- **WHEN** `NifiQueue.resolve_queue_name("ocr_processing_job", "producer")` is called
- **THEN** it SHALL return `"ocr_processing_job_input"`

#### Scenario: Consumer queue name resolution
- **WHEN** `NifiQueue.resolve_queue_name("ocr_processing_job", "consumer")` is called
- **THEN** it SHALL return `"ocr_processing_job_output"`

#### Scenario: Queue name validation
- **WHEN** `NifiQueue.resolve_queue_name("ocr_processing_job_input", "producer")` is called with a name already ending in `_input`
- **THEN** it SHALL raise `ValueError` to prevent ambiguous queue names

#### Scenario: Push to input queue
- **WHEN** `NifiQueue.push("ocr_processing_job", {"job_id": "abc"})` is called
- **THEN** the data SHALL be LPUSH'd to `"ocr_processing_job_input"` in Redis

#### Scenario: Pop from output queue
- **WHEN** `NifiQueue.pop("ocr_processing_job", timeout=5)` is called
- **THEN** it SHALL BRPOP from `"ocr_processing_job_output"` in Redis

### Requirement: Messaging factory function
The system SHALL provide a `get_messaging()` factory function that returns the appropriate `MessageQueue` implementation based on the `MESSAGING_IMPL` environment variable. The factory SHALL cache the instance per process (fork-safe with PID checking).

#### Scenario: Default to Redis
- **WHEN** `MESSAGING_IMPL` is not set or set to `"redis"` (case-insensitive)
- **THEN** `get_messaging()` SHALL return a `RedisQueue` instance

#### Scenario: Select NiFi
- **WHEN** `MESSAGING_IMPL` is set to `"nifi"` (case-insensitive)
- **THEN** `get_messaging()` SHALL return a `NifiQueue` instance

#### Scenario: Unknown implementation
- **WHEN** `MESSAGING_IMPL` is set to an unrecognized value (e.g., `"kafka"`)
- **THEN** `get_messaging()` SHALL raise `ValueError` with a message listing valid options

#### Scenario: Fork safety
- **WHEN** a process forks after `get_messaging()` has been called
- **THEN** the child process SHALL create a new `MessageQueue` instance (detected via PID change)

### Requirement: Reply keys bypass NiFi
The system SHALL route reply key operations (`push_reply`, `wait_reply`) directly to Redis regardless of `MESSAGING_IMPL` setting. Reply keys are ephemeral UUID-based and cannot be routed through NiFi's static port model.

#### Scenario: OCR reply push with NiFi mode
- **WHEN** `MESSAGING_IMPL=nifi` and `NifiQueue.push_reply("ocr_reply:abc-123", result)` is called
- **THEN** the data SHALL be LPUSH'd directly to `"ocr_reply:abc-123"` in Redis (no suffix transformation)

#### Scenario: OCR reply wait with NiFi mode
- **WHEN** `MESSAGING_IMPL=nifi` and `NifiQueue.wait_reply("ocr_reply:abc-123", timeout=30)` is called
- **THEN** it SHALL BLPOP directly from `"ocr_reply:abc-123"` in Redis (no suffix transformation)

### Requirement: Configuration via shared config
The system SHALL register `MESSAGING_IMPL` and `NIFI_ENDPOINT` in `shared/config.py` following the existing lazy-loaded settings pattern. Environment variable names SHALL be defined in `shared/env_names.py` and defaults in `shared/defaults.py`.

#### Scenario: MESSAGING_IMPL default
- **WHEN** `MESSAGING_IMPL` environment variable is not set
- **THEN** `settings.MESSAGING_IMPL` SHALL return `"redis"`

#### Scenario: NIFI_ENDPOINT override
- **WHEN** `NIFI_ENDPOINT` is set to `"http://nifi-core01:8080"`
- **THEN** `settings.NIFI_ENDPOINT` SHALL return `"http://nifi-core01:8080"`

### Requirement: Consolidated Redis client
The messaging module SHALL provide a single fork-safe `get_redis_client()` function that replaces the 6 existing per-module Redis client factories. All messaging operations SHALL use this shared client.

#### Scenario: Single client per process
- **WHEN** `get_redis_client()` is called multiple times within the same process
- **THEN** it SHALL return the same `redis.Redis` instance

#### Scenario: Fork-safe reset
- **WHEN** a process forks after `get_redis_client()` has been called
- **THEN** the child process SHALL create a new `redis.Redis` instance (detected via PID change)

### Requirement: Logging and observability
All messaging operations SHALL log queue name, operation type, and data size (not content) at DEBUG level. Errors SHALL be logged at ERROR level with full exception context.

#### Scenario: Push logging
- **WHEN** `messaging.push("ocr_processing_job", data)` is called
- **THEN** a DEBUG log entry SHALL be emitted containing the queue name and `len(json.dumps(data))` character count

#### Scenario: Pop timeout logging
- **WHEN** `messaging.pop("ocr_processing_job", timeout=5)` times out with no data
- **THEN** a DEBUG log entry SHALL be emitted indicating the timeout
