## ADDED Requirements

### Requirement: RedisQueueConsumer processor
The system SHALL provide a NiFi Python processor named `RedisQueueConsumer` that performs blocking pop (BRPOP/BLPOP) from a configured Redis queue using the native Python `redis` library and emits the result as a FlowFile with the popped data as content and queue metadata as attributes.

#### Scenario: Successful pop
- **WHEN** the processor executes and data is available on the configured Redis queue
- **THEN** it SHALL create a FlowFile with the popped data as content, set attributes `redis.queue` (source queue name), `redis.pop.time` (ISO timestamp), and route to the `success` relationship

#### Scenario: Empty queue (timeout)
- **WHEN** the processor executes and no data is available within the configured timeout
- **THEN** it SHALL yield the processor for 1 second and produce no FlowFile

#### Scenario: Redis connection failure
- **WHEN** the processor cannot connect to Redis
- **THEN** it SHALL log the error, yield for 5 seconds, and route no FlowFile

### Requirement: RedisQueueConsumer configurable properties
The `RedisQueueConsumer` SHALL expose the following NiFi PropertyDescriptors: `Redis Host` (string, required, default: `localhost`), `Redis Port` (integer, required, default: `6379`), `Redis DB` (integer, optional, default: `0`), `Redis List Key` (string, required, supports expression language), `Pop Operation` (allowable values: `BRPOP`, `BLPOP`, `LPOP`, default: `BRPOP`), `Timeout Seconds` (integer, default: `5`). The processor SHALL use the native Python `redis` library with `redis.ConnectionPool` for connection management.

#### Scenario: Expression language queue name
- **WHEN** `Redis List Key` is set to `${queue.base.name}_input`
- **THEN** the processor SHALL resolve the expression at runtime using FlowFile attributes or NiFi variable registry

#### Scenario: Direct Redis connection
- **WHEN** the processor initializes
- **THEN** it SHALL create a `redis.Redis` client using `redis.ConnectionPool(host=..., port=..., db=..., decode_responses=True)` without using NiFi's `RedisConnectionPoolService`

#### Scenario: Single queue per processor
- **WHEN** the processor is configured on the NiFi canvas
- **THEN** it SHALL listen on a single Redis queue (configured via `Redis List Key` property). One processor instance per queue.

### Requirement: RedisQueueProducer processor
The system SHALL provide a NiFi Python processor named `RedisQueueProducer` that reads FlowFile content and pushes it to a configured Redis queue using LPUSH or RPUSH via the native Python `redis` library.

#### Scenario: Successful push
- **WHEN** the processor receives a FlowFile
- **THEN** it SHALL read the FlowFile content, push it to the configured Redis queue, and route the FlowFile to the `success` relationship

#### Scenario: Redis connection failure on push
- **WHEN** the processor cannot connect to Redis during push
- **THEN** it SHALL route the FlowFile to the `failure` relationship with attribute `redis.error` containing the exception message

#### Scenario: Batch push
- **WHEN** multiple FlowFiles arrive in a single execution cycle
- **THEN** the processor SHALL push all FlowFile contents to Redis in a single pipeline operation for efficiency

### Requirement: RedisQueueProducer configurable properties
The `RedisQueueProducer` SHALL expose the following NiFi PropertyDescriptors: `Redis Host` (string, required, default: `localhost`), `Redis Port` (integer, required, default: `6379`), `Redis DB` (integer, optional, default: `0`), `Redis List Key` (string, required, supports expression language), `Push Operation` (allowable values: `LPUSH`, `RPUSH`, default: `LPUSH`), `TTL Seconds` (integer, optional — if set, applies EXPIRE to the queue key after push). The processor SHALL use the native Python `redis` library.

#### Scenario: TTL on reply keys
- **WHEN** `TTL Seconds` is set to `300` and `Redis List Key` is `ocr_reply:${reply_id}`
- **THEN** after pushing, the processor SHALL apply `EXPIRE 300` to the resolved queue key

### Requirement: Processor dependencies declaration
Both processors SHALL declare their Python dependencies in `ProcessorDetails.dependencies`. The only external dependency SHALL be `redis>=5.0.0`.

#### Scenario: NiFi auto-install
- **WHEN** NiFi loads the processor for the first time
- **THEN** NiFi SHALL pip install `redis>=5.0.0` into the processor's isolated virtual environment

### Requirement: FlowFile attribute propagation
Both processors SHALL propagate existing FlowFile attributes through the transformation. The `RedisQueueConsumer` SHALL add new attributes without removing existing ones. The `RedisQueueProducer` SHALL pass all attributes through to the output FlowFile.

#### Scenario: Attribute preservation through consumer
- **WHEN** a FlowFile with attribute `trace_id=abc-123` passes through `RedisQueueConsumer`
- **THEN** the output FlowFile SHALL retain `trace_id=abc-123` and additionally have `redis.queue` and `redis.pop.time`

### Requirement: Processor logging
Both processors SHALL use NiFi's logging framework. Successful operations SHALL log at INFO level. Errors SHALL log at ERROR level with full exception traceback.

#### Scenario: Consumer pop logging
- **WHEN** `RedisQueueConsumer` successfully pops a FlowFile from `ocr_processing_job_input`
- **THEN** it SHALL log: `"Popped FlowFile from ocr_processing_job_input ({size} bytes)"`

#### Scenario: Producer push logging
- **WHEN** `RedisQueueProducer` successfully pushes a FlowFile to `ocr_processing_job_output`
- **THEN** it SHALL log: `"Pushed FlowFile to ocr_processing_job_output ({size} bytes)"`

### Requirement: Connection pooling
Both processors SHALL use `redis.ConnectionPool` for efficient connection management. The pool SHALL be created once per processor instance and reused across invocations.

#### Scenario: Pool reuse
- **WHEN** the processor executes multiple times
- **THEN** it SHALL reuse the same `redis.ConnectionPool` instance rather than creating new connections

#### Scenario: Pool configuration
- **WHEN** the processor initializes the connection pool
- **THEN** it SHALL configure `decode_responses=True` for automatic JSON string handling

### Requirement: One processor per queue
The system SHALL deploy one `RedisQueueConsumer` and one `RedisQueueProducer` per Redis queue. Each processor instance SHALL be configured with a single `Redis List Key` property.

#### Scenario: Multiple queues
- **WHEN** the system has 5 queues (`ocr_processing_job`, `whisper_processing_job`, `chunk_ingest_queue:0`, `chunk_ingest_queue:1`, `chunk_ingest_queue:2`)
- **THEN** NiFi SHALL have 5 `RedisQueueConsumer` instances (one per `_input` queue) and 5 `RedisQueueProducer` instances (one per `_output` queue)

#### Scenario: Processor naming
- **WHEN** processors are created on the NiFi canvas
- **THEN** they SHALL be named descriptively (e.g., `RedisConsumer - ocr_processing_job_input`, `RedisProducer - ocr_processing_job_output`)
