## ADDED Requirements

### Requirement: RedisSourceProcessor
The system SHALL provide a NiFi Python processor named `RedisSourceProcessor` that performs blocking pop (BRPOP/BLPOP) from a configured Redis queue and emits the result as a FlowFile with the popped data as content and queue metadata as attributes.

#### Scenario: Successful pop
- **WHEN** the processor executes and data is available on the configured Redis queue
- **THEN** it SHALL create a FlowFile with the popped data as content, set attributes `redis.queue` (source queue name), `redis.pop.time` (ISO timestamp), and route to the `success` relationship

#### Scenario: Empty queue (timeout)
- **WHEN** the processor executes and no data is available within the configured timeout
- **THEN** it SHALL yield the processor for 1 second and produce no FlowFile

#### Scenario: Redis connection failure
- **WHEN** the processor cannot connect to Redis
- **THEN** it SHALL log the error, yield for 5 seconds, and route no FlowFile

### Requirement: RedisSourceProcessor configurable properties
The `RedisSourceProcessor` SHALL expose the following NiFi PropertyDescriptors: `Redis Connection Pool` (controller service reference, required), `Queue Name` (string, required, supports expression language), `Pop Operation` (allowable values: `BRPOP`, `BLPOP`, default: `BRPOP`), `Timeout Seconds` (integer, default: `5`).

#### Scenario: Expression language queue name
- **WHEN** `Queue Name` is set to `${queue.base.name}_output`
- **THEN** the processor SHALL resolve the expression at runtime using FlowFile attributes or NiFi variable registry

### Requirement: RedisSinkProcessor
The system SHALL provide a NiFi Python processor named `RedisSinkProcessor` that reads FlowFile content and pushes it to a configured Redis queue using LPUSH or RPUSH.

#### Scenario: Successful push
- **WHEN** the processor receives a FlowFile on the `success` relationship
- **THEN** it SHALL read the FlowFile content, push it to the configured Redis queue, and route the FlowFile to the `success` relationship

#### Scenario: Redis connection failure on push
- **WHEN** the processor cannot connect to Redis during push
- **THEN** it SHALL route the FlowFile to the `failure` relationship with attribute `redis.error` containing the exception message

#### Scenario: Batch push
- **WHEN** multiple FlowFiles arrive in a single execution cycle
- **THEN** the processor SHALL push all FlowFile contents to Redis in a single pipeline operation for efficiency

### Requirement: RedisSinkProcessor configurable properties
The `RedisSinkProcessor` SHALL expose the following NiFi PropertyDescriptors: `Redis Connection Pool` (controller service reference, required), `Queue Name` (string, required, supports expression language), `Push Operation` (allowable values: `LPUSH`, `RPUSH`, default: `LPUSH`), `TTL Seconds` (integer, optional — if set, applies EXPIRE to the queue key after push).

#### Scenario: TTL on reply keys
- **WHEN** `TTL Seconds` is set to `300` and `Queue Name` is `ocr_reply:${reply_id}`
- **THEN** after pushing, the processor SHALL apply `EXPIRE 300` to the resolved queue key

### Requirement: Processor dependencies declaration
Both processors SHALL declare their Python dependencies in `ProcessorDetails.dependencies` so NiFi auto-installs them. The only external dependency SHALL be `redis>=5.0.0`.

#### Scenario: NiFi auto-install
- **WHEN** NiFi loads the processor for the first time
- **THEN** NiFi SHALL pip install `redis>=5.0.0` into the processor's isolated virtual environment

### Requirement: FlowFile attribute propagation
Both processors SHALL propagate existing FlowFile attributes through the transformation. The `RedisSourceProcessor` SHALL add new attributes without removing existing ones. The `RedisSinkProcessor` SHALL pass all attributes through to the output FlowFile.

#### Scenario: Attribute preservation through source
- **WHEN** a FlowFile with attribute `trace_id=abc-123` passes through `RedisSourceProcessor`
- **THEN** the output FlowFile SHALL retain `trace_id=abc-123` and additionally have `redis.queue` and `redis.pop.time`

### Requirement: Processor logging
Both processors SHALL use NiFi's logging framework. Successful operations SHALL log at INFO level. Errors SHALL log at ERROR level with full exception traceback.

#### Scenario: Source pop logging
- **WHEN** `RedisSourceProcessor` successfully pops a FlowFile from `ocr_processing_job_input`
- **THEN** it SHALL log: `"Popped FlowFile from ocr_processing_job_input ({size} bytes)"`

#### Scenario: Sink push logging
- **WHEN** `RedisSinkProcessor` successfully pushes a FlowFile to `ocr_processing_job_output`
- **THEN** it SHALL log: `"Pushed FlowFile to ocr_processing_job_output ({size} bytes)"`
