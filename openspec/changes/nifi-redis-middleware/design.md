## Context

The current pipeline uses Redis as a message broker between 6 worker types (Gatekeeper, Producer, Consumer, OCR, WhisperX, API). Workers communicate via Redis List queues using LPUSH/BRPOP patterns. There are 6 separate Redis client factories across the codebase, no centralized messaging abstraction, and no flow-level observability.

NiFi 2.x is already partially adopted: two Python processors exist in `nifi/python/extensions/` (`MarkdownSplitter.py`, `PythonHttpProcessor.py`), and planning documents in `planning/` describe a full NiFi-native pipeline. This design covers Phase 1: establishing NiFi as middleware without changing worker logic.

**Current queue topology:**

| Queue | Producer | Consumer | Pattern |
|-------|----------|----------|---------|
| `ocr_processing_job` | `ocr_utils.py` (LPUSH) | `ocr_worker.py` (BRPOP) | Job dispatch |
| `whisper_processing_job` | `whisper_utils.py` (LPUSH) | `whisperx_worker.py` (BRPOP) | Job dispatch |
| `chunk_ingest_queue:0`, `:1` | `producer_graph.py` (Lua RPUSH) | `consumer_worker.py` (BLPOP) | Chunk streaming |
| `ocr_reply:{uuid}` | `ocr_graph.py` (LPUSH) | `ocr_utils.py` (BLPOP) | Request-reply |
| `whisper_reply:{uuid}` | `whisperx_worker.py` (RPUSH) | `whisper_utils.py` (BLPOP) | Streaming reply |
| `session:{uuid}` | `chat_session_service.py` (RPUSH) | `chat_session_service.py` (LRANGE) | Session store |

**Constraints:**
- Workers must remain unchanged in their business logic
- Redis remains the underlying transport (NiFi is middleware, not replacement)
- Air-gapped environment: no external package downloads at runtime
- NiFi Python processors must declare dependencies in `ProcessorDetails.dependencies`

## Goals / Non-Goals

**Goals:**
- Establish `MessageQueue` abstraction that workers use instead of direct Redis calls
- Support two implementations: `RedisQueue` (direct, current behavior) and `NifiQueue` (via NiFi middleware)
- Queue name resolution: `NifiQueue` transforms base names to `{name}_input` / `{name}_output` to prevent infinite loops
- NiFi Python processors (`RedisSourceProcessor`, `RedisSinkProcessor`) that bridge NiFi flows to Redis
- Programmatic NiFi flow setup via REST API (no manual XML management)
- Docker-compose integration with `NIFI_ENDPOINT` override pattern
- Full backward compatibility: `MESSAGING_IMPL=redis` (default) is identical to current behavior

**Non-Goals:**
- Replacing Redis with NiFi connections (Phase 2+)
- Routing ephemeral reply keys (`ocr_reply:{uuid}`, `whisper_reply:{uuid}`) through NiFi
- Session store migration through NiFi
- DuckDB writes through NiFi
- Refactoring OCR/Whisper to async request-reply pattern
- Native NiFi processors replacing Python workers
- NiFi Registry integration

## Decisions

### Decision 1: Strategy pattern with Abstract Factory

**Choice:** `MessageQueue` ABC with `RedisQueue` and `NifiQueue` concrete implementations, instantiated via `get_messaging()` factory function based on `MESSAGING_IMPL` env var.

**Rationale:** Follows the existing pattern in the codebase (`get_env_strategy()` in `config/env_strategy.py`). The factory is called once at worker startup and cached. Workers call `messaging.push(queue, data)` / `messaging.pop(queue, timeout)` without knowing the underlying implementation.

**Alternatives considered:**
- **Adapter pattern (wrap Redis client):** Too tightly coupled to Redis API surface. NiFi REST API has different semantics (transactions, FlowFile attributes).
- **Facade pattern (single class with internal branching):** Violates Open/Closed principle. Adding a third implementation (e.g., Kafka) would require modifying the facade.

### Decision 2: Queue name suffix transformation

**Choice:** `NifiQueue.resolve_queue_name(base_name, role)` appends `_input` for producers and `_output` for consumers.

```
MESSAGING_IMPL=redis:
  Producer → ocr_processing_job → Consumer

MESSAGING_IMPL=nifi:
  Producer → ocr_processing_job_input → [NiFi] → ocr_processing_job_output → Consumer
```

**Rationale:** Prevents infinite loops where NiFi pops its own push. Workers don't need to know about suffixes — the strategy handles it transparently. The `_input`/`_output` convention is self-documenting in NiFi's UI.

**Alternatives considered:**
- **Separate config for NiFi queue names:** More configuration burden, error-prone.
- **NiFi internal routing (no Redis intermediary):** This is Phase 2+, not Phase 1.

### Decision 3: Two separate NiFi processors (Source + Sink)

**Choice:** `RedisSourceProcessor` (BRPOP → FlowFile) and `RedisSinkProcessor` (FlowFile → LPUSH) as separate processors, connected by NiFi connections.

**Rationale:** NiFi's execution model requires source processors (generate FlowFiles) and sink processors (consume FlowFiles) to be separate. A single bidirectional processor would violate NiFi's threading model. Between source and sink, NiFi provides backpressure, load balancing, prioritization, and provenance — all for free.

**Alternatives considered:**
- **Single processor with both push and pop:** Not possible in NiFi's `FlowFileTransform` model. A processor either generates or consumes FlowFiles, not both in the same invocation.

### Decision 4: Programmatic flow setup via REST API

**Choice:** Python script (`nifi/setup_flow.py`) that runs at NiFi container startup, creates process groups, processors, connections, and controller services via NiFi REST API.

**Rationale:** NiFi UI exports produce large XML blobs that are not version-controllable. The REST API approach:
- Keeps flow definition in code (version-controlled, reviewable)
- Is idempotent (checks if flow exists before creating)
- Follows the existing pattern (`infra/haproxy-entrypoint.sh` generates HAProxy config from env vars)
- Supports dynamic queue names from `QUEUE_NAMES` env var

**Alternatives considered:**
- **NiFi Registry:** Additional service to manage, overkill for Phase 1.
- **Static template XML:** Fragile, hard to parameterize, not reviewable in diffs.
- **NiFi CLI (nifi-toolkit):** Java dependency, heavier than a Python script.

### Decision 5: Reply keys excluded from NiFi middleware

**Choice:** Ephemeral reply keys (`ocr_reply:{uuid}`, `whisper_reply:{uuid}`) remain on direct Redis. Only job dispatch queues go through NiFi.

**Rationale:** Reply keys are UUID-based and created dynamically per request. NiFi cannot dynamically create input/output ports for each reply key. Routing all replies through a single port would require changing how workers push replies (adding attributes), which violates the "workers unchanged" constraint.

**Future phases:** Refactor OCR/Whisper to async pattern where results flow through a named queue with `job_id` attribute, enabling NiFi routing.

### Decision 6: Consolidate Redis client factories

**Choice:** The `MessageQueue` abstraction uses a single `get_redis_client()` from `services/messaging/redis_client.py` with fork-safe PID checking. Existing per-module factories (`ocr_utils.py`, `whisper_utils.py`, `producer_utils.py`) are replaced with calls through the messaging facade.

**Rationale:** The current 6 separate Redis client factories are a maintenance burden and fork-safety risk. The messaging facade is the single point of Redis client creation. This is a side benefit of the abstraction, not its primary goal.

## Risks / Trade-offs

**[Risk] NiFi REST API transaction overhead** → The NiFi input port transaction API requires 3 HTTP round-trips per push (create transaction, send FlowFiles, commit). For high-throughput chunk ingest (thousands of chunks per file), this could be slow.
**Mitigation:** Batch multiple FlowFiles in a single transaction. The `NifiQueue.push_batch()` method sends up to 100 FlowFiles per transaction. NiFi supports this natively.

**[Risk] NiFi container resource usage** → The `apache/nifi:2.x` image is ~2GB and requires significant memory (default: 2GB heap).
**Mitigation:** NiFi is optional (`MESSAGING_IMPL=redis` by default). For dev/test, `NIFI_ENDPOINT` allows connecting to a remote NiFi instance instead of running locally. Docker-compose only starts NiFi when `MESSAGING_IMPL=nifi`.

**[Risk] Blocking BRPOP in NiFi source processor** → `BRPOP` with timeout blocks the NiFi thread, potentially reducing throughput.
**Mitigation:** Configure processor Run Schedule to 0s (run continuously) and Concurrent Tasks to 1 per queue. The blocking pop IS the throttle. On timeout (no data), the processor yields for 1 second to avoid busy-looping.

**[Risk] Queue name collision** → If a base queue name already ends with `_input` or `_output`, the suffix transformation could create ambiguous names.
**Mitigation:** `resolve_queue_name()` validates that base names don't end with `_input` or `_output` and raises `ValueError` if they do.

**[Trade-off] Partial provenance** → Phase 1 only provides provenance for the NiFi middle segment. Worker push and consumer pop are invisible to NiFi.
**Accepted:** This is the explicit trade-off for zero worker changes. Phase 2 (workers push to NiFi input ports via REST API) will extend provenance to the edges.

## Migration Plan

**Deployment steps:**
1. Deploy code changes (messaging facade, queue name resolution)
2. Set `MESSAGING_IMPL=redis` (default, no behavior change)
3. Deploy NiFi container or configure `NIFI_ENDPOINT`
4. Run `nifi/setup_flow.py` to create the flow
5. Switch `MESSAGING_IMPL=nifi` on a single worker type (e.g., OCR) for validation
6. Monitor NiFi provenance and queue depths
7. Roll out to remaining worker types

**Rollback strategy:**
- Set `MESSAGING_IMPL=redis` on any worker to bypass NiFi immediately
- No data migration needed (Redis is the transport in both modes)
- NiFi flow can be stopped independently without affecting workers

## Open Questions

1. **NiFi version:** Should we pin to a specific NiFi 2.x version (e.g., 2.2.0) or use `latest`?
2. **Backpressure thresholds:** What FlowFile count and total size thresholds should be configured on NiFi connections? (Current Redis backpressure uses 50,000 items max.)
3. **NiFi flow monitoring:** Should we add a health check endpoint that verifies NiFi flow status, or rely on NiFi's built-in REST API health checks?
