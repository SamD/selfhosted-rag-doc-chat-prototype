## Context

The current pipeline uses Redis as a message broker between 6 worker types (Gatekeeper, Producer, Consumer, OCR, WhisperX, API). Workers communicate via Redis List queues using LPUSH/BRPOP patterns. There are 6 separate Redis client factories across the codebase, no centralized messaging abstraction, and no flow-level observability.

NiFi 2.x is already partially adopted: two Python processors exist in `nifi/python/extensions/` (`MarkdownSplitter.py`, `PythonHttpProcessor.py`), and planning documents in `planning/` describe a full NiFi-native pipeline. A remote NiFi instance is already deployed and accessible via HTTPS with a self-signed cert. This design covers Phase 1: establishing NiFi as transparent middleware ("NiFi sandwich") without changing worker logic, connecting to the remote NiFi via `NIFI_ENDPOINT`.

**Current queue topology:**

| Queue | Producer | Consumer | Pattern |
|-------|----------|----------|---------|
| `ocr_processing_job` | `ocr_utils.py` (LPUSH) | `ocr_worker.py` (BRPOP) | Job dispatch |
| `whisper_processing_job` | `whisper_utils.py` (LPUSH) | `whisperx_worker.py` (BRPOP) | Job dispatch |
| `chunk_ingest_queue:0`, `:1` | `producer_graph.py` (Lua RPUSH) | `consumer_worker.py` (BLPOP) | Chunk streaming |
| `ocr_reply:{uuid}` | `ocr_graph.py` (LPUSH) | `ocr_utils.py` (BLPOP) | Request-reply |
| `whisper_reply:{uuid}` | `whisperx_worker.py` (RPUSH) | `whisper_utils.py` (BLPOP) | Streaming reply |
| `session:{uuid}` | `chat_session_service.py` (RPUSH) | `chat_session_service.py` (LRANGE) | Session store |

**NiFi sandwich topology (Phase 1):**

| Queue | Producer | NiFi | Consumer | Pattern |
|-------|----------|------|----------|---------|
| `ocr_processing_job_input` | `ocr_utils.py` (LPUSH) | → `ocr_processing_job_output` | `ocr_worker.py` (BRPOP) | Job dispatch |
| `whisper_processing_job_input` | `whisper_utils.py` (LPUSH) | → `whisper_processing_job_output` | `whisperx_worker.py` (BRPOP) | Job dispatch |
| `chunk_ingest_queue:0_input` | `producer_graph.py` (RPUSH) | → `chunk_ingest_queue:0_output` | `consumer_worker.py` (BLPOP) | Chunk streaming |

**Constraints:**
- Workers must remain completely unchanged — no code modifications
- Redis remains the underlying transport (NiFi is middleware, not replacement)
- Air-gapped environment: no external package downloads at runtime
- NiFi Python processors must declare dependencies in `ProcessorDetails.dependencies`
- NiFi is deployed remotely (not containerized locally); all communication via `NIFI_ENDPOINT`
- Remote NiFi uses self-signed TLS certificate; clients must disable cert verification

## Goals / Non-Goals

**Goals:**
- Establish NiFi as transparent middleware between Redis `_input` and `_output` queues
- NiFi Python processors (`RedisQueueConsumer`, `RedisQueueProducer`) that bridge NiFi flows to Redis
- Programmatic NiFi flow setup via NiPyAPI against remote NiFi (no manual XML management)
- `NIFI_ENDPOINT` configuration following the same pattern as other remote services
- Self-signed certificate support via `NIFI_SSL_VERIFY=false` (default for dev/self-hosted)
- Zero worker changes — workers continue using direct Redis calls
- Full backward compatibility: without NiFi running, workers can still use direct Redis

**Non-Goals:**
- Running NiFi as a local Docker container (NiFi is deployed remotely)
- Replacing Redis with NiFi connections (Phase 2+)
- Routing ephemeral reply keys (`ocr_reply:{uuid}`, `whisper_reply:{uuid}`) through NiFi
- Session store migration through NiFi
- DuckDB writes through NiFi
- Refactoring OCR/Whisper to async request-reply pattern
- Native NiFi processors replacing Python workers
- NiFi Registry integration
- HAProxy load balancing for NiFi (single remote instance)
- MessageQueue abstraction layer or strategy pattern

## Decisions

### Decision 1: NiFi sandwich pattern (no worker changes)

**Choice:** Workers continue using direct Redis LPUSH/BRPOP calls. NiFi sits between `_input` and `_output` queues, transparently moving messages. Workers push to `{queue}_input`, NiFi pops and pushes to `{queue}_output`, consumers pop from `{queue}_output`.

**Rationale:** Zero worker changes is the simplest path to Phase 1. Workers don't need to know NiFi exists. The `_input`/`_output` naming convention is self-documenting in NiFi's UI. If NiFi is unavailable, workers can fall back to direct Redis (no `_input`/`_output` suffixes).

**Alternatives considered:**
- **Strategy pattern (MessageQueue ABC):** Adds abstraction layer, requires worker changes, more complex for Phase 1.
- **Workers push to NiFi input ports via REST API:** Requires worker changes, adds HTTP overhead to every push.
- **NiFi replaces Redis entirely:** Too risky for Phase 1, requires full migration.

### Decision 2: Queue name suffix transformation

**Choice:** For each base queue name (e.g., `ocr_processing_job`), create two Redis queues: `{name}_input` (workers push here) and `{name}_output` (consumers pop from here). NiFi pops from `_input` and pushes to `_output`.

```
Without NiFi (direct Redis):
  Producer → ocr_processing_job → Consumer

With NiFi (sandwich):
  Producer → ocr_processing_job_input → [NiFi] → ocr_processing_job_output → Consumer
```

**Rationale:** Prevents infinite loops where NiFi pops its own push. Workers don't need to know about suffixes — they just push to the base queue name (which is now `_input`). The `_input`/`_output` convention is self-documenting in NiFi's UI.

**Alternatives considered:**
- **Separate config for NiFi queue names:** More configuration burden, error-prone.
- **NiFi internal routing (no Redis intermediary):** This is Phase 2+, not Phase 1.

### Decision 3: Correlation ID and queue name tracking

**Choice:** The reply mechanism remains unchanged — workers read `reply_key` from the job payload. NiFi processors set `redis.source.queue` as a FlowFile attribute for observability, but this is not used by workers.

**Rationale:** The existing reply mechanism already embeds the reply key in the payload:
1. Producer pushes job to `ocr_processing_job_input` with `reply_key` in payload
2. NiFi pops from `_input`, creates FlowFile (queue name set as attribute, not in payload)
3. NiFi pushes to `_output`
4. Worker pops from `_output`, reads `reply_key` from payload, processes, pushes result to `reply_key`

The queue name is tracked as a FlowFile attribute for NiFi observability (provenance, monitoring), but workers don't need it — they use the `reply_key` from the payload.

**Alternatives considered:**
- **Add `queue_name` to payload:** Requires worker code changes, breaks backward compatibility.
- **Single multi-queue processor:** More complex, harder to monitor per-queue metrics.

### Decision 4: Two separate NiFi processors (Consumer + Producer) with native Redis library, one per queue

**Choice:** `RedisQueueConsumer` (BRPOP → FlowFile) and `RedisQueueProducer` (FlowFile → LPUSH) as separate processors using the native Python `redis` library with `redis.ConnectionPool`. Processors connect directly to Redis without using NiFi's `RedisConnectionPoolService` controller service. One processor instance per queue.

**Rationale:** 
- NiFi's execution model requires source processors (generate FlowFiles) and sink processors (consume FlowFiles) to be separate
- The native `redis` library is recommended over NiFi's `RedisConnectionPoolService` for Python processors — it's more flexible, better documented, and aligns with the existing `redis_service.py` pattern
- One processor per queue is simpler than multi-queue polling, provides better NiFi UI visibility, and makes per-queue backpressure configuration easier
- Memory overhead is minimal (~1-2MB per processor instance, ~10MB total for 5 queues)
- Between consumer and producer, NiFi provides backpressure, load balancing, prioritization, and provenance — all for free

**Alternatives considered:**
- **Single processor with both push and pop:** Not possible in NiFi's `FlowFileTransform` model. A processor either generates or consumes FlowFiles, not both in the same invocation.
- **NiFi RedisConnectionPoolService:** Less flexible than native redis library, not recommended for Python processors per NiFi best practices.
- **Single multi-queue processor:** More complex, need to track which queue each message came from, harder to monitor per-queue metrics.

### Decision 5: Programmatic flow setup via NiPyAPI (remote NiFi)

**Choice:** Python script (`nifi/setup_flow.py`) that connects to the remote NiFi instance via `NIFI_ENDPOINT` and creates process groups, processors, connections using NiPyAPI 1.x (for NiFi 2.0 support). Authentication via `NIFI_USERNAME`/`NIFI_PASSWORD` using single-user basic auth with `nipyapi.utils.set_endpoint()`. SSL verification controlled by `NIFI_SSL_VERIFY` (default: `false`). Registry configured via `REGISTRY_ENDPOINT` (HTTP = no auth, HTTPS = same credentials).

**Rationale:** NiPyAPI provides a high-level Python SDK that abstracts NiFi REST API complexity. Using 1.x version ensures NiFi 2.0 compatibility. `utils.set_endpoint()` is the correct method for establishing authenticated sessions (not `set_service_auth()`). The URL must include `/nifi-api` suffix. Registry also needs configuration even if using HTTP (no auth).

**Alternatives considered:**
- **Raw REST API with requests:** More code, manual JSON payload construction, harder to maintain.
- **NiFi Registry:** Additional service to manage, overkill for Phase 1.
- **Static template XML:** Fragile, hard to parameterize, not reviewable in diffs.
- **NiFi CLI (nifi-toolkit):** Java dependency, heavier than Python SDK.
- **Profiles system (`nipyapi.profiles.switch()`):** Recommended by NiPyAPI docs but adds complexity for our simple single-environment use case. Direct config is sufficient.

### Decision 6: Reply keys excluded from NiFi middleware

**Choice:** Ephemeral reply keys (`ocr_reply:{uuid}`, `whisper_reply:{uuid}`) remain on direct Redis. Only job dispatch queues go through NiFi.

**Rationale:** Reply keys are UUID-based and created dynamically per request. NiFi cannot dynamically create input/output ports for each reply key. Routing all replies through a single port would require changing how workers push replies (adding attributes), which violates the "workers unchanged" constraint.

**Future phases:** Refactor OCR/Whisper to async pattern where results flow through a named queue with `job_id` attribute, enabling NiFi routing.

### Decision 7: Remote-only NiFi deployment with self-signed TLS

**Choice:** NiFi runs as an externally managed service (not containerized in docker-compose). Workers and setup scripts connect via `NIFI_ENDPOINT` (configurable HTTPS URL). A dedicated `nifi/nifi_client.py` module wraps NiPyAPI configuration with `verify_ssl=False` when `NIFI_SSL_VERIFY=false` and basic auth via `NIFI_USERNAME`/`NIFI_PASSWORD`.

**Rationale:** NiFi is a heavyweight service (~2GB image, 2GB+ heap). Running it remotely on dedicated infrastructure avoids resource contention with GPU workers. The self-signed cert is standard for internal LAN services. NiPyAPI 1.x supports NiFi 2.0. Single-user basic auth is simplest for trusted LAN.

**Alternatives considered:**
- **Local Docker container:** Resource-heavy, unnecessary when remote instance already exists.
- **Proper CA-signed cert:** Adds operational overhead for a LAN-only service. `verify=false` is acceptable for internal traffic.
- **HAProxy in front of NiFi:** Single instance, no load balancing needed. Adds complexity without benefit.
- **LDAP/OAuth auth:** Overkill for single-user LAN deployment.

## Risks / Trade-offs

**[Risk] NiFi REST API transaction overhead** → The NiFi input port transaction API requires 3 HTTP round-trips per push (create transaction, send FlowFiles, commit). For high-throughput chunk ingest (thousands of chunks per file), this could be slow.
**Mitigation:** Batch multiple FlowFiles in a single transaction. The `RedisQueueProducer` can be configured to batch multiple FlowFiles per Redis operation. NiFi supports this natively via pipeline operations.

**[Risk] Self-signed certificate security** → Disabling SSL verification (`verify=False`) removes protection against MITM attacks.
**Mitigation:** Acceptable for LAN-only traffic on a trusted network. `NIFI_SSL_VERIFY` defaults to `false` for dev/self-hosted but can be set to `true` if a proper CA cert is installed. `urllib3.InsecureRequestWarning` is suppressed to avoid log spam.

**[Risk] Remote NiFi network dependency** → Workers depend on network connectivity to the remote NiFi instance. Network partition = NiFi middleware unavailable.
**Mitigation:** Workers can fall back to direct Redis (no `_input`/`_output` suffixes) if NiFi is unavailable. `setup_flow.py` retries with exponential backoff on connection failures.

**[Risk] NiPyAPI version compatibility** → NiPyAPI 1.x is required for NiFi 2.0 support. Using 0.x will fail.
**Mitigation:** Pin `nipyapi>=1.0.0` in dependencies. Test against actual NiFi 2.x instance during development.

**[Risk] Blocking BRPOP in NiFi source processor** → `BRPOP` with timeout blocks the NiFi thread, potentially reducing throughput.
**Mitigation:** Configure processor Run Schedule to 0s (run continuously) and Concurrent Tasks to 1 per queue. The blocking pop IS the throttle. On timeout (no data), the processor yields for 1 second to avoid busy-looping.

**[Risk] Queue name collision** → If a base queue name already ends with `_input` or `_output`, the suffix transformation could create ambiguous names.
**Mitigation:** `resolve_queue_name()` validates that base names don't end with `_input` or `_output` and raises `ValueError` if they do.

**[Trade-off] Partial provenance** → Phase 1 only provides provenance for the NiFi middle segment. Worker push and consumer pop are invisible to NiFi.
**Accepted:** This is the explicit trade-off for zero worker changes. Phase 2 (workers push to NiFi input ports via REST API) will extend provenance to the edges.

## Migration Plan

**Deployment steps:**
1. Deploy code changes (NiFi processors, setup script, NiPyAPI client)
2. Configure `NIFI_ENDPOINT`, `NIFI_USERNAME`, `NIFI_PASSWORD`, `NIFI_SSL_VERIFY`
3. Run `nifi/setup_flow.py` to create the flow on remote NiFi
4. Update worker queue names to use `_input` suffix (e.g., `ocr_processing_job` → `ocr_processing_job_input`)
5. Update consumer queue names to use `_output` suffix (e.g., `ocr_processing_job` → `ocr_processing_job_output`)
6. Monitor NiFi provenance and queue depths
7. Roll out to remaining queue types

**Rollback strategy:**
- Revert worker queue names to base names (remove `_input`/`_output` suffixes)
- NiFi flow can be stopped independently without affecting workers
- No data migration needed (Redis is the transport in both modes)

## Open Questions

1. **NiFi version:** Remote instance is NiFi 2.x — should we pin the REST API client to a specific minor version?
2. **Backpressure thresholds:** What FlowFile count and total size thresholds should be configured on NiFi connections? (Current Redis backpressure uses 50,000 items max.)
3. **NiFi flow monitoring:** Should we add a health check endpoint that verifies NiFi flow status via the remote REST API, or rely on NiFi's built-in UI?
4. **TLS upgrade path:** When should we migrate from self-signed cert to CA-signed? Should `NIFI_SSL_VERIFY` default change to `true` for production?
5. **Queue name migration:** Should we update worker queue names in a single PR or incrementally per queue type?
