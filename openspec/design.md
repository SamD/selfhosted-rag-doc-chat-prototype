## Context

The WhisperX worker (`doc-ingest-chat/workers/whisperx_worker.py`, 187 lines) is a Redis-polling loop that listens on `REDIS_WHISPER_JOB_QUEUE`, pops JSON job messages, transcribes media files via either a local `whisperx` model or a remote API (`RemoteWhisper`), and pushes transcription segments back to a per-job reply key. The sender side (`send_media_to_whisperx()` in `utils/whisper_utils.py`, 94 lines) creates a UUID-based job, pushes it to Redis, then blocks-waits (up to 30 minutes) on the reply key for segments or completion/error signals.

Currently there is **zero durability**: if the worker container crashes mid-transcription, the job is silently lost — the sender waits until its 30-minute timeout and raises `TimeoutError`. There is no retry mechanism, no visibility into in-progress work, and no way to distinguish "still processing" from "lost." Transcription is the longest-running step in the pipeline (minutes per file), making these failures costly.

The WhisperX worker is the **best candidate** for Temporal migration because:
- It has no DuckDB coupling (unlike Gatekeeper/Producer/Consumer)
- It has no LangGraph dependency (unlike OCR worker)
- Clean single-Activity boundary: one input file → one transcription result
- Clear failure modes that map directly to Temporal retry policies

## Goals / Non-Goals

**Goals:**
- Wrap the WhisperX transcription logic as a Temporal Activity with durable retry (3 attempts, exponential backoff)
- Add `temporal-server` + `temporal-web` as Docker Compose services for local/production deployment
- Support dual-run mode via `USE_TEMPORAL_WHISPER` feature flag — both paths run simultaneously during transition
- Provide workflow visibility: operators can see what files are being transcribed, their status, and any errors via Temporal Web UI
- Zero loss of existing functionality: all config/settings, model loading, and `RemoteWhisper`/local whisperx logic preserved

**Non-Goals:**
- Migrating other workers (Gatekeeper, Producer, Consumer, OCR) to Temporal
- Removing the existing Redis queue infrastructure
- Converting complex multi-step workers to LangGraph subgraphs
- Redesigning the DuckDB state machine
- Adding distributed tracing (OpenTelemetry) in Phase 1 — scope for later

## Decisions

### Decision 1: Wrap as Activity, don't rewrite the worker loop
**CHOICE:** Extract just the transcription call (`model.transcribe_file()` / local equivalent) into a standalone Activity function. Keep the existing `RemoteWhisper` class and model loading logic untouched.

**Rationale:** The council unanimously rated this as highest ROI. The existing `whisperx_worker.py` has ~120 lines of Redis plumbing and signal handling; only ~50 lines are the actual transcription logic. Wrapping that as an Activity gives us durability without touching model loading or handler chains. A full rewrite would be 2-3 weeks; wrapping is 3-5 days (spike).

**Alternatives considered:**
- *Full worker rewrite in Temporal*: More control but higher risk and effort. Deferred.
- *Redis Streams + consumer groups*: Would give basic retry but no workflow history, no Web UI, no structured timeouts. Inferior to Temporal.

### Decision 2: Activity worker runs as a separate Docker container
**CHOICE:** `run_temporal_worker.py` runs in its own Docker service (`whisperx-temporal-worker`), distinct from the existing `whisperx-worker` Redis-based container. Both coexist during dual-run.

**Rationale:** Isolated failure domain — a bug in the Temporal path doesn't affect the Redis path or vice versa. Gradual traffic shift via feature flag. Each container has its own resource limits and health checks.

**Alternatives considered:**
- *Single container running both*: Simpler compose file but harder to isolate failures and gradually switch traffic.
- *Sidecar pattern*: Adds Kubernetes complexity (not needed yet).

### Decision 3: Activity input uses serializable dataclasses
**CHOICE:** Define `TranscriptionInput(file_path: str, language: str, mime_type: str | None)` and `TranscriptionResult(segments: list[dict], source_file: str)` as plain dataclasses in `doc-ingest-chat/models/transcription_input.py`. These are automatically serialized by the Temporal SDK.

**Rationale:** Type-safe, self-documenting, and compatible with Temporal's built-in serialization. No need for custom encoders/decoders.

### Decision 4: Dual-run via feature flag, not traffic splitting
**CHOICE:** `USE_TEMPORAL_WHISPER` env var controls whether `send_media_to_whisperx()` dispatches to Redis or Temporal. Both paths process the same files independently — results are compared via logs during the transition period.

**Rationale:** Simple to implement, easy to roll back (just flip the flag). No complex load balancer logic needed. The sender (`whisper_utils.py`) becomes a thin dispatcher: call the appropriate backend based on the flag.

**Alternatives considered:**
- *Random 50/50 split*: Harder to debug — can't control which path processes which file during comparison.
- *Per-file routing via metadata*: More complex flagging system; unnecessary for initial migration.

### Decision 5: Temporal server runs single-node with embedded storage
**CHOICE:** `temporal-server start --environment dev` (or production-ready single-node with PostgreSQL). For local dev, use `temporalite` (in-memory Temporal server) via a Docker Compose service.

**Rationale:** The system is self-hosted by design. A single Temporal node is sufficient for the expected throughput (~10-20 transcription jobs per day). PostgreSQL backend can be added later when scaling is needed.

### Decision 6: Retry policy — 3 attempts, exponential backoff, 90-minute activity timeout
**CHOICE:**
```python
@activity.defn
async def transcribe_media(input: TranscriptionInput) -> TranscriptionResult:
    # ...
```
With `@activity.task_queue("whisperx")` and retry policy: `maximum_attempts=3`, `initial_interval=5s`, `backoff_coefficient=2.0`, `maximum_interval=60s`. Activity timeout: 90 minutes (covers longest known transcription ~10min video × 6x speed). Workflow timeout: 4 hours.

**Rationale:** Transcription failures are typically transient (network blips, server restarts). 3 attempts catches 99% of transient errors. Exponential backoff prevents thundering herd on retry. 90-minute timeout is generous but not infinite — catches hung processes.

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Temporal SDK serialization fails on large return values (many segments) | Medium | `TranscriptionResult.segments` could be 100+ segments for long files. Use temp file path instead of in-memory list if segment count > 500. |
| Feature flag not propagated to all code paths | High | Audit: `whisper_utils.py`, `gatekeeper_logic.py` (calls `send_media_to_whisperx`), and any direct imports. Add explicit unit test that asserts correct path selection. |
| Temporal server crash during dual-run | Low | `USE_TEMPORAL_WHISPER=false` falls back to Redis instantly. No Redis dependency on Temporal health. |
| Latency overhead of Activity wrapper | Low (expected <50ms) | Benchmark in spike. If >100ms, investigate gRPC connection pool. Acceptable trade-off for durability. |
| Docker Compose file grows with new services | Low (manageable) | Use `profiles: [temporal]` for Temporal services — only start when needed. |

## Migration Plan

### Phase 1: Spike (3-5 days)
1. Add `temporalio>=1.10.0` to dependencies
2. Create `temporalite` Docker service in compose
3. Implement `transcribe_media()` Activity wrapper around existing `RemoteWhisper.transcribe_file()`
4. Create `run_temporal_worker.py` entry point
5. Run end-to-end test with a real 10-second MP3 file — measure latency overhead
6. Write up spike results: latency numbers, pain points, recommendations

### Phase 2: Production Dual-Run (1-2 weeks)
1. Add `temporal-server` + `temporal-web` services to `docker-compose.yml` (with `profiles: [temporal]`)
2. Implement `send_media_to_whisperx_temporal()` dispatcher in `whisper_utils.py`
3. Feature flag `USE_TEMPORAL_WHISPER` gates dispatch path
4. Deploy both workers running simultaneously
5. Run for 7 days with monitoring — compare transcription results via checksums
6. Update operations docs (day-1.md, day-2.md)

### Phase 3: Cut-over and Redis removal (1 week)
1. After successful dual-run period, set `USE_TEMPORAL_WHISPER=true` as default
2. Monitor for 7 days — no issues
3. Remove `whisperx-worker` Redis container from compose
4. Clean up dead Redis queue references
5. Update documentation to reflect Temporal as primary path

## Rollback Strategy

If the Temporal path exhibits bugs in production:
1. Set `USE_TEMPORAL_WHISPER=false` — immediate rollback to Redis path
2. No state migration needed — the two paths are independent during dual-run
3. Orphaned Temporal workflows can be replayed or cancelled manually via Temporal CLI/Web UI
4. If `temporal-server` itself is unstable, restart it without affecting Redis worker

## Open Questions

1. **How will the sender know when to use Temporal vs Redis?**
   - Answer: `USE_TEMPORAL_WHISPER` boolean env var. Default `false`. Sender checks this in `send_media_to_whisperx()` and dispatches accordingly.

2. **What happens to in-progress Redis jobs during transition?**
   - Answer: They complete normally. The Redis worker and Temporal worker are independent — no coordination needed.

3. **Should the Activity write transcription results somewhere (DuckDB, file system)?**
   - Answer: No — the existing flow has the *sender* receive segments via the reply key / Activity return value. The Activity just returns segments; the caller decides what to do with them. This preserves the existing architecture.

4. **How many Temporal activity workers should run concurrently?**
   - Answer: Initially, 1 worker process = 1 concurrent transcription. Scale to N based on media queue depth. Each worker is a separate container in Docker Compose, started via `deploy.replicas`.

5. **Will Temporal metrics be collected?**
   - Answer: Temporal exposes Prometheus metrics on port 9090 by default. These will be scraped if Prometheus is available; otherwise, they're available for ad-hoc inspection.