# Day 2 — Ongoing Operations

Symptom-driven runbook. Each entry: **Symptom** → **Diagnosis** → **Fix**.

---

## 1. DuckDB — Lifecycle State Machine

### Symptom: Document stuck in PREPROCESSING or INGESTING for >1 hour

**Diagnosis:**

```sql
SELECT id, original_filename, status, worker_id, new_at
FROM ingestion_lifecycle
WHERE status NOT LIKE '%SUCCESS%'
  AND status NOT LIKE '%FAILED%'
  AND new_at < (CURRENT_TIMESTAMP - INTERVAL 1 HOUR);
```

If `worker_id` is set, that worker likely crashed mid-job.

**Fix:**

```sql
-- Force-fail the stuck job so it can be re-processed
UPDATE ingestion_lifecycle
SET status = 'INGEST_FAILED', error_log = 'Stuck — manually failed', finalized_at = CURRENT_TIMESTAMP
WHERE id = '<job_id>';
```

Then re-place the original file in `staging/` to trigger re-discovery.

---

### Symptom: "Data not found" but file was ingested

**Diagnosis:**

```sql
SELECT status, error_log, pdf_path, md_path
FROM ingestion_lifecycle
WHERE original_filename = '<filename>';
```

If status is `INGEST_SUCCESS`, check if the chunk actually made it to Qdrant:

```bash
curl -s -X POST http://<vector-db-host>:6333/collections/vector_base_collection/points/count \
  -H "Content-Type: application/json" \
  -d '{"filter": {"must": [{"key": "source_file", "match": {"text": "<filename>"}}]}}'
```

**Fix:** Re-trigger ingestion by re-placing the file in `staging/`. The duplicate check rejects `INGEST_SUCCESS` filenames, so first:

```sql
UPDATE ingestion_lifecycle
SET status = 'INGEST_FAILED'
WHERE original_filename = '<filename>' AND status = 'INGEST_SUCCESS';
```

---

### Symptom: DuckDB and Qdrant out of sync

**Diagnosis:**

```sql
-- Find chunks in DuckDB whose IDs suggest they weren't stored in Qdrant
SELECT id, source_file
FROM parquet_chunks
WHERE id NOT LIKE 'DOC_%';
```

**Fix:** Re-run the consumer for the affected files by re-placing them in `ingestion/` (Producer directory).

---

### Symptom: All ingestion jobs failing

**Diagnosis:**

```sql
SELECT original_filename, error_log
FROM ingestion_lifecycle
WHERE status = 'INGEST_FAILED'
ORDER BY finalized_at DESC
LIMIT 20;
```

Common errors:
- `"Model not found"` — EMBEDDING_ENDPOINTS or LLM_PATH is wrong
- `"Redis connection refused"` — Redis container not running
- `"DuckDB lock"` — Multiple workers contending, normally self-resolving

**Fix:** Address the root cause from the error log, then re-fail + re-ingest stuck files.

---

## 2. Redis — Queue Inspection

### Symptom: Ingestion stalled, queues growing

**Diagnosis:**

```bash
docker exec -it $(docker ps -q -f name=redis) redis-cli LLEN chunk_ingest_queue:0
docker exec -it $(docker ps -q -f name=redis) redis-cli LLEN chunk_ingest_queue:1
docker exec -it $(docker ps -q -f name=redis) redis-cli LLEN ocr_processing_job
docker exec -it $(docker ps -q -f name=redis) redis-cli LLEN whisper_processing_job
```

If queues are growing unbounded, the corresponding consumer worker may be down.

**Fix:**

```bash
docker logs <worker-name> 2>&1 | tail -30
docker restart <worker-name>
```

---

## 3. HAProxy — Load Balancer Monitoring

### Symptom: All traffic hitting one backend

**Diagnosis:**

```bash
docker logs haproxy_supervisor 2>&1 | grep "be_supervisor/" | tail -10
```

If only `srv0` appears, the client may be using keep-alive connections that pin to one backend.

**Fix:** `option httpclose` is already configured — verify the client isn't sending `Connection: keep-alive`. Workers should use short-lived connections.

### Symptom: Backend marked DOWN

**Diagnosis:**

```bash
docker exec haproxy_supervisor cat /tmp/haproxy.cfg | grep "server srv"
```

Check if the backend's health endpoint responds:

```bash
curl http://<backend-host>:<port>/models
# or
curl http://<backend-host>:<port>/health
```

**Fix:** Restart the failing backend. HAProxy auto-recovers when the backend passes 2 consecutive health checks.

### Symptom: 503 Service Unavailable

**Diagnosis:** 0 endpoints configured for the service.

```bash
docker exec haproxy_supervisor env | grep SUPERVISOR_LLM_ENDPOINTS
```

**Fix:** Set `SUPERVISOR_LLM_ENDPOINTS` (or other `*_ENDPOINTS`) to at least one URL, or bypass HAProxy by setting `*_PATH` directly.

---

## 4. Qdrant — Vector Inspection

### Symptom: Chat returns no results for ingested content

**Diagnosis:** Check points for the document:

```bash
curl -s -X POST http://<vector-db-host>:6333/collections/vector_base_collection/points/count \
  -H "Content-Type: application/json" \
  -d '{"filter": {"must": [{"key": "source_file", "match": {"text": "<filename>"}}]}}'
```

**Fix:** If count is 0, the consumer may not have upserted. Check consumer logs and re-process.

### Inspect sample payloads

```bash
curl -s -X POST http://<vector-db-host>:6333/collections/vector_base_collection/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit": 3, "with_payload": true, "with_vector": false}'
```

**Dashboard:** Visit `http://<vector-db-host>:6333/dashboard`.

---

## 5. Chat — Session Troubleshooting

### Symptom: Chat history not persisting between queries

**Diagnosis:** Check if Redis has the session key:

```bash
docker exec -it $(docker ps -q -f name=redis) redis-cli EXISTS session:<session_id>
docker exec -it $(docker ps -q -f name=redis) redis-cli LRANGE session:<session_id> 0 -1
```

**Fix:** If key doesn't exist, session expired or was never created. Generate a new session_id. If key exists but history is wrong, check `MAX_SESSION_TURNS` — old entries may have been trimmed.

### Symptom: "Data not found" from LLM

**Diagnosis:** The retrieved context didn't contain relevant information. Check how many chunks were retrieved:

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"<test question>"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['debug'])"
```

The `debug` field shows `"✅ Grounding successful. N citations mapped."` or `"⚠️ Grounding failure: No valid citation tags found."`

**Fix:** If grounding failure, the retrieved chunks had no matching content. Increase `RETRIEVER_TOP_K` (default: 4) or check that the document was ingested with proper chunking.

### Symptom: API query fails after code update

**Diagnosis:** Check API logs:

```bash
docker logs api 2>&1 | tail -30
```

**Fix:** Common issues:
- `ModuleNotFoundError` — new service file not in Docker image (rebuild needed)
- `KeyError: 'chat_history'` — frontend still sending old format (update frontend)
- `KeyError: 'session_id'` — frontend not sending session_id

---

## 6. Whisper — Audio Transcription

### Symptom: 400 Bad Request on audio files

**Diagnosis:**

```bash
docker logs whisperx-worker 2>&1 | tail -20
```

**Fix:** The whisper server must be started with `--convert` flag to handle non-WAV formats. Verify:

```bash
docker inspect <whisper-container> 2>&1 | grep -A2 "cmd\|entrypoint"
```

### Symptom: Transcription returns empty

**Diagnosis:** Audio too quiet or no speech detected.

**Fix:** Adjust `no_speech_thold` parameter (default: 0.6). Lower values are more aggressive.

### Convert MP4 to WAV manually (debugging)

```bash
ffmpeg -i /path/to/file.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 /tmp/test.wav
curl http://<whisper-host>:1145/inference \
  -F "file=@/tmp/test.wav" -F "temperature=0.0" -F "response_format=json"
```

---

## 7. Schema Evolution

### Add a column

```sql
ALTER TABLE ingestion_lifecycle ADD COLUMN language VARCHAR DEFAULT 'en';
```

### Create an index

```sql
CREATE INDEX idx_source_file ON parquet_chunks (source_file);
```

### Wipe all state (fresh start)

**Warning**: Permanently deletes ingestion state and chunk data. Qdrant vectors are not affected by DuckDB deletes.

```sql
DELETE FROM ingestion_lifecycle;
DELETE FROM parquet_chunks;
DELETE FROM staged_chunks;
DELETE FROM file_ingestion_jobs;
DELETE FROM gatekeeper_history;
```

---

## 8. Metrics (JSONL)

Metrics are recorded in `$DEFAULT_DOC_INGEST_ROOT/metrics.jsonl`.

### Average normalization time

```bash
jq -r 'select(.event == "file_processing_complete") | .metrics.total_processing_time_ms' \
  $DEFAULT_DOC_INGEST_ROOT/metrics.jsonl | \
  awk '{sum+=$1; count+=1} END {print "Avg: " sum/count " ms"}'
```
