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

**Fix (automatic):** On restart, workers now reclaim orphaned jobs:
- Gatekeeper resets `PREPROCESSING` → `NEW`
- Producer resets `INGESTING` → `PREPROCESSING_COMPLETE`
- Consumer resets `CONSUMING` → `INGESTING`

This happens automatically within `STUCK_JOB_TIMEOUT_HOURS` (default: 1) of the crash. Simply restart the worker container.

**Fix (manual, if auto-reclaim doesn't trigger):**

```sql
-- Force-fail the stuck job so it can be re-processed
UPDATE ingestion_lifecycle
SET status = 'INGEST_FAILED', error_log = 'Stuck — manually failed', finalized_at = CURRENT_TIMESTAMP
WHERE id = '<job_id>';
```

Then move the file from `failed/` back to `staging/` to trigger re-discovery.

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

---

## 9. NiFi — Middleware Troubleshooting

### Symptom: nifi_bootstrap service failed

**Diagnosis:**

```bash
docker logs nifi_bootstrap 2>&1 | tail -50
```

Common errors:
- `NIFI_ENDPOINT environment variable is required` — Missing env var
- `NiFi not available` — NiFi is down or unreachable
- `Failed to configure NiFi endpoint` — Authentication or SSL error
- `RedisQueueConsumer processor type not found` — Processors not deployed to NiFi

**Fix:**

1. Check environment variables:
   ```bash
   docker exec nifi_bootstrap env | grep NIFI
   ```

2. Verify NiFi is accessible:
   ```bash
   curl -k https://<nifi-host>:8443/nifi-api/access/config
   ```

3. Verify processors are deployed:
   ```bash
   ssh <nifi-host> "ls -la /opt/nifi/nifi-current/python/extensions/Redis*.py"
   ```

4. Manually run bootstrap:
   ```bash
   cd nifi/
   python nifi_bootstrap.py
   ```

---

### Symptom: NiFi processors not found in bootstrap

**Diagnosis:**

```bash
cd nifi/
python -c "
from nifi_client import NifiClient
import os, nipyapi
from nipyapi import canvas

client = NifiClient(
    base_url=os.getenv('NIFI_ENDPOINT'),
    username=os.getenv('NIFI_USERNAME'),
    password=os.getenv('NIFI_PASSWORD'),
    ssl_verify=os.getenv('NIFI_SSL_VERIFY', 'false').lower() == 'true',
)

processor_types = canvas.list_all_processor_types()
redis_processors = [pt.type for pt in processor_types.processor_types if 'redis' in pt.type.lower() and 'queue' in pt.type.lower()]
print('Redis queue processors:', redis_processors)
"
```

If output is empty or missing `RedisQueueConsumer`/`RedisQueueProducer`, the processors aren't loaded.

**Fix:**

1. Verify processor files exist in NiFi's Python extensions directory:
   ```bash
   ssh <nifi-host> "ls -la /opt/nifi/nifi-current/python/extensions/Redis*.py"
   ```

2. If missing, deploy them:
   ```bash
   scp nifi/python/extensions/RedisQueueConsumer.py <nifi-host>:/opt/nifi/nifi-current/python/extensions/
   scp nifi/python/extensions/RedisQueueProducer.py <nifi-host>:/opt/nifi/nifi-current/python/extensions/
   ```

3. Restart NiFi:
   ```bash
   ssh <nifi-host> "docker restart nifi"
   ```

4. Wait 30-60 seconds for NiFi to load the processors.

5. Re-run bootstrap:
   ```bash
   cd nifi/
   python nifi_bootstrap.py
   ```

---

### Symptom: NiFi connection refused or SSL errors

**Diagnosis:**

```bash
curl -k https://<nifi-host>:8443/nifi-api/access/config
```

If this fails, NiFi is down or the URL is wrong.

**Fix:**

1. Check NiFi container status:
   ```bash
   ssh <nifi-host> "docker ps | grep nifi"
   ```

2. Check NiFi logs:
   ```bash
   ssh <nifi-host> "docker logs nifi 2>&1 | tail -50"
   ```

3. Verify `NIFI_ENDPOINT` includes `/nifi-api` suffix:
   ```bash
   echo $NIFI_ENDPOINT
   # Should be: https://<nifi-host>:8443/nifi-api
   ```

4. For self-signed certificates, ensure `NIFI_SSL_VERIFY=false`:
   ```bash
   export NIFI_SSL_VERIFY="false"
   ```

---

### Symptom: NiFi authentication failed

**Diagnosis:**

```bash
cd nifi/
python -c "
from nifi_client import NifiClient
import os

try:
    client = NifiClient(
        base_url=os.getenv('NIFI_ENDPOINT'),
        username=os.getenv('NIFI_USERNAME'),
        password=os.getenv('NIFI_PASSWORD'),
        ssl_verify=os.getenv('NIFI_SSL_VERIFY', 'false').lower() == 'true',
    )
    print('✅ Authentication successful')
except Exception as e:
    print(f'❌ Authentication failed: {e}')
"
```

**Fix:**

1. Verify credentials match NiFi's single-user configuration:
   ```bash
   ssh <nifi-host> "docker inspect nifi | grep -A2 SINGLE_USER_CREDENTIALS"
   ```

2. If credentials are wrong, update environment variables:
   ```bash
   export NIFI_USERNAME="admin"
   export NIFI_PASSWORD="<correct-password>"
   ```

3. If NiFi uses a different auth method (LDAP, certificate), update `nifi_client.py` accordingly.

---

### Symptom: Processors stuck in "Invalid" state

**Diagnosis:**

Open NiFi UI at `https://<nifi-host>:8443/nifi` and check processor status. Invalid processors show a yellow warning icon.

Click on the processor and check the "Properties" tab for validation errors.

**Common issues:**
- Missing required properties (Redis Host, Redis Port, Redis List Key)
- Invalid property values (non-numeric port, empty queue name)
- Processor type not found (see "processors not found" above)

**Fix:**

1. Check processor properties in NiFi UI
2. Update missing/invalid properties
3. Right-click processor → "Start" to activate

Or delete and recreate the flow:

```bash
cd nifi/
python -c "
from nifi_client import NifiClient
import os

client = NifiClient(
    base_url=os.getenv('NIFI_ENDPOINT'),
    username=os.getenv('NIFI_USERNAME'),
    password=os.getenv('NIFI_PASSWORD'),
    ssl_verify=os.getenv('NIFI_SSL_VERIFY', 'false').lower() == 'true',
)

pg_id = client.check_flow_exists('RAG Pipeline')
if pg_id:
    client.delete_process_group(pg_id)
    print('Deleted old flow')

# Re-run bootstrap
import subprocess
subprocess.run(['python', 'nifi_bootstrap.py'])
"
```

---

### Symptom: Messages not flowing through NiFi (queues empty on output side)

**Diagnosis:**

1. Check Redis queue depths:
   ```bash
   docker exec -it $(docker ps -q -f name=redis) redis-cli LLEN ocr_processing_job_input
   docker exec -it $(docker ps -q -f name=redis) redis-cli LLEN ocr_processing_job_output
   ```

2. If `_input` queue has messages but `_output` is empty, NiFi isn't processing.

3. Check NiFi UI for processor status:
   - Are processors running?
   - Are there errors in the processor logs?
   - Is the connection between consumer and producer valid?

**Fix:**

1. Start processors if stopped:
   ```bash
   cd nifi/
   python -c "
   from nifi_client import NifiClient
   import os

   client = NifiClient(
       base_url=os.getenv('NIFI_ENDPOINT'),
       username=os.getenv('NIFI_USERNAME'),
       password=os.getenv('NIFI_PASSWORD'),
       ssl_verify=os.getenv('NIFI_SSL_VERIFY', 'false').lower() == 'true',
   )

   pg_id = client.check_flow_exists('RAG Pipeline')
   if pg_id:
       client.start_process_group(pg_id)
       print('Started process group')
   "
   ```

2. Check processor logs in NiFi UI (right-click → "View data provenance" or "Show details")

3. Verify Redis connection properties in processor configuration match your Redis host/port

---

### Symptom: Need to rollback NiFi middleware

**Diagnosis:** NiFi is causing issues and you need to revert to direct Redis.

**Fix:**

1. Stop and delete the NiFi flow:
   ```bash
   cd nifi/
   python -c "
   from nifi_client import NifiClient
   import os, time

   client = NifiClient(
       base_url=os.getenv('NIFI_ENDPOINT'),
       username=os.getenv('NIFI_USERNAME'),
       password=os.getenv('NIFI_PASSWORD'),
       ssl_verify=os.getenv('NIFI_SSL_VERIFY', 'false').lower() == 'true',
   )

   pg_id = client.check_flow_exists('RAG Pipeline')
   if pg_id:
       # Stop the process group
       from nipyapi import canvas
       canvas.schedule_process_group(pg_id, scheduled=False)
       time.sleep(2)
       client.delete_process_group(pg_id)
       print('✅ Deleted RAG Pipeline')
   "
   ```

2. Revert worker code to use base queue names (remove `_input`/`_output` suffixes):
   - `ocr_utils.py`: `lpush(REDIS_OCR_JOB_QUEUE, ...)` instead of `lpush(f"{REDIS_OCR_JOB_QUEUE}_input", ...)`
   - `whisper_utils.py`: `lpush(REDIS_WHISPER_JOB_QUEUE, ...)` instead of `lpush(f"{REDIS_WHISPER_JOB_QUEUE}_input", ...)`
   - `producer_graph.py`: `queue_name` instead of `f"{queue_name}_input"`
   - `ocr_worker.py`: `brpop(REDIS_OCR_JOB_QUEUE, ...)` instead of `brpop(f"{REDIS_OCR_JOB_QUEUE}_output", ...)`
   - `whisperx_worker.py`: `brpop(REDIS_WHISPER_JOB_QUEUE, ...)` instead of `brpop(f"{REDIS_WHISPER_JOB_QUEUE}_output", ...)`
   - `consumer_worker.py`: `blpop(queue_name, ...)` instead of `blpop(f"{queue_name}_output", ...)`

3. Restart workers:
   ```bash
   ./doc-ingest-chat/run-compose.sh --build
   ```

---

### NiFi UI Access

Open `https://<nifi-host>:8443/nifi` in a browser.

**What to check:**
- Process group "RAG Pipeline" exists on root canvas
- Inside it: consumer/producer pairs for each queue
- All processors show "Running" status (green play icon)
- Queue depths visible on connections (should be low/zero during normal operation)
- No red error icons on processors

**Useful actions:**
- Right-click processor → "View data provenance" to see message flow
- Right-click processor → "Show details" to see configuration and logs
- Right-click connection → "List queue" to see queued FlowFiles
