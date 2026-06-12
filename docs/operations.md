**[< Architecture](overview.md) | [Deep Dive](deep-dive.md) | [Quick Start](quickstart.md)**

# Operations, Debugging & Metrics

This document covers operational procedures for monitoring, debugging, and maintaining the Self-Hosted RAG system.

---

## DuckDB — Lifecycle Inspection

The `chunks.duckdb` database (in `$DEFAULT_DOC_INGEST_ROOT`) is the primary source of truth for all document state.

```bash
duckdb $DEFAULT_DOC_INGEST_ROOT/chunks.duckdb
```

### Document Status Overview

```sql
SELECT original_filename, status, worker_id, new_at
FROM ingestion_lifecycle
ORDER BY new_at DESC;
```

### Find Stuck Jobs (In progress > 1 hour)

```sql
SELECT id, original_filename, status
FROM ingestion_lifecycle
WHERE status NOT LIKE '%SUCCESS%'
  AND status NOT LIKE '%FAILED%'
  AND new_at < (CURRENT_TIMESTAMP - INTERVAL 1 HOUR);
```

### Timing Breakdown (Per Document)

```sql
SELECT
    original_filename,
    preprocessing_complete_at - preprocessing_at AS normalization_time,
    ingesting_at - preprocessing_complete_at AS chunking_time,
    consuming_at - ingesting_at AS queue_time,
    finalized_at - consuming_at AS persistence_time,
    finalized_at - new_at AS total_turnaround
FROM ingestion_lifecycle
WHERE status = 'INGEST_SUCCESS';
```

### Inspect Errors

```sql
SELECT original_filename, error_log
FROM ingestion_lifecycle
WHERE status = 'INGEST_FAILED'
ORDER BY finalized_at DESC;
```

### Verify Physical File Locations

```sql
SELECT status, pdf_path, md_path
FROM ingestion_lifecycle
WHERE original_filename LIKE '%my_document%';
```

### Lock Contention Audit

```sql
SELECT slug, status, error
FROM gatekeeper_history
WHERE status = 'FAILURE'
ORDER BY timestamp DESC;
```

---

## DuckDB — Chunk Distribution

### Count Chunks by Type

```sql
SELECT type, count(*) AS chunk_count
FROM parquet_chunks
GROUP BY type;
```

### Largest Documents (by Chunk Count)

```sql
SELECT source_file, count(*) AS chunks
FROM parquet_chunks
GROUP BY source_file
ORDER BY chunks DESC
LIMIT 10;
```

### Page-Level Distribution

```sql
SELECT page, count(*) AS chunks_per_page
FROM parquet_chunks
WHERE source_file = 'my_document.pdf'
GROUP BY page
ORDER BY page ASC;
```

---

## DuckDB — Staging Inspection

The `staged_chunks` table acts as the Write-Ahead Log for chunks before Qdrant persistence.

### Current Buffer Size

```sql
SELECT count(*) AS enqueued_chunks, count(DISTINCT source_file) AS active_files
FROM staged_chunks;
```

### Chunk Size Check (Character-Length Proxy)

```sql
SELECT id, length(chunk) AS chars
FROM staged_chunks
ORDER BY chars DESC
LIMIT 10;
```

### Integrity Check — Non-Deterministic IDs

```sql
SELECT id, source_file
FROM parquet_chunks
WHERE id NOT LIKE 'DOC_%';
```

---

## Redis — Queue Inspection

With NiFi middleware, each queue has `_input` and `_output` suffix pairs. Workers write to `*_input` and read from `*_output` — NiFi's bridge processors move data between them. Queue length on `_input` represents backpressure: if it's growing, NiFi may be overwhelmed or a downstream worker is stalled.

```bash
docker exec -it doc-ingest-chat-redis-1 redis-cli
```

```redis
> LLEN chunk_ingest_input:0
> LLEN chunk_ingest_output:0
> LLEN chunk_ingest_input:1
> LLEN chunk_ingest_output:1
> LLEN ocr_job_input
> LLEN ocr_job_output
> LLEN whisper_job_input
> LLEN whisper_job_output
```

Also check that there are no orphaned `_output` queues (NiFi should handle draining them):

```redis
> LLEN chunk_ingest_queue:0
> LLEN chunk_ingest_queue:1
```

These legacy queue names should be empty after the NiFi migration — if they contain data, NiFi's bridge processors aren't consuming correctly.

---

## NiFi — Flow Monitoring

### Access the NiFi UI

Navigate to `https://<nifi-host>:<nifi-port>/nifi` (default `8443`) and log in. The "RAG Pipeline" process group contains all bridge processors organized into sub-groups: OCR Processing, Whisper Processing, Retype to Markdown LLM, and Chunk and Tokenize Consumer.

### Check Processor Health

In the NiFi UI, each processor shows:
- **Status icon**: Green (running), yellow (warning), red (stopped/error)
- **Tasks/Time**: How many flowfiles are being processed
- **In/Out**: Queue counts showing data flow

Right-click any processor → **View data provenance** to trace individual flowfiles through the pipeline. Each flowfile carries the original `trace_id` from the worker as an attribute, preserved end-to-end.

### Verify Flow Health via API

```bash
curl -k -u "$NIFI_USERNAME:$NIFI_PASSWORD" \
  "https://<nifi-host>:8443/nifi-api/process-groups/root/process-groups" | \
  jq '.processGroups[] | select(.component.name == "RAG Pipeline") | .id'
```

```bash
# Get processor status for the RAG Pipeline group
PG_ID="<pg-id-from-above>"
curl -k -u "$NIFI_USERNAME:$NIFI_PASSWORD" \
  "https://<nifi-host>:8443/nifi-api/process-groups/$PG_ID/processors" | \
  jq '.processors[] | {name: .component.name, state: .component.state, tasks: .status.aggregateSnapshot.tasks}'
```

### Check Bootstrap Logs

The `nifi_bootstrap` service deploys the flow on first start. Check its logs if processors aren't running:

```bash
docker logs nifi_bootstrap
```

### Restart the Flow

If NiFi was restarted, re-deploy:

```bash
cd nifi/ && PYTHONPATH=.. python ./nifi_bootstrap.py
```

The bootstrap is idempotent — it skips creation if the "RAG Pipeline" process group already exists and just starts all processors.

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Processors stuck yellow/stopped | NiFi restarted without re-bootstrap | Run `nifi_bootstrap.py` |
| `_input` queues growing, `_output` empty | Consumer worker down or slow | Check Consumer worker logs |
| `_output` queues growing, `_input` empty | NiFi processors running but downstream not consuming | Check downstream worker logs |
| NiFi UI shows 0/0 in/out | No data flowing — check worker logs for Redis connectivity | Check `REDIS_HOST`/`REDIS_PORT` in worker env |
| Provenance shows flowfile stuck | Processor error (check bulletins in NiFi UI) | Right-click processor → View data provenance → Error tab |

---

## Whisper.cpp — Server Debugging

The whisper.cpp server must be started with `--convert` to handle non-WAV formats (MP4, MP3, M4A, etc.). Without it, the server returns `400 Bad Request` and logs `failed to decode audio data from memory buffer`.

### Verify Server

```bash
curl http://<whisper-host>:1145/
```

Should return an HTML page with the API documentation.

### Test with WAV (basic)

```bash
curl http://<whisper-host>:1145/inference \
  -F "file=@/path/to/test.wav" \
  -F "temperature=0.0" \
  -F "response_format=json"
```

### Test with MP4 (requires --convert)

```bash
curl http://<whisper-host>:1145/inference \
  -F "file=@/path/to/test.mp4" \
  -F "temperature=0.0" \
  -F "temperature_inc=0.2" \
  -F "no_speech_thold=0.6" \
  -F "response_format=json"
```

### Convert MP4 to WAV manually

```bash
ffmpeg -i /path/to/test.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 /tmp/test.wav
```

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `400 Bad Request` + `failed to decode audio data` | Server missing `--convert` flag | Restart server with `--convert` |
| `Connection refused` | Server not running or wrong host/port | Check `WHISPER_MODEL_ENDPOINTS` env var |
| Transcription empty | Audio too quiet or no speech | Check `no_speech_thold` parameter |

---

## HAProxy — Load Balancer Monitoring

When multi-endpoint `*_ENDPOINTS` env vars are set, HAProxy containers handle request distribution across backends.

### Stats UI

| Service | URL |
|---------|-----|
| Supervisor LLM | `http://localhost:8404/stats` |
| Embedding | `http://localhost:8405/stats` |
| Whisper | `http://localhost:8406/stats` |
| OCR | `http://localhost:8407/stats` |

### Check HAProxy Logs

```bash
docker logs haproxy_supervisor 2>&1 | grep "POST\|GET" | tail -20
docker logs haproxy_embd 2>&1 | grep "POST\|GET" | tail -20
docker logs haproxy_whisper 2>&1 | grep "POST\|GET" | tail -20
docker logs haproxy_ocr 2>&1 | grep "POST\|GET" | tail -20
```

### Verify Traffic Distribution

Requests should alternate between backends (`srv0`, `srv1`, etc.):

```bash
docker logs haproxy_supervisor 2>&1 | grep "be_supervisor/" | tail -10
```

Expected output shows alternating backends:
```
be_supervisor/srv0 ... "POST /v1/chat/completions HTTP/1.1"
be_supervisor/srv1 ... "POST /v1/chat/completions HTTP/1.1"
be_supervisor/srv0 ... "POST /v1/chat/completions HTTP/1.1"
```

### Check Backend Health

HAProxy health-checks backends every 2s via `GET /models` (or `GET /health`). A backend is marked down after 3 consecutive failures and up after 2 successes.

```bash
docker exec haproxy_supervisor cat /tmp/haproxy.cfg | grep "server srv"
```

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| All traffic to one backend | Keep-alive connection pinning | `option httpclose` is set — check if client is reusing connections |
| Backend marked DOWN | Health check failing | Check if the backend's `/models` or `/health` endpoint responds |
| 503 from HAProxy | 0 endpoints configured | Set `*_ENDPOINTS` env var or point `*_PATH` directly to backend |

---

## Qdrant — Vector Inspection

### Point Count for a Document

```bash
curl -X POST http://<vector-db-host>:6333/collections/vector_base_collection/points/count \
  -H "Content-Type: application/json" \
  -d '{
    "filter": {
      "must": [{"key": "source_file", "match": {"text": "my_document"}}]
    }
  }'
```

Replace `<vector-db-host>` with your Qdrant REST API endpoint (default port 6333, or as configured via `VECTOR_DB_URL`).

### Sample Payloads

```bash
curl -X POST http://<vector-db-host>:6333/collections/vector_base_collection/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit": 3, "with_payload": true, "with_vector": false}'
```

### Qdrant Dashboard

Visit `http://<vector-db-host>:6333/dashboard` for the built-in web UI.

---

## Metrics (JSONL)

Metrics are recorded in `$DEFAULT_DOC_INGEST_ROOT/metrics.jsonl`.

### Average Normalization Time

```bash
jq -r 'select(.event == "file_processing_complete") | .metrics.total_processing_time_ms' \
  $DEFAULT_DOC_INGEST_ROOT/metrics.jsonl | \
  awk '{sum+=$1; count+=1} END {print "Avg: " sum/count " ms"}'
```

---

## Schema Evolution

The canonical schema is in `doc-ingest-chat/sql/schema.sql`. Update that file first, then apply changes:

### Add a Column

```sql
ALTER TABLE ingestion_lifecycle ADD COLUMN language VARCHAR DEFAULT 'en';
```

### Create an Index

```sql
CREATE INDEX idx_source_file ON parquet_chunks (source_file);
```

### Wipe All State (Fresh Start)

**Warning**: This permanently deletes all ingestion state and chunk data. Back up `chunks.duckdb` before running these commands. Qdrant vectors are not affected by DuckDB deletes and must be cleaned separately.

```sql
DELETE FROM ingestion_lifecycle;
DELETE FROM parquet_chunks;
DELETE FROM staged_chunks;
DELETE FROM file_ingestion_jobs;
DELETE FROM gatekeeper_history;
```

---

## War Room Scenarios

### "Data not found" but file was ingested

```sql
SELECT status, error_log, pdf_path, md_path
FROM ingestion_lifecycle
WHERE original_filename = 'missing_file.pdf';
```

### DuckDB and Qdrant out of sync

```sql
-- Find chunks in DuckDB without deterministic IDs
SELECT id, source_file
FROM parquet_chunks
WHERE id NOT LIKE 'DOC_%';
```

### Ingestion stalled

Check for lock contention using the query in [Lock Contention Audit](#lock-contention-audit) above. Also check NiFi queue backpressure — if `LLEN chunk_ingest_input:N` is growing without bound, the Consumer may have crashed or be blocked. In the NiFi UI, check that the "RAG Pipeline" processors are green (running) and not showing errors in the bulletins.
