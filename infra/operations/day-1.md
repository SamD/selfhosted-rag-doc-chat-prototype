# Day 1 — Initial Setup & Deployment

Linear checklist for first-time deployment and startup verification.

## 1. Environment Variables

### Required — will cause `sys.exit(1)` if missing

```bash
export DEFAULT_DOC_INGEST_ROOT=/path/to/rag-data
export EMBEDDING_ENDPOINTS=/path/to/e5-large-v2
export LLM_PATH=/path/to/phi-3.5-mini-instruct-q6_k.gguf
export SUPERVISOR_LLM_ENDPOINTS=/path/to/qwen3.5-4b-mtp-ud-q4_k_xl.gguf
```

### Optional — sensible defaults exist

```bash
export VECTOR_DB_PROFILE=qdrant          # or chroma
export OCR_ENDPOINTS=LOCAL               # or remote docling-serve URL
export WHISPER_MODEL_ENDPOINTS=NOT_SET   # or remote URL
export MAX_SESSION_TURNS=20              # chat history limit
export SESSION_TTL_HOURS=24              # session expiry
```

Full reference: `shared/env_names.py` and `shared/defaults.py`.

## 2. Start Services

```bash
# Qdrant (GPU assumed, cuda profile applied automatically)
VECTOR_DB_PROFILE=qdrant ./run-compose.sh

# Chroma
VECTOR_DB_PROFILE=chroma ./run-compose.sh
```

### Verify all containers are running

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

Expected containers: `redis`, `vector-db` (qdrant/chroma), `gatekeeper`, `producer`, `consumer-0`, `consumer-1`, `ocr-worker`, `whisperx-worker`, `api`, `frontend`. Plus HAProxy containers if multi-endpoint configured.

## 3. Verify Health

### API health

```bash
curl http://localhost:8000/api/v1/health
# {"status":"healthy","message":"API is running"}
```

### API status (checks vector DB connectivity)

```bash
curl http://localhost:8000/api/v1/status
# {"status":"operational","collection_count":42,"model_info":{...}}
```

### Frontend

Open `http://localhost:4321` in a browser. Should show chat UI with dark theme by default.

### Worker logs — check for startup errors

```bash
docker logs api 2>&1 | tail -20
docker logs gatekeeper 2>&1 | tail -20
docker logs producer 2>&1 | tail -20
docker logs consumer-0 2>&1 | tail -20
```

### Redis queues — confirm workers are connected

```bash
docker exec -it $(docker ps -q -f name=redis) redis-cli LLEN chunk_ingest_queue:0
docker exec -it $(docker ps -q -f name=redis) redis-cli LLEN chunk_ingest_queue:1
```

Should return `(integer) 0` on a fresh system.

## 4. Smoke Test — Ingestion

### Place a test file in staging

```bash
echo "Test document content" > $DEFAULT_DOC_INGEST_ROOT/staging/test.txt
```

### Monitor lifecycle progression

```bash
duckdb $DEFAULT_DOC_INGEST_ROOT/chunks.duckdb \
  "SELECT original_filename, status, new_at, finalized_at \
   FROM ingestion_lifecycle ORDER BY new_at DESC LIMIT 5;"
```

Expected progression: `NEW` → `PREPROCESSING` → `PREPROCESSING_COMPLETE` → `INGESTING` → `CONSUMING` → `INGEST_SUCCESS`. Each transition typically takes 2–10 seconds depending on file size and model speeds.

### Verify success

```bash
duckdb $DEFAULT_DOC_INGEST_ROOT/chunks.duckdb \
  "SELECT original_filename, status, finalized_at - new_at AS duration \
   FROM ingestion_lifecycle WHERE status = 'INGEST_SUCCESS' ORDER BY finalized_at DESC LIMIT 5;"
```

### Verify chat can find the content

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What does the test document say?"}'
```

## 5. Smoke Test — Chat Session

### First query (no session_id — server generates one)

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Hello"}' | python3 -m json.tool
```

Response includes `session_id` — save this for follow-up queries.

### Follow-up query (uses session_id for history)

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What did I just say?","session_id":"<session_id_from_previous>"}' | python3 -m json.tool
```

Should reference the previous question, proving session history works.

## 6. HAProxy (if multi-endpoint configured)

### Verify stats pages

```bash
curl http://localhost:8404/stats   # Supervisor LLM
curl http://localhost:8405/stats   # Embeddings
curl http://localhost:8406/stats   # Whisper
curl http://localhost:8407/stats   # OCR
```

### Verify traffic distribution

```bash
docker logs haproxy_supervisor 2>&1 | grep "be_supervisor/" | tail -10
```

Expected: alternating `srv0`, `srv1`, etc. across requests.

---

## 7. NiFi Middleware (Required)

NiFi is deployed as transparent middleware between Redis queues. The `nifi_bootstrap` service automatically deploys the flow on startup.

### Prerequisites

- NiFi 2.x deployed and accessible (see [quickstart.md](../../docs/quickstart.md#nifi-docker-setup) for Docker setup)
- Python processors deployed to NiFi's extensions directory
- NiFi restarted after processor deployment

### Environment Variables

```bash
export NIFI_ENDPOINT="https://<nifi-host>:8443/nifi-api"
export NIFI_USERNAME="admin"
export NIFI_PASSWORD="<your-password>"
export NIFI_SSL_VERIFY="false"  # For self-signed certificates
```

### Deploy Python Processors

Copy the Redis bridge processors to NiFi's Python extensions directory:

```bash
# From the project root
scp nifi/python/extensions/RedisQueueConsumer.py <nifi-host>:/opt/nifi/nifi-current/python/extensions/
scp nifi/python/extensions/RedisQueueProducer.py <nifi-host>:/opt/nifi/nifi-current/python/extensions/

# Restart NiFi to load new processors
ssh <nifi-host> "docker restart nifi"
```

Wait 30-60 seconds for NiFi to restart and load the processors.

### Start the Stack

```bash
./doc-ingest-chat/run-compose.sh --build
```

The `nifi_bootstrap` service will automatically:
1. Wait for NiFi to become available (with exponential backoff)
2. Check if "RAG Pipeline" process group exists
3. If not, create the process group with consumer/producer pairs for each queue
4. Start all processors
5. Verify flow health
6. Exit (one-shot service, doesn't restart)

### Verify Bootstrap Success

Check the bootstrap service logs:

```bash
docker logs nifi_bootstrap
```

Expected output:
```
✅ NiFi endpoint configured successfully
✅ NiFi is available. Root process group ID: <id>
✅ Created process group 'RAG Pipeline' (ID: <id>)
✅ Created consumer processor for ocr_processing_job_input
✅ Created producer processor for ocr_processing_job_output
✅ Created connection: ocr_processing_job_input → ocr_processing_job_output
... (for each queue)
✅ Flow started successfully: RAG Pipeline
✅ All processors are running
✅ Flow deployed, started, and healthy
```

### Verify Flow Health

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
    healthy = client.verify_flow_health(pg_id)
    print('Flow health:', '✅ All processors running' if healthy else '❌ Some processors not running')
else:
    print('❌ RAG Pipeline not found')
"
```

### Access NiFi UI

Open `https://<nifi-host>:8443/nifi` in a browser. You should see:
- The "RAG Pipeline" process group on the root canvas
- Inside it: consumer/producer pairs for each queue
- All processors showing "Running" status
- Queue depths visible on connections

### Manual Bootstrap (if needed)

If the bootstrap service fails or you need to redeploy:

```bash
cd nifi/
python nifi_bootstrap.py
```

### Delete the Flow (Rollback)

To remove the NiFi middleware and revert to direct Redis:

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
    # Stop the process group first
    client.start_process_group(pg_id)  # This toggles, so call again to stop
    import time
    time.sleep(2)
    client.delete_process_group(pg_id)
    print('✅ Deleted RAG Pipeline process group')
else:
    print('RAG Pipeline not found')
"
```

Then revert worker queue names to remove `_input`/`_output` suffixes (or use the pre-NiFi code).

---

## Checklist

- [ ] All 4 required env vars set
- [ ] NiFi deployed and accessible
- [ ] Python processors deployed to NiFi
- [ ] `docker ps` shows all expected containers as healthy
- [ ] `nifi_bootstrap` service completed successfully
- [ ] `curl /health` returns `"healthy"`
- [ ] `curl /status` returns `"operational"`
- [ ] Frontend loads at `http://localhost:4321`
- [ ] Test file ingested to `INGEST_SUCCESS`
- [ ] Chat API returns answer with citations
- [ ] Session ID works across follow-up queries
- [ ] (If HAProxy) Stats pages respond and traffic distributes
- [ ] All processors showing "Running" in NiFi UI
