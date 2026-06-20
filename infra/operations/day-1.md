# Day 1 — Initial Setup & Deployment

Linear checklist for first-time deployment and startup verification.

## 1. Environment Variables

### Required — will cause `sys.exit(1)` if missing

```bash
export DEFAULT_DOC_INGEST_ROOT=/path/to/rag-data
export EMBEDDING_ENDPOINTS=http://<embedding-host>:11434/v1/embeddings
export LLM_PATH=http://<llm-host>:11435/v1/chat/completions
export SUPERVISOR_LLM_ENDPOINTS=http://<llm-host>:11534/v1/chat/completions
export TOKENIZER_MODEL_PATH=/path/to/mxbai-embed-large-v1
export REDIS_HOST=<redis-host>
export REDIS_PORT=6380
```

### Optional — sensible defaults exist

```bash
export VECTOR_DB_PROFILE=qdrant          # or chroma
export VECTOR_DB_URL=http://<vector-db-host>:6334
export VECTOR_DB_USE_GRPC=true
export OCR_ENDPOINTS=http://<ocr-host>:5001/v1/convert/file  # or LOCAL
export WHISPER_MODEL_ENDPOINTS=http://<whisper-host>:1145/inference  # or NOT_SET
export PDF_FORCE_OCR=false
export HA_INTERLEAVE=false
export FORCE_MARKDOWN_LLM=false
export EMBEDDING_BATCH_SIZE=25
export USE_TEMPORAL_WHISPER=false
export TEMPORAL_HOST=<temporal-host>
export TEMPORAL_PORT=7233
export TEMPORAL_WHISPER_TASK_QUEUE=whisperx
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

## 7. Temporal Services (if USE_TEMPORAL_WHISPER=true)

Temporal is a remote service — no Docker Compose services are provided. Deploy Temporal separately (e.g., Temporal Cloud, self-hosted Kubernetes, or a standalone `temporalite` process on another host).

### Set Temporal env vars

```bash
export USE_TEMPORAL_WHISPER=true
export TEMPORAL_HOST=<temporal-host>       # e.g. temporal.example.com
export TEMPORAL_PORT=7233                  # gRPC port (default: 7233)
export TEMPORAL_WHISPER_TASK_QUEUE=whisperx  # task queue (default: whisperx)
```

### Start with Temporal enabled

```bash
USE_TEMPORAL_WHISPER=true TEMPORAL_HOST=<temporal-host> ./run-compose.sh
```

### Verify Temporal connectivity

```bash
# Check the Temporal worker container logs for connection
docker logs whisperx_worker 2>&1 | grep -i temporal
```

### Expected containers
- `whisperx_worker` connects to the remote Temporal server via gRPC
- No local Temporal server containers (temporal-server, temporal-web, temporalite) are started

## Checklist

- [ ] All required env vars set (`DEFAULT_DOC_INGEST_ROOT`, `EMBEDDING_ENDPOINTS`, `LLM_PATH`, `SUPERVISOR_LLM_ENDPOINTS`, `TOKENIZER_MODEL_PATH`, `REDIS_HOST`, `REDIS_PORT`)
- [ ] `docker ps` shows all expected containers as healthy
- [ ] `curl /health` returns `"healthy"`
- [ ] `curl /status` returns `"operational"`
- [ ] Frontend loads at `http://localhost:4321`
- [ ] Test file ingested to `INGEST_SUCCESS`
- [ ] Chat API returns answer with citations
- [ ] Session ID works across follow-up queries
- [ ] (If HAProxy) Stats pages respond and traffic distributes
- [ ] (If Temporal) `USE_TEMPORAL_WHISPER=true` + `TEMPORAL_HOST` set, worker connects to remote server
