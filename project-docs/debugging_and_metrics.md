# Debugging, Inspection & Metrics

## 🛠️ Debugging and Inspection

### 🧬 Lifecycle Inspection (DuckDB)
The `ingestion_lifecycle` table in `chunks.duckdb` is the primary source for debugging document flow.

```bash
# Connect to the lifecycle database
duckdb Docs/chunks.duckdb
```

#### Monitoring Document Velocity
```sql
-- Track how long each document spent in normalization vs chunking
SELECT 
    original_filename,
    preprocessing_complete_at - preprocessing_at as normalization_time,
    consuming_at - ingesting_at as chunking_time,
    finalized_at - new_at as total_turnaround
FROM ingestion_lifecycle 
WHERE status = 'INGEST_SUCCESS';
```

#### Inspecting Errors & Tracebacks
```sql
-- See detailed error logs for failed ingestions
SELECT original_filename, error_log 
FROM ingestion_lifecycle 
WHERE status = 'INGEST_FAILED' 
ORDER BY finalized_at DESC;
```

#### Verifying Physical File Locations
```sql
-- Find where the PDF and MD are currently located for an active job
SELECT status, pdf_path, md_path 
FROM ingestion_lifecycle 
WHERE original_filename LIKE '%history%';
```

---

### 📥 Redis Queue Inspection
Check the state of the ingestion queues using the unified port **6379**.

```bash
# Connect to the Redis container
docker exec -it doc-ingest-chat-redis-1 redis-cli

# Check lengths of partitioned queues
> LLEN chunk_ingest_queue:0
> LLEN chunk_ingest_queue:1
```

---

### 🧠 Qdrant Vector Inspection
Verify data is indexed and reachable in the vector store.

#### 1. Quick Count for a Document
```bash
curl -X POST http://localhost:9002/collections/vector_base_collection/points/count \
  -H "Content-Type: application/json" \
  -d '{
    "filter": {
        "must": [{"key": "metadata.slug", "match": {"value": "outline-of-history-pt1-4e82d6f5"}}]
    }
  }'
```

#### 2. Web UI (Dashboard)
Visit: `http://localhost:9002/dashboard`

---

## 📊 Performance Metrics

### Analyzing Metrics with JQ
Metrics are stored in `Docs/metrics.jsonl`.

```bash
# Average normalization time per document
jq -r 'select(.event == "file_processing_complete") | .metrics.total_processing_time_ms' Docs/metrics.jsonl | awk '{sum+=$1; count+=1} END {print "Avg Normalization: " sum/count " ms"}'
```
