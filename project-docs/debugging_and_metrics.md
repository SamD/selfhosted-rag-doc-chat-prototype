# Debugging, Inspection & Metrics

## 🛠️ Debugging and Inspection

### Redis Queue Inspection

To inspect if ingestion is progressing:

- Use the Redis CLI or GUI to check the state of producer and consumer queues:

```bash
redis-cli -p 6379
redis-cli -p 6380

> LRANGE chunk_ingest_queue:0 0 -1
> LRANGE chunk_ingest_queue:1 0 -1
```

If the queues remain full or never drain, check consumer logs. Each chunk is pushed with metadata, followed by a sentinel `file_end` message.

### DuckDB Inspection

To explore or debug ingested data and track job status:

```bash
# Connect to DuckDB shell
duckdb Docs/chunks.duckdb
```

#### Monitoring Ingestion Progress

The system tracks the lifecycle of every file in the `file_ingestion_jobs` table.

```sql
-- View counts of files by status
SELECT status, COUNT(*) FROM file_ingestion_jobs GROUP BY status;

-- List all failed files and their error messages
SELECT file_path, error_message, updated_at 
FROM file_ingestion_jobs 
WHERE status = 'failed'
ORDER BY updated_at DESC;

-- List files that are currently being processed or enqueued
SELECT file_path, status, updated_at 
FROM file_ingestion_jobs 
WHERE status IN ('processing', 'chunking', 'enqueuing', 'enqueued')
ORDER BY updated_at DESC;

-- List successfully completed files
SELECT file_path, updated_at 
FROM file_ingestion_jobs 
WHERE status = 'completed'
ORDER BY updated_at DESC;
```

#### Inspecting Ingested Chunks

The actual text data is stored in the `parquet_chunks` table.

```sql
-- View chunk counts by engine type
SELECT engine, COUNT(*) FROM parquet_chunks GROUP BY engine;

-- Check if specific content (e.g. "Bretton") exists
SELECT * FROM parquet_chunks WHERE chunk ILIKE '%bretton%';

-- View per-file chunk totals
SELECT source_file, COUNT(*) FROM parquet_chunks GROUP BY source_file ORDER BY COUNT(*) DESC;
```

#### Advanced Queries for Troubleshooting

```sql
-- Show all chunk rows from files with 'Bretton' OR 'Woods' in them
SELECT * FROM parquet_chunks WHERE chunk ILIKE '%bretton%' OR chunk ILIKE '%woods%';

-- Find chunk counts for files that likely mention monetary systems
SELECT source_file, COUNT(*) FROM parquet_chunks 
WHERE chunk ILIKE '%currency%' OR chunk ILIKE '%exchange rate%' OR chunk ILIKE '%gold standard%'
GROUP BY source_file ORDER BY COUNT(*) DESC;

-- Check for empty or too-short text chunks
SELECT * FROM parquet_chunks WHERE length(chunk) < 10;

-- Check chunk length distribution to detect overly small or excessively large chunks
SELECT length(chunk) AS token_count, COUNT(*) 
FROM parquet_chunks 
GROUP BY token_count 
ORDER BY token_count DESC;

-- Find chunks with unusual or null metadata (e.g., missing engine label)
SELECT * 
FROM parquet_chunks 
WHERE engine IS NULL OR engine = '';

-- Detect duplicate chunk texts (possible over-splitting or OCR duplication)
SELECT chunk, COUNT(*)
FROM parquet_chunks
GROUP BY chunk
HAVING COUNT(*) > 1
ORDER BY COUNT(*) DESC
LIMIT 10;
```

---

## 📊 Performance Metrics

The system collects performance metrics at key points in the document processing pipeline to help identify bottlenecks and monitor system health.

### Metrics Overview

Metrics are emitted as structured JSON logs to both stdout and a dedicated metrics file (`metrics.jsonl` by default). Three event types are tracked:

- **`file_processing_complete`** - Producer worker metrics (document extraction, OCR, chunking, Redis enqueue)
- **`file_storage_complete`** - Consumer worker metrics (ChromaDB embedding/storage, Parquet writes)
- **`ocr_job_complete`** - OCR worker metrics (Tesseract execution, image processing)

### Example Metrics Output

```jsonl
{"event":"file_processing_complete","timestamp":"2025-12-26T10:30:45.123456","worker":"producer","file":"docs/sample.pdf","metrics":{"total_processing_time_ms":15234.5,"text_extraction_time_ms":8120.3,"redis_enqueue_time_ms":120.8,"ocr_operations":[{"page":3,"ocr_roundtrip_time_ms":2340.5,"engine":"tesseract","success":true}],"chunks_produced":47,"pages_processed":12}}
{"event":"file_storage_complete","timestamp":"2025-12-26T10:30:50.789012","worker":"consumer","file":"docs/sample.pdf","queue":"chunk_ingest_queue:0","metrics":{"total_storage_time_ms":4523.7,"chromadb_embedding_time_ms":3890.2,"parquet_write_time_ms":183.0,"chunks_stored":47,"batches_processed":1}}
{"event":"ocr_job_complete","timestamp":"2025-12-26T10:30:48.456789","worker":"ocr","file":"docs/sample.pdf","page":3,"job_id":"abc123","metrics":{"total_processing_time_ms":2340.5,"tesseract_execution_time_ms":2280.1,"image_decode_time_ms":45.3,"engine":"tesseract","text_length":1024,"success":true}}
```

### Analyzing Metrics

Use `jq` to analyze the metrics file:

```bash
# Average ChromaDB storage time per file
jq -r 'select(.event == "file_storage_complete") | .metrics.chromadb_embedding_time_ms' metrics.jsonl | awk '{sum+=$1; count+=1} END {print "Avg ChromaDB time: " sum/count " ms"}'

# Files with longest processing times (top 20)
jq -r 'select(.event == "file_processing_complete") | "\(.metrics.total_processing_time_ms)\t\(.file)"' metrics.jsonl | sort -rn | head -20

# OCR roundtrip timing statistics (min/max/avg)
jq -r 'select(.event == "ocr_job_complete") | .metrics.total_processing_time_ms' metrics.jsonl | awk '{sum+=$1; count+=1; if(min==""){min=max=$1} if($1>max){max=$1} if($1<min){min=$1}} END {print "OCR timing - Min: " min "ms, Max: " max "ms, Avg: " sum/count "ms"}'

# Total chunks processed
jq -s '[.[] | select(.event == "file_storage_complete") | .metrics.chunks_stored] | add' metrics.jsonl
```

### Configuration

Metrics collection can be configured via environment variables in `ingest-svc.env`:

```bash
# Enable or disable metrics collection
METRICS_ENABLED=true

# File path for metrics output (JSONL format)
METRICS_LOG_FILE=${INGEST_FOLDER}/metrics.jsonl

# Also log metrics to stdout
METRICS_LOG_TO_STDOUT=true
```
