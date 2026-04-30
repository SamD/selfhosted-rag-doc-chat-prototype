# 💾 DuckDB Operational Guide

This document provides a comprehensive list of SQL commands for monitoring, debugging, and maintaining the Self-Hosted RAG database.

---

## 🔍 1. Document Lifecycle Monitoring
The `ingestion_lifecycle` table is the primary state machine for the entire pipeline.

### **Check current status of all files:**
```sql
SELECT original_filename, status, worker_id, updated_at 
FROM ingestion_lifecycle 
ORDER BY updated_at DESC;
```

### **Identify "Stuck" jobs (Jobs in progress for > 1 hour):**
```sql
SELECT id, original_filename, status 
FROM ingestion_lifecycle 
WHERE status NOT LIKE '%SUCCESS%' 
  AND status NOT LIKE '%FAILED%'
  AND updated_at < (CURRENT_TIMESTAMP - INTERVAL 1 HOUR);
```

### **View detailed timing for a specific document:**
```sql
SELECT 
    original_filename,
    preprocessing_at - new_at AS normalization_latency,
    ingesting_at - preprocessing_complete_at AS chunking_latency,
    finalized_at - consuming_at AS persistence_latency
FROM ingestion_lifecycle 
WHERE status = 'INGEST_SUCCESS';
```

---

## 📊 2. Chunk & Vector Distribution
The `parquet_chunks` table contains the semantic units ready for archival.

### **Count chunks by media type:**
```sql
SELECT type, count(*) as chunk_count 
FROM parquet_chunks 
GROUP BY type;
```

### **Find the largest documents (by chunk count):**
```sql
SELECT source_file, count(*) as chunks 
FROM parquet_chunks 
GROUP BY source_file 
ORDER BY chunks DESC 
LIMIT 10;
```

### **Verify page-level distribution for a PDF:**
```sql
SELECT page, count(*) as chunks_per_page
FROM parquet_chunks 
WHERE source_file = 'outline_of_history_pt1.pdf'
GROUP BY page 
ORDER BY page ASC;
```

---

## 🛠️ 3. Persistence & Staging (Zero-Memory Path)
The `staged_chunks` table acts as the persistent buffer for large documents.

### **Inspect current staging buffer size:**
```sql
SELECT count(*) as enqueued_chunks, count(DISTINCT source_file) as active_files
FROM staged_chunks;
```

### **Verify chunk integrity before persistence:**
```sql
-- Checks if any staged chunk exceeds the 512 token limit
-- (Note: exact token count requires the Python tokenizer, this is a char-length proxy)
SELECT id, length(chunk) as chars 
FROM staged_chunks 
ORDER BY chars DESC 
LIMIT 10;
```

---

## 🚨 4. Production Debugging (The "War Room" Queries)

### **Scenario: The UI says "Data not found" but I ingested the file.**
Check if the file is actually in the terminal state:
```sql
SELECT status, error_log 
FROM ingestion_lifecycle 
WHERE original_filename = 'my_missing_file.pdf';
```

### **Scenario: Ingestion is slow. Is there lock contention?**
Check the historical audit for failures:
```sql
SELECT slug, status, error 
FROM gatekeeper_history 
WHERE status = 'FAILURE' 
ORDER BY timestamp DESC;
```

### **Scenario: The Vector DB and DuckDB are out of sync.**
Find chunks that exist in DuckDB but might be missing their deterministic ID:
```sql
SELECT id, source_file 
FROM parquet_chunks 
WHERE id NOT LIKE 'DOC_%';
```

---

## 📈 5. Schema Evolution
Since we use a declarative `schema.sql`, always update that file first. To apply changes to a live production DB, use these patterns:

### **Adding a new column (e.g., for language detection):**
```sql
ALTER TABLE ingestion_lifecycle ADD COLUMN language VARCHAR DEFAULT 'en';
```

### **Creating an index for faster RAG lookups:**
```sql
CREATE INDEX idx_source_file ON parquet_chunks (source_file);
```

### **Wiping local state for a fresh start:**
```sql
DELETE FROM ingestion_lifecycle;
DELETE FROM parquet_chunks;
DELETE FROM staged_chunks;
```
