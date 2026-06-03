**[< Architecture](overview.md) | [Operations & Debugging](operations.md) | [Quick Start](quickstart.md)**

# Ingestion Analytics

The `ingestion_lifecycle` table in DuckDB (`$DEFAULT_DOC_INGEST_ROOT/chunks.duckdb`)
records precise timestamps at every pipeline phase. These queries produce the metrics
needed for throughput reporting, trend analysis, and bottleneck detection — ready
for dashboards, daily digests, or automated alerts.

## Schema — Timestamp Columns

| Column | Set when |
|---|---|
| `new_at` | File discovered in `staging/` and job created |
| `preprocessing_at` | Gatekeeper claims job, begins text extraction + LLM normalization |
| `preprocessing_complete_at` | Markdown output written to `ingestion/` |
| `ingesting_at` | Producer picks up the Markdown file |
| `consuming_at` | Consumer worker starts embedding chunks |
| `finalized_at` | Terminal state — `INGEST_SUCCESS` or `INGEST_FAILED` |

---

## Throughput — Files Per Time Window

Files successfully ingested per day, week, or hour:

```sql
-- per day (last 30)
SELECT
    strftime(new_at, '%Y-%m-%d')          AS day,
    count(*)                              AS files_ingested,
    sum(strftime(finalized_at, new_at))   AS total_seconds
FROM ingestion_lifecycle
WHERE status = 'INGEST_SUCCESS'
  AND new_at >= CURRENT_TIMESTAMP - INTERVAL 30 DAY
GROUP BY day
ORDER BY day DESC;

-- per hour (last 24)
SELECT
    strftime(new_at, '%Y-%m-%d %H:00') AS hour,
    count(*)                            AS files_ingested
FROM ingestion_lifecycle
WHERE status = 'INGEST_SUCCESS'
  AND new_at >= CURRENT_TIMESTAMP - INTERVAL 1 DAY
GROUP BY hour
ORDER BY hour DESC;
```

---

## Phase Latency Distribution

Show the spread of preprocessing and total turnaround times across all successful files:

```sql
SELECT
    count(*)                                                     AS files,
    avg(strftime(preprocessing_complete_at, preprocessing_at))   AS avg_preproc_sec,
    min(strftime(preprocessing_complete_at, preprocessing_at))   AS min_preproc_sec,
    median(strftime(preprocessing_complete_at, preprocessing_at)) AS med_preproc_sec,
    max(strftime(preprocessing_complete_at, preprocessing_at))   AS max_preproc_sec,
    avg(strftime(finalized_at, new_at))                          AS avg_total_sec,
    median(strftime(finalized_at, new_at))                       AS med_total_sec,
    max(strftime(finalized_at, new_at))                          AS max_total_sec
FROM ingestion_lifecycle
WHERE status = 'INGEST_SUCCESS';
```

---

## Pipeline Efficiency Breakdown

Where time is spent — as a percentage of total turnaround:

```sql
SELECT
    count(*)                                                          AS files,
    avg(strftime(preprocessing_complete_at, preprocessing_at))        AS avg_preproc_sec,
    avg(strftime(ingesting_at, preprocessing_complete_at))            AS avg_producer_queue_sec,
    avg(strftime(consuming_at, ingesting_at))                         AS avg_chunking_sec,
    avg(strftime(finalized_at, consuming_at))                         AS avg_persistence_sec,
    avg(strftime(finalized_at, new_at))                               AS avg_total_sec,
    round(avg(strftime(preprocessing_complete_at, preprocessing_at))
          / nullif(avg(strftime(finalized_at, new_at)), 0) * 100, 1) AS preproc_pct,
    round(avg(strftime(ingesting_at, preprocessing_complete_at))
          / nullif(avg(strftime(finalized_at, new_at)), 0) * 100, 1) AS queue_pct,
    round(avg(strftime(consuming_at, ingesting_at))
          / nullif(avg(strftime(finalized_at, new_at)), 0) * 100, 1) AS chunking_pct,
    round(avg(strftime(finalized_at, consuming_at))
          / nullif(avg(strftime(finalized_at, new_at)), 0) * 100, 1) AS persistence_pct
FROM ingestion_lifecycle
WHERE status = 'INGEST_SUCCESS';
```

---

## Error Rate Report

Success vs. failure counts and rates over time:

```sql
SELECT
    strftime(new_at, '%Y-%m-%d')                   AS day,
    count(*)                                        AS total,
    sum(CASE WHEN status = 'INGEST_SUCCESS' THEN 1 ELSE 0 END)  AS succeeded,
    sum(CASE WHEN status = 'INGEST_FAILED' THEN 1 ELSE 0 END)   AS failed,
    round(sum(CASE WHEN status = 'INGEST_FAILED' THEN 1 ELSE 0 END)
          / nullif(count(*), 0) * 100, 1)                        AS failure_pct
FROM ingestion_lifecycle
WHERE new_at >= CURRENT_TIMESTAMP - INTERVAL 30 DAY
GROUP BY day
ORDER BY day DESC;
```

---

## Largest Files by Preprocessing Time

Identify files that dominated preprocessing, indicating heavy OCR, large page counts, or slow
remote endpoints:

```sql
SELECT
    original_filename,
    strftime(preprocessing_complete_at, preprocessing_at) AS preprocessing_sec,
    strftime(finalized_at, new_at)                        AS total_sec,
    new_at                                                 AS started
FROM ingestion_lifecycle
WHERE status = 'INGEST_SUCCESS'
ORDER BY preprocessing_sec DESC
LIMIT 20;
```

---

## Top-Level Summary (Current State)

A single-row snapshot of the entire pipeline — total files, aggregate time, active count:

```sql
SELECT
    (SELECT count(*) FROM ingestion_lifecycle WHERE status = 'INGEST_SUCCESS')   AS total_succeeded,
    (SELECT count(*) FROM ingestion_lifecycle WHERE status = 'INGEST_FAILED')    AS total_failed,
    (SELECT count(*) FROM ingestion_lifecycle
     WHERE status NOT IN ('INGEST_SUCCESS', 'INGEST_FAILED'))                    AS in_flight,
    (SELECT avg(strftime(finalized_at, new_at))
     FROM ingestion_lifecycle WHERE status = 'INGEST_SUCCESS')                   AS avg_total_sec,
    (SELECT sum(strftime(finalized_at, new_at))
     FROM ingestion_lifecycle WHERE status = 'INGEST_SUCCESS')                   AS cumulative_sec;
```

---

## Weekly Comparison

Compare this week's performance against the previous week:

```sql
WITH weekly AS (
    SELECT
        strftime(new_at, '%Y-%W')                     AS week,
        count(*)                                       AS files,
        avg(strftime(finalized_at, new_at))            AS avg_sec,
        avg(strftime(preprocessing_complete_at, preprocessing_at)) AS avg_preproc_sec
    FROM ingestion_lifecycle
    WHERE status = 'INGEST_SUCCESS'
    GROUP BY week
)
SELECT week, files, avg_sec, avg_preproc_sec,
       lag(avg_sec) OVER (ORDER BY week)   AS prev_week_avg_sec,
       avg_sec - lag(avg_sec) OVER (ORDER BY week) AS delta_sec
FROM weekly
ORDER BY week DESC
LIMIT 12;
```

---

## Per-File Timing Breakdown

Every phase of every successfully ingested file, with computed durations in seconds:

```sql
SELECT
    original_filename,
    new_at,
    finalized_at,
    strftime(preprocessing_complete_at, preprocessing_at)    AS preprocessing_sec,
    strftime(ingesting_at, preprocessing_complete_at)        AS queue_to_producer_sec,
    strftime(consuming_at, ingesting_at)                     AS chunking_sec,
    strftime(finalized_at, consuming_at)                     AS persistence_sec,
    strftime(finalized_at, new_at)                           AS total_sec
FROM ingestion_lifecycle
WHERE status = 'INGEST_SUCCESS'
ORDER BY new_at DESC;
```

---

## Content Distribution

Chunk types and volume — useful for sizing reports and storage cost estimation:

```sql
-- Chunks by extraction type
SELECT type, count(*) AS chunk_count
FROM parquet_chunks
GROUP BY type;

-- Total chunk inventory
SELECT count(*) AS total_chunks,
       count(DISTINCT source_file) AS unique_files,
       round(avg(length(chunk))) AS avg_chunk_chars
FROM parquet_chunks;

-- Largest files by chunk count (storage cost drivers)
SELECT source_file, count(*) AS chunks
FROM parquet_chunks
GROUP BY source_file
ORDER BY chunks DESC
LIMIT 20;
```

---

## Staging Buffer Metrics

Current pipeline backlog — how many chunks are queued for embedding:

```sql
SELECT
    count(*)                            AS enqueued_chunks,
    count(DISTINCT source_file)         AS active_files,
    max(timestamp)                      AS oldest_chunk_age
FROM staged_chunks;
```

---

## Notes

- All time values are in seconds (`strftime(a, b)` subtraction returns seconds).
- Queries assume timestamps are in UTC.
- Replace `INTERVAL 30 DAY` with any range — `1 HOUR`, `7 DAY`, `90 DAY`, etc.
- For large datasets, add `WHERE finalized_at IS NOT NULL` to exclude in-flight jobs.
- The `median()` function is available in DuckDB natively.
