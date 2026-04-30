CREATE TABLE IF NOT EXISTS ingestion_lifecycle (
    id VARCHAR PRIMARY KEY,
    status VARCHAR,
    original_filename VARCHAR,
    pdf_path VARCHAR,
    md_path VARCHAR,
    worker_id VARCHAR,
    error_log TEXT,
    trace_id VARCHAR,
    new_at TIMESTAMP,
    preprocessing_at TIMESTAMP,
    preprocessing_complete_at TIMESTAMP,
    ingesting_at TIMESTAMP,
    consuming_at TIMESTAMP,
    finalized_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS parquet_chunks (
    id VARCHAR PRIMARY KEY,
    chunk TEXT,
    source_file VARCHAR,
    document_id VARCHAR,
    type VARCHAR,
    chunk_index INTEGER,
    engine VARCHAR,
    hash VARCHAR,
    page INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS staged_chunks (
    id VARCHAR PRIMARY KEY,
    source_file VARCHAR,
    document_id VARCHAR,
    chunk TEXT,
    metadata JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS file_ingestion_jobs (
    file_path VARCHAR PRIMARY KEY,
    job_id VARCHAR,
    document_id VARCHAR,
    status VARCHAR,
    error_message VARCHAR,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gatekeeper_history (
    slug VARCHAR,
    timestamp TIMESTAMP,
    status VARCHAR,
    metadata TEXT,
    error VARCHAR
);
