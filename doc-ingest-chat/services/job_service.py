#!/usr/bin/env python3
"""
Job service for tracking file ingestion states in DuckDB.
This replaces the old text-file based tracking (ingested_files.txt/failed_files.txt)
with a relational schema for better state management and restart resilience.
"""

import datetime
import random
import time
from typing import Any, Dict, Optional

import duckdb
from config.settings import DUCKDB_FILE
from utils.logging_config import setup_logging

log = setup_logging("job_service.log")

# Standardized job statuses used throughout the IngestState graph
STATUS_PENDING = "pending"      # File discovered but not yet started
STATUS_PROCESSING = "processing" # File currently being handled by a worker
STATUS_CHUNKING = "chunking"     # Text extraction successful, splitting into chunks
STATUS_ENQUEUING = "enqueuing"   # Chunks being pushed to Redis
STATUS_ENQUEUED = "enqueued"     # All chunks and sentinel successfully enqueued
STATUS_COMPLETED = "completed"   # Consumer has successfully stored all chunks in Vector DB
STATUS_FAILED = "failed"         # An error occurred during any stage of ingestion


class JobService:
    """
    Job service for tracking ingestion states as static methods.
    Encapsulates all DuckDB interactions for file-level status tracking.
    Includes retry logic to handle DuckDB file lock contention in concurrent environments.
    """

    @staticmethod
    def _execute_with_retry(query: str, params: tuple = (), fetch: bool = False) -> Any:
        """
        Executes a query with exponential backoff to handle 'Database is locked' errors.
        """
        max_retries = 10
        base_delay = 0.1
        
        for attempt in range(max_retries):
            con = None
            try:
                con = duckdb.connect(DUCKDB_FILE)
                if fetch:
                    res = con.execute(query, params).fetchone()
                    # Get column names if result exists
                    cols = [desc[0] for desc in con.description] if con.description else []
                    return res, cols
                else:
                    con.execute(query, params)
                    return True
            except (duckdb.IOException, duckdb.InternalException) as e:
                if "lock" in str(e).lower() or "used by another process" in str(e).lower():
                    delay = base_delay * (2 ** attempt) + (random.random() * 0.1)
                    log.warning(f"⏳ DuckDB locked, retrying in {delay:.2f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                    continue
                raise e
            finally:
                if con:
                    con.close()
        
        raise RuntimeError(f"💥 Failed to acquire DuckDB lock after {max_retries} attempts.")

    @staticmethod
    def ensure_schema() -> None:
        """
        Ensure the file_ingestion_jobs table exists in DuckDB.
        """
        JobService._execute_with_retry("""
            CREATE TABLE IF NOT EXISTS file_ingestion_jobs (
                file_path VARCHAR PRIMARY KEY,
                job_id VARCHAR,
                status VARCHAR,
                error_message VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        log.info("✅ Ensured file_ingestion_jobs schema in DuckDB")

    @staticmethod
    def update_job(file_path: str, status: str, job_id: Optional[str] = None, error_message: Optional[str] = None) -> None:
        """
        Update or insert a job record. 
        """
        now = datetime.datetime.now()
        JobService._execute_with_retry("""
            INSERT OR REPLACE INTO file_ingestion_jobs (file_path, job_id, status, error_message, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (file_path, job_id, status, error_message, now))
        log.info(f"🔄 Updated job: {file_path} -> {status}")

    @staticmethod
    def get_job(file_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a complete job record by file_path.
        """
        try:
            res_tuple, cols = JobService._execute_with_retry(
                "SELECT * FROM file_ingestion_jobs WHERE file_path = ?", 
                (file_path,), 
                fetch=True
            )
            if res_tuple:
                return dict(zip(cols, res_tuple))
            return None
        except Exception as e:
            log.error(f"💥 Failed to get job {file_path}: {e}")
            return None

    @staticmethod
    def is_processed(file_path: str) -> bool:
        """
        Helper method used by the tree watcher to skip already handled files.
        """
        job = JobService.get_job(file_path)
        return job is not None and job["status"] == STATUS_COMPLETED


# Module-level convenience functions for cleaner imports
init_job_db = JobService.ensure_schema
update_job_status = JobService.update_job
get_job_status = JobService.get_job
is_file_processed = JobService.is_processed
