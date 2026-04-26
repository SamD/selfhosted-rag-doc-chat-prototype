#!/usr/bin/env python3
"""
Job service for tracking file ingestion states in DuckDB.
Implements a database-driven state machine for atomic lifecycle management.
"""

import datetime
import logging
import os
import random
import time
import uuid
from typing import Any, Dict, Optional

import duckdb
from config.settings import DUCKDB_FILE

log = logging.getLogger("ingest.job_service")

# --- NEW LIFECYCLE STATUSES ---
STATUS_NEW = "NEW"  # Discovered in staging
STATUS_PREPROCESSING = "PREPROCESSING"  # Gatekeeper is normalizing
STATUS_PREPROCESSING_COMPLETE = "PREPROCESSING_COMPLETE"  # Markdown finished, waiting for Producer
STATUS_INGESTING = "INGESTING"  # Producer is chunking/enqueuing
STATUS_CONSUMING = "CONSUMING"  # Sentinels sent, Consumer is persisting to Vector DB
STATUS_INGEST_SUCCESS = "INGEST_SUCCESS"  # Fully stored and archived
STATUS_INGEST_FAILED = "INGEST_FAILED"  # Permanent failure

# --- LEGACY STATUSES (for compatibility with tests) ---
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_CHUNKING = "chunking"
STATUS_ENQUEUING = "enqueuing"
STATUS_ENQUEUED = "enqueued"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class JobService:
    """
    Job service for tracking ingestion states as static methods.
    Encapsulates all DuckDB interactions for file-level status tracking.
    """

    @staticmethod
    def _execute_with_retry(query: str, params: tuple = (), fetch: bool = False, fetch_all: bool = False) -> Any:
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
                    cols = [desc[0] for desc in con.description] if con.description else []
                    return res, cols
                elif fetch_all:
                    res = con.execute(query, params).fetchall()
                    cols = [desc[0] for desc in con.description] if con.description else []
                    return res, cols
                else:
                    con.execute(query, params)
                    return True
            except (duckdb.IOException, duckdb.InternalException) as e:
                if "lock" in str(e).lower() or "used by another process" in str(e).lower():
                    delay = base_delay * (2**attempt) + (random.random() * 0.1)
                    # Use INFO for first 3 attempts to reduce log noise, only WARNING if it persists
                    log_func = log.info if attempt < 3 else log.warning
                    log_func(f"⏳ DuckDB locked, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                raise e
            finally:
                if con:
                    con.close()

        raise RuntimeError(f"💥 Failed to acquire DuckDB lock after {max_retries} attempts.")

    @staticmethod
    def ensure_schema(quiet: bool = False) -> None:
        """
        Ensure the ingestion_lifecycle and legacy file_ingestion_jobs tables exist in DuckDB.
        """
        JobService._execute_with_retry("""
            CREATE TABLE IF NOT EXISTS ingestion_lifecycle (
                id VARCHAR PRIMARY KEY,
                status VARCHAR,
                original_filename VARCHAR,
                pdf_path VARCHAR,
                md_path VARCHAR,
                worker_id VARCHAR,
                error_log TEXT,
                new_at TIMESTAMP,
                preprocessing_at TIMESTAMP,
                preprocessing_complete_at TIMESTAMP,
                ingesting_at TIMESTAMP,
                consuming_at TIMESTAMP,
                finalized_at TIMESTAMP
            )
        """)
        # Keep legacy table for compatibility
        JobService._execute_with_retry("""
            CREATE TABLE IF NOT EXISTS file_ingestion_jobs (
                file_path VARCHAR PRIMARY KEY,
                job_id VARCHAR,
                document_id VARCHAR,
                status VARCHAR,
                error_message VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        if not quiet:
            log.info("✅ Ensured database schemas in DuckDB")

    @staticmethod
    def create_job(original_pdf_path: str) -> Optional[str]:
        """
        [STAGE 1] Gatekeeper creates a NEW record for a discovered PDF.
        """
        filename = os.path.basename(original_pdf_path)

        # Duplicate Check
        existing, _ = JobService._execute_with_retry("SELECT id FROM ingestion_lifecycle WHERE original_filename = ? AND status != ?", (filename, STATUS_INGEST_FAILED), fetch=True)
        if existing:
            return None

        job_id = str(uuid.uuid4())
        now = datetime.datetime.now()

        JobService._execute_with_retry(
            """
            INSERT INTO ingestion_lifecycle (id, status, original_filename, pdf_path, new_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, STATUS_NEW, filename, original_pdf_path, now),
        )
        log.info(f"🆕 Created job {job_id} for {filename}")
        return job_id

    @staticmethod
    def claim_job(current_status: str, next_status: str) -> Optional[Dict[str, Any]]:
        """
        ATOMIC CLAIM: Updates one record from current to next status and returns it.
        """
        timestamp_col = f"{next_status.lower()}_at"
        worker_id = str(os.getpid())

        try:
            query = f"""
                UPDATE ingestion_lifecycle 
                SET status = ?, worker_id = ?, {timestamp_col} = CURRENT_TIMESTAMP
                WHERE id = (
                    SELECT id FROM ingestion_lifecycle 
                    WHERE status = ? 
                    ORDER BY new_at ASC
                    LIMIT 1
                )
                RETURNING *
            """
            res, cols = JobService._execute_with_retry(query, (next_status, worker_id, current_status), fetch=True)
            if res:
                return dict(zip(cols, res))
            return None
        except Exception:
            return None

    @staticmethod
    def transition_job(job_id: str, next_status: str, new_pdf_path: str = None, new_md_path: str = None, error: str = None) -> bool:
        """
        [ATOMIC HANDOFF] Updates DB state.
        """
        timestamp_col = f"{next_status.lower()}_at"
        if next_status in (STATUS_INGEST_SUCCESS, STATUS_INGEST_FAILED):
            timestamp_col = "finalized_at"

        updates = ["status = ?", f"{timestamp_col} = CURRENT_TIMESTAMP"]
        params = [next_status]

        if new_pdf_path:
            updates.append("pdf_path = ?")
            params.append(new_pdf_path)
        if new_md_path:
            updates.append("md_path = ?")
            params.append(new_md_path)
        if error:
            updates.append("error_log = ?")
            params.append(error)

        params.append(job_id)

        query = f"UPDATE ingestion_lifecycle SET {', '.join(updates)} WHERE id = ?"
        return JobService._execute_with_retry(query, tuple(params))

    # --- LEGACY METHODS (for compatibility) ---
    @staticmethod
    def update_job(file_path: str, status: str, job_id: Optional[str] = None, error_message: Optional[str] = None, document_id: Optional[str] = None) -> None:
        now = datetime.datetime.now()
        JobService._execute_with_retry(
            "INSERT OR REPLACE INTO file_ingestion_jobs (file_path, job_id, document_id, status, error_message, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (file_path, job_id, document_id, status, error_message, now),
        )

    @staticmethod
    def get_job(file_path: str) -> Optional[Dict[str, Any]]:
        res_tuple, cols = JobService._execute_with_retry("SELECT * FROM file_ingestion_jobs WHERE file_path = ?", (file_path,), fetch=True)
        return dict(zip(cols, res_tuple)) if res_tuple else None

    @staticmethod
    def is_processed(file_path: str) -> bool:
        job = JobService.get_job(file_path)
        return job["status"] in (STATUS_COMPLETED, STATUS_ENQUEUED, STATUS_PROCESSING) if job else False


# Module-level convenience functions
init_job_db = JobService.ensure_schema
update_job_status = JobService.update_job
get_job_status = JobService.get_job
is_file_processed = JobService.is_processed
