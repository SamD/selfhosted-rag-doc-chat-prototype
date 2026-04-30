#!/usr/bin/env python3
"""
Job service for tracking file ingestion states in DuckDB.
Implements a database-driven state machine for atomic lifecycle management.
"""

import datetime
import logging
import os
import uuid
from typing import Any, Dict, Optional

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
    def _execute_with_retry(sql: str, params: tuple = (), fetch: bool = False, fetch_all: bool = False) -> Any:
        """
        Delegates to the centralized DatabaseService with robust multi-process safety.
        """
        from services.database import execute, query

        if fetch or fetch_all:
            return query(sql, params, fetch_all=fetch_all)
        return execute(sql, params)

    @staticmethod
    def ensure_schema(quiet: bool = False, db_path: str = None) -> None:
        """
        Delegates database initialization to the centralized DatabaseService.
        """
        from services.database import DatabaseService

        DatabaseService.init_db(db_path=db_path)

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
        from utils.trace_utils import generate_trace_id
        trace_id = generate_trace_id()
        now = datetime.datetime.now()

        JobService._execute_with_retry(
            """
            INSERT INTO ingestion_lifecycle (id, status, original_filename, pdf_path, new_at, trace_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_id, STATUS_NEW, filename, original_pdf_path, now, trace_id),
        )
        log.info(f"🆕 [{trace_id}] Created job {job_id} for {filename}")
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
