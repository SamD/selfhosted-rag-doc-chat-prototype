#!/usr/bin/env python3
"""
Parquet service for data storage operations.
Optimized for incremental DuckDB writes and disk-based Parquet exportation.
"""

import logging
from typing import Any, Dict, List

import duckdb
import pandas as pd
from config.settings import DUCKDB_FILE, PARQUET_FILE
from services.job_service import JobService

log = logging.getLogger("ingest.parquet")


class ParquetService:
    """Parquet service handling DuckDB persistence and Parquet exports."""

    @staticmethod
    def ensure_schema() -> None:
        """Ensures the persistent table exists in DuckDB."""
        JobService._execute_with_retry("""
            CREATE TABLE IF NOT EXISTS parquet_chunks (
              id VARCHAR PRIMARY KEY,
              chunk TEXT,
              source_file VARCHAR,
              document_id VARCHAR,
              type VARCHAR,
              chunk_index INTEGER,
              engine VARCHAR,
              hash VARCHAR,
              page INTEGER
            )
            """)
        log.info("✅ Ensured parquet_chunks schema in DuckDB")

    @staticmethod
    def append_chunks(entries: List[Dict[str, Any]]) -> None:
        """
        Incrementally appends chunks to the DuckDB table.
        This allows us to clear them from Python memory immediately.
        """
        if not entries:
            return

        df = pd.DataFrame(entries)
        desired_cols = ["id", "chunk", "source_file", "document_id", "type", "chunk_index", "engine", "hash", "page"]
        for col in desired_cols:
            if col not in df.columns:
                df[col] = -1 if col == "page" else None
        df = df[desired_cols]

        max_retries = 10
        for attempt in range(max_retries):
            con = None
            try:
                con = duckdb.connect(DUCKDB_FILE)
                con.register("df_temp", df)
                con.execute("INSERT OR REPLACE INTO parquet_chunks SELECT * FROM df_temp")
                return
            except (duckdb.IOException, duckdb.InternalException) as e:
                if "lock" in str(e).lower():
                    import random
                    import time

                    delay = 0.1 * (2**attempt) + (random.random() * 0.1)
                    log.warning(f"⏳ DuckDB locked during append, retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
                raise e
            finally:
                if con:
                    con.close()

    @staticmethod
    def commit_to_parquet() -> None:
        """
        Exports the current state of the DuckDB table to a Parquet file.
        This operation happens on-disk and does not load chunks into Python RAM.
        """
        JobService._execute_with_retry(f"COPY parquet_chunks TO '{PARQUET_FILE}' (FORMAT PARQUET)")
        log.info(f"💾 Exported all chunks from DuckDB to {PARQUET_FILE}")


# Convenience aliases
init_schema = ParquetService.ensure_schema
append_chunks = ParquetService.append_chunks
commit_to_parquet = ParquetService.commit_to_parquet
