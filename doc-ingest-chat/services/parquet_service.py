#!/usr/bin/env python3
"""
Parquet service for data storage operations.
Optimized for incremental DuckDB writes and disk-based Parquet exportation.
"""

import json
import logging
from typing import Any, Dict, List

import pandas as pd
from config.settings import PARQUET_FILE
from services.database import DatabaseService

log = logging.getLogger("ingest.parquet")


class ParquetService:
    """Parquet service handling DuckDB persistence and Parquet exports."""

    @staticmethod
    def ensure_schema() -> None:
        """Delegates database initialization to the centralized DatabaseService."""
        DatabaseService.init_db()

    @staticmethod
    def _execute_protected_query(sql: str, params: tuple = (), fetch_all: bool = False) -> Any:
        """
        Delegates to the centralized DatabaseService with robust multi-process safety.
        """
        from services.database import execute, query

        if fetch_all:
            return query(sql, params, fetch_all=True)
        return execute(sql, params)

    @staticmethod
    def stage_chunks(entries: List[Dict[str, Any]]) -> None:
        """Batch persists multiple chunks to the staging table using lock retries."""
        if not entries:
            return

        # Ensure required fields exist in every entry
        for entry in entries:
            entry["type"] = entry.get("type", "unknown")
            entry["engine"] = entry.get("engine", "llamacpp")
            entry["page"] = entry.get("page", 1)
            # Store full metadata as JSON string for the DB
            entry["metadata_json"] = json.dumps(entry)

        df = pd.DataFrame(entries)
        # Map to DuckDB schema
        df_staged = pd.DataFrame({"id": df["id"], "source_file": df["source_file"], "document_id": df["document_id"], "chunk": df["chunk"], "metadata": df["metadata_json"]})

        # Use protected logic for the registration/insertion
        max_retries = 20
        import random
        import time

        for attempt in range(max_retries):
            con = None
            try:
                con = DatabaseService.get_duckdb()
                con.register("df_staged_temp", df_staged)
                # Explicit column names ensure we don't hit column-count mismatches with timestamps
                con.execute("INSERT OR REPLACE INTO staged_chunks (id, source_file, document_id, chunk, metadata) SELECT * FROM df_staged_temp")
                return
            except Exception as e:
                err_msg = str(e).lower()
                if "lock" in err_msg or "used by another process" in err_msg:
                    delay = 0.2 * (2**attempt) + (random.random() * 0.2)
                    log.warning(f"⏳ DuckDB locked during batch stage, retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
                raise e
            finally:
                if con:
                    con.close()

    @staticmethod
    def stage_chunk(entry: Dict[str, Any]) -> None:
        """Persists a single chunk to the staging table using central execution helper."""
        entry["type"] = entry.get("type", "unknown")
        entry["engine"] = entry.get("engine", "llamacpp")
        entry["page"] = entry.get("page", 1)

        query = "INSERT OR REPLACE INTO staged_chunks (id, source_file, document_id, chunk, metadata) VALUES (?, ?, ?, ?, ?)"
        params = (entry.get("id"), entry.get("source_file"), entry.get("document_id"), entry.get("chunk"), json.dumps(entry))
        ParquetService._execute_protected_query(query, params)

    @staticmethod
    def get_staged_chunks(source_file: str, purge: bool = True) -> List[Dict[str, Any]]:
        """Retrieves and optionally deletes all staged chunks for a file."""
        chunks = []
        try:
            # 1. Fetch
            query_select = "SELECT metadata FROM staged_chunks WHERE source_file = ? ORDER BY timestamp ASC"
            # execute_query returns (results, columns)
            res, _ = ParquetService._execute_protected_query(query_select, (source_file,), fetch_all=True)

            if res:
                for row in res:
                    chunks.append(json.loads(row[0]))

            # 2. Optional Purge
            if purge and chunks:
                query_delete = "DELETE FROM staged_chunks WHERE source_file = ?"
                ParquetService._execute_protected_query(query_delete, (source_file,))

        except Exception as e:
            log.error(f"💥 Failed to retrieve staged chunks for {source_file}: {e}")

        return chunks

    @staticmethod
    def append_chunks(entries: List[Dict[str, Any]]) -> None:
        """Incrementally appends chunks to the DuckDB table."""
        if not entries:
            return

        df = pd.DataFrame(entries)
        desired_cols = ["id", "chunk", "source_file", "document_id", "type", "chunk_index", "engine", "hash", "page"]
        for col in desired_cols:
            if col not in df.columns:
                if col == "page" or col == "chunk_index":
                    df[col] = 1
                elif col == "type":
                    df[col] = "unknown"
                elif col == "engine":
                    df[col] = "llamacpp"
                else:
                    df[col] = None

        df = df[desired_cols]

        max_retries = 20
        import random
        import time

        for attempt in range(max_retries):
            con = None
            try:
                con = DatabaseService.get_duckdb()
                con.register("df_temp", df)
                # EXPLICIT COLUMN LIST: Prevents 'excluded' count mismatch with auto-timestamp columns
                cols_str = ", ".join(desired_cols)
                con.execute(f"INSERT OR REPLACE INTO parquet_chunks ({cols_str}) SELECT * FROM df_temp")
                return
            except Exception as e:
                err_msg = str(e).lower()
                if "lock" in err_msg or "used by another process" in err_msg:
                    delay = 0.2 * (2**attempt) + (random.random() * 0.2)
                    time.sleep(delay)
                    continue
                raise e
            finally:
                if con:
                    con.close()

    @staticmethod
    def commit_to_parquet() -> None:
        """Exports the current state of the DuckDB table to a Parquet file."""
        ParquetService._execute_protected_query(f"COPY parquet_chunks TO '{PARQUET_FILE}' (FORMAT PARQUET)")
        log.info(f"💾 Exported all chunks from DuckDB to {PARQUET_FILE}")


# Convenience aliases
init_schema = ParquetService.ensure_schema
append_chunks = ParquetService.append_chunks
commit_to_parquet = ParquetService.commit_to_parquet
stage_chunk = ParquetService.stage_chunk
stage_chunks = ParquetService.stage_chunks
get_staged_chunks = ParquetService.get_staged_chunks
