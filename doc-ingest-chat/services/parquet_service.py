#!/usr/bin/env python3
"""
Parquet service for data storage operations.
Optimized for incremental DuckDB writes and disk-based Parquet exportation.
"""

import json
import logging
import random
import time
from typing import Any, Dict, List

import duckdb
import pandas as pd
from config.settings import DUCKDB_FILE, PARQUET_FILE

log = logging.getLogger("ingest.parquet")


class ParquetService:
    """Parquet service handling DuckDB persistence and Parquet exports."""

    @staticmethod
    def ensure_schema() -> None:
        """Ensures the persistent table exists in DuckDB."""
        # 1. Main persistent archival table
        ParquetService._execute_protected_query("""
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
        # 2. Transient staging table for zero-memory consumer
        ParquetService._execute_protected_query("""
            CREATE TABLE IF NOT EXISTS staged_chunks (
                id VARCHAR PRIMARY KEY,
                source_file VARCHAR,
                document_id VARCHAR,
                chunk TEXT,
                metadata JSON,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        log.info("✅ Ensured parquet_chunks and staged_chunks schemas in DuckDB")

    @staticmethod
    def _execute_protected_query(query: str, params: tuple = (), fetch_all: bool = False) -> Any:
        """Internal helper with aggressive locking retries for persistence nodes."""
        max_retries = 20  # Increased ceiling for heavy parallel load
        base_delay = 0.2

        for attempt in range(max_retries):
            con = None
            try:
                con = duckdb.connect(DUCKDB_FILE)
                if fetch_all:
                    res = con.execute(query, params).fetchall()
                    return res
                else:
                    con.execute(query, params)
                    return True
            except (duckdb.IOException, duckdb.InternalException) as e:
                if "lock" in str(e).lower() or "used by another process" in str(e).lower():
                    delay = base_delay * (2**attempt) + (random.random() * 0.2)
                    log.info(f"⏳ Persistence layer locked, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                raise e
            finally:
                if con:
                    con.close()
        raise RuntimeError(f"💥 Persistence layer failed to acquire lock after {max_retries} attempts.")

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
        for attempt in range(max_retries):
            con = None
            try:
                con = duckdb.connect(DUCKDB_FILE)
                con.register("df_staged_temp", df_staged)
                con.execute("INSERT OR REPLACE INTO staged_chunks (id, source_file, document_id, chunk, metadata) SELECT * FROM df_staged_temp")
                return
            except (duckdb.IOException, duckdb.InternalException) as e:
                if "lock" in str(e).lower():
                    import random
                    import time

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
        """Persists a single chunk to the staging table using lock retries and idempotent replacement."""
        # ENSURE TYPE and ENGINE exist to prevent 'Storage error' in consumer
        entry["type"] = entry.get("type", "unknown")
        entry["engine"] = entry.get("engine", "llamacpp")
        entry["page"] = entry.get("page", 1)

        query = "INSERT OR REPLACE INTO staged_chunks (id, source_file, document_id, chunk, metadata) VALUES (?, ?, ?, ?, ?)"
        params = [entry.get("id"), entry.get("source_file"), entry.get("document_id"), entry.get("chunk"), json.dumps(entry)]
        try:
            ParquetService._execute_protected_query(query, tuple(params))
        except Exception as e:
            log.error(f"💥 Critical failure staging chunk {entry.get('id')}: {e}")

    @staticmethod
    def get_staged_chunks(source_file: str, purge: bool = True) -> List[Dict[str, Any]]:
        """Retrieves and optionally deletes all staged chunks for a file using lock retries."""
        chunks = []
        try:
            # 1. Fetch
            query_select = "SELECT metadata FROM staged_chunks WHERE source_file = ? ORDER BY timestamp ASC"
            res = ParquetService._execute_protected_query(query_select, (source_file,), fetch_all=True)

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
                # Default values to prevent schema mismatch
                if col == "page" or col == "chunk_index":
                    df[col] = 1
                elif col == "type":
                    df[col] = "unknown"
                elif col == "engine":
                    df[col] = "llamacpp"
                else:
                    df[col] = None

        df = df[desired_cols]

        # Use protected logic for the registration/insertion
        max_retries = 20
        for attempt in range(max_retries):
            con = None
            try:
                con = duckdb.connect(DUCKDB_FILE)
                con.register("df_temp", df)
                con.execute("INSERT OR REPLACE INTO parquet_chunks SELECT * FROM df_temp")
                return
            except (duckdb.IOException, duckdb.InternalException) as e:
                if "lock" in str(e).lower():
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
