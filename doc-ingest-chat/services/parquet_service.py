#!/usr/bin/env python3
"""
Parquet service for data storage operations.
"""
from typing import Any, Dict, List

import duckdb
import pandas as pd
from config.settings import DUCKDB_FILE
from utils.logging_config import setup_logging

log = setup_logging("parquet_service.log")


class ParquetService:
    """Parquet service for data storage operations as static methods."""
    
    @staticmethod
    def write_to_parquet(entries: List[Dict[str, Any]], path: str, lock=None):
        """Write entries to parquet file."""
        if not entries:
            return

        if lock:
            with lock:
                ParquetService._do_write(entries, path)
        else:
            ParquetService._do_write(entries, path)

    @staticmethod
    def _do_write(entries: List[Dict[str, Any]], path: str):
        """Internal write function."""
        df = pd.DataFrame(entries)

        desired_cols = [
            "id",
            "chunk",
            "source_file",
            "type",
            "chunk_index",
            "engine",
            "hash",
            "page"
        ]
        for col in desired_cols:
            if col not in df.columns:
                df[col] = -1 if col == "page" else None
        df = df[desired_cols]
        source_file = df['source_file'][0]

        try:
            # Use a persistent DB instead of in-memory table
            con = duckdb.connect(DUCKDB_FILE)

            # Register the new data
            con.register("df", df)

            # Create the table if it doesn't exist
            con.execute("""
                CREATE TABLE IF NOT EXISTS parquet_chunks AS SELECT * FROM df LIMIT 0
            """)

            # Append new data
            con.execute("INSERT INTO parquet_chunks SELECT * FROM df")

            # Write the ENTIRE DB to Parquet (overwrite only once)
            con.execute(f"""
                COPY parquet_chunks TO '{path}' (FORMAT PARQUET)
            """)
            con.close()

            log.info(f"✅ Appended {len(df)} entries to {path} for source_file {source_file}")
        except Exception as e:
            log.error(f"💥 Failed to write Parquet file {path}: {e}", exc_info=True) 

# Expose static methods as module-level functions after class definition
write_to_parquet = ParquetService.write_to_parquet 