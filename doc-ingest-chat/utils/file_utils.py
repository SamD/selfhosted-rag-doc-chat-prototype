#!/usr/bin/env python3
"""
File utility functions.
"""
import fcntl
import os
import hashlib
import json
from pathlib import Path
from typing import Set

from config.settings import FAILED_FILES, INGESTED_FILE


class FileUtils:
    """File utility functions as static methods."""
    
    @staticmethod
    def normalize_rel_path(p: str) -> str:
        """Normalize a relative path."""
        return os.path.normpath(p)

    @staticmethod
    def update_ingested_files(file: str, ingested_file: str = INGESTED_FILE):
        """Update the ingested files tracking file."""
        file = FileUtils.normalize_rel_path(file)
        with open(ingested_file, "a", encoding="utf-8") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(file + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    @staticmethod
    def update_failed_files(file: str, failed_files: str = FAILED_FILES):
        """Update the failed files tracking file."""
        file = FileUtils.normalize_rel_path(file)
        with open(failed_files, "a", encoding="utf-8") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(file + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    @staticmethod
    def load_tracked(filepath: str) -> Set[str]:
        """Load tracked files from a file."""
        if not Path(filepath).exists():
            return set()
        return set(FileUtils.normalize_rel_path(line.strip()) for line in Path(filepath).read_text().splitlines() if line.strip())

    @staticmethod
    def mark_tracked(filepath: str, rel_path: str):
        """Mark a file as tracked."""
        rel_path = FileUtils.normalize_rel_path(rel_path)
        with open(filepath, "a") as f:
            f.write(rel_path + "\n")

    @staticmethod
    def md5(text: str) -> str:
        """Generate MD5 hash of text."""
        return hashlib.md5(text.encode()).hexdigest()

    @staticmethod
    def md5_from_int_list(int_list) -> str:
        """
        Generates an MD5 hash from a list of integers.

        Args:
          int_list: A list of integers.

        Returns:
          A string representing the MD5 hash in hexadecimal format.
        """
        # Convert the list of integers to a JSON string and encode it to bytes
        json_string = json.dumps(int_list, sort_keys=True).encode('utf-8')

        # Create an MD5 hash object
        md5_hash = hashlib.md5()

        # Update the hash object with the encoded JSON string
        md5_hash.update(json_string)

        # Return the hexadecimal representation of the hash
        return md5_hash.hexdigest() 

normalize_rel_path = FileUtils.normalize_rel_path
update_ingested_files = FileUtils.update_ingested_files
update_failed_files = FileUtils.update_failed_files 
load_tracked = FileUtils.load_tracked 