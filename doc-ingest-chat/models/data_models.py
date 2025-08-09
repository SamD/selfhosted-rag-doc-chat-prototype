#!/usr/bin/env python3
"""
Data models for the document ingestion system.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ChunkEntry:
    """Represents a text chunk with metadata."""
    chunk: str
    id: str
    source_file: str
    type: str
    hash: str
    engine: str
    page: int
    chunk_index: int


@dataclass
class OCRJob:
    """Represents an OCR job."""
    job_id: str
    rel_path: str
    page_num: int
    image_shape: List[int]
    image_dtype: str
    image_base64: str
    reply_key: str


@dataclass
class OCRResponse:
    """Represents an OCR response."""
    text: Optional[str]
    rel_path: str
    page_num: int
    engine: str
    job_id: str


@dataclass
class FileEndMessage:
    """Represents a file end message."""
    type: str = "file_end"
    source_file: str = ""
    expected_chunks: int = 0


@dataclass
class ProcessingResult:
    """Represents a processing result."""
    success: bool
    message: str
    chunks_processed: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = [] 