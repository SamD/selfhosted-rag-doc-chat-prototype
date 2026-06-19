from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptionInput:
    file_path: str
    language: str = "en"
    mime_type: Optional[str] = None

@dataclass  
class TranscriptionResult:
    segments: list[dict]
    source_file: str
    job_id: str