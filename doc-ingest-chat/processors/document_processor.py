#!/usr/bin/env python3
"""
Document processing functionality.
"""

import re
from typing import List, Optional

from bs4 import BeautifulSoup
from charset_normalizer import from_path
from config.settings import SUPPORTED_MEDIA_EXT
from utils.logging_config import setup_logging

log = setup_logging("document_processor.log")


class DocumentProcessor:
    """Document processing functionality as static methods."""

    @staticmethod
    def extract_text_from_html(full_path: str) -> Optional[str]:
        """Extract text from HTML file."""
        try:
            match = from_path(full_path).best()
            if not match:
                raise ValueError(f"[ERROR] Could not detect encoding for: {full_path}")

            html = str(match)  # Decoded text (charset-normalizer >= 3.x)
            soup = BeautifulSoup(html, "html5lib")  # Most forgiving parser

            text = soup.get_text(separator="\n", strip=True)
            text = re.sub(r"\n\s*\n+", "\n\n", text)  # Collapse extra blank lines
            return text

        except Exception as e:
            log.error(f"[ERROR] extract_text_from_html failed for {full_path}: {e}", exc_info=True)
            return None

    @staticmethod
    def extract_text_from_media(filepath: str) -> Optional[List]:
        """Extract text from media files by delegating to WhisperX worker."""
        if not filepath.lower().endswith(SUPPORTED_MEDIA_EXT):
            raise ValueError(f"Unsupported file type: {filepath}")

        log.info(f" 🎥 Delegating media transcription for {filepath}")

        try:
            from utils.whisper_utils import send_media_to_whisperx
            return list(send_media_to_whisperx(filepath))
        except Exception as e:
            log.error(f"Transcription delegation failed for {filepath}: {e}", exc_info=True)
            return None


# Expose static methods as module-level functions after class definition
extract_text_from_html = DocumentProcessor.extract_text_from_html
