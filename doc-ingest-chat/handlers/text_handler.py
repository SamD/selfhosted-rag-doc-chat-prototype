import logging
from typing import Generator

from processors.document_processor import extract_text_from_html

from .base_handler import BaseContentTypeHandler

log = logging.getLogger("ingest.handlers.text")


class TextContentTypeHandler(BaseContentTypeHandler):
    """
    Handler for text-based files (txt, md, html).
    """

    def can_handle(self, file_path: str) -> bool:
        ext = file_path.lower()
        return ext.endswith((".txt", ".md", ".html", ".htm"))

    def stream_content(self, file_path: str) -> Generator[str, None, None]:
        """
        Extracts content from text or HTML files.
        For simplicity in this initial version, we yield the entire content as one 'page'.
        """
        log.info(f"📄 Extracting text content from {file_path}")
        try:
            if file_path.lower().endswith((".html", ".htm")):
                content = extract_text_from_html(file_path)
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

            if content:
                yield content

        except Exception as e:
            log.error(f"❌ Text extraction failed: {e}")
            raise
