#!/usr/bin/env python3
"""
Logging configuration utilities.
"""

import logging
import sys
from typing import Optional


class FlushFileHandler(logging.FileHandler):
    """File handler that flushes after each log record."""

    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO, include_default_filters: bool = False) -> logging.Logger:
    """
    Set up logging configuration. Should be called ONCE in the entry point (main).
    """
    root_logger = logging.getLogger()
    
    # If handlers are already set, don't re-configure (prevents 0-byte orphaned files)
    if root_logger.hasHandlers():
        return logging.getLogger("ingest")

    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(FlushFileHandler(log_file))

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=handlers, force=True)

    if include_default_filters:
        setup_pdf_logging()

    return logging.getLogger("ingest")


class SuppressCropBoxWarnings(logging.Filter):
    """Filter to suppress CropBox missing warnings from pdfminer."""

    def filter(self, record):
        return "CropBox missing from" not in record.getMessage()


def setup_pdf_logging():
    """Set up PDF-specific logging filters."""
    logging.getLogger("pdfminer.pdfpage").addFilter(SuppressCropBoxWarnings())
