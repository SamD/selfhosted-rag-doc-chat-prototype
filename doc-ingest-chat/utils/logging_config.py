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
    Forced configuration ensures that our handlers are always applied even if
    other libraries have already called basicConfig.
    """
    root_logger = logging.getLogger()

    # Clear existing handlers to ensure our configuration is applied
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(FlushFileHandler(log_file))

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)

    # Use basicConfig with force=True for extra safety
    logging.basicConfig(level=level, handlers=handlers, force=True)

    if include_default_filters:
        setup_pdf_logging()

    # Ensure the root logger itself is set to the correct level
    root_logger.setLevel(level)

    return root_logger


class SuppressCropBoxWarnings(logging.Filter):
    """Filter to suppress CropBox missing warnings from pdfminer."""

    def filter(self, record):
        return "CropBox missing from" not in record.getMessage()


def setup_pdf_logging():
    """Set up PDF-specific logging filters."""
    logging.getLogger("pdfminer.pdfpage").addFilter(SuppressCropBoxWarnings())
