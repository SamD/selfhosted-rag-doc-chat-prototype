#!/usr/bin/env python3
"""
Logging configuration utilities.
"""
import logging
import sys


class FlushFileHandler(logging.FileHandler):
    """File handler that flushes after each log record."""
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(log_file: str, level: int = logging.INFO, include_default_filters: bool = False) -> logging.Logger:
    """
    Set up logging configuration with both file and console handlers.
    
    Args:
        log_file: Path to the log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create handlers
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = FlushFileHandler(log_file)

    # Set consistent formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Apply to root logger
    logging.basicConfig(level=level, handlers=[stream_handler, file_handler])

    if include_default_filters:
        setup_pdf_logging()
    
    return logging.getLogger(__name__)


class SuppressCropBoxWarnings(logging.Filter):
    """Filter to suppress CropBox missing warnings from pdfminer."""
    def filter(self, record):
        return "CropBox missing from" not in record.getMessage()


def setup_pdf_logging():
    """Set up PDF-specific logging filters."""
    logging.getLogger("pdfminer.pdfpage").addFilter(SuppressCropBoxWarnings()) 