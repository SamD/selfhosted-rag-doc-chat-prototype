import logging
import sys
from typing import Optional


class FlushFileHandler(logging.FileHandler):
    """File handler that flushes after each log record."""

    def emit(self, record):
        super().emit(record)
        self.flush()


class FlushStreamHandler(logging.StreamHandler):
    """Stream handler (console) that flushes after each log record."""

    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO, include_default_filters: bool = False) -> logging.Logger:
    """
    Set up logging configuration to write to both console and file.
    Uses flushing handlers to ensure immediate visibility in Docker/CLI.
    """
    root_logger = logging.getLogger()

    # Clear existing handlers to ensure our configuration is applied
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console Handler (forced flush)
    handlers = [FlushStreamHandler(sys.stdout)]

    # Optional File Handler (forced flush)
    if log_file:
        handlers.append(FlushFileHandler(log_file))

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)

    # Apply configuration
    logging.basicConfig(level=level, handlers=handlers, force=True)
    root_logger.setLevel(level)

    if include_default_filters:
        setup_pdf_logging()

    return root_logger


class SuppressCropBoxWarnings(logging.Filter):
    """Filter to suppress CropBox missing warnings from pdfminer."""

    def filter(self, record):
        return "CropBox missing from" not in record.getMessage()


def setup_pdf_logging():
    """Set up PDF-specific logging filters."""
    logging.getLogger("pdfminer.pdfpage").addFilter(SuppressCropBoxWarnings())
