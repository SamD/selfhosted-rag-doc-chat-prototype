#!/usr/bin/env python3
"""
Performance metrics collection for the document ingestion pipeline.

Provides lightweight timing instrumentation with structured JSON logging.
Zero external dependencies - uses only stdlib (time, json, logging, datetime).
"""
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional


class Timer:
    """
    Context manager for timing operations with high-resolution timing.

    Usage:
        with Timer() as t:
            # do work
            pass
        elapsed_ms = t.elapsed_ms
    """

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            self.elapsed_ms = (self.end_time - self.start_time) * 1000.0
        return False  # Don't suppress exceptions


class FileMetrics:
    """
    Aggregates file-level metrics for document processing.

    Tracks timing for multiple operations within a file's processing lifecycle
    and emits a single JSON log event when complete.

    Usage:
        metrics = FileMetrics(worker="producer", file="docs/sample.pdf")

        with metrics.timer("total_processing"):
            with metrics.timer("text_extraction"):
                # extract text
                pass
            metrics.add_counter("chunks_produced", 47)

        metrics.emit(log)
    """

    def __init__(self, worker: str, file: str, **kwargs):
        """
        Initialize file-level metrics collector.

        Args:
            worker: Worker name (e.g., "producer", "consumer", "ocr")
            file: Relative file path being processed
            **kwargs: Additional fields to include in the metrics event (e.g., queue="chunk_ingest_queue:0")
        """
        self.worker = worker
        self.file = file
        self.extra_fields = kwargs
        self.timers: Dict[str, Timer] = {}
        self.metrics: Dict[str, Any] = {}
        self.active_timer: Optional[str] = None

    def timer(self, name: str) -> Timer:
        """
        Create a named timer context manager.

        Args:
            name: Timer name (will be used as metric key with _time_ms suffix)

        Returns:
            Timer context manager
        """
        if name not in self.timers:
            self.timers[name] = Timer()
        return _TimerAdapter(self, name, self.timers[name])

    def add_counter(self, name: str, value: int):
        """Add a counter metric (e.g., chunks_produced, pages_processed)."""
        self.metrics[name] = value

    def add_field(self, name: str, value: Any):
        """Add an arbitrary field to the metrics."""
        self.metrics[name] = value

    def add_ocr_operation(self, page: int, ocr_roundtrip_time_ms: float, engine: str, success: bool = True):
        """Track an OCR operation for aggregation."""
        if "ocr_operations" not in self.metrics:
            self.metrics["ocr_operations"] = []

        self.metrics["ocr_operations"].append({
            "page": page,
            "ocr_roundtrip_time_ms": ocr_roundtrip_time_ms,
            "engine": engine,
            "success": success
        })

    def _finalize_metrics(self):
        """Convert timer objects to millisecond values."""
        final_metrics = dict(self.metrics)

        for name, timer in self.timers.items():
            metric_key = f"{name}_time_ms"
            final_metrics[metric_key] = round(timer.elapsed_ms, 3)

        return final_metrics

    def emit(self, logger: logging.Logger):
        """
        Emit structured JSON log event.

        Args:
            logger: Logger instance to emit the event through
        """
        try:
            # Import settings here to avoid circular imports
            from config.settings import METRICS_ENABLED, METRICS_LOG_TO_STDOUT, METRICS_LOG_FILE

            if not METRICS_ENABLED:
                return

            # Map worker names to event names
            event_names = {
                "producer": "file_processing_complete",
                "consumer": "file_storage_complete"
            }
            event_name = event_names.get(self.worker, f"file_{self.worker}_complete")

            event = {
                "event": event_name,
                "timestamp": datetime.utcnow().isoformat(),
                "worker": self.worker,
                "file": self.file,
                "metrics": self._finalize_metrics()
            }

            # Add extra fields at top level
            event.update(self.extra_fields)

            # Emit to stdout via logger
            if METRICS_LOG_TO_STDOUT:
                logger.info(f"METRICS: {json.dumps(event)}")

            # Optionally write to metrics file
            if METRICS_LOG_FILE:
                try:
                    with open(METRICS_LOG_FILE, 'a') as f:
                        f.write(json.dumps(event) + '\n')
                except Exception as e:
                    logger.debug(f"Failed to write metrics to file: {e}")

        except Exception as e:
            # Never let metrics collection crash the worker
            logger.debug(f"Failed to emit metrics: {e}")


class _TimerAdapter:
    """
    Adapter that records timer results in parent FileMetrics when context exits.
    """

    def __init__(self, parent: FileMetrics, name: str, timer: Timer):
        self.parent = parent
        self.name = name
        self.timer = timer

    def __enter__(self):
        return self.timer.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        result = self.timer.__exit__(exc_type, exc_val, exc_tb)
        # Timer result is automatically recorded via parent's _finalize_metrics
        return result


class JobMetrics:
    """
    Tracks job-level metrics for OCR worker operations.

    Simpler than FileMetrics - designed for individual OCR job processing.

    Usage:
        metrics = JobMetrics(worker="ocr", job_id="abc123")

        with metrics.timer("total_processing"):
            with metrics.timer("image_decode"):
                # decode image
                pass
            metrics.add_field("engine", "tesseract")
            metrics.add_field("text_length", 1024)

        metrics.emit(log)
    """

    def __init__(self, worker: str, job_id: str, **kwargs):
        """
        Initialize job-level metrics collector.

        Args:
            worker: Worker name (typically "ocr")
            job_id: Unique job identifier
            **kwargs: Additional fields to include (e.g., file, page)
        """
        self.worker = worker
        self.job_id = job_id
        self.extra_fields = kwargs
        self.timers: Dict[str, Timer] = {}
        self.metrics: Dict[str, Any] = {}

    def timer(self, name: str) -> Timer:
        """Create a named timer context manager."""
        if name not in self.timers:
            self.timers[name] = Timer()
        return _JobTimerAdapter(self, name, self.timers[name])

    def add_field(self, name: str, value: Any):
        """Add a field to the metrics."""
        self.metrics[name] = value

    def _finalize_metrics(self):
        """Convert timer objects to millisecond values."""
        final_metrics = dict(self.metrics)

        for name, timer in self.timers.items():
            metric_key = f"{name}_time_ms"
            final_metrics[metric_key] = round(timer.elapsed_ms, 3)

        return final_metrics

    def emit(self, logger: logging.Logger):
        """Emit structured JSON log event."""
        try:
            from config.settings import METRICS_ENABLED, METRICS_LOG_TO_STDOUT, METRICS_LOG_FILE

            if not METRICS_ENABLED:
                return

            event = {
                "event": "ocr_job_complete",
                "timestamp": datetime.utcnow().isoformat(),
                "worker": self.worker,
                "job_id": self.job_id,
                "metrics": self._finalize_metrics()
            }

            # Add extra fields at top level
            event.update(self.extra_fields)

            if METRICS_LOG_TO_STDOUT:
                logger.info(f"METRICS: {json.dumps(event)}")

            if METRICS_LOG_FILE:
                try:
                    with open(METRICS_LOG_FILE, 'a') as f:
                        f.write(json.dumps(event) + '\n')
                except Exception as e:
                    logger.debug(f"Failed to write metrics to file: {e}")

        except Exception as e:
            logger.debug(f"Failed to emit metrics: {e}")


class _JobTimerAdapter:
    """Adapter that records timer results in parent JobMetrics."""

    def __init__(self, parent: JobMetrics, name: str, timer: Timer):
        self.parent = parent
        self.name = name
        self.timer = timer

    def __enter__(self):
        return self.timer.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.timer.__exit__(exc_type, exc_val, exc_tb)
