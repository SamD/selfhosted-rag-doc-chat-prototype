#!/usr/bin/env python3
"""
Traceability utilities for the ingestion pipeline.
Provides context-aware logging with a global trace_id.
"""

import contextvars
import logging
import uuid
from functools import wraps
from typing import Optional

# Context variable to store the current trace_id
_TRACE_ID_VAR = contextvars.ContextVar("trace_id", default=None)

def get_trace_id() -> Optional[str]:
    """Retrieve the current trace_id from the context."""
    return _TRACE_ID_VAR.get()

def set_trace_id(trace_id: str) -> contextvars.Token:
    """Set the trace_id for the current context."""
    return _TRACE_ID_VAR.set(trace_id)

def reset_trace_id(token: contextvars.Token):
    """Reset the trace_id to its previous value."""
    _TRACE_ID_VAR.reset(token)

class TraceLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically injects the current trace_id into logs.
    """
    def process(self, msg, kwargs):
        trace_id = get_trace_id()
        if trace_id:
            return f"[{trace_id}] {msg}", kwargs
        return msg, kwargs

def get_logger(name: str) -> logging.Logger:
    """Returns a trace-aware logger."""
    logger = logging.getLogger(name)
    return TraceLoggerAdapter(logger, {})

def with_trace(trace_id_arg_name="trace_id"):
    """
    Decorator to automatically set the trace_id context for a function.
    Assumes the trace_id is passed as a keyword argument or positional argument.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract trace_id from arguments
            trace_id = kwargs.get(trace_id_arg_name)
            if not trace_id and args:
                # This is a bit brittle, but handles common cases
                # Better to pass trace_id explicitly as kwarg
                pass
            
            if trace_id:
                token = set_trace_id(trace_id)
                try:
                    return func(*args, **kwargs)
                finally:
                    reset_trace_id(token)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def generate_trace_id() -> str:
    """Generates a new short trace ID (8 chars)."""
    return str(uuid.uuid4())[:8]
