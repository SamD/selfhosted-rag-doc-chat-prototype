"""Shared utilities for the monorepo."""

import os


def parse_endpoints(env_value: str) -> list[str]:
    """Parse a comma-separated list of endpoint URLs from an env var.

    Strips whitespace, removes empty entries, and normalizes trailing slashes.
    Returns a list of clean URLs, or an empty list if none found.
    """
    if not env_value:
        return []
    return [e.strip().rstrip("/") for e in env_value.split(",") if e.strip()]


def resolve_supervisor_endpoint() -> str:
    """Resolve the supervisor LLM endpoint URL.

    Priority:
    1. SUPERVISOR_LLM_ENDPOINTS (single entry only — multi-entry needs HAProxy)
    2. Local model path (non-URL)
    """
    endpoints = parse_endpoints(os.getenv("SUPERVISOR_LLM_ENDPOINTS", ""))
    if len(endpoints) == 1:
        url = endpoints[0]
        if url.startswith(("http://", "https://")) and not url.endswith("/v1"):
            url = url.rstrip("/") + "/v1"
        return url

    path = os.getenv("SUPERVISOR_LLM_ENDPOINTS", "")
    if path.startswith(("http://", "https://")):
        if not path.endswith("/v1"):
            path = path.rstrip("/") + "/v1"
        return path
    return path


def resolve_embedding_endpoint() -> str | None:
    """Resolve the embedding endpoint URL.

    Priority:
    1. EMBEDDING_ENDPOINTS (single entry only — multi-entry needs HAProxy)
    2. None (local model path)
    """
    endpoints = parse_endpoints(os.getenv("EMBEDDING_ENDPOINTS", ""))
    if len(endpoints) == 1:
        if endpoints[0].startswith(("http://", "https://")):
            return endpoints[0]

    path = os.getenv("EMBEDDING_ENDPOINTS", "")
    if path.startswith(("http://", "https://")):
        return path
    return None
