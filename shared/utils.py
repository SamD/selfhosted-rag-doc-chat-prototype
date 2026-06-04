"""Shared utilities for the monorepo."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable


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


class EndpointDispatcher:
    """Dispatches batched calls across multiple HA backends with configurable interleaving.

    When interleaving is enabled, batches are dispatched concurrently via a
    ThreadPoolExecutor, round-robining across backends per call. This spreads
    load evenly but each backend processes one request at a time.

    When interleaving is disabled, all batches within a job are sent sequentially
    to the same backend, one batch at a time. After the job completes, the next
    job pins to the next backend. This keeps each backend's prompt cache warm
    for consecutive batches but leaves other backends idle during a job.

    Designed for benchmarking TPS with both strategies.
    """

    def __init__(self, endpoints: list[str], interleave: bool = True, max_workers: int = 4):
        self.endpoints = endpoints
        self.interleave = interleave
        self.max_workers = max_workers
        self._counter = 0
        self._lock = threading.Lock()
        if not endpoints:
            raise ValueError("EndpointDispatcher requires at least one endpoint")

    def _next_endpoint(self) -> str:
        """Return the next endpoint in round-robin order."""
        with self._lock:
            idx = self._counter % len(self.endpoints)
            self._counter += 1
            return self.endpoints[idx]

    def dispatch(
        self,
        fn: Callable,
        args_list: list[tuple],
        job_label: str = "",
    ) -> list:
        """Dispatch a list of (args) across endpoints.

        When interleave=True: each call goes to the next endpoint via ThreadPool.
        When interleave=False: all calls in this dispatch go to the same endpoint,
        executed sequentially.

        Args:
            fn: Callable whose first positional argument is the endpoint URL.
            args_list: List of extra positional arg tuples to pass after endpoint.
            job_label: Optional label for logging.

        Returns:
            List of results in the same order as args_list.
        """
        if not args_list:
            return []

        if self.interleave:
            return self._dispatch_interleaved(fn, args_list, job_label)
        else:
            return self._dispatch_pinned(fn, args_list, job_label)

    def _dispatch_interleaved(self, fn: Callable, args_list: list[tuple], job_label: str = "") -> list:
        """Concurrent dispatch: each call round-robins to the next backend."""
        n = len(args_list)
        if n == 1:
            url = self._next_endpoint()
            return [fn(url, *args_list[0])]

        results: list = [None] * n
        with ThreadPoolExecutor(max_workers=min(self.max_workers, n)) as pool:
            futures = {}
            for i, args in enumerate(args_list):
                url = self._next_endpoint()
                future = pool.submit(fn, url, *args)
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    raise RuntimeError(
                        f"Interleaved batch {idx}/{n} failed{f' ({job_label})' if job_label else ''}: {e}"
                    ) from e
        return results

    def _dispatch_pinned(self, fn: Callable, args_list: list[tuple], job_label: str = "") -> list:
        """Sequential dispatch: all calls go to the same endpoint."""
        url = self._next_endpoint()
        results = []
        for i, args in enumerate(args_list):
            try:
                results.append(fn(url, *args))
            except Exception as e:
                raise RuntimeError(
                    f"Pinned batch {i}/{len(args_list)} to {url}{f' ({job_label})' if job_label else ''}: {e}"
                ) from e
        return results
