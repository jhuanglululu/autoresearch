"""Shared HTTP plumbing for both provider clients: a POST with exponential
backoff on transient failures.

Retries 429, any 5xx, and httpx transport errors (connect/read/write) up to
MAX_TRIES times. A 429/5xx honours the server's ``Retry-After`` header when
present; otherwise it backs off exponentially with jitter. Non-retryable 4xx
responses raise immediately, and exhausting the retries raises with the last
error attached — the caller always gets a clear failure, never a silent hang.
"""
from __future__ import annotations

import asyncio
import random

import httpx

MAX_TRIES = 5
BASE_BACKOFF = 0.5  # seconds; grows as BASE_BACKOFF * 2**attempt
MAX_BACKOFF = 30.0


class LLMError(RuntimeError):
    """Raised when an LLM HTTP request fails non-retryably or exhausts retries."""


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    """Parse a ``Retry-After`` header. Supports the delta-seconds form; the
    HTTP-date form is ignored (we fall back to computed backoff)."""
    raw = resp.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _backoff_seconds(attempt: int) -> float:
    delay = min(BASE_BACKOFF * (2**attempt), MAX_BACKOFF)
    return delay + random.uniform(0.0, BASE_BACKOFF)


async def post_json(
    http: httpx.AsyncClient, path: str, body: dict, *, provider: str
) -> dict:
    """POST ``body`` as JSON to ``path`` and return the parsed JSON response,
    retrying transient failures. ``provider`` labels errors (e.g. "anthropic")."""
    last_error: BaseException | None = None
    for attempt in range(MAX_TRIES):
        try:
            resp = await http.post(path, json=body)
        except httpx.TransportError as exc:  # connect/read/write/pool errors
            last_error = exc
            if attempt == MAX_TRIES - 1:
                break
            await asyncio.sleep(_backoff_seconds(attempt))
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = LLMError(
                f"{provider} HTTP {resp.status_code}: {resp.text[:500]}"
            )
            if attempt == MAX_TRIES - 1:
                break
            delay = _retry_after_seconds(resp)
            if delay is None:
                delay = _backoff_seconds(attempt)
            await asyncio.sleep(delay)
            continue

        if resp.status_code >= 400:  # non-retryable client error
            raise LLMError(f"{provider} HTTP {resp.status_code}: {resp.text[:1000]}")

        return resp.json()

    raise LLMError(
        f"{provider} request failed after {MAX_TRIES} attempts: {last_error}"
    )
