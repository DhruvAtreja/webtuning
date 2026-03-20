"""Shared transient Supabase/httpx error classification utilities.

This module is the single source of truth for determining whether an
exception should be treated as transient (retryable) across Supabase
access paths.
"""

import httpx

_TRANSIENT_MESSAGE_MARKERS = (
    "read timed out",
    "timed out",
    "connect timed out",
    "pool timeout",
    "connection reset",
    "connection refused",
    "server disconnected",
    "broken pipe",
    "client has been closed",
)


def is_transient_error(exc: Exception) -> bool:
    """Return whether an exception represents a transient infrastructure error.

    Args:
        exc: The exception to classify.

    Returns:
        True if the error is considered transient and retryable.
    """
    if isinstance(
        exc,
        (
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.PoolTimeout,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True

    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (
        502,
        503,
        504,
    ):
        return True

    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MESSAGE_MARKERS)
