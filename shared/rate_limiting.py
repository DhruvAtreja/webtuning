"""Rate limiting utilities for the Pioneer API.

Provides IP extraction from proxied requests and limiter factory
for ElastiCache-backed distributed rate limiting with graceful fallback.

Architecture (5 layers, inspired by Stripe's rate-limiter stack):

Layer 1 — **WAF** (infrastructure/cdk): IP-based abuse protection.
    Blocks IPs exceeding ``WAF_RATE_LIMIT`` req/min across all endpoints.
    Pure DDoS / bot defence; never affects legitimate per-user traffic.

Layer 2 — **Request Rate Limiter** (this module, SlowAPI + Redis):
    Per-user and global RPM throttling on generation endpoints.
    Redis-backed via ElastiCache for consistency across ECS tasks.

Layer 3 — **Global Admission Control** (shared/global_admission.py):
    Fleet-wide in-flight concurrency caps for expensive synchronous paths.
    Uses Redis counters for cross-task coordination and returns 429 immediately
    when capacity is saturated.

Layer 4 — **Concurrent Generation Limiter** (services/generation/admission.py):
    Per-user safety cap on in-flight generation jobs (queued + generating).
    High default (100) — only catches runaway loops, not normal batch use.
    Day-to-day throughput is governed by queue depth + worker scaling.

Layer 5 — **FelixClient 429 Retry** (mle_agent/tools/felix/client.py):
    Defence-in-depth: agent retries with exponential backoff on 429.

Environment variables (all optional, with safe defaults):

    REDIS_URL
        Redis connection URI for distributed counters.
        Default: in-memory (local dev).

    FELIX_GENERATION_RATE_LIMIT_ENABLED
        Master kill-switch.  Set to "false"/"0"/"off" to bypass all
        SlowAPI generation limits.  Default: true.

    FELIX_GENERATION_PER_USER_LIMIT_PER_MINUTE
        Max generation requests per user per minute.
        Default: 120.

    FELIX_GENERATION_GLOBAL_LIMIT_PER_MINUTE
        Max generation requests across all users per minute.
        Default: 500. Should match backend capacity.

    GENERATION_MAX_CONCURRENT_PER_USER
        Max in-flight generation jobs (status in queued/generating) per user.
        Default: 100. Safety valve only — normal batch use should never hit it.
        Enforced in services/generation/admission.py.
"""

import logging
import os
import inspect
from typing import Any, Callable
from urllib.parse import urlparse

from fastapi import Request
from slowapi import Limiter

logger = logging.getLogger(__name__)

DEFAULT_RATE_LIMITS = ["1000/hour", "100/minute"]


def _redact_url(url: str) -> str:
    """Return a URL with any embedded credentials stripped out for safe logging.

    Args:
        url: A Redis connection URL that may contain a username and password.

    Returns:
        The URL with credentials removed (e.g. ``rediss://host:6379/0``),
        or ``"<unparseable>"`` if the URL cannot be parsed.
    """
    parsed = urlparse(url)
    if parsed.hostname:
        port = parsed.port or 6379
        path = parsed.path.lstrip("/")
        return f"{parsed.scheme}://{parsed.hostname}:{port}/{path}"
    return "<unparseable>"


def get_real_client_ip(request: Request) -> str:
    """Extract real client IP from X-Forwarded-For, falling back to remote address.

    Behind CloudFront -> WAF -> ALB, the leftmost X-Forwarded-For entry
    is the original client IP.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The client's real IP address, or "127.0.0.1" if undetermined.
    """
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for and forwarded_for.strip():
        # CloudFront is our outermost proxy and always sets the leftmost
        # X-Forwarded-For entry to the real client IP. WAF blocks direct
        # ALB access, so this value cannot be spoofed in our architecture.
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "127.0.0.1"


def create_limiter(redis_url: str | None = None) -> Limiter:
    """Create a SlowAPI Limiter with Redis or in-memory storage.

    When a Redis URL is provided, uses Redis as the shared counter store
    across all ECS tasks, with in-memory fallback for graceful degradation
    if Redis becomes unreachable.

    When no Redis URL is provided (local dev), uses in-memory storage.

    Args:
        redis_url: Redis connection URI (e.g. "rediss://host:6379/0").
            None or empty string falls back to in-memory storage.

    Returns:
        Configured SlowAPI Limiter instance.
    """
    use_redis = bool(redis_url)

    if use_redis:
        logger.info("Rate limiter using Redis backend: %s", _redact_url(redis_url))
        return Limiter(
            key_func=get_real_client_ip,
            default_limits=DEFAULT_RATE_LIMITS,
            storage_uri=redis_url,
            in_memory_fallback_enabled=True,
            in_memory_fallback=DEFAULT_RATE_LIMITS,
        )

    logger.info("Rate limiter using in-memory backend (no REDIS_URL)")
    return Limiter(
        key_func=get_real_client_ip,
        default_limits=DEFAULT_RATE_LIMITS,
        storage_uri="memory://",
    )


def _install_signature_preserving_limit_wrapper(rate_limiter: Limiter) -> Limiter:
    """Patch ``rate_limiter.limit`` to preserve endpoint call signatures.

    SlowAPI wraps handlers and, for body-bearing POST routes, this can hide the
    original FastAPI signature. Preserving ``__signature__`` keeps OpenAPI and
    body parsing correct for decorated endpoints.

    Args:
        rate_limiter: Limiter instance to patch.

    Returns:
        The same limiter instance with a patched ``limit`` decorator.
    """
    original_limit = rate_limiter.limit

    def signature_preserving_limit(
        *limit_args: Any,
        **limit_kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a SlowAPI decorator that preserves the wrapped signature."""
        slowapi_decorator = original_limit(*limit_args, **limit_kwargs)

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            wrapped = slowapi_decorator(func)
            wrapped.__signature__ = inspect.signature(func, eval_str=True)
            return wrapped

        return decorator

    rate_limiter.limit = signature_preserving_limit  # type: ignore[method-assign]
    return rate_limiter


# System-wide singleton imported by all routers for @limiter.limit() decorators.
# server_fastapi.py registers this instance as app.state.limiter so SlowAPI
# enforces limits using the same counter store across every route.
limiter = _install_signature_preserving_limit_wrapper(
    create_limiter(redis_url=os.getenv("REDIS_URL"))
)


# ── Generation endpoint rate-limit helpers ──────────────────────────────────
# Used by felix_generate_router to apply per-user and global throttling on
# all data-generation endpoints (NER, classification, records, custom, etc.).

DEFAULT_FELIX_GENERATION_PER_USER_LIMIT_PER_MINUTE = 120
DEFAULT_FELIX_GENERATION_GLOBAL_LIMIT_PER_MINUTE = 500

GENERATION_USER_SCOPE = "felix-generation-user"
GENERATION_GLOBAL_SCOPE = "felix-generation-global"


def _parse_positive_int(name: str, default: int) -> int:
    """Parse a positive integer environment variable with safe fallback.

    Args:
        name: Environment variable name.
        default: Fallback value when missing or invalid.

    Returns:
        Parsed positive integer value.
    """
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = int(raw)
        if value <= 0:
            raise ValueError("must be positive")
        return value
    except Exception:
        logger.warning("Invalid %s=%r. Falling back to %s.", name, raw, default)
        return default


def _parse_enabled(name: str, default: bool = True) -> bool:
    """Parse an on/off flag from environment.

    Args:
        name: Environment variable name.
        default: Fallback value when unset.

    Returns:
        True when enabled.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def generation_rate_limit_exempt(*_: object, **__: object) -> bool:
    """Return whether generation route limits should be skipped.

    Wired into SlowAPI's ``exempt_when`` callback.

    Env var:
        FELIX_GENERATION_RATE_LIMIT_ENABLED

    Returns:
        True when rate limiting should be bypassed.
    """
    enabled = _parse_enabled("FELIX_GENERATION_RATE_LIMIT_ENABLED", default=True)
    return not enabled


def generation_per_user_limit(*_: object, **__: object) -> str:
    """Return configured per-user generation limit string for SlowAPI.

    Env var:
        FELIX_GENERATION_PER_USER_LIMIT_PER_MINUTE

    Returns:
        SlowAPI-compatible limit string.
    """
    limit = _parse_positive_int(
        "FELIX_GENERATION_PER_USER_LIMIT_PER_MINUTE",
        DEFAULT_FELIX_GENERATION_PER_USER_LIMIT_PER_MINUTE,
    )
    return f"{limit}/minute"


def generation_global_limit(*_: object, **__: object) -> str:
    """Return configured global generation limit string for SlowAPI.

    Env var:
        FELIX_GENERATION_GLOBAL_LIMIT_PER_MINUTE

    Returns:
        SlowAPI-compatible limit string.
    """
    limit = _parse_positive_int(
        "FELIX_GENERATION_GLOBAL_LIMIT_PER_MINUTE",
        DEFAULT_FELIX_GENERATION_GLOBAL_LIMIT_PER_MINUTE,
    )
    return f"{limit}/minute"


def generation_user_key(request: Request) -> str:
    """Compute per-user rate-limit key for generation endpoints.

    Uses authenticated user id written by auth dependency, with IP fallback.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Stable user-scoped limiter key.
    """
    user_id = getattr(request.state, "api_key_user_id", None)
    if user_id:
        return f"user:{user_id}"

    return f"ip:{get_real_client_ip(request)}"


def generation_global_key(request: Request) -> str:
    """Return a single shared key for global generation throttling.

    Args:
        request: Incoming request (unused).

    Returns:
        Shared key across all callers.
    """
    _ = request
    return GENERATION_GLOBAL_SCOPE
