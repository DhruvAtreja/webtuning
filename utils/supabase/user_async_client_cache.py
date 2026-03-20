"""Token-scoped cache for async user Supabase clients."""

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass

import httpx
from supabase import AsyncClient, acreate_client
from supabase._async.client import ClientOptions

logger = logging.getLogger(__name__)


def _get_positive_int_env(name: str, default: int) -> int:
    """Read a positive integer env var with fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s value '%s'; using %s", name, raw_value, default)
        return default
    if parsed < 1:
        logger.warning("Invalid %s value '%s'; using %s", name, raw_value, default)
        return default
    return parsed


def _get_positive_float_env(name: str, default: float) -> float:
    """Read a positive float env var with fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s value '%s'; using %.2f", name, raw_value, default)
        return default
    if parsed < 0.01:
        logger.warning("Invalid %s value '%s'; using %.2f", name, raw_value, default)
        return default
    return parsed


USER_ASYNC_CLIENT_CACHE_SIZE = _get_positive_int_env(
    "SUPABASE_USER_ASYNC_CLIENT_CACHE_SIZE", 256
)
USER_ASYNC_CLIENT_CACHE_TTL_SECONDS = _get_positive_float_env(
    "SUPABASE_USER_ASYNC_CLIENT_CACHE_TTL_SECONDS", 900.0
)


@dataclass
class _UserAsyncClientCacheEntry:
    """Cached async user-scoped Supabase client and its isolated transport."""

    client: AsyncClient
    httpx_client: httpx.AsyncClient
    last_used_monotonic: float


_user_async_client_cache: dict[str, _UserAsyncClientCacheEntry] = {}
_user_async_client_cache_lock: asyncio.Lock | None = None


def _get_user_async_client_cache_lock() -> asyncio.Lock:
    """Create lock lazily so import-time does not require an active event loop."""
    global _user_async_client_cache_lock
    if _user_async_client_cache_lock is None:
        _user_async_client_cache_lock = asyncio.Lock()
    return _user_async_client_cache_lock


def _build_user_cache_key(
    supabase_url: str | None, supabase_key: str | None, auth_header: str
) -> str:
    """Build a stable cache key without storing raw credentials."""
    seed = f"{supabase_url}|{supabase_key}|{auth_header}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _is_user_async_cache_entry_expired(
    entry: _UserAsyncClientCacheEntry, now_monotonic: float
) -> bool:
    """Return whether a cached entry has exceeded the configured TTL."""
    return (
        now_monotonic - entry.last_used_monotonic
    ) > USER_ASYNC_CLIENT_CACHE_TTL_SECONDS


def _collect_expired_user_async_cache_entries_locked(
    now_monotonic: float,
) -> list[_UserAsyncClientCacheEntry]:
    """Pop and return expired cache entries (caller must hold cache lock)."""
    expired_keys: list[str] = []
    for key, entry in _user_async_client_cache.items():
        if _is_user_async_cache_entry_expired(entry, now_monotonic):
            expired_keys.append(key)

    expired_entries: list[_UserAsyncClientCacheEntry] = []
    for key in expired_keys:
        entry = _user_async_client_cache.pop(key, None)
        if entry is not None:
            expired_entries.append(entry)
    return expired_entries


def _collect_lru_user_async_cache_entries_locked() -> list[_UserAsyncClientCacheEntry]:
    """Pop and return oldest entries while cache size exceeds configured limit."""
    overflow = len(_user_async_client_cache) - USER_ASYNC_CLIENT_CACHE_SIZE
    if overflow <= 0:
        return []

    oldest_items = sorted(
        _user_async_client_cache.items(),
        key=lambda item: item[1].last_used_monotonic,
    )[:overflow]

    evicted_entries: list[_UserAsyncClientCacheEntry] = []
    for key, _ in oldest_items:
        entry = _user_async_client_cache.pop(key, None)
        if entry is not None:
            evicted_entries.append(entry)
    return evicted_entries


def _build_isolated_httpx_async_client(
    timeout: httpx.Timeout,
    limits: httpx.Limits,
    connect_retries: int,
) -> httpx.AsyncClient:
    """Create isolated httpx client for one auth context."""
    return httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(retries=connect_retries),
        timeout=timeout,
        limits=limits,
    )


async def _close_user_async_cache_entries(
    entries: list[_UserAsyncClientCacheEntry], *, reason: str
) -> None:
    """Close isolated httpx clients for evicted cache entries."""
    for entry in entries:
        try:
            await entry.httpx_client.aclose()
        except Exception as exc:  # pragma: no cover - defensive logging branch
            logger.warning("Failed to close user async httpx client (%s): %s", reason, exc)


async def get_cached_user_async_client(
    *,
    supabase_url: str | None,
    supabase_key: str | None,
    auth_header: str,
    timeout: httpx.Timeout,
    limits: httpx.Limits,
    connect_retries: int,
) -> AsyncClient:
    """Return an isolated user client with token-scoped cache reuse."""
    cache_key = _build_user_cache_key(supabase_url, supabase_key, auth_header)
    now_monotonic = time.monotonic()
    cache_lock = _get_user_async_client_cache_lock()
    entries_to_close: list[_UserAsyncClientCacheEntry] = []

    async with cache_lock:
        entries_to_close.extend(
            _collect_expired_user_async_cache_entries_locked(now_monotonic)
        )
        cached_entry = _user_async_client_cache.get(cache_key)
        if cached_entry is not None:
            cached_entry.last_used_monotonic = now_monotonic
            cached_client = cached_entry.client
        else:
            cached_client = None

    await _close_user_async_cache_entries(entries_to_close, reason="expired")
    if cached_client is not None:
        return cached_client

    isolated_httpx_client = _build_isolated_httpx_async_client(
        timeout=timeout,
        limits=limits,
        connect_retries=connect_retries,
    )
    options = ClientOptions(httpx_client=isolated_httpx_client)
    if auth_header:
        options.headers = {"Authorization": auth_header}

    created_client = await acreate_client(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        options=options,
    )
    created_entry = _UserAsyncClientCacheEntry(
        client=created_client,
        httpx_client=isolated_httpx_client,
        last_used_monotonic=time.monotonic(),
    )

    entries_to_close = []
    async with cache_lock:
        now_monotonic = time.monotonic()
        existing_entry = _user_async_client_cache.get(cache_key)
        if existing_entry is not None and not _is_user_async_cache_entry_expired(
            existing_entry, now_monotonic
        ):
            existing_entry.last_used_monotonic = now_monotonic
            entries_to_close.append(created_entry)
            result_client = existing_entry.client
        else:
            if existing_entry is not None:
                _user_async_client_cache.pop(cache_key, None)
                entries_to_close.append(existing_entry)
            _user_async_client_cache[cache_key] = created_entry
            entries_to_close.extend(_collect_lru_user_async_cache_entries_locked())
            result_client = created_entry.client

    await _close_user_async_cache_entries(entries_to_close, reason="replaced_or_lru")
    return result_client


async def close_cached_user_async_clients() -> None:
    """Close and clear all cached user-scoped async clients."""
    global _user_async_client_cache

    async with _get_user_async_client_cache_lock():
        entries = list(_user_async_client_cache.values())
        _user_async_client_cache = {}

    await _close_user_async_cache_entries(entries, reason="shutdown")
