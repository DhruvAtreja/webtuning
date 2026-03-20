"""
Supabase Client
Simplified client for Supabase operations - FastAPI only.
"""

import asyncio
import os
import logging
from typing import Awaitable, Callable, TypeVar

import httpx
from supabase import ClientOptions, create_client, Client, acreate_client, AsyncClient
from dotenv import load_dotenv
from fastapi import Request as FastAPIRequest
from utils.supabase.transient_errors import is_transient_error
from utils.supabase.user_async_client_cache import (
    close_cached_user_async_clients,
    get_cached_user_async_client,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)

# ============================================================================
# HTTPX CONNECTION POOL CONFIGURATION
# ============================================================================
# These settings prevent 522/503 errors under load by properly configuring
# timeouts and connection pooling for the underlying httpx client.

SUPABASE_TIMEOUT = httpx.Timeout(
    30.0,  # Total timeout for request
    connect=30.0,  # Connection timeout
)

SUPABASE_LIMITS = httpx.Limits(
    max_connections=100,  # Maximum concurrent connections
    max_keepalive_connections=20,  # Keep-alive connections to reuse
)

# Retries on connection-level failures (ConnectTimeout, ConnectError).
# httpx transports only retry the connection phase, not reads/writes,
# so this is safe for non-idempotent requests.
SUPABASE_CONNECT_RETRIES = 3

# Shared httpx clients (singleton pattern for connection pooling)
_httpx_sync_client: httpx.Client | None = None
_httpx_async_client: httpx.AsyncClient | None = None

# Serializes reset operations so concurrent coroutines don't cascade-reset
# healthy clients. The generation counter lets callers detect whether another
# coroutine already performed a reset, making their own reset a no-op.
_reset_lock: asyncio.Lock | None = None
_client_generation: int = 0


def _get_reset_lock() -> asyncio.Lock:
    """Lazily create the reset lock (must happen inside a running event loop)."""
    global _reset_lock
    if _reset_lock is None:
        _reset_lock = asyncio.Lock()
    return _reset_lock


def _get_httpx_sync_client() -> httpx.Client:
    """Get or create the shared sync httpx client with connection pooling."""
    global _httpx_sync_client
    if _httpx_sync_client is None:
        transport = httpx.HTTPTransport(
            retries=SUPABASE_CONNECT_RETRIES,
        )
        _httpx_sync_client = httpx.Client(
            transport=transport,
            timeout=SUPABASE_TIMEOUT,
            limits=SUPABASE_LIMITS,
        )
        logger.info("Created shared sync httpx client with connection pooling")
    return _httpx_sync_client


def _get_httpx_async_client() -> httpx.AsyncClient:
    """Get or create the shared async httpx client with connection pooling."""
    global _httpx_async_client
    if _httpx_async_client is None:
        transport = httpx.AsyncHTTPTransport(
            retries=SUPABASE_CONNECT_RETRIES,
        )
        _httpx_async_client = httpx.AsyncClient(
            transport=transport,
            timeout=SUPABASE_TIMEOUT,
            limits=SUPABASE_LIMITS,
        )
        logger.info("Created shared async httpx client with connection pooling")
    return _httpx_async_client


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class TokenExpiredError(Exception):
    """Raised when JWT token has expired but could be refreshed."""

    pass


def _load_config():
    """Load configuration from environment."""
    # Try multiple .env locations
    env_paths = ["../../.env", "../.env", ".env"]
    for path in env_paths:
        if os.path.exists(path):
            load_dotenv(dotenv_path=path)
            break

    return {
        "url": os.getenv("SUPABASE_URL"),
        "key": os.getenv("SUPABASE_PUBLISHABLE_KEY"),
        "secret_key": os.getenv("SUPABASE_SECRET_KEY"),
    }


def get_supabase_client() -> Client:
    """Get basic Supabase client with publishable key."""
    config = _load_config()
    if not config["url"] or not config["key"]:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY must be set in environment"
        )

    return create_client(config["url"], config["key"])


_sync_service_role_client: Client | None = None


def reset_sync_service_role_client() -> None:
    """Reset the sync service role client singleton.

    Discards the current client so the next call to get_service_role_client()
    creates a fresh instance. Useful in tests and after connection errors.
    """
    global _sync_service_role_client
    _sync_service_role_client = None


def get_service_role_client() -> Client:
    """Get Supabase client with service role key (bypasses RLS).

    Returns a singleton instance, reusing the connection pool across requests.
    Uses shared httpx client with connection pooling and timeouts.
    """
    global _sync_service_role_client
    if _sync_service_role_client is None:
        config = _load_config()
        if not config["url"] or not config["secret_key"]:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SECRET_KEY must be set in environment"
            )
        options = ClientOptions(httpx_client=_get_httpx_sync_client())
        _sync_service_role_client = create_client(
            config["url"], config["secret_key"], options=options
        )
    return _sync_service_role_client


def authenticate_user_fastapi(request: FastAPIRequest) -> str:
    """
    Authenticate user from FastAPI request.

    Args:
        request: FastAPI Request object

    Returns:
        str: User ID

    Raises:
        Exception: If authentication fails
    """
    # Get Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        raise ValueError("No Authorization header provided")

    # Extract token
    if not auth_header.startswith("Bearer "):
        raise ValueError("Invalid Authorization header format")

    token = auth_header.removeprefix("Bearer ")

    # Create Supabase client with token
    config = _load_config()
    options = ClientOptions()
    options.headers = {"Authorization": f"Bearer {token}"}

    supabase = create_client(
        supabase_url=config["url"],
        supabase_key=config["key"],
        options=options,
    )

    # Get user from token
    try:
        user_response = supabase.auth.get_user(jwt=token)
        if getattr(user_response, "user", None):
            return user_response.user.id
        else:
            raise ValueError("No user found in token response")
    except ValueError as e:
        error_msg = str(e).lower()
        if "expired" in error_msg or "token has invalid claims" in error_msg:
            logger.info(f"JWT token expired: {e}")
            raise TokenExpiredError("Session expired") from e
        logger.info(f"Failed to authenticate user: {e}")
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "expired" in error_msg or "token has invalid claims" in error_msg:
            logger.info(f"JWT token expired: {e}")
            raise TokenExpiredError("Session expired") from e
        logger.warning(f"Unexpected auth error ({type(e).__name__}): {e}")
        raise


def get_user_supabase_client_fastapi(request: FastAPIRequest) -> Client:
    """
    Get user-specific Supabase client for FastAPI requests.

    Args:
        request: FastAPI Request object

    Returns:
        Client: Supabase client with user's auth token
    """
    config = _load_config()
    auth_header = request.headers.get("Authorization", "")

    options = ClientOptions()
    if auth_header:
        options.headers = {"Authorization": auth_header}

    return create_client(
        supabase_url=config["url"],
        supabase_key=config["key"],
        options=options,
    )


# ============================================================================
# ASYNC CLIENT FUNCTIONS
# ============================================================================

# Singleton for async service role client (shared across all requests)
_async_service_role_client: AsyncClient | None = None


async def get_async_supabase_client() -> AsyncClient:
    """Get async Supabase client with publishable key."""
    config = _load_config()
    if not config["url"] or not config["key"]:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY must be set in environment"
        )

    return await acreate_client(config["url"], config["key"])


async def get_async_service_role_client() -> AsyncClient:
    """Get async Supabase client with service role key (bypasses RLS).

    Uses singleton pattern since service role client is shared across all requests.
    Configured with connection pooling and timeouts to prevent 522/503 errors.
    """
    global _async_service_role_client

    if _async_service_role_client is None:
        config = _load_config()
        if not config["url"] or not config["secret_key"]:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SECRET_KEY must be set in environment"
            )

        options = ClientOptions(httpx_client=_get_httpx_async_client())
        _async_service_role_client = await acreate_client(
            config["url"], config["secret_key"], options=options
        )
        logger.info("Created singleton async service role client with connection pooling")

    return _async_service_role_client


async def reset_async_service_role_client(*, close_httpx_client: bool = False) -> None:
    """Reset the async service role client singleton.

    Called when the client enters a bad state (connection pool exhausted, timeouts).
    By default, this does NOT close the old shared httpx client because in-flight
    requests may still hold references to Supabase clients backed by that transport.
    Closing it eagerly can surface RuntimeError("client has been closed") in other
    coroutines. The next call to get_async_service_role_client() still creates a
    fresh singleton.

    Set ``close_httpx_client=True`` only for controlled teardown paths where no
    in-flight request should still rely on the previous transport.

    Thread-safe: uses an asyncio.Lock to prevent concurrent resets from
    cascading (thundering-herd problem at scale).

    Args:
        close_httpx_client: Whether to close the previous shared httpx client.
    """
    global _async_service_role_client, _httpx_async_client, _client_generation

    async with _get_reset_lock():
        old_httpx_client = _httpx_async_client

        _async_service_role_client = None
        _httpx_async_client = None
        _client_generation += 1

        if close_httpx_client and old_httpx_client is not None:
            try:
                await old_httpx_client.aclose()
                logger.info("Closed old async httpx client during reset")
            except Exception as e:
                logger.warning(f"Failed to close old async httpx client: {e}")

        logger.warning("Reset async service role client — will recreate on next use")


def _is_recoverable_client_error(exc: Exception) -> bool:
    """Check whether client recovery should retry with a fresh singleton.

    Uses ``is_transient_error`` as the single source of truth so the
    legacy recovery path and the newer retry wrapper classify failures
    consistently.
    """
    return is_transient_error(exc)


async def execute_with_client_recovery(
    operation: Callable[[AsyncClient], Awaitable[T]],
) -> T:
    """Execute a Supabase operation with one retry on transient errors.

    Long-running background tasks may hold a stale reference to the singleton
    AsyncClient after another coroutine resets it. This helper catches
    transient Supabase/httpx errors, resets the singleton, and retries once
    with a fresh client.

    Scale-safe: uses a generation counter to skip redundant resets. If another
    coroutine already reset the client between our failure and our reset
    attempt, we skip the reset and just grab the fresh client.

    Args:
        operation: An async callable that receives an AsyncClient and returns a result.

    Returns:
        The result of the operation.

    Raises:
        The original exception if it is not transient,
        or the retry exception if the retry also fails.
    """
    gen_before: int = _client_generation
    try:
        client = await get_async_service_role_client()
        return await operation(client)
    except Exception as first_err:
        if not _is_recoverable_client_error(first_err):
            raise
        logger.warning(
            "Supabase transient error (%s), resetting and retrying once: %s",
            type(first_err).__name__,
            first_err,
        )
        # Only reset if nobody else already did (prevents thundering-herd resets)
        if _client_generation == gen_before:
            await reset_async_service_role_client()
        client = await get_async_service_role_client()
        return await operation(client)


async def close_async_clients() -> None:
    """Close all async Supabase/httpx clients for clean shutdown.

    Called from the FastAPI lifespan shutdown to avoid leaked connection warnings.
    """
    global _async_service_role_client, _httpx_async_client

    await close_cached_user_async_clients()

    if _httpx_async_client is not None:
        try:
            await _httpx_async_client.aclose()
            logger.info("Closed async httpx client on shutdown")
        except Exception as e:
            logger.warning(f"Failed to close async httpx client on shutdown: {e}")
        _httpx_async_client = None

    _async_service_role_client = None


async def authenticate_user_fastapi_async(request: FastAPIRequest) -> str:
    """
    Authenticate user from FastAPI request (async version).

    Args:
        request: FastAPI Request object

    Returns:
        str: User ID

    Raises:
        Exception: If authentication fails
    """
    # Get Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        raise ValueError("No Authorization header provided")

    # Extract token
    if not auth_header.startswith("Bearer "):
        raise ValueError("Invalid Authorization header format")

    token = auth_header.removeprefix("Bearer ")

    # Create async Supabase client with token
    config = _load_config()
    options = ClientOptions()
    options.headers = {"Authorization": f"Bearer {token}"}

    supabase = await acreate_client(
        supabase_url=config["url"],
        supabase_key=config["key"],
        options=options,
    )

    # Get user from token
    try:
        user_response = await supabase.auth.get_user(jwt=token)
        if getattr(user_response, "user", None):
            return user_response.user.id
        else:
            raise ValueError("No user found in token response")
    except ValueError as e:
        error_msg = str(e).lower()
        if "expired" in error_msg or "token has invalid claims" in error_msg:
            logger.info(f"JWT token expired: {e}")
            raise TokenExpiredError("Session expired") from e
        logger.info(f"Failed to authenticate user: {e}")
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "expired" in error_msg or "token has invalid claims" in error_msg:
            logger.info(f"JWT token expired: {e}")
            raise TokenExpiredError("Session expired") from e
        logger.warning(f"Unexpected auth error ({type(e).__name__}): {e}")
        raise


async def get_async_user_supabase_client_fastapi(
    request: FastAPIRequest,
) -> AsyncClient:
    """Get user-scoped async Supabase client with token-scoped safe pooling.

    Args:
        request: FastAPI Request object containing the Authorization header.

    Returns:
        AsyncClient: Async Supabase client scoped to the authenticated user.
    """
    config = _load_config()
    auth_header = request.headers.get("Authorization", "")
    return await get_cached_user_async_client(
        supabase_url=config["url"],
        supabase_key=config["key"],
        auth_header=auth_header,
        timeout=SUPABASE_TIMEOUT,
        limits=SUPABASE_LIMITS,
        connect_retries=SUPABASE_CONNECT_RETRIES,
    )
