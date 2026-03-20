"""
FastAPI Dependencies
Centralized authentication and dependency injection for API endpoints.
"""

import base64
import json
import logging
import hashlib
import time
from typing import Tuple, Literal
from fastapi import Request, HTTPException, WebSocket
from supabase import Client, AsyncClient

from datetime import datetime, timezone
from utils.supabase.supabase_client import (
    authenticate_user_fastapi as _authenticate_user,
    get_user_supabase_client_fastapi as _get_user_client,
    get_service_role_client,
    authenticate_user_fastapi_async as _authenticate_user_async,
    get_async_user_supabase_client_fastapi as _get_async_user_client,
    get_async_service_role_client,
    reset_async_service_role_client,
    TokenExpiredError,
)
from utils.supabase.api_keys_manager import (
    APIKeysManager,
    APIKeyValidationServiceError,
)
from utils.monitoring.token_tracking import check_user_payment_status_async
from utils.caching import api_key_cache
from services.analytics.service import get_analytics_service
from utils.stripe.stripe_service import PLAN_CREDIT_LIMITS

logger = logging.getLogger(__name__)


def _extract_usage_source(request: Request) -> Literal["user", "agent"]:
    """Extract usage source marker from request headers.

    Args:
        request: FastAPI request object.

    Returns:
        "agent" when explicitly marked by trusted internal callers, otherwise "user".
    """
    raw_source = (
        request.headers.get("X-Pioneer-Usage-Source")
        or request.headers.get("x-pioneer-usage-source")
        or "user"
    )
    return "agent" if raw_source == "agent" else "user"


class AuthResult:
    """Container for authentication results with both sync and async clients."""

    def __init__(
        self,
        user_id: str,
        supabase_client: Client,
        api_key_id: str = None,
        async_client: AsyncClient = None,
        auth_method: str = "jwt",  # 'api_key' or 'jwt'
        api_key: str = None,  # The actual API key value (for Felix, etc.)
    ):
        self.user_id = user_id
        self.client = supabase_client  # Keep for backward compatibility
        self.async_client = async_client  # New async client
        self.api_key_id = api_key_id  # Set when authenticated via API key
        self.auth_method = auth_method  # Track authentication method
        self.api_key = api_key  # The actual API key value


async def get_current_user_with_client(request: Request) -> AuthResult:
    """
    FastAPI dependency to authenticate user and return both user_id and Supabase client.

    Args:
        request: FastAPI Request object

    Returns:
        AuthResult: Object containing user_id and supabase_client

    Raises:
        HTTPException: 401 if authentication fails
    """
    try:
        user_id = _authenticate_user(request)
        supabase_client = _get_user_client(request)
        return AuthResult(user_id=user_id, supabase_client=supabase_client)
    except TokenExpiredError:
        logger.info("JWT token expired, client should refresh")
        raise HTTPException(
            status_code=401,
            detail={"message": "Session expired", "code": "token_expired"},
        )
    except ValueError as e:
        logger.info(f"Authentication failed: {e}")
        raise HTTPException(
            status_code=401,
            detail={"message": "Authentication required", "code": "invalid_credentials"},
        )
    except Exception as e:
        logger.warning(f"Unexpected auth error ({type(e).__name__}): {e}")
        raise HTTPException(
            status_code=401,
            detail={"message": "Authentication required", "code": "invalid_credentials"},
        )


def _compute_free_tier_status(user_info: dict | None) -> dict | None:
    """Compute free tier status from cached user info.

    Uses current_period_usage (already fetched via the joined query) and the
    user's payment_plan to determine credit limits without an extra DB call.

    Args:
        user_info: User info dict from the api_keys + users joined query,
            expected to contain payment_plan and current_period_usage.

    Returns:
        Dict with total_usage, credit_limit, free_tier_remaining,
        exceeds_free_tier, and payment_plan, or None if user_info is absent.
        Returning None (rather than {}) ensures token_tracking middleware
        triggers a fresh Stripe check instead of treating an empty dict as valid.
    """
    if not user_info:
        return None
    payment_plan = user_info.get("payment_plan", "hobby") or "hobby"
    current_period_usage = float(user_info.get("current_period_usage") or 0)
    credit_limit = PLAN_CREDIT_LIMITS.get(payment_plan, PLAN_CREDIT_LIMITS["hobby"])
    return {
        "total_usage": current_period_usage,
        "payment_plan": payment_plan,
        "credit_limit": credit_limit,
        "free_tier_remaining": max(0.0, credit_limit - current_period_usage),
        "exceeds_free_tier": current_period_usage > credit_limit,
    }


async def get_api_key_auth(
    api_key: str, request: Request, use_cache: bool = True
) -> Tuple[bool, str | None, str | None, AsyncClient | None]:
    """
    Check if request uses API key authentication (via X-API-Key header).

    Now with caching! Checks cache first, falls back to DB on miss.

    Args:
        api_key: API key from X-API-Key header
        request: FastAPI Request object for caching results
        use_cache: Whether to use cache (default True)

    Returns:
        Tuple[bool, str | None, str | None, AsyncClient | None]: (is_api_key_auth, user_id, api_key_id, supabase_client)
            - is_api_key_auth: True if valid API key authentication
            - user_id: User ID if API key auth is valid
            - api_key_id: API key ID for request logging
            - supabase_client: Async service role client if API key auth is valid

    Raises:
        HTTPException: 500 if server configuration error (e.g., missing Supabase credentials)
        HTTPException: 401 if API key is blocked

    Note:
        This is a helper function for endpoints that support API key authentication.
        Uses service role client to bypass RLS since API keys don't have JWT tokens.
    """
    if not api_key:
        return False, None, None, None

    # Validate API key format before database lookup
    if not api_key.startswith("pio_sk_"):
        logger.warning("Invalid API key format - missing 'pio_sk_' prefix")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key format. API keys must start with 'pio_sk_'. Please check your X-API-Key header.",
        )

    # Hash API key for cache lookup (cache key derivation, not password storage)
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()  # nosec B324

    # Check cache first (if enabled)
    if use_cache:
        cached_data = await api_key_cache.get(api_key_hash)
        if cached_data:
            # Cache hit! Use cached data
            logger.debug("✅ API key cache HIT")

            # Check if blocked
            if cached_data.get("is_blocked", False):
                logger.warning("🚫 Blocked API key attempted access")
                raise HTTPException(
                    status_code=401,
                    detail="API key is blocked. Please contact support or use the billing page to resolve payment issues.",
                )

            # Check if deleted
            if cached_data.get("is_deleted", False):
                logger.warning("Deleted API key attempted access")
                return False, None, None, None

            # Check if expired
            if cached_data.get("expires_at"):
                expires_at = datetime.fromisoformat(
                    cached_data["expires_at"].replace("Z", "+00:00")
                )
                if expires_at < datetime.now(timezone.utc):
                    logger.warning("Expired API key attempted access")
                    return False, None, None, None

            # Valid cached key - use it
            user_id = cached_data["user_id"]
            api_key_id = cached_data["api_key_id"]
            user_info = cached_data.get("user_info")

            # Cache results in request.state
            request.state.api_key_validated = True
            request.state.api_key_user_id = user_id
            request.state.api_key_id = api_key_id
            request.state.user_info = user_info
            request.state.api_key_hash = api_key_hash
            request.state.cached_payment_status = cached_data.get("has_paid", True)
            request.state.cached_free_tier_status = cached_data.get("free_tier_status")
            request.state.usage_source = _extract_usage_source(request)

            service_client: AsyncClient = await get_async_service_role_client()
            return True, user_id, api_key_id, service_client

    # Cache miss - do DB lookup
    logger.debug("⚠️ API key cache MISS, querying database")

    try:
        service_client: AsyncClient = await get_async_service_role_client()
        key_info = await _validate_api_key_with_recovery(api_key)

        if not key_info:
            logger.warning("API key validation failed")
            return False, None, None, None

        user_id = key_info["user_id"]
        api_key_id = key_info["key_id"]
        user_info = key_info.get("user_info")

        # Check if blocked in DB (should have is_blocked field after migration)
        # For now, assume not blocked if not in field (backward compatible)
        is_blocked = False  # Will be populated from DB after migration

        logger.info("🔑 API key authenticated")

        # Pre-compute free_tier_status from the joined user data so subsequent
        # cache hits can skip the payment check entirely.
        free_tier_status = _compute_free_tier_status(user_info)

        # Cache the result for next time
        if use_cache:
            cache_data = {
                "user_id": user_id,
                "api_key_id": api_key_id,
                "user_info": user_info,
                "is_blocked": is_blocked,
                "is_deleted": False,  # validate_key_async already filtered deleted keys
                "expires_at": None,  # Could add this from key_info if needed
                "has_paid": True,  # Optimistic default; synchronous check runs below on first use
                "payment_checked_at": time.time(),
                "free_tier_status": free_tier_status,
            }
            await api_key_cache.set(api_key_hash, cache_data)
            logger.debug("📝 Cached API key")

        # Cache results in request.state to avoid duplicate lookups in middleware
        request.state.api_key_validated = True
        request.state.api_key_user_id = user_id
        request.state.api_key_id = api_key_id
        request.state.user_info = user_info
        request.state.api_key_hash = api_key_hash
        request.state.usage_source = _extract_usage_source(request)

        return True, user_id, api_key_id, service_client

    except ValueError as e:
        # Server configuration error (missing Supabase credentials)
        logger.error(f"Server configuration error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error. Please contact support.",
        )
    except APIKeyValidationServiceError as e:
        logger.warning("Transient API key validation dependency failure: %s", e)
        raise HTTPException(
            status_code=503,
            detail=(
                "Authentication service temporarily unavailable. "
                "Please retry shortly."
            ),
        )
    except Exception as e:
        # Other unexpected errors during authentication
        logger.warning(f"Failed to authenticate with API key: {e}")
        # For other errors, we can't distinguish between invalid key and server error
        # Return False to indicate authentication failed (401)
        return False, None, None, None


async def _validate_api_key_with_recovery(
    api_key: str,
) -> dict | None:
    """Validate API key with one singleton-client recovery retry.

    Args:
        api_key: API key to validate.

    Returns:
        Validation payload when key is valid, otherwise None.

    Raises:
        APIKeyValidationServiceError: When validation dependencies remain unavailable
            after one recovery retry.
    """
    async_service_client = await get_async_service_role_client()
    api_keys_manager = APIKeysManager(async_service_client)

    try:
        return await api_keys_manager.validate_key_async(api_key)
    except APIKeyValidationServiceError:
        logger.warning(
            "API key validation dependency unavailable; resetting async service role client and retrying once."
        )
        await reset_async_service_role_client()
        retried_async_service_client = await get_async_service_role_client()
        retried_api_keys_manager = APIKeysManager(retried_async_service_client)
        return await retried_api_keys_manager.validate_key_async(api_key)


async def _authenticate_with_api_key(
    request: Request, api_key: str, enforce_payment: bool
) -> AuthResult:
    """Authenticate a request using an API key.

    Args:
        request: FastAPI request object.
        api_key: API key from request headers.
        enforce_payment: Whether to enforce payment checks.

    Returns:
        AuthResult containing authenticated user and clients.

    Raises:
        HTTPException: If API key is invalid or payment is required.
    """
    (
        is_api_key,
        authenticated_user_id,
        api_key_id,
        sync_client,
    ) = await get_api_key_auth(api_key, request)
    if not is_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please check your credentials.",
        )

    if enforce_payment:
        cached_status = getattr(request.state, "cached_payment_status", None)

        if cached_status is False:
            raise HTTPException(status_code=401, detail="Payment required.")

        if cached_status is True:
            # Cache hit — use pre-computed free_tier_status; defer payment re-validation
            # to background so the request proceeds without a Supabase round-trip.
            request.state.free_tier_status = getattr(
                request.state, "cached_free_tier_status", None
            )
            request.state.deferred_payment_check = True
        else:
            # Cache miss — run synchronous check and write result back to cache
            # so the next request for this key can skip it.
            has_paid, reason, free_tier_status = await check_user_payment_status_async(
                authenticated_user_id, request
            )
            if not has_paid:
                raise HTTPException(
                    status_code=401,
                    detail=f"Payment required: {reason}",
                )
            request.state.free_tier_status = free_tier_status

            api_key_hash = getattr(request.state, "api_key_hash", None)
            if api_key_hash:
                await api_key_cache.update_payment_status(api_key_hash, has_paid)
                await api_key_cache.update_free_tier_status(api_key_hash, free_tier_status)

    request.state.usage_source = _extract_usage_source(request)

    async_client: AsyncClient = await get_async_service_role_client()
    return AuthResult(
        user_id=authenticated_user_id,
        supabase_client=sync_client,
        api_key_id=api_key_id,
        async_client=async_client,
        auth_method="api_key",
    )


def _decode_jwt_claims(request: Request) -> dict:
    """Extract claims from the JWT payload without cryptographic verification.

    The JWT has already been validated by ``supabase.auth.get_user()``, so
    we only need to read the claims.  Returns an empty dict on any failure
    so callers never need to handle exceptions.

    Args:
        request: FastAPI request carrying the ``Authorization: Bearer ...`` header.

    Returns:
        Decoded JWT payload as a dict, or empty dict on failure.
    """
    try:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return {}
        token = auth_header.removeprefix("Bearer ")
        payload_segment = token.split(".")[1]
        padding = 4 - len(payload_segment) % 4
        if padding != 4:
            payload_segment += "=" * padding
        return json.loads(base64.urlsafe_b64decode(payload_segment))
    except Exception:
        return {}


async def _authenticate_with_jwt(
    request: Request, enforce_payment: bool
) -> AuthResult:
    """Authenticate using standard Supabase JWT auth.

    Args:
        request: FastAPI request object.
        enforce_payment: Whether to enforce payment checks.

    Returns:
        AuthResult containing authenticated user and clients.

    Raises:
        HTTPException: If auth fails or payment is required.
    """
    authenticated_user_id = await _authenticate_user_async(request)
    sync_client = _get_user_client(request)
    async_client = await _get_async_user_client(request)

    if enforce_payment:
        has_paid, reason, free_tier_status = await check_user_payment_status_async(
            authenticated_user_id, request
        )
        if not has_paid:
            raise HTTPException(
                status_code=401,
                detail=f"Payment required: {reason}",
            )

        # Set billing state for middleware (enables usage tracking for JWT requests)
        request.state.api_key_validated = True
        request.state.api_key_user_id = authenticated_user_id
        request.state.api_key_id = None  # No API key for JWT auth
        request.state.user_info = None  # Will be fetched by billing if needed
        request.state.api_key_hash = None
        request.state.free_tier_status = free_tier_status
        request.state.usage_source = _extract_usage_source(request)

    try:
        claims = _decode_jwt_claims(request)
        app_meta = claims.get("app_metadata", {})
        provider = app_meta.get("provider")
        analytics = get_analytics_service()
        is_new_session = analytics.identify_user(
            user_id=authenticated_user_id,
            auth_method="jwt",
            provider=provider,
        )
        if is_new_session:
            analytics.track_event(
                event_type="user_authenticated",
                user_id=authenticated_user_id,
                event_properties={
                    "auth_method": "jwt",
                    "provider": provider or "email",
                },
            )
    except Exception:
        logger.debug("Non-critical: failed to track user in Amplitude", exc_info=True)

    return AuthResult(
        user_id=authenticated_user_id,
        supabase_client=sync_client,
        async_client=async_client,
        auth_method="jwt",
    )


PIONEER_API_KEY_PREFIX = "pio_sk_"


def _extract_pioneer_key_from_bearer(request: Request) -> str | None:
    """Extract a Pioneer API key from the Authorization Bearer header.

    The OpenAI SDK sends ``Authorization: Bearer <api_key>``. When the
    bearer token starts with the Pioneer API key prefix (``pio_sk_``),
    we treat it as an API key rather than a JWT so the OpenAI-compat
    endpoint works as a drop-in replacement.

    Args:
        request: FastAPI request object.

    Returns:
        The API key string if detected, otherwise None.
    """
    auth_header = request.headers.get("authorization") or ""
    if not auth_header.lower().startswith("bearer "):
        return None
    token = auth_header[len("bearer "):].strip()
    if token.startswith(PIONEER_API_KEY_PREFIX):
        return token
    return None


class FlexibleAuth:
    """Callable auth dependency with optional read-only mode.

    Use ``FlexibleAuth(readonly=True)`` for routes that should authenticate users
    without enforcing payment checks.
    """

    def __init__(self, readonly: bool = False):
        self.readonly = readonly

    async def __call__(self, request: Request) -> AuthResult:
        """Authenticate a request with optional payment enforcement.

        Args:
            request: FastAPI request object.

        Returns:
            AuthResult for the authenticated user.
        """
        enforce_payment = not self.readonly
        try:
            api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")

            # OpenAI SDK sends API keys as "Authorization: Bearer <key>".
            # Detect Pioneer API keys in the Bearer token so the OpenAI-compat
            # endpoint works as a drop-in replacement.
            if not api_key:
                api_key = _extract_pioneer_key_from_bearer(request)

            if api_key:
                return await _authenticate_with_api_key(
                    request=request, api_key=api_key, enforce_payment=enforce_payment
                )

            return await _authenticate_with_jwt(
                request=request, enforce_payment=enforce_payment
            )
        except TokenExpiredError:
            logger.info("JWT token expired, client should refresh")
            raise HTTPException(
                status_code=401,
                detail={"message": "Session expired", "code": "token_expired"},
            )
        except HTTPException:
            raise
        except ValueError as e:
            logger.info(f"Authentication failed: {e}")
            raise HTTPException(
                status_code=401,
                detail={"message": "Authentication required", "code": "invalid_credentials"},
            )
        except Exception as e:
            logger.warning(f"Unexpected auth error ({type(e).__name__}): {e}")
            raise HTTPException(
                status_code=401,
                detail={"message": "Authentication required", "code": "invalid_credentials"},
            )


async def authenticate_websocket(websocket: WebSocket) -> tuple[str, str, bool]:
    """
    Authenticate WebSocket connection using X-API-Key header or query parameters.

    Uses the same authentication system as HTTP endpoints.
    For browser clients that cannot send headers, returns requires_first_message_auth=True
    to indicate that authentication should be done via first message.

    Args:
        websocket: WebSocket connection

    Returns:
        tuple[user_id, api_key, requires_first_message_auth]
        - If header auth succeeds: (user_id, api_key, False)
        - If no header but should try first-message auth: (None, None, True)
        - On auth failure (invalid key): (None, None, False) after closing connection

    Note:
        Does NOT close WebSocket if no header provided - allows first-message auth flow.
        Closes WebSocket with error code 4001 only if API key is invalid.
    """
    # Check header for API key (works for CLI/SDK clients)
    api_key = websocket.headers.get("X-API-Key") or websocket.headers.get("x-api-key")

    if not api_key:
        # No header - allow first-message authentication (for browser clients)
        logger.debug("No X-API-Key header, will try first-message auth")
        return None, None, True

    # Validate API key from header
    # Minimal request wrapper for get_api_key_auth
    class MockRequest:
        def __init__(self, ws: WebSocket):
            self.headers = ws.headers
            self.state = type("obj", (object,), {})()

    try:
        mock_request = MockRequest(websocket)
        (is_valid, user_id, _, _) = await get_api_key_auth(api_key, mock_request)
        if not is_valid:
            await websocket.close(code=4001, reason="Invalid API key")
            return None, None, False

        # Enforce payment checks for websocket entry, matching HTTP behavior.
        has_paid, reason, _ = await check_user_payment_status_async(user_id, mock_request)
        if not has_paid:
            await websocket.close(code=4001, reason=f"Payment required: {reason}")
            return None, None, False

        return user_id, api_key, False
    except Exception as e:
        await websocket.close(code=4001, reason=f"Authentication failed: {str(e)}")
        return None, None, False


async def validate_websocket_auth_message_detailed(
    credential: str,
) -> tuple[tuple[str, str] | None, str | None]:
    """Validate websocket first-message auth and return failure reason.

    Args:
        credential: Either an API key or Supabase access token.

    Returns:
        Tuple of (auth_result, error_message):
            - auth_result: (user_id, api_key_or_token) on success, otherwise None
            - error_message: User-facing error reason when auth_result is None
    """
    if not credential:
        return None, "Authentication required"

    # Try API key auth first (if it looks like an API key)
    if credential.startswith("pio_sk_"):

        class MockRequest:
            def __init__(self):
                self.headers = {}
                self.state = type("obj", (object,), {})()

        try:
            mock_request = MockRequest()
            (is_valid, user_id, _, _) = await get_api_key_auth(credential, mock_request)
            if not is_valid:
                return None, "Invalid API key"

            has_paid, reason, _ = await check_user_payment_status_async(
                user_id, mock_request
            )
            if not has_paid:
                return None, f"Payment required: {reason}"

            return (user_id, credential), None
        except Exception as e:
            logger.error(f"API key auth failed: {e}")
            return None, "Invalid API key"

    # Try Supabase access token auth (JWT)
    try:
        supabase = get_service_role_client()
        user_response = supabase.auth.get_user(credential)

        if user_response and user_response.user:
            user_id = user_response.user.id
            mock_request = type("MockRequest", (), {})()
            mock_request.state = type("obj", (object,), {})()
            mock_request.headers = {}
            has_paid, reason, _ = await check_user_payment_status_async(user_id, mock_request)
            if not has_paid:
                return None, f"Payment required: {reason}"
            logger.info(f"WebSocket auth via Supabase token for user {user_id}")
            return (user_id, credential), None
    except Exception as e:
        logger.debug(f"Supabase token auth failed: {e}")

    logger.warning("Invalid credential - neither valid API key nor Supabase token")
    return None, "Invalid API key"


async def validate_websocket_auth_message(credential: str) -> tuple[str, str] | None:
    """
    Validate a credential received via WebSocket first-message authentication.

    Supports two authentication methods:
    1. API key (pio_sk_...) - preferred, returns user_id and api_key
    2. Supabase access token (JWT) - fallback for frontend, returns user_id and empty api_key

    Args:
        credential: Either an API key or Supabase access token

    Returns:
        tuple[user_id, api_key_or_empty] if valid, None if invalid
    """
    auth_result, _ = await validate_websocket_auth_message_detailed(credential)
    return auth_result
