"""
API Key Authentication Middleware.

Provides API key authentication with multiple strategies and management.
"""
import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from starlette.middleware.base import BaseHTTPMiddleware

from pdf_splitter.api.config import config

logger = logging.getLogger(__name__)


class APIKey:
    """API Key model."""

    def __init__(
        self,
        key: str,
        name: str,
        scopes: List[str] = None,
        rate_limit: Optional[int] = None,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize API key."""
        self.key = key
        self.key_hash = self._hash_key(key)
        self.name = name
        self.scopes = scopes or []
        self.rate_limit = rate_limit
        self.expires_at = expires_at
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.last_used = None
        self.usage_count = 0

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()

    def is_valid(self) -> bool:
        """Check if key is still valid."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def has_scope(self, scope: str) -> bool:
        """Check if key has required scope."""
        return scope in self.scopes or "*" in self.scopes

    def record_usage(self):
        """Record key usage."""
        self.last_used = datetime.utcnow()
        self.usage_count += 1


class APIKeyManager:
    """Manages API keys."""

    def __init__(self):
        """Initialize API key manager."""
        # In-memory storage (replace with database in production)
        self.keys: Dict[str, APIKey] = {}

        # Default API keys for development
        if config.debug:
            self._add_default_keys()

    def _add_default_keys(self):
        """Add default development keys."""
        dev_key = self.create_key(
            name="Development Key", scopes=["*"], metadata={"type": "development"}
        )
        logger.info(f"Development API key created: {dev_key.key}")

    def create_key(
        self,
        name: str,
        scopes: List[str] = None,
        rate_limit: Optional[int] = None,
        expires_in: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> APIKey:
        """
        Create a new API key.

        Args:
            name: Key name/description
            scopes: List of allowed scopes
            rate_limit: Custom rate limit for this key
            expires_in: Key expiration time
            metadata: Additional metadata

        Returns:
            Created API key
        """
        # Generate secure random key
        key_value = f"sk_{secrets.token_urlsafe(32)}"

        # Calculate expiry
        expires_at = None
        if expires_in:
            expires_at = datetime.utcnow() + expires_in

        # Create key object
        api_key = APIKey(
            key=key_value,
            name=name,
            scopes=scopes,
            rate_limit=rate_limit,
            expires_at=expires_at,
            metadata=metadata,
        )

        # Store by hash
        self.keys[api_key.key_hash] = api_key

        logger.info(f"API key created: {name}")

        return api_key

    def get_key(self, key_value: str) -> Optional[APIKey]:
        """Get API key by value."""
        key_hash = APIKey._hash_key(key_value)
        return self.keys.get(key_hash)

    def validate_key(
        self, key_value: str, required_scope: Optional[str] = None
    ) -> APIKey:
        """
        Validate API key and check scope.

        Args:
            key_value: API key value
            required_scope: Required scope

        Returns:
            Valid API key

        Raises:
            HTTPException: If key is invalid
        """
        # Get key
        api_key = self.get_key(key_value)

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
            )

        # Check validity
        if not api_key.is_valid():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="API key expired"
            )

        # Check scope if required
        if required_scope and not api_key.has_scope(required_scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key lacks required scope: {required_scope}",
            )

        # Record usage
        api_key.record_usage()

        return api_key

    def revoke_key(self, key_value: str) -> bool:
        """Revoke an API key."""
        key_hash = APIKey._hash_key(key_value)
        if key_hash in self.keys:
            del self.keys[key_hash]
            logger.info(f"API key revoked: {key_hash[:8]}...")
            return True
        return False

    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without key values)."""
        return [
            {
                "name": key.name,
                "scopes": key.scopes,
                "created_at": key.created_at.isoformat(),
                "last_used": key.last_used.isoformat() if key.last_used else None,
                "usage_count": key.usage_count,
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "is_valid": key.is_valid(),
            }
            for key in self.keys.values()
        ]


# Global API key manager
api_key_manager = APIKeyManager()


# FastAPI dependencies
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def get_api_key(
    header_key: Optional[str] = Depends(api_key_header),
    query_key: Optional[str] = Depends(api_key_query),
) -> str:
    """Get API key from header or query parameter."""
    api_key = header_key or query_key

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required"
        )

    return api_key


async def require_api_key(
    api_key: str = Depends(get_api_key), required_scope: Optional[str] = None
) -> APIKey:
    """Require valid API key with optional scope."""
    return api_key_manager.validate_key(api_key, required_scope)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""

    def __init__(
        self,
        app,
        exclude_paths: Optional[Set[str]] = None,
        optional_paths: Optional[Set[str]] = None,
    ):
        """Initialize API key auth middleware."""
        super().__init__(app)

        # Paths that don't require authentication
        self.exclude_paths = exclude_paths or {
            "/",
            "/api/health",
            "/api/docs",
            "/api/openapi.json",
            "/api/redoc",
        }

        # Paths where auth is optional
        self.optional_paths = optional_paths or {"/api/websocket"}

    async def dispatch(self, request: Request, call_next):
        """Apply API key authentication."""
        path = request.url.path

        # Skip excluded paths
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)

        # Check if authentication is required
        is_optional = any(path.startswith(optional) for optional in self.optional_paths)

        # Get API key
        api_key = None

        # Check header
        api_key = request.headers.get("X-API-Key")

        # Check query parameter
        if not api_key:
            api_key = request.query_params.get("api_key")

        # Check authorization header
        if not api_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]

        # Validate if we have a key or if it's required
        if api_key:
            try:
                validated_key = api_key_manager.validate_key(api_key)

                # Add to request state
                request.state.api_key = validated_key

                # Add custom rate limit if specified
                if validated_key.rate_limit:
                    request.state.custom_rate_limit = validated_key.rate_limit

            except HTTPException:
                if not is_optional:
                    raise
        elif not is_optional and config.require_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Process request
        response = await call_next(request)

        return response


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for JWT authentication (alternative to API keys)."""

    def __init__(self, app, secret_key: Optional[str] = None):
        """Initialize JWT auth middleware."""
        super().__init__(app)
        self.secret_key = secret_key or config.secret_key
        self.algorithm = "HS256"

    async def dispatch(self, request: Request, call_next):
        """Apply JWT authentication."""
        # Get token from header
        auth_header = request.headers.get("Authorization", "")

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

            try:
                # Decode token
                payload = jwt.decode(
                    token, self.secret_key, algorithms=[self.algorithm]
                )

                # Add to request state
                request.state.jwt_payload = payload
                request.state.user_id = payload.get("sub")

            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
                )
            except jwt.InvalidTokenError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
                )

        return await call_next(request)
