"""
WebSocket Authentication Middleware

Provides authentication and authorization for WebSocket connections.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
from fastapi import HTTPException, WebSocket, status

from pdf_splitter.api.config import config
from pdf_splitter.api.services.session_service import SessionService

logger = logging.getLogger(__name__)


class WebSocketAuthMiddleware:
    """Middleware for WebSocket authentication."""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        token_expiry: int = 3600,  # 1 hour
    ):
        self.secret_key = secret_key or config.secret_key or "dev-secret-key"
        self.algorithm = algorithm
        self.token_expiry = token_expiry
        self.session_service = SessionService()

    def generate_token(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a JWT token for WebSocket authentication.

        Args:
            session_id: Session ID to authorize
            user_id: Optional user identifier
            metadata: Optional additional metadata

        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        payload = {
            "session_id": session_id,
            "user_id": user_id,
            "metadata": metadata or {},
            "iat": now,
            "exp": now + timedelta(seconds=self.token_expiry),
            "type": "websocket_auth",
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded token payload

        Raises:
            ValueError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Verify token type
            if payload.get("type") != "websocket_auth":
                raise ValueError("Invalid token type")

            return payload

        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {str(e)}")

    async def authenticate_websocket(
        self, websocket: WebSocket, session_id: str, token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Authenticate a WebSocket connection.

        Args:
            websocket: WebSocket connection
            session_id: Requested session ID
            token: Optional authentication token

        Returns:
            Authentication context

        Raises:
            HTTPException: If authentication fails
        """
        # If no token provided, check if authentication is required
        if not token and config.require_websocket_auth:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        auth_context = {
            "session_id": session_id,
            "authenticated": False,
            "user_id": None,
            "metadata": {},
        }

        # If token provided, verify it
        if token:
            try:
                payload = self.verify_token(token)

                # Verify session ID matches
                if payload.get("session_id") != session_id:
                    raise ValueError("Session ID mismatch")

                auth_context.update(
                    {
                        "authenticated": True,
                        "user_id": payload.get("user_id"),
                        "metadata": payload.get("metadata", {}),
                    }
                )

                logger.info(
                    f"WebSocket authenticated: session={session_id}, "
                    f"user={auth_context['user_id']}"
                )

            except ValueError as e:
                logger.warning(f"WebSocket auth failed: {e}")
                if config.require_websocket_auth:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e)
                    )

        # Verify session exists and is accessible
        try:
            session = self.session_service.get_session_details(session_id)

            # Check if session is active
            if session["status"] == "expired":
                raise HTTPException(
                    status_code=status.HTTP_410_GONE, detail="Session has expired"
                )

            auth_context["session_details"] = session

        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Invalid session"
            )

        return auth_context

    def create_session_token(
        self, session_id: str, duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a session-specific token for WebSocket access.

        Args:
            session_id: Session ID
            duration: Optional token duration in seconds

        Returns:
            Token information including token string and expiry
        """
        # Use custom duration or default
        if duration:
            self.token_expiry = duration

        token = self.generate_token(session_id)

        return {
            "token": token,
            "expires_in": self.token_expiry,
            "expires_at": (
                datetime.utcnow() + timedelta(seconds=self.token_expiry)
            ).isoformat(),
            "session_id": session_id,
            "websocket_url": f"{config.websocket_url}/ws/enhanced/{session_id}",
        }


# Global auth middleware instance
websocket_auth = WebSocketAuthMiddleware()


async def require_websocket_auth(
    websocket: WebSocket, session_id: str, token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Dependency to require WebSocket authentication.

    Use in WebSocket endpoints:
    ```python
    @router.websocket("/ws/{session_id}")
    async def websocket_endpoint(
        websocket: WebSocket,
        session_id: str,
        auth: dict = Depends(require_websocket_auth)
    ):
        # auth contains authentication context
    ```
    """
    return await websocket_auth.authenticate_websocket(websocket, session_id, token)
