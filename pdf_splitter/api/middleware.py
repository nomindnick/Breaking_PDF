"""Custom middleware for the API.

This module provides middleware for request logging, error handling,
and other cross-cutting concerns.
"""

import time
import traceback
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from pdf_splitter.api.exceptions import APIError
from pdf_splitter.core.logging import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request details and response status.

        Args:
            request: FastAPI request object
            call_next: Next middleware in chain

        Returns:
            Response object
        """
        # Generate request ID
        request_id = str(uuid4())
        request.state.request_id = request_id

        # Log request
        start_time = time.time()
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        try:
            # Process request
            response = await call_next(request)

            # Log response
            duration = time.time() - start_time
            logger.info(
                "Request completed",
                request_id=request_id,
                status_code=response.status_code,
                duration=f"{duration:.3f}s",
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # Log error
            duration = time.time() - start_time
            logger.error(
                "Request failed",
                request_id=request_id,
                error=str(e),
                duration=f"{duration:.3f}s",
                traceback=traceback.format_exc(),
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling exceptions and converting them to JSON responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle exceptions and convert to appropriate responses.

        Args:
            request: FastAPI request object
            call_next: Next middleware in chain

        Returns:
            Response object
        """
        try:
            return await call_next(request)
        except APIError as e:
            # Handle custom API errors
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "success": False,
                    "error_code": e.error_code,
                    "message": e.message,
                    "details": e.details,
                    "request_id": getattr(request.state, "request_id", None),
                },
            )
        except Exception as e:
            # Handle unexpected errors
            logger.error(
                "Unhandled exception",
                error=str(e),
                traceback=traceback.format_exc(),
                request_id=getattr(request.state, "request_id", None),
            )

            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error_code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "details": {"error": str(e)} if logger.level == "DEBUG" else {},
                    "request_id": getattr(request.state, "request_id", None),
                },
            )


class CORSMiddleware(BaseHTTPMiddleware):
    """Enhanced CORS middleware with preflight handling."""

    def __init__(self, app, allowed_origins: list = None):
        """Initialize CORS middleware.

        Args:
            app: FastAPI application
            allowed_origins: List of allowed origins
        """
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle CORS headers.

        Args:
            request: FastAPI request object
            call_next: Next middleware in chain

        Returns:
            Response object
        """
        # Handle preflight requests
        if request.method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization",
                    "Access-Control-Max-Age": "86400",
                },
            )

        # Process request
        response = await call_next(request)

        # Add CORS headers
        origin = request.headers.get("origin")
        if origin in self.allowed_origins or "*" in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""

    def __init__(self, app, requests_per_minute: int = 60):
        """Initialize rate limit middleware.

        Args:
            app: FastAPI application
            requests_per_minute: Maximum requests per minute per IP
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}  # IP -> list of timestamps

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limits.

        Args:
            request: FastAPI request object
            call_next: Next middleware in chain

        Returns:
            Response object
        """
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Skip rate limiting for local development
        if client_ip in ["127.0.0.1", "localhost"]:
            return await call_next(request)

        # Check rate limit
        current_time = time.time()
        minute_ago = current_time - 60

        # Clean old entries and count recent requests
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []

        self.request_counts[client_ip] = [
            t for t in self.request_counts[client_ip] if t > minute_ago
        ]

        # Check if limit exceeded
        if len(self.request_counts[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute.",
                },
            )

        # Record request
        self.request_counts[client_ip].append(current_time)

        # Process request
        return await call_next(request)
