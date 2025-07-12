"""
Request Validation and Sanitization Middleware

Provides comprehensive request validation, input sanitization, and security checks.
"""
import html
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

from fastapi import HTTPException, Request, status
from starlette.datastructures import Headers, QueryParams
from starlette.middleware.base import BaseHTTPMiddleware

from pdf_splitter.api.config import config

logger = logging.getLogger(__name__)


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request validation and sanitization."""

    def __init__(self, app):
        super().__init__(app)

        # Validation rules
        self.max_request_size = config.max_upload_size
        self.allowed_methods = {
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "OPTIONS",
            "HEAD",
        }
        self.allowed_content_types = {
            "application/json",
            "application/pdf",
            "multipart/form-data",
            "application/x-www-form-urlencoded",
            "text/plain",
        }

        # Security patterns
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER)\b)",
            r"(-{2}|\/\*|\*\/)",  # SQL comments
            r"(\bOR\b\s*\d+\s*=\s*\d+)",  # OR 1=1
            r"(\bAND\b\s*\d+\s*=\s*\d+)",  # AND 1=1
        ]

        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",  # Event handlers
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
        ]

        self.path_traversal_patterns = [
            r"\.\./",  # ../
            r"\.\.",  # ..
            r"%2e%2e",  # URL encoded ..
            r"%252e%252e",  # Double encoded
            r"\.\.\\",  # Windows style
        ]

        # Headers to remove for security
        self.headers_to_remove = {"x-powered-by", "server", "x-aspnet-version"}

    async def dispatch(self, request: Request, call_next):
        """Validate and sanitize request."""
        try:
            # Validate request method
            self._validate_method(request)

            # Validate content type
            self._validate_content_type(request)

            # Validate request size
            await self._validate_request_size(request)

            # Validate and sanitize path
            self._validate_path(request)

            # Validate and sanitize query parameters
            self._validate_query_params(request)

            # Validate and sanitize headers
            self._validate_headers(request)

            # Process request
            response = await call_next(request)

            # Sanitize response headers
            self._sanitize_response_headers(response)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request"
            )

    def _validate_method(self, request: Request):
        """Validate HTTP method."""
        if request.method not in self.allowed_methods:
            raise HTTPException(
                status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                detail=f"Method {request.method} not allowed",
            )

    def _validate_content_type(self, request: Request):
        """Validate content type."""
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "").split(";")[0].strip()

            if not content_type:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Content-Type header required",
                )

            # Allow any content type for file uploads
            if request.url.path.startswith("/api/upload"):
                return

            if content_type not in self.allowed_content_types:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Content type {content_type} not supported",
                )

    async def _validate_request_size(self, request: Request):
        """Validate request size."""
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Request size {size} exceeds maximum {self.max_request_size}",
                    )
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid Content-Length header",
                )

    def _validate_path(self, request: Request):
        """Validate and sanitize request path."""
        path = str(request.url.path)

        # Check for path traversal attempts
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                logger.warning(f"Path traversal attempt detected: {path}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path"
                )

        # Validate path characters
        if not re.match(r"^[a-zA-Z0-9/_\-\.]+$", path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid characters in path",
            )

    def _validate_query_params(self, request: Request):
        """Validate and sanitize query parameters."""
        for key, value in request.query_params.items():
            # Check parameter names
            if not re.match(r"^[a-zA-Z0-9_\-]+$", key):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid parameter name: {key}",
                )

            # Check for injection attempts
            self._check_injection_attempts(value, "query parameter")

    def _validate_headers(self, request: Request):
        """Validate and sanitize headers."""
        for key, value in request.headers.items():
            # Skip standard headers
            if key.lower() in [
                "host",
                "user-agent",
                "accept",
                "content-type",
                "content-length",
            ]:
                continue

            # Check header names
            if not re.match(r"^[a-zA-Z0-9\-]+$", key):
                logger.warning(f"Invalid header name: {key}")
                continue

            # Check for injection attempts in custom headers
            if key.lower().startswith("x-"):
                self._check_injection_attempts(value, f"header {key}")

    def _check_injection_attempts(self, value: str, context: str):
        """Check for SQL injection and XSS attempts."""
        # SQL injection
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"SQL injection attempt in {context}: {value}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid input detected",
                )

        # XSS
        for pattern in self.xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"XSS attempt in {context}: {value}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid input detected",
                )

    def _sanitize_response_headers(self, response):
        """Remove sensitive headers from response."""
        for header in self.headers_to_remove:
            if header in response.headers:
                del response.headers[header]

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"


class InputSanitizationMiddleware(BaseHTTPMiddleware):
    """Middleware for input sanitization."""

    def __init__(self, app):
        super().__init__(app)

        # Fields to skip sanitization (e.g., passwords, tokens)
        self.skip_fields = {"password", "token", "api_key", "secret"}

    async def dispatch(self, request: Request, call_next):
        """Sanitize request inputs."""
        # Only process JSON requests
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                # Get body
                body = await request.body()
                if body:
                    # Parse JSON
                    data = json.loads(body)

                    # Sanitize
                    sanitized_data = self._sanitize_data(data)

                    # Replace body with sanitized version
                    request._body = json.dumps(sanitized_data).encode()

            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON"
                )
            except Exception as e:
                logger.error(f"Sanitization error: {e}")

        return await call_next(request)

    def _sanitize_data(self, data: Any, path: str = "") -> Any:
        """Recursively sanitize data."""
        if isinstance(data, dict):
            return {
                key: self._sanitize_data(value, f"{path}.{key}")
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [
                self._sanitize_data(item, f"{path}[{i}]") for i, item in enumerate(data)
            ]
        elif isinstance(data, str):
            # Skip certain fields
            field_name = path.split(".")[-1] if path else ""
            if field_name.lower() in self.skip_fields:
                return data

            # Sanitize string
            return self._sanitize_string(data)
        else:
            return data

    def _sanitize_string(self, value: str) -> str:
        """Sanitize a string value."""
        # HTML escape
        value = html.escape(value)

        # Remove null bytes
        value = value.replace("\x00", "")

        # Trim whitespace
        value = value.strip()

        # Limit length
        max_length = 10000
        if len(value) > max_length:
            value = value[:max_length]

        return value


class ContentSecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for content security policies."""

    def __init__(self, app):
        super().__init__(app)

        # CSP directives
        self.csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Adjust based on needs
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: blob:",
            "font-src 'self'",
            "connect-src 'self' ws: wss:",
            "media-src 'none'",
            "object-src 'none'",
            "frame-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
        ]

    async def dispatch(self, request: Request, call_next):
        """Add content security headers."""
        response = await call_next(request)

        # Add CSP header
        csp = "; ".join(self.csp_directives)
        response.headers["Content-Security-Policy"] = csp

        # Add other security headers
        response.headers[
            "Strict-Transport-Security"
        ] = "max-age=31536000; includeSubDomains"
        response.headers[
            "Permissions-Policy"
        ] = "geolocation=(), microphone=(), camera=()"

        return response
