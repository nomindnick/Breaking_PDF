"""
API Middleware Package.

Provides comprehensive middleware for the PDF Splitter API.
"""
import logging
import time
import uuid

from fastapi import Request

# Legacy imports for backward compatibility
from starlette.middleware.base import BaseHTTPMiddleware

from pdf_splitter.api.middleware.auth import (
    APIKeyAuthMiddleware,
    JWTAuthMiddleware,
    api_key_manager,
    get_api_key,
    require_api_key,
)
from pdf_splitter.api.middleware.error_handler import (
    EnhancedErrorHandlingMiddleware,
    ErrorRecoveryMiddleware,
    ErrorResponse,
)
from pdf_splitter.api.middleware.logging import (
    AuditLoggingMiddleware,
    MetricsExportMiddleware,
    PerformanceMonitoringMiddleware,
    StructuredLoggingMiddleware,
)
from pdf_splitter.api.middleware.rate_limiter import (
    AdaptiveRateLimitMiddleware,
    EnhancedRateLimitMiddleware,
    FixedWindowStrategy,
    RateLimitStrategy,
    SlidingWindowStrategy,
    TokenBucketStrategy,
)
from pdf_splitter.api.middleware.validation import (
    ContentSecurityMiddleware,
    InputSanitizationMiddleware,
    RequestValidationMiddleware,
)
from pdf_splitter.api.middleware.websocket_auth import (
    WebSocketAuthMiddleware,
    require_websocket_auth,
    websocket_auth,
)

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Legacy error handling middleware for backward compatibility."""

    async def dispatch(self, request: Request, call_next):
        """Use enhanced error handling."""
        middleware = EnhancedErrorHandlingMiddleware(self.app)
        return await middleware.dispatch(request, call_next)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Legacy request logging middleware for backward compatibility."""

    async def dispatch(self, request: Request, call_next):
        """Log request information."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()

        logger.info(f"Request {request_id}: {request.method} {request.url.path}")

        response = await call_next(request)

        duration = time.time() - start_time
        logger.info(
            f"Request {request_id} completed: "
            f"status={response.status_code} duration={duration:.3f}s"
        )

        response.headers["X-Request-ID"] = request_id

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Legacy rate limit middleware for backward compatibility."""

    def __init__(self, app, requests_per_minute: int = 60):
        """Initialize rate limit middleware."""
        super().__init__(app)
        self.requests_per_minute = requests_per_minute

    async def dispatch(self, request: Request, call_next):
        """Use enhanced rate limiting."""
        strategy = SlidingWindowStrategy(requests=self.requests_per_minute, window=60)
        middleware = EnhancedRateLimitMiddleware(self.app, strategy=strategy)
        return await middleware.dispatch(request, call_next)


__all__ = [
    # Enhanced middleware
    "EnhancedErrorHandlingMiddleware",
    "ErrorRecoveryMiddleware",
    "RequestValidationMiddleware",
    "InputSanitizationMiddleware",
    "ContentSecurityMiddleware",
    "EnhancedRateLimitMiddleware",
    "AdaptiveRateLimitMiddleware",
    "APIKeyAuthMiddleware",
    "JWTAuthMiddleware",
    "StructuredLoggingMiddleware",
    "PerformanceMonitoringMiddleware",
    "AuditLoggingMiddleware",
    "MetricsExportMiddleware",
    "WebSocketAuthMiddleware",
    # Utilities
    "api_key_manager",
    "require_api_key",
    "get_api_key",
    "websocket_auth",
    "require_websocket_auth",
    # Strategies
    "RateLimitStrategy",
    "FixedWindowStrategy",
    "SlidingWindowStrategy",
    "TokenBucketStrategy",
    # Legacy middleware
    "ErrorHandlingMiddleware",
    "RequestLoggingMiddleware",
    "RateLimitMiddleware",
]
