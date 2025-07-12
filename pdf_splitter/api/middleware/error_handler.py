"""
Enhanced Error Handling Middleware.

Provides comprehensive error handling with recovery, logging, and client-friendly responses.
"""
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Union

from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from pdf_splitter.api.config import config
from pdf_splitter.api.utils.exceptions import (
    APIException,
    FileNotFoundError,
    ProcessingError,
    SecurityError,
    SessionNotFoundError,
)
from pdf_splitter.api.utils.exceptions import ValidationError as CustomValidationError

logger = logging.getLogger(__name__)


class ErrorResponse:
    """Standardized error response format."""

    def __init__(
        self,
        error_type: str,
        message: str,
        status_code: int,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        self.error_type = error_type
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.request_id = request_id
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        response = {
            "error": {
                "type": self.error_type,
                "message": self.message,
                "timestamp": self.timestamp.isoformat(),
            }
        }

        if self.request_id:
            response["error"]["request_id"] = self.request_id

        if self.details:
            response["error"]["details"] = self.details

        # Add helpful information based on error type
        if self.status_code == 404:
            response["error"]["help"] = "Check the resource ID and ensure it exists"
        elif self.status_code == 401:
            response["error"][
                "help"
            ] = "Ensure you have valid authentication credentials"
        elif self.status_code == 429:
            response["error"][
                "help"
            ] = "You are being rate limited. Please slow down your requests"
        elif self.status_code >= 500:
            response["error"][
                "help"
            ] = "This is a server error. Please try again later or contact support"

        return response


class EnhancedErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for comprehensive error handling."""

    def __init__(self, app, enable_debug: bool = None):
        super().__init__(app)
        self.enable_debug = enable_debug if enable_debug is not None else config.debug
        self.error_handlers = self._setup_error_handlers()

    def _setup_error_handlers(self) -> Dict[type, callable]:
        """Set up specific error handlers for different exception types."""
        return {
            # Custom exceptions
            FileNotFoundError: self._handle_not_found,
            SessionNotFoundError: self._handle_not_found,
            SecurityError: self._handle_security_error,
            ProcessingError: self._handle_processing_error,
            CustomValidationError: self._handle_validation_error,
            # Pydantic validation
            ValidationError: self._handle_pydantic_validation_error,
            # Standard exceptions
            ValueError: self._handle_value_error,
            KeyError: self._handle_key_error,
            AttributeError: self._handle_attribute_error,
            TypeError: self._handle_type_error,
            # File operations
            FileNotFoundError: self._handle_file_not_found,
            PermissionError: self._handle_permission_error,
            # Network errors
            ConnectionError: self._handle_connection_error,
            TimeoutError: self._handle_timeout_error,
        }

    async def dispatch(self, request: Request, call_next):
        """Process request with error handling."""
        request_id = getattr(request.state, "request_id", None)

        try:
            response = await call_next(request)
            return response

        except Exception as exc:
            # Log the error
            self._log_error(exc, request, request_id)

            # Get specific handler or use generic
            handler = self.error_handlers.get(type(exc), self._handle_generic_error)
            error_response = handler(exc, request_id)

            # Return JSON error response
            return JSONResponse(
                status_code=error_response.status_code, content=error_response.to_dict()
            )

    def _log_error(self, exc: Exception, request: Request, request_id: Optional[str]):
        """Log error with context."""
        error_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

        if isinstance(exc, APIException) and exc.status_code < 500:
            # Client errors - log as warning
            logger.warning(f"Client error: {error_data}")
        else:
            # Server errors - log as error with traceback
            logger.error(f"Server error: {error_data}", exc_info=True)

    def _handle_not_found(
        self, exc: Exception, request_id: Optional[str]
    ) -> ErrorResponse:
        """Handle not found errors."""
        return ErrorResponse(
            error_type="not_found",
            message=str(exc),
            status_code=status.HTTP_404_NOT_FOUND,
            request_id=request_id,
        )

    def _handle_security_error(
        self, exc: SecurityError, request_id: Optional[str]
    ) -> ErrorResponse:
        """Handle security errors."""
        return ErrorResponse(
            error_type="security_error",
            message=str(exc),
            status_code=status.HTTP_403_FORBIDDEN,
            details={"security_violation": True},
            request_id=request_id,
        )

    def _handle_processing_error(
        self, exc: ProcessingError, request_id: Optional[str]
    ) -> ErrorResponse:
        """Handle processing errors."""
        return ErrorResponse(
            error_type="processing_error",
            message=str(exc),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            request_id=request_id,
        )

    def _handle_validation_error(
        self,
        exc: Union[CustomValidationError, ValidationError],
        request_id: Optional[str],
    ) -> ErrorResponse:
        """Handle validation errors."""
        details = {}

        if isinstance(exc, ValidationError):
            # Pydantic validation error
            details["validation_errors"] = [
                {
                    "field": ".".join(str(x) for x in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"],
                }
                for error in exc.errors()
            ]

        return ErrorResponse(
            error_type="validation_error",
            message="Invalid request data",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
            request_id=request_id,
        )

    def _handle_pydantic_validation_error(
        self, exc: ValidationError, request_id: Optional[str]
    ) -> ErrorResponse:
        """Handle Pydantic validation errors."""
        return self._handle_validation_error(exc, request_id)

    def _handle_value_error(
        self, exc: ValueError, request_id: Optional[str]
    ) -> ErrorResponse:
        """Handle value errors."""
        return ErrorResponse(
            error_type="invalid_value",
            message=str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
            request_id=request_id,
        )

    def _handle_key_error(
        self, exc: KeyError, request_id: Optional[str]
    ) -> ErrorResponse:
        """Handle missing key errors."""
        return ErrorResponse(
            error_type="missing_field",
            message=f"Required field missing: {str(exc)}",
            status_code=status.HTTP_400_BAD_REQUEST,
            request_id=request_id,
        )

    def _handle_attribute_error(
        self, exc: AttributeError, request_id: Optional[str]
    ) -> ErrorResponse:
        """Handle attribute errors."""
        return ErrorResponse(
            error_type="invalid_attribute",
            message="Invalid request structure",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"error": str(exc)} if self.enable_debug else {},
            request_id=request_id,
        )

    def _handle_type_error(
        self, exc: TypeError, request_id: Optional[str]
    ) -> ErrorResponse:
        """Handle type errors."""
        return ErrorResponse(
            error_type="type_error",
            message="Invalid data type provided",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"error": str(exc)} if self.enable_debug else {},
            request_id=request_id,
        )

    def _handle_file_not_found(
        self, exc: FileNotFoundError, request_id: Optional[str]
    ) -> ErrorResponse:
        """Handle file not found errors."""
        return ErrorResponse(
            error_type="file_not_found",
            message="The requested file does not exist",
            status_code=status.HTTP_404_NOT_FOUND,
            request_id=request_id,
        )

    def _handle_permission_error(
        self, exc: PermissionError, request_id: Optional[str]
    ) -> ErrorResponse:
        """Handle permission errors."""
        return ErrorResponse(
            error_type="permission_denied",
            message="Insufficient permissions to access this resource",
            status_code=status.HTTP_403_FORBIDDEN,
            request_id=request_id,
        )

    def _handle_connection_error(
        self, exc: ConnectionError, request_id: Optional[str]
    ) -> ErrorResponse:
        """Handle connection errors."""
        return ErrorResponse(
            error_type="connection_error",
            message="Failed to connect to external service",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            request_id=request_id,
        )

    def _handle_timeout_error(
        self, exc: TimeoutError, request_id: Optional[str]
    ) -> ErrorResponse:
        """Handle timeout errors."""
        return ErrorResponse(
            error_type="timeout",
            message="Operation timed out",
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            request_id=request_id,
        )

    def _handle_generic_error(
        self, exc: Exception, request_id: Optional[str]
    ) -> ErrorResponse:
        """Handle any other errors."""
        # Log full traceback for unexpected errors
        logger.error(
            f"Unexpected error: {type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}"
        )

        # Prepare error details
        details = {}
        if self.enable_debug:
            details = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc().split("\n"),
            }

        return ErrorResponse(
            error_type="internal_error",
            message="An unexpected error occurred",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
            request_id=request_id,
        )


class ErrorRecoveryMiddleware(BaseHTTPMiddleware):
    """Middleware for error recovery and retry logic."""

    def __init__(self, app, max_retries: int = 3, retry_delay: float = 1.0):
        super().__init__(app)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retryable_errors = {ConnectionError, TimeoutError, ProcessingError}

    async def dispatch(self, request: Request, call_next):
        """Process request with retry logic for certain errors."""
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                response = await call_next(request)
                return response

            except Exception as exc:
                last_error = exc

                # Check if error is retryable
                if not any(
                    isinstance(exc, error_type) for error_type in self.retryable_errors
                ):
                    raise

                # Check if we've exhausted retries
                if retries >= self.max_retries:
                    logger.error(
                        f"Max retries ({self.max_retries}) exhausted for request"
                    )
                    raise

                # Log retry attempt
                retries += 1
                logger.warning(
                    f"Retry {retries}/{self.max_retries} after {type(exc).__name__}: {str(exc)}"
                )

                # Wait before retry
                await asyncio.sleep(self.retry_delay * retries)

        # Should not reach here, but just in case
        if last_error:
            raise last_error
