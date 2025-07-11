"""API-specific exceptions for the PDF Splitter application.

This module defines custom exceptions for handling various error scenarios
in the API layer.
"""

from typing import Optional


class APIError(Exception):
    """Base exception for all API-related errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """Initialize API error.

        Args:
            message: Human-readable error message
            status_code: HTTP status code
            error_code: Machine-readable error code for client handling
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class UploadError(APIError):
    """Raised when file upload fails."""

    def __init__(self, message: str, details: Optional[dict] = None):
        """Initialize upload error."""
        super().__init__(message, status_code=400, details=details)


class ValidationError(APIError):
    """Raised when request validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        """Initialize validation error."""
        details = {"field": field} if field else {}
        super().__init__(message, status_code=422, details=details)


class SessionNotFoundError(APIError):
    """Raised when requested session doesn't exist."""

    def __init__(self, session_id: str):
        """Initialize session not found error."""
        super().__init__(
            f"Session {session_id} not found",
            status_code=404,
            details={"session_id": session_id},
        )


class SessionExpiredError(APIError):
    """Raised when session has expired."""

    def __init__(self, session_id: str):
        """Initialize session expired error."""
        super().__init__(
            f"Session {session_id} has expired",
            status_code=410,
            details={"session_id": session_id},
        )


class SessionStateError(APIError):
    """Raised when session is in invalid state for requested operation."""

    def __init__(self, session_id: str, current_state: str, required_state: str):
        """Initialize session state error."""
        super().__init__(
            f"Session {session_id} is in state '{current_state}', "
            f"but '{required_state}' is required",
            status_code=409,
            details={
                "session_id": session_id,
                "current_state": current_state,
                "required_state": required_state,
            },
        )


class DetectionError(APIError):
    """Raised when boundary detection fails."""

    def __init__(self, message: str, session_id: Optional[str] = None):
        """Initialize detection error."""
        details = {"session_id": session_id} if session_id else {}
        super().__init__(
            message, status_code=500, error_code="DETECTION_FAILED", details=details
        )


class SplitError(APIError):
    """Raised when PDF splitting fails."""

    def __init__(self, message: str, session_id: Optional[str] = None):
        """Initialize split error."""
        details = {"session_id": session_id} if session_id else {}
        super().__init__(
            message, status_code=500, error_code="SPLIT_FAILED", details=details
        )


class ResourceNotFoundError(APIError):
    """Raised when requested resource doesn't exist."""

    def __init__(self, resource_type: str, resource_id: str):
        """Initialize resource not found error."""
        super().__init__(
            f"{resource_type} '{resource_id}' not found",
            status_code=404,
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class FileSizeError(APIError):
    """Raised when uploaded file exceeds size limits."""

    def __init__(self, file_size: int, max_size: int):
        """Initialize file size error."""
        super().__init__(
            f"File size {file_size} bytes exceeds maximum allowed size of {max_size} bytes",
            status_code=413,
            details={"file_size": file_size, "max_size": max_size},
        )


class FileTypeError(APIError):
    """Raised when uploaded file has invalid type."""

    def __init__(self, file_type: str, allowed_types: list[str]):
        """Initialize file type error."""
        super().__init__(
            f"File type '{file_type}' is not allowed. Allowed types: {', '.join(allowed_types)}",
            status_code=415,
            details={"file_type": file_type, "allowed_types": allowed_types},
        )


class ConcurrentOperationError(APIError):
    """Raised when concurrent operations conflict."""

    def __init__(self, operation: str, session_id: str):
        """Initialize concurrent operation error."""
        super().__init__(
            f"Another {operation} operation is already in progress for session {session_id}",
            status_code=409,
            details={"operation": operation, "session_id": session_id},
        )


class QuotaExceededError(APIError):
    """Raised when user exceeds usage quotas."""

    def __init__(self, quota_type: str, limit: int, current: int):
        """Initialize quota exceeded error."""
        super().__init__(
            f"{quota_type} quota exceeded. Limit: {limit}, Current: {current}",
            status_code=429,
            details={"quota_type": quota_type, "limit": limit, "current": current},
        )
