"""
Custom API Exceptions

Defines custom exceptions for better error handling and user feedback.
"""
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException, status


class APIException(HTTPException):
    """Base API Exception with consistent error format."""

    def __init__(
        self,
        status_code: int,
        error_type: str,
        message: str,
        detail: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status_code,
            detail={
                "error": {
                    "type": error_type,
                    "message": message,
                    "detail": detail or {},
                    "timestamp": datetime.utcnow().isoformat(),
                }
            },
        )


class FileUploadError(APIException):
    """Raised when file upload fails."""

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_type="file_upload_error",
            message=message,
            detail=detail,
        )


class FileSizeError(APIException):
    """Raised when uploaded file exceeds size limit."""

    def __init__(self, file_size: int, max_size: int):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            error_type="file_size_error",
            message=f"File size {file_size} bytes exceeds maximum allowed size of {max_size} bytes",
            detail={"file_size": file_size, "max_size": max_size},
        )


class FileTypeError(APIException):
    """Raised when uploaded file has invalid type."""

    def __init__(self, file_type: str, allowed_types: list[str]):
        super().__init__(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            error_type="file_type_error",
            message=f"File type '{file_type}' is not allowed",
            detail={"file_type": file_type, "allowed_types": allowed_types},
        )


class FileNotFoundError(APIException):
    """Raised when requested file is not found."""

    def __init__(self, file_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_type="file_not_found",
            message=f"File with ID '{file_id}' not found",
            detail={"file_id": file_id},
        )


class SessionNotFoundError(APIException):
    """Raised when requested session is not found."""

    def __init__(self, session_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_type="session_not_found",
            message=f"Session with ID '{session_id}' not found",
            detail={"session_id": session_id},
        )


class ProcessingError(APIException):
    """Raised when PDF processing fails."""

    def __init__(self, message: str, stage: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_type="processing_error",
            message=message,
            detail={"stage": stage} if stage else {},
        )


class AuthenticationError(APIException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_type="authentication_error",
            message=message,
            detail={},
        )


class SecurityError(APIException):
    """Raised when security validation fails."""

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error_type="security_error",
            message=message,
            detail=detail or {},
        )


class ValidationError(APIException):
    """Raised when request validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_type="validation_error",
            message=message,
            detail={"field": field, **(detail or {})},
        )
