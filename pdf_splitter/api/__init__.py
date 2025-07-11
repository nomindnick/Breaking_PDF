"""API module for PDF Splitter application.

This module provides REST API endpoints and WebSocket support for the PDF
splitting application, including file upload, boundary detection, session
management, and real-time progress updates.
"""

from pdf_splitter.api.exceptions import (
    APIError,
    DetectionError,
    SessionExpiredError,
    SessionNotFoundError,
    SplitError,
    UploadError,
    ValidationError,
)
from pdf_splitter.api.router import api_router

__all__ = [
    "api_router",
    "APIError",
    "UploadError",
    "ValidationError",
    "SessionNotFoundError",
    "SessionExpiredError",
    "DetectionError",
    "SplitError",
]
