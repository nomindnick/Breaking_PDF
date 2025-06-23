"""Custom exception classes for PDF Splitter application."""

from typing import Any, Dict, Optional


class PDFSplitterError(Exception):
    """Base exception class for PDF Splitter application."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize PDFSplitterError with message and optional details."""
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class ConfigurationError(PDFSplitterError):
    """Raised when there's a configuration issue."""

    pass


class PDFProcessingError(PDFSplitterError):
    """Base class for PDF processing errors."""

    pass


class PDFReadError(PDFProcessingError):
    """Raised when a PDF file cannot be read or opened."""

    pass


class PDFValidationError(PDFProcessingError):
    """Raised when a PDF file fails validation."""

    pass


class PDFHandlerError(PDFProcessingError):
    """Raised when general PDF handler operations fail."""

    pass


class PDFRenderError(PDFProcessingError):
    """Raised when PDF page rendering fails."""

    pass


class PDFTextExtractionError(PDFProcessingError):
    """Raised when text extraction from PDF fails."""

    pass


class OCRError(PDFProcessingError):
    """Base class for OCR-related errors."""

    pass


class OCREngineError(OCRError):
    """Raised when an OCR engine fails to initialize or process."""

    pass


class OCRProcessingError(OCRError):
    """Raised when OCR processing fails for a specific page or document."""

    pass


class DetectionError(PDFSplitterError):
    """Base class for document boundary detection errors."""

    pass


class LLMError(DetectionError):
    """Raised when LLM-based detection fails."""

    pass


class InsufficientSignalsError(DetectionError):
    """Raised when not enough detection signals are available."""

    pass


class SplittingError(PDFSplitterError):
    """Base class for PDF splitting errors."""

    pass


class InvalidBoundaryError(SplittingError):
    """Raised when document boundaries are invalid."""

    pass


class FileSystemError(PDFSplitterError):
    """Base class for file system related errors."""

    pass


class InsufficientStorageError(FileSystemError):
    """Raised when there's not enough storage space."""

    pass


class PermissionError(FileSystemError):
    """Raised when there are permission issues with files or directories."""

    pass


class PDFCacheError(PDFHandlerError):
    """Raised when cache operations fail."""

    pass
