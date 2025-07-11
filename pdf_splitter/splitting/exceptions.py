"""Custom exceptions for the splitting module."""


class SplittingError(Exception):
    """Base exception for all splitting-related errors."""

    pass


class InvalidSegmentError(SplittingError):
    """Raised when a segment configuration is invalid."""

    pass


class PDFSplitError(SplittingError):
    """Raised when PDF splitting operation fails."""

    pass


class SessionNotFoundError(SplittingError):
    """Raised when a session ID is not found."""

    pass


class SessionExpiredError(SplittingError):
    """Raised when attempting to use an expired session."""

    pass


class InvalidSessionStateError(SplittingError):
    """Raised when session is in an invalid state for the requested operation."""

    pass


class FilenameSuggestionError(SplittingError):
    """Raised when filename suggestion fails."""

    pass


class PreviewGenerationError(SplittingError):
    """Raised when preview generation fails."""

    pass
