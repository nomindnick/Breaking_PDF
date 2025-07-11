"""PDF splitting module for breaking PDFs into individual documents.

This module provides functionality for:
- Converting boundary detection results into document segments
- Suggesting filenames based on document content
- Splitting PDFs into individual files
- Managing stateful split operations with session tracking
- Generating previews of document segments
"""

from pdf_splitter.splitting.exceptions import (
    FilenameSuggestionError,
    InvalidSegmentError,
    InvalidSessionStateError,
    PDFSplitError,
    PreviewGenerationError,
    SessionExpiredError,
    SessionNotFoundError,
    SplittingError,
)
from pdf_splitter.splitting.models import (
    DocumentSegment,
    SplitProposal,
    SplitResult,
    SplitSession,
    UserModification,
)
from pdf_splitter.splitting.pdf_splitter import PDFSplitter
from pdf_splitter.splitting.session_manager import SplitSessionManager

__all__ = [
    # Core service classes
    "PDFSplitter",
    "SplitSessionManager",
    # Data models
    "DocumentSegment",
    "SplitProposal",
    "SplitResult",
    "SplitSession",
    "UserModification",
    # Exceptions
    "SplittingError",
    "InvalidSegmentError",
    "PDFSplitError",
    "SessionNotFoundError",
    "SessionExpiredError",
    "InvalidSessionStateError",
    "FilenameSuggestionError",
    "PreviewGenerationError",
]
