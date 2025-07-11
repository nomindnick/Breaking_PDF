"""Data models for the splitting module.

This module defines the core data structures used for representing
document segments, split proposals, and session management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pdf_splitter.detection.base_detector import BoundaryResult


@dataclass
class DocumentSegment:
    """Represents a proposed document split within a PDF.

    Attributes:
        start_page: Starting page number (0-indexed)
        end_page: Ending page number (inclusive, 0-indexed)
        document_type: Type of document (Email, Invoice, Plans, etc.)
        suggested_filename: AI-suggested filename for this segment
        confidence: Confidence score for this segment (0-1)
        summary: Optional summary of the document content
        metadata: Additional metadata (dates, subjects, etc.)
        segment_id: Unique identifier for this segment
        is_user_defined: Whether this segment was manually created by user
    """

    start_page: int
    end_page: int
    document_type: str
    suggested_filename: str
    confidence: float
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    segment_id: str = field(default_factory=lambda: str(uuid4()))
    is_user_defined: bool = False

    def __post_init__(self):
        """Validate segment data."""
        if self.start_page < 0:
            raise ValueError(f"start_page must be >= 0, got {self.start_page}")
        if self.end_page < self.start_page:
            raise ValueError(
                f"end_page ({self.end_page}) must be >= start_page ({self.start_page})"
            )
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"confidence must be between 0 and 1, got {self.confidence}"
            )

    @property
    def page_count(self) -> int:
        """Number of pages in this segment."""
        return self.end_page - self.start_page + 1

    @property
    def page_range(self) -> str:
        """Human-readable page range (1-indexed)."""
        if self.start_page == self.end_page:
            return f"Page {self.start_page + 1}"
        return f"Pages {self.start_page + 1}-{self.end_page + 1}"


@dataclass
class SplitProposal:
    """Complete split proposal for user review.

    Represents the full splitting plan for a PDF, including all segments
    and the original boundary detection results.
    """

    pdf_path: Path
    total_pages: int
    segments: List[DocumentSegment]
    detection_results: List[BoundaryResult]  # Original boundaries from detection
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None
    proposal_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        """Validate proposal data."""
        if not self.pdf_path.exists():
            raise ValueError(f"PDF file not found: {self.pdf_path}")
        if self.total_pages <= 0:
            raise ValueError(f"total_pages must be > 0, got {self.total_pages}")

        # Validate segments don't overlap and cover all pages
        self._validate_segments()

    def _validate_segments(self):
        """Ensure segments are valid and non-overlapping."""
        if not self.segments:
            return

        sorted_segments = sorted(self.segments, key=lambda s: s.start_page)

        for i, segment in enumerate(sorted_segments):
            # Check within bounds
            if segment.end_page >= self.total_pages:
                raise ValueError(
                    f"Segment {segment.segment_id} end_page ({segment.end_page}) "
                    f"exceeds total pages ({self.total_pages})"
                )

            # Check for overlaps
            if i > 0:
                prev_segment = sorted_segments[i - 1]
                if segment.start_page <= prev_segment.end_page:
                    raise ValueError(
                        f"Segments overlap: {prev_segment.segment_id} ends at "
                        f"{prev_segment.end_page}, {segment.segment_id} starts at "
                        f"{segment.start_page}"
                    )

    @property
    def segment_count(self) -> int:
        """Number of segments in the proposal."""
        return len(self.segments)

    def get_segment(self, segment_id: str) -> Optional[DocumentSegment]:
        """Get a segment by ID."""
        for segment in self.segments:
            if segment.segment_id == segment_id:
                return segment
        return None

    def add_segment(self, segment: DocumentSegment):
        """Add a new segment, maintaining sort order."""
        self.segments.append(segment)
        self.segments.sort(key=lambda s: s.start_page)
        self._validate_segments()
        self.modified_at = datetime.now()

    def remove_segment(self, segment_id: str) -> bool:
        """Remove a segment by ID. Returns True if removed."""
        original_count = len(self.segments)
        self.segments = [s for s in self.segments if s.segment_id != segment_id]
        if len(self.segments) < original_count:
            self.modified_at = datetime.now()
            return True
        return False

    def update_segment(self, segment_id: str, **kwargs) -> bool:
        """Update a segment's attributes. Returns True if updated."""
        segment = self.get_segment(segment_id)
        if not segment:
            return False

        # Update allowed fields
        allowed_fields = {
            "suggested_filename",
            "document_type",
            "summary",
            "metadata",
            "start_page",
            "end_page",
        }

        for field_name, value in kwargs.items():
            if field_name in allowed_fields and hasattr(segment, field_name):
                setattr(segment, field_name, value)

        self._validate_segments()
        self.modified_at = datetime.now()
        return True


@dataclass
class UserModification:
    """Tracks a user modification to the split proposal."""

    modification_type: str  # add, remove, update, rename
    segment_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitSession:
    """Manages stateful split operation.

    Tracks the full lifecycle of a split operation from initial
    proposal through user modifications to final execution.
    """

    session_id: str
    proposal: SplitProposal
    user_modifications: List[UserModification] = field(default_factory=list)
    status: str = "pending"  # pending, modified, confirmed, completed, cancelled
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    output_directory: Optional[Path] = None
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # For storing extra info like upload_id

    def add_modification(self, modification: UserModification):
        """Add a user modification to the session."""
        self.user_modifications.append(modification)
        self.status = "modified"
        self.updated_at = datetime.now()

    def confirm(self, output_directory: Path):
        """Mark session as confirmed and ready for execution."""
        self.status = "confirmed"
        self.output_directory = output_directory
        self.updated_at = datetime.now()

    def complete(self):
        """Mark session as completed."""
        self.status = "completed"
        self.updated_at = datetime.now()

    def cancel(self):
        """Mark session as cancelled."""
        self.status = "cancelled"
        self.updated_at = datetime.now()

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        if self.expires_at and datetime.now() > self.expires_at:
            return True
        return False

    @property
    def is_active(self) -> bool:
        """Check if session is still active."""
        return (
            self.status in ("pending", "modified", "confirmed") and not self.is_expired
        )


@dataclass
class SplitResult:
    """Result of a successful split operation."""

    session_id: str
    input_pdf: Path
    output_files: List[Path]
    segments: List[DocumentSegment]
    completed_at: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    @property
    def file_count(self) -> int:
        """Number of files created."""
        return len(self.output_files)
