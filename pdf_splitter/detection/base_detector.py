"""
Base detector interface and data models for document boundary detection.

This module provides the abstract base class for all detection strategies
and defines the common data structures used throughout the detection system.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pdf_splitter.core.config import PDFConfig


class BoundaryType(Enum):
    """Types of document boundaries that can be detected."""

    DOCUMENT_START = "document_start"
    DOCUMENT_END = "document_end"
    SECTION_BREAK = "section_break"
    PAGE_CONTINUATION = "page_continuation"
    UNCERTAIN = "uncertain"


class DetectorType(Enum):
    """Types of detectors available in the system."""

    LLM = "llm"
    VISUAL = "visual"
    HEURISTIC = "heuristic"
    EMBEDDINGS = "embeddings"
    COMBINED = "combined"


@dataclass
class ProcessedPage:
    """
    Represents a page that has been processed by the preprocessing module.

    This is the input format that detectors receive from the preprocessing pipeline.
    """

    page_number: int
    text: str
    ocr_confidence: Optional[float] = None
    page_type: str = "unknown"  # From PDFHandler: SEARCHABLE, IMAGE_BASED, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Visual information from OCR/text extraction
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    layout_info: Dict[str, Any] = field(default_factory=dict)

    # Extracted features
    has_header: bool = False
    has_footer: bool = False
    has_page_number: bool = False
    dominant_font_size: Optional[float] = None
    
    # Rendered image for visual detection (optional)
    rendered_image: Optional[bytes] = None  # PNG/JPEG bytes of the rendered page

    @property
    def is_empty(self) -> bool:
        """Check if the page has meaningful content."""
        return len(self.text.strip()) < 10


@dataclass
class BoundaryResult:
    """
    Represents a detected document boundary.

    This is the output format that detectors produce.
    """

    page_number: int  # Page where boundary occurs
    boundary_type: BoundaryType
    confidence: float  # 0.0 to 1.0
    detector_type: DetectorType

    # Supporting evidence
    evidence: Dict[str, Any] = field(default_factory=dict)
    reasoning: Optional[str] = None

    # For boundaries between pages
    is_between_pages: bool = False
    next_page_number: Optional[int] = None

    timestamp: datetime = field(default_factory=datetime.now)
    
    # Track original confidence before any merging/boosting
    original_confidence: Optional[float] = None

    def __post_init__(self):
        """Validate confidence is in valid range and set original confidence."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        
        # Set original confidence if not already set
        if self.original_confidence is None:
            self.original_confidence = self.confidence


@dataclass
class DetectionContext:
    """
    Context information for detection operations.

    Provides configuration and state that detectors may need.
    """

    config: PDFConfig
    total_pages: int
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    previous_boundaries: List[BoundaryResult] = field(default_factory=list)

    # Performance tracking
    start_time: datetime = field(default_factory=datetime.now)
    pages_processed: int = 0

    def update_progress(self, pages: int) -> None:
        """Update processing progress."""
        self.pages_processed += pages


class BaseDetector(ABC):
    """
    Abstract base class for all document boundary detectors.

    Defines the interface that all detection strategies must implement.
    """

    def __init__(self, config: Optional[PDFConfig] = None):
        """
        Initialize the detector with optional configuration.

        Args:
            config: PDF processing configuration
        """
        self.config = config or PDFConfig()
        self._last_detection_time: Optional[float] = None
        self._detection_history: List[BoundaryResult] = []

    @abstractmethod
    def detect_boundaries(
        self, pages: List[ProcessedPage], context: Optional[DetectionContext] = None
    ) -> List[BoundaryResult]:
        """
        Detect document boundaries in a list of processed pages.

        Args:
            pages: List of processed pages to analyze
            context: Optional context information for detection

        Returns:
            List of detected boundaries with confidence scores
        """
        pass

    @abstractmethod
    def get_detector_type(self) -> DetectorType:
        """Return the type of this detector."""
        pass

    def get_confidence_threshold(self) -> float:
        """
        Get the minimum confidence threshold for this detector.

        Returns:
            Confidence threshold between 0.0 and 1.0
        """
        return 0.7  # Default threshold, can be overridden

    def filter_by_confidence(
        self, results: List[BoundaryResult], threshold: Optional[float] = None
    ) -> List[BoundaryResult]:
        """
        Filter detection results by confidence threshold.

        Args:
            results: List of boundary results
            threshold: Optional custom threshold (uses default if not provided)

        Returns:
            Filtered list of results above threshold
        """
        threshold = threshold or self.get_confidence_threshold()
        return [r for r in results if r.confidence >= threshold]

    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about detection performance.

        Returns:
            Dictionary with detection statistics
        """
        if not self._detection_history:
            return {"detections": 0, "avg_confidence": 0.0}

        confidences = [r.confidence for r in self._detection_history]
        return {
            "detections": len(self._detection_history),
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "last_detection_time": self._last_detection_time,
        }

    def extract_context_window(
        self, pages: List[ProcessedPage], center_idx: int, window_size: int = 3
    ) -> Tuple[List[ProcessedPage], int]:
        """
        Extract a context window around a specific page.

        Args:
            pages: List of all pages
            center_idx: Index of the center page
            window_size: Total size of the window (must be odd)

        Returns:
            Tuple of (window pages, index of center in window)
        """
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd")

        half_window = window_size // 2
        start_idx = max(0, center_idx - half_window)
        end_idx = min(len(pages), center_idx + half_window + 1)

        window = pages[start_idx:end_idx]
        center_in_window = center_idx - start_idx

        return window, center_in_window

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity between two strings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0

        # Simple word-based Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def find_date_patterns(self, text: str) -> List[str]:
        """
        Find date patterns in text that might indicate document boundaries.

        Args:
            text: Text to search

        Returns:
            List of found date patterns
        """
        import re

        # Common date patterns
        patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # MM/DD/YYYY or MM-DD-YYYY
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",  # YYYY/MM/DD or YYYY-MM-DD
            # Month DD, YYYY
            (
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r"[a-z]* \d{1,2},? \d{4}\b"
            ),
            # DD Month YYYY
            (
                r"\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r"[a-z]* \d{4}\b"
            ),
        ]

        dates = []
        for pattern in patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))

        return dates

    def identify_document_markers(self, text: str) -> Dict[str, bool]:
        """
        Identify common document start/end markers.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of marker types and whether they were found
        """
        text_lower = text.lower()

        markers = {
            "email_header": any(
                marker in text_lower for marker in ["from:", "to:", "subject:", "date:"]
            ),
            "letter_greeting": any(
                marker in text_lower for marker in ["dear ", "to whom it may concern"]
            ),
            "letter_closing": any(
                marker in text_lower
                for marker in ["sincerely", "regards", "respectfully"]
            ),
            "invoice_header": any(
                marker in text_lower
                for marker in ["invoice", "bill to", "invoice number"]
            ),
            "memo_header": any(
                marker in text_lower
                for marker in ["memorandum", "memo to", "memo from"]
            ),
            "form_header": any(
                marker in text_lower
                for marker in ["form ", "application", "request for"]
            ),
            "page_number": bool(re.search(r"page \d+", text_lower)),
            "document_title": bool(
                re.search(r"^[A-Z][A-Z\s]{5,}$", text.strip(), re.MULTILINE)
            ),
        }

        return markers
