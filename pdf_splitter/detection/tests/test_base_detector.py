"""Tests for base detector interface and data models."""

from datetime import datetime
from typing import List, Optional

import pytest

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.base_detector import (
    BaseDetector,
    BoundaryResult,
    BoundaryType,
    DetectionContext,
    DetectorType,
    ProcessedPage,
)


class MockDetector(BaseDetector):
    """Mock detector for testing base class functionality."""

    def detect_boundaries(
        self, pages: List[ProcessedPage], context: Optional[DetectionContext] = None
    ) -> List[BoundaryResult]:
        """Find boundaries at page 5 and 10."""
        results = []
        for page in pages:
            if page.page_number in [5, 10]:
                results.append(
                    BoundaryResult(
                        page_number=page.page_number,
                        boundary_type=BoundaryType.DOCUMENT_START,
                        confidence=0.85,
                        detector_type=self.get_detector_type(),
                    )
                )
        return results

    def get_detector_type(self) -> DetectorType:
        """Return mock detector type."""
        return DetectorType.HEURISTIC


class TestProcessedPage:
    """Test ProcessedPage data model."""

    def test_creation_with_defaults(self):
        """Test creating a ProcessedPage with default values."""
        page = ProcessedPage(page_number=1, text="Sample text")

        assert page.page_number == 1
        assert page.text == "Sample text"
        assert page.ocr_confidence is None
        assert page.page_type == "unknown"
        assert page.metadata == {}
        assert page.bounding_boxes == []
        assert not page.has_header
        assert not page.has_footer

    def test_creation_with_all_fields(self):
        """Test creating a ProcessedPage with all fields specified."""
        page = ProcessedPage(
            page_number=5,
            text="Full page text",
            ocr_confidence=0.95,
            page_type="SEARCHABLE",
            metadata={"source": "pdf"},
            bounding_boxes=[{"x": 10, "y": 20}],
            layout_info={"columns": 2},
            has_header=True,
            has_footer=True,
            has_page_number=True,
            dominant_font_size=12.0,
        )

        assert page.page_number == 5
        assert page.ocr_confidence == 0.95
        assert page.page_type == "SEARCHABLE"
        assert page.metadata["source"] == "pdf"
        assert len(page.bounding_boxes) == 1
        assert page.layout_info["columns"] == 2
        assert page.has_header
        assert page.dominant_font_size == 12.0

    def test_is_empty_property(self):
        """Test the is_empty property."""
        # Empty page
        empty_page = ProcessedPage(page_number=1, text="   \n  ")
        assert empty_page.is_empty

        # Page with minimal text
        minimal_page = ProcessedPage(page_number=1, text="abc")
        assert minimal_page.is_empty

        # Page with sufficient text
        full_page = ProcessedPage(
            page_number=1, text="This is a full page with content"
        )
        assert not full_page.is_empty


class TestBoundaryResult:
    """Test BoundaryResult data model."""

    def test_creation_with_defaults(self):
        """Test creating a BoundaryResult with minimum required fields."""
        result = BoundaryResult(
            page_number=5,
            boundary_type=BoundaryType.DOCUMENT_START,
            confidence=0.9,
            detector_type=DetectorType.LLM,
        )

        assert result.page_number == 5
        assert result.boundary_type == BoundaryType.DOCUMENT_START
        assert result.confidence == 0.9
        assert result.detector_type == DetectorType.LLM
        assert result.evidence == {}
        assert result.reasoning is None
        assert not result.is_between_pages
        assert isinstance(result.timestamp, datetime)

    def test_creation_with_all_fields(self):
        """Test creating a BoundaryResult with all fields."""
        result = BoundaryResult(
            page_number=10,
            boundary_type=BoundaryType.DOCUMENT_END,
            confidence=0.75,
            detector_type=DetectorType.VISUAL,
            evidence={"layout_change": True},
            reasoning="Major layout shift detected",
            is_between_pages=True,
            next_page_number=11,
        )

        assert result.page_number == 10
        assert result.boundary_type == BoundaryType.DOCUMENT_END
        assert result.evidence["layout_change"] is True
        assert result.reasoning == "Major layout shift detected"
        assert result.is_between_pages
        assert result.next_page_number == 11

    def test_confidence_validation(self):
        """Test that confidence values are validated."""
        # Valid confidence
        result = BoundaryResult(
            page_number=1,
            boundary_type=BoundaryType.UNCERTAIN,
            confidence=0.5,
            detector_type=DetectorType.COMBINED,
        )
        assert result.confidence == 0.5

        # Invalid confidence - too high
        with pytest.raises(ValueError, match="Confidence must be between"):
            BoundaryResult(
                page_number=1,
                boundary_type=BoundaryType.UNCERTAIN,
                confidence=1.5,
                detector_type=DetectorType.COMBINED,
            )

        # Invalid confidence - negative
        with pytest.raises(ValueError, match="Confidence must be between"):
            BoundaryResult(
                page_number=1,
                boundary_type=BoundaryType.UNCERTAIN,
                confidence=-0.1,
                detector_type=DetectorType.COMBINED,
            )


class TestDetectionContext:
    """Test DetectionContext data model."""

    def test_creation_and_update(self):
        """Test creating and updating detection context."""
        config = PDFConfig()
        context = DetectionContext(
            config=config, total_pages=50, document_metadata={"title": "Test Document"}
        )

        assert context.config == config
        assert context.total_pages == 50
        assert context.document_metadata["title"] == "Test Document"
        assert context.pages_processed == 0

        # Update progress
        context.update_progress(10)
        assert context.pages_processed == 10

        context.update_progress(5)
        assert context.pages_processed == 15


class TestBaseDetector:
    """Test BaseDetector abstract class and utilities."""

    @pytest.fixture
    def detector(self):
        """Create a mock detector instance."""
        return MockDetector()

    @pytest.fixture
    def sample_pages(self):
        """Create sample pages for testing."""
        return [
            ProcessedPage(page_number=i, text=f"Page {i} content") for i in range(1, 11)
        ]

    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert isinstance(detector.config, PDFConfig)
        assert detector._last_detection_time is None
        assert detector._detection_history == []

    def test_detect_boundaries(self, detector, sample_pages):
        """Test the mock detect_boundaries implementation."""
        results = detector.detect_boundaries(sample_pages)

        assert len(results) == 2
        assert results[0].page_number == 5
        assert results[1].page_number == 10
        assert all(r.confidence == 0.85 for r in results)

    def test_get_detector_type(self, detector):
        """Test getting detector type."""
        assert detector.get_detector_type() == DetectorType.HEURISTIC

    def test_filter_by_confidence(self, detector):
        """Test filtering results by confidence."""
        results = [
            BoundaryResult(1, BoundaryType.DOCUMENT_START, 0.9, DetectorType.LLM),
            BoundaryResult(2, BoundaryType.DOCUMENT_START, 0.6, DetectorType.LLM),
            BoundaryResult(3, BoundaryType.DOCUMENT_START, 0.8, DetectorType.LLM),
            BoundaryResult(4, BoundaryType.DOCUMENT_START, 0.5, DetectorType.LLM),
        ]

        # Use default threshold (0.7)
        filtered = detector.filter_by_confidence(results)
        assert len(filtered) == 2
        assert filtered[0].page_number == 1
        assert filtered[1].page_number == 3

        # Use custom threshold
        filtered = detector.filter_by_confidence(results, threshold=0.85)
        assert len(filtered) == 1
        assert filtered[0].page_number == 1

    def test_extract_context_window(self, detector, sample_pages):
        """Test extracting context windows."""
        # Normal window in the middle
        window, center_idx = detector.extract_context_window(
            sample_pages, 5, window_size=5
        )
        assert len(window) == 5
        assert window[0].page_number == 4
        assert window[4].page_number == 8
        assert center_idx == 2

        # Window at the beginning (can't get full window)
        window, center_idx = detector.extract_context_window(
            sample_pages, 0, window_size=3
        )
        assert len(window) == 2  # Only 2 pages available at the beginning
        assert window[0].page_number == 1
        assert window[1].page_number == 2
        assert center_idx == 0

        # Window at the end (can't get full window)
        window, center_idx = detector.extract_context_window(
            sample_pages, 9, window_size=3
        )
        assert len(window) == 2  # Only 2 pages available at the end
        assert window[0].page_number == 9
        assert window[1].page_number == 10
        assert center_idx == 1  # Index 9 is at position 1 in the 2-element window

        # Even window size should raise error
        with pytest.raises(ValueError, match="Window size must be odd"):
            detector.extract_context_window(sample_pages, 5, window_size=4)

    def test_calculate_text_similarity(self, detector):
        """Test text similarity calculation."""
        # Identical texts
        sim = detector.calculate_text_similarity("hello world", "hello world")
        assert sim == 1.0

        # Completely different texts
        sim = detector.calculate_text_similarity("hello world", "foo bar")
        assert sim == 0.0

        # Partial overlap
        sim = detector.calculate_text_similarity("hello world test", "hello world foo")
        assert 0.4 < sim < 0.6  # 2 common words out of 4 unique

        # Empty texts
        sim = detector.calculate_text_similarity("", "hello")
        assert sim == 0.0

        sim = detector.calculate_text_similarity("hello", "")
        assert sim == 0.0

    def test_find_date_patterns(self, detector):
        """Test finding date patterns in text."""
        text = """
        Meeting scheduled for 12/25/2023.
        Previous meeting was on 2023-11-15.
        Invoice dated January 15, 2024.
        Report from 5 Feb 2024.
        """

        dates = detector.find_date_patterns(text)
        assert len(dates) >= 4
        assert "12/25/2023" in dates
        assert "2023-11-15" in dates
        assert any("January" in date for date in dates)
        assert any("Feb" in date for date in dates)

    def test_identify_document_markers(self, detector):
        """Test identifying document markers."""
        # Email
        email_text = "From: john@example.com\nTo: jane@example.com\nSubject: Test"
        markers = detector.identify_document_markers(email_text)
        assert markers["email_header"]
        assert not markers["invoice_header"]

        # Letter
        letter_text = (
            "Dear Mr. Smith,\n\nThank you for your inquiry.\n\nSincerely,\nJohn Doe"
        )
        markers = detector.identify_document_markers(letter_text)
        assert markers["letter_greeting"]
        assert markers["letter_closing"]

        # Invoice
        invoice_text = "INVOICE #12345\nBill To: ABC Company"
        markers = detector.identify_document_markers(invoice_text)
        assert markers["invoice_header"]

        # Page number
        page_text = "Some content here\nPage 5 of 10"
        markers = detector.identify_document_markers(page_text)
        assert markers["page_number"]

    def test_detection_stats(self, detector, sample_pages):
        """Test detection statistics tracking."""
        # Initial stats
        stats = detector.get_detection_stats()
        assert stats["detections"] == 0
        assert stats["avg_confidence"] == 0.0

        # Add some detections to history
        detector._detection_history = [
            BoundaryResult(1, BoundaryType.DOCUMENT_START, 0.9, DetectorType.LLM),
            BoundaryResult(5, BoundaryType.DOCUMENT_END, 0.8, DetectorType.LLM),
            BoundaryResult(8, BoundaryType.DOCUMENT_START, 0.7, DetectorType.LLM),
        ]
        detector._last_detection_time = 1.5

        stats = detector.get_detection_stats()
        assert stats["detections"] == 3
        assert (
            abs(stats["avg_confidence"] - 0.8) < 0.0001
        )  # Handle floating point precision
        assert stats["min_confidence"] == 0.7
        assert stats["max_confidence"] == 0.9
        assert stats["last_detection_time"] == 1.5
