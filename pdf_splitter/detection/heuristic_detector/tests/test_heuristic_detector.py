"""Tests for the heuristic detector."""

from pdf_splitter.detection.base_detector import BoundaryResult, ProcessedPage
from pdf_splitter.detection.heuristic_detector import (
    HeuristicConfig,
    HeuristicDetector,
    PatternConfig,
)


class TestHeuristicDetector:
    """Test suite for HeuristicDetector."""

    def test_initialization(self):
        """Test detector initialization with default config."""
        detector = HeuristicDetector()
        assert detector.config is not None
        assert len(detector.config.patterns) > 0

    def test_custom_config(self):
        """Test detector with custom configuration."""
        config = HeuristicConfig(
            patterns={
                "test_pattern": PatternConfig(
                    name="test_pattern", weight=1.0, enabled=True
                )
            },
            min_confidence_threshold=0.5,
        )
        detector = HeuristicDetector(config)
        assert "test_pattern" in detector.config.patterns

    def test_date_pattern_detection(self):
        """Test date pattern detection."""
        detector = HeuristicDetector()

        # Page with date at top
        page_with_date = ProcessedPage(
            page_number=1,
            text="January 15, 2024\n\nDear Sir/Madam,\n\nThis is a test letter.",
            ocr_confidence=0.95,
        )

        # Page without date
        page_without_date = ProcessedPage(
            page_number=2,
            text="This is the continuation of the letter.\n\nMore content here.",
            ocr_confidence=0.95,
        )

        result = detector.detect_boundary(page_without_date, page_with_date)
        assert result.confidence > 0.5  # Should detect date pattern
        assert "date_pattern" in result.evidence["active_patterns"]

    def test_document_keywords(self):
        """Test document keyword detection."""
        detector = HeuristicDetector()

        # Page with document keyword
        page_with_keyword = ProcessedPage(
            page_number=1,
            text="MEMORANDUM\n\nTo: All Staff\nFrom: Management\n\nSubject: Test",
            ocr_confidence=0.95,
        )

        page_regular = ProcessedPage(
            page_number=2,
            text="This is regular content without keywords.",
            ocr_confidence=0.95,
        )

        result = detector.detect_boundary(page_regular, page_with_keyword)
        assert result.confidence > 0.6
        assert "document_keywords" in result.evidence["active_patterns"]

    def test_email_header_detection(self):
        """Test email header detection."""
        detector = HeuristicDetector()

        # Page with email header
        email_page = ProcessedPage(
            page_number=1,
            text="From: john.doe@example.com\nTo: jane.smith@example.com\nSubject: Test Email\n\nHello Jane,",
            ocr_confidence=0.95,
        )

        regular_page = ProcessedPage(
            page_number=2, text="Regular document content here.", ocr_confidence=0.95
        )

        result = detector.detect_boundary(regular_page, email_page)
        assert result.confidence > 0.8
        assert "email_header" in result.evidence["active_patterns"]

    def test_page_numbering_reset(self):
        """Test page numbering reset detection."""
        detector = HeuristicDetector()

        # Last page of document
        last_page = ProcessedPage(
            page_number=10,
            text="This is the end of the document.\n\n\n\nPage 10 of 10",
            ocr_confidence=0.95,
        )

        # First page of new document
        first_page = ProcessedPage(
            page_number=11,
            text="New Document Title\n\nContent begins here.\n\n\n\nPage 1 of 5",
            ocr_confidence=0.95,
        )

        result = detector.detect_boundary(last_page, first_page)
        assert result.confidence > 0.8
        assert "page_numbering" in result.evidence["active_patterns"]

    def test_terminal_phrases(self):
        """Test terminal phrase detection."""
        detector = HeuristicDetector()

        # Page ending with terminal phrase
        ending_page = ProcessedPage(
            page_number=1,
            text="Thank you for your consideration.\n\nSincerely,\n\nJohn Doe",
            ocr_confidence=0.95,
        )

        new_page = ProcessedPage(
            page_number=2,
            text="INVOICE #12345\n\nBill To: ABC Company",
            ocr_confidence=0.95,
        )

        result = detector.detect_boundary(ending_page, new_page)
        assert result.confidence > 0.4
        assert "terminal_phrases" in result.evidence["signals"]

    def test_whitespace_detection(self):
        """Test whitespace ratio detection."""
        detector = HeuristicDetector()

        # Page with lots of whitespace at end
        page_with_whitespace = ProcessedPage(
            page_number=1,
            text="Short content.\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            ocr_confidence=0.95,
        )

        new_doc = ProcessedPage(
            page_number=2,
            text="New Document\n\nFull of content here...",
            ocr_confidence=0.95,
        )

        result = detector.detect_boundary(page_with_whitespace, new_doc)
        # Should detect whitespace at end of previous page
        assert "whitespace_end" in result.evidence["signals"]

    def test_multiple_signals(self):
        """Test combination of multiple signals."""
        detector = HeuristicDetector()

        # Page with terminal phrase and page number
        ending_page = ProcessedPage(
            page_number=5,
            text="End of report.\n\nSincerely,\nThe Team\n\n\nPage 5 of 5",
            ocr_confidence=0.95,
        )

        # New document with keyword and date
        new_doc = ProcessedPage(
            page_number=6,
            text="MEMORANDUM\n\nDate: March 1, 2024\n\nTo: All Staff\n\nPage 1 of 3",
            ocr_confidence=0.95,
        )

        result = detector.detect_boundary(ending_page, new_doc)
        from pdf_splitter.detection.base_detector import BoundaryType

        assert result.boundary_type != BoundaryType.PAGE_CONTINUATION
        assert result.confidence > 0.8
        assert len(result.evidence["active_patterns"]) >= 3

    def test_disabled_patterns(self):
        """Test that disabled patterns don't contribute."""
        config = HeuristicConfig()
        config.patterns["email_header"].enabled = False
        detector = HeuristicDetector(config)

        email_page = ProcessedPage(
            page_number=1,
            text="From: test@example.com\nTo: user@example.com\nSubject: Test",
            ocr_confidence=0.95,
        )

        regular_page = ProcessedPage(
            page_number=0, text="Regular content", ocr_confidence=0.95
        )

        result = detector.detect_boundary(regular_page, email_page)
        assert "email_header" not in result.evidence["active_patterns"]

    def test_confidence_thresholds(self):
        """Test confidence threshold behavior."""
        config = HeuristicConfig(min_confidence_threshold=0.8)
        detector = HeuristicDetector(config)

        # Weak signal page
        page1 = ProcessedPage(
            page_number=1, text="Some content here", ocr_confidence=0.95
        )

        page2 = ProcessedPage(
            page_number=2, text="Different content there", ocr_confidence=0.95
        )

        result = detector.detect_boundary(page1, page2)
        # With high threshold, weak signals shouldn't trigger boundary
        from pdf_splitter.detection.base_detector import BoundaryType

        assert result.boundary_type == BoundaryType.PAGE_CONTINUATION

    def test_empty_pages(self):
        """Test handling of empty pages."""
        detector = HeuristicDetector()

        empty_page = ProcessedPage(page_number=1, text="", ocr_confidence=0.0)

        content_page = ProcessedPage(
            page_number=2, text="Document content here", ocr_confidence=0.95
        )

        # Should handle empty pages gracefully
        result = detector.detect_boundary(empty_page, content_page)
        assert isinstance(result, BoundaryResult)

    def test_detect_all_boundaries(self):
        """Test detecting boundaries across multiple pages."""
        detector = HeuristicDetector()

        pages = [
            ProcessedPage(
                page_number=1,
                text="First document content\n\nPage 1 of 2",
                ocr_confidence=0.95,
            ),
            ProcessedPage(
                page_number=2,
                text="More content\n\nSincerely,\nAuthor\n\nPage 2 of 2",
                ocr_confidence=0.95,
            ),
            ProcessedPage(
                page_number=3,
                text="INVOICE #123\n\nDate: Jan 1, 2024\n\nPage 1 of 1",
                ocr_confidence=0.95,
            ),
            ProcessedPage(
                page_number=4,
                text="MEMORANDUM\n\nTo: Staff\n\nPage 1 of 3",
                ocr_confidence=0.95,
            ),
        ]

        results = detector.detect_all_boundaries(pages)
        assert len(results) == 3  # n-1 boundaries for n pages

        # Should detect boundary between doc 1 and invoice
        assert results[1].confidence > 0.7
        # Should detect boundary between invoice and memo
        assert results[2].confidence > 0.7
