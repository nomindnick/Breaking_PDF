"""Integration tests for edge cases and error scenarios."""

from pathlib import Path
from typing import List

import pytest

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.base_detector import (
    BoundaryResult,
    BoundaryType,
    DetectorType,
    ProcessedPage,
)
from pdf_splitter.preprocessing import PDFHandler
from pdf_splitter.splitting import PDFSplitter, SplitSessionManager
from pdf_splitter.splitting.exceptions import SessionExpiredError
from pdf_splitter.splitting.models import (
    DocumentSegment,
    SplitProposal,
    UserModification,
)


class TestEdgeCases:
    """Test edge cases and error scenarios in the pipeline."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return PDFConfig(debug=True)

    @pytest.fixture
    def mock_pages(self) -> List[ProcessedPage]:
        """Create mock processed pages."""
        return [
            ProcessedPage(
                page_number=i,
                text=f"Page {i+1} content\nSome text here",
                page_type="SEARCHABLE",
            )
            for i in range(10)
        ]

    @pytest.fixture
    def mock_boundaries(self) -> List[BoundaryResult]:
        """Create mock boundary results."""
        return [
            BoundaryResult(
                page_number=0,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.9,
                detector_type=DetectorType.EMBEDDINGS,
            ),
            BoundaryResult(
                page_number=5,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.85,
                detector_type=DetectorType.EMBEDDINGS,
            ),
            BoundaryResult(
                page_number=8,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.7,
                detector_type=DetectorType.EMBEDDINGS,
            ),
        ]

    def test_single_page_documents(self, config, mock_pages, tmp_path):
        """Test handling of single-page documents."""
        # Create boundaries for each page
        boundaries = [
            BoundaryResult(
                page_number=i,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.8,
                detector_type=DetectorType.HEURISTIC,
            )
            for i in range(len(mock_pages))
        ]

        # Create test PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf")

        splitter = PDFSplitter(config=config)
        proposal = splitter.generate_proposal(
            boundaries=boundaries, pages=mock_pages, pdf_path=pdf_path
        )

        # Each page should be its own document
        assert len(proposal.segments) == len(mock_pages)

        for i, segment in enumerate(proposal.segments):
            assert segment.start_page == i
            assert segment.end_page == i
            assert segment.page_count == 1

    def test_no_boundaries_detected(self, config, mock_pages, tmp_path):
        """Test handling when no boundaries are detected."""
        # Empty boundaries list
        boundaries = []

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf")

        splitter = PDFSplitter(config=config)
        proposal = splitter.generate_proposal(
            boundaries=boundaries, pages=mock_pages, pdf_path=pdf_path
        )

        # Should create no segments
        assert len(proposal.segments) == 0

    def test_overlapping_boundaries(self, config, mock_pages, tmp_path):
        """Test handling of overlapping boundary detections."""
        # Create boundaries with some overlap
        boundaries = [
            BoundaryResult(
                page_number=0,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.9,
                detector_type=DetectorType.EMBEDDINGS,
            ),
            BoundaryResult(
                page_number=5,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.85,
                detector_type=DetectorType.HEURISTIC,
            ),
            BoundaryResult(
                page_number=5,  # Duplicate boundary
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.8,
                detector_type=DetectorType.LLM,
            ),
        ]

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf")

        splitter = PDFSplitter(config=config)
        proposal = splitter.generate_proposal(
            boundaries=boundaries, pages=mock_pages, pdf_path=pdf_path
        )

        # Should handle duplicates gracefully
        assert len(proposal.segments) == 2
        assert proposal.segments[0].start_page == 0
        assert proposal.segments[1].start_page == 5

    def test_empty_pages(self, config, tmp_path):
        """Test handling of empty pages."""
        # Create pages with some empty
        pages = [
            ProcessedPage(page_number=0, text="Content", page_type="SEARCHABLE"),
            ProcessedPage(page_number=1, text="", page_type="EMPTY"),  # Empty
            ProcessedPage(
                page_number=2, text="   ", page_type="EMPTY"
            ),  # Whitespace only
            ProcessedPage(page_number=3, text="More content", page_type="SEARCHABLE"),
        ]

        boundaries = [
            BoundaryResult(
                page_number=0,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.9,
                detector_type=DetectorType.EMBEDDINGS,
            ),
        ]

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf")

        splitter = PDFSplitter(config=config)
        proposal = splitter.generate_proposal(
            boundaries=boundaries, pages=pages, pdf_path=pdf_path
        )

        # Should still create segment including empty pages
        assert len(proposal.segments) == 1
        assert proposal.segments[0].end_page == 3

    def test_invalid_page_ranges(self, config, tmp_path):
        """Test handling of invalid page ranges."""
        pages = [ProcessedPage(page_number=i, text=f"Page {i}") for i in range(5)]

        # Boundary beyond page count
        boundaries = [
            BoundaryResult(
                page_number=10,  # Invalid - only 5 pages
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.9,
                detector_type=DetectorType.EMBEDDINGS,
            ),
        ]

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf")

        splitter = PDFSplitter(config=config)
        proposal = splitter.generate_proposal(
            boundaries=boundaries, pages=pages, pdf_path=pdf_path
        )

        # Should skip invalid boundary
        assert len(proposal.segments) == 0

    def test_special_characters_in_content(self, config, tmp_path):
        """Test handling of special characters in document content."""
        # Pages with special characters
        pages = [
            ProcessedPage(
                page_number=0,
                text='Invoice #12345\n<>:"/\\|?*\nSpecial chars: €£¥',
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=1, text="Normal content here", page_type="SEARCHABLE"
            ),
        ]

        boundaries = [
            BoundaryResult(
                page_number=0,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.9,
                detector_type=DetectorType.EMBEDDINGS,
            ),
        ]

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf")

        splitter = PDFSplitter(config=config)
        proposal = splitter.generate_proposal(
            boundaries=boundaries, pages=pages, pdf_path=pdf_path
        )

        # Should sanitize filename properly
        assert len(proposal.segments) == 1
        segment = proposal.segments[0]

        # Filename should be sanitized
        assert "<" not in segment.suggested_filename
        assert ">" not in segment.suggested_filename
        assert ":" not in segment.suggested_filename
        assert '"' not in segment.suggested_filename
        assert "/" not in segment.suggested_filename
        assert "\\" not in segment.suggested_filename
        assert "|" not in segment.suggested_filename
        assert "?" not in segment.suggested_filename
        assert "*" not in segment.suggested_filename

    @pytest.mark.asyncio
    async def test_corrupt_pdf_handling(self, config):
        """Test handling of corrupt PDFs."""
        pdf_handler = PDFHandler(config=config)

        # Create a fake corrupt PDF
        corrupt_pdf = Path("corrupt.pdf")
        try:
            # Write invalid PDF content
            corrupt_pdf.write_bytes(b"This is not a valid PDF")

            # Should raise an error
            with pytest.raises(Exception):  # PyMuPDF will raise an error
                await pdf_handler.load_pdf(corrupt_pdf)

        finally:
            if corrupt_pdf.exists():
                corrupt_pdf.unlink()
            await pdf_handler.cleanup()

    def test_session_edge_cases(self, config, tmp_path):
        """Test session management edge cases."""
        from datetime import timedelta

        db_path = tmp_path / "sessions.db"
        manager = SplitSessionManager(config=config, db_path=db_path)

        # Create test proposal
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy")

        proposal = SplitProposal(
            pdf_path=pdf_path,
            total_pages=5,
            segments=[
                DocumentSegment(
                    start_page=0,
                    end_page=4,
                    document_type="Test",
                    suggested_filename="test.pdf",
                    confidence=0.9,
                )
            ],
            detection_results=[],
        )

        # Test 1: Expired session
        expired_session = manager.create_session(
            proposal, lifetime=timedelta(seconds=-1)  # Already expired
        )

        with pytest.raises(SessionExpiredError):
            manager.get_session(expired_session.session_id)

        # Test 2: Invalid state transitions
        active_session = manager.create_session(proposal)

        from pdf_splitter.splitting.exceptions import InvalidSessionStateError

        # Can't go from pending to completed directly
        with pytest.raises(InvalidSessionStateError):
            manager.update_session(active_session.session_id, status="completed")

        # Test 3: Concurrent modifications
        session1 = manager.create_session(proposal)

        # Simulate concurrent modification
        mod1 = UserModification(
            modification_type="rename",
            segment_id=proposal.segments[0].segment_id,
            details={"new_filename": "renamed1.pdf"},
        )

        mod2 = UserModification(
            modification_type="rename",
            segment_id=proposal.segments[0].segment_id,
            details={"new_filename": "renamed2.pdf"},
        )

        # Both modifications should be recorded
        manager.update_session(session1.session_id, modifications=[mod1, mod2])

        retrieved = manager.get_session(session1.session_id)
        assert len(retrieved.user_modifications) == 2

    def test_extreme_document_counts(self, config, tmp_path):
        """Test handling of PDFs with extreme document counts."""
        # Test with 100 single-page documents
        pages = [ProcessedPage(page_number=i, text=f"Document {i}") for i in range(100)]

        boundaries = [
            BoundaryResult(
                page_number=i,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.8,
                detector_type=DetectorType.HEURISTIC,
            )
            for i in range(100)
        ]

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf")

        splitter = PDFSplitter(config=config)
        proposal = splitter.generate_proposal(
            boundaries=boundaries, pages=pages, pdf_path=pdf_path
        )

        assert len(proposal.segments) == 100

        # All segments should be valid
        for segment in proposal.segments:
            assert segment.start_page == segment.end_page
            assert 0 <= segment.start_page < 100
