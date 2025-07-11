"""Tests for splitting module data models."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from pdf_splitter.detection.base_detector import BoundaryResult
from pdf_splitter.splitting.models import (
    DocumentSegment,
    SplitProposal,
    SplitResult,
    SplitSession,
    UserModification,
)


class TestDocumentSegment:
    """Test DocumentSegment model."""

    def test_valid_segment_creation(self):
        """Test creating a valid document segment."""
        segment = DocumentSegment(
            start_page=0,
            end_page=5,
            document_type="Invoice",
            suggested_filename="invoice_2024_01.pdf",
            confidence=0.95,
            summary="Invoice from January 2024",
        )

        assert segment.start_page == 0
        assert segment.end_page == 5
        assert segment.document_type == "Invoice"
        assert segment.suggested_filename == "invoice_2024_01.pdf"
        assert segment.confidence == 0.95
        assert segment.summary == "Invoice from January 2024"
        assert segment.page_count == 6
        assert segment.page_range == "Pages 1-6"
        assert not segment.is_user_defined
        assert segment.segment_id  # Should have auto-generated ID

    def test_single_page_segment(self):
        """Test segment with single page."""
        segment = DocumentSegment(
            start_page=5,
            end_page=5,
            document_type="Email",
            suggested_filename="email.pdf",
            confidence=0.8,
        )

        assert segment.page_count == 1
        assert segment.page_range == "Page 6"

    def test_invalid_page_range(self):
        """Test validation of invalid page ranges."""
        # Negative start page
        with pytest.raises(ValueError, match="start_page must be >= 0"):
            DocumentSegment(
                start_page=-1,
                end_page=5,
                document_type="Test",
                suggested_filename="test.pdf",
                confidence=0.5,
            )

        # End page before start page
        with pytest.raises(ValueError, match="end_page .* must be >= start_page"):
            DocumentSegment(
                start_page=5,
                end_page=3,
                document_type="Test",
                suggested_filename="test.pdf",
                confidence=0.5,
            )

    def test_invalid_confidence(self):
        """Test validation of confidence scores."""
        # Too high
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            DocumentSegment(
                start_page=0,
                end_page=5,
                document_type="Test",
                suggested_filename="test.pdf",
                confidence=1.5,
            )

        # Too low
        with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
            DocumentSegment(
                start_page=0,
                end_page=5,
                document_type="Test",
                suggested_filename="test.pdf",
                confidence=-0.1,
            )

    def test_user_defined_segment(self):
        """Test creating a user-defined segment."""
        segment = DocumentSegment(
            start_page=10,
            end_page=15,
            document_type="Custom",
            suggested_filename="user_doc.pdf",
            confidence=1.0,
            is_user_defined=True,
        )

        assert segment.is_user_defined
        assert segment.confidence == 1.0


class TestSplitProposal:
    """Test SplitProposal model."""

    @pytest.fixture
    def test_pdf_path(self, tmp_path):
        """Create a temporary PDF file."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")
        return pdf_path

    @pytest.fixture
    def sample_segments(self):
        """Create sample segments."""
        return [
            DocumentSegment(
                start_page=0,
                end_page=5,
                document_type="Invoice",
                suggested_filename="invoice.pdf",
                confidence=0.95,
            ),
            DocumentSegment(
                start_page=6,
                end_page=10,
                document_type="Email",
                suggested_filename="email.pdf",
                confidence=0.85,
            ),
        ]

    @pytest.fixture
    def sample_boundaries(self):
        """Create sample boundary results."""
        from pdf_splitter.detection.base_detector import BoundaryType, DetectorType

        return [
            BoundaryResult(
                page_number=0,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.95,
                detector_type=DetectorType.EMBEDDINGS,
            ),
            BoundaryResult(
                page_number=6,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.85,
                detector_type=DetectorType.EMBEDDINGS,
            ),
        ]

    def test_valid_proposal_creation(
        self, test_pdf_path, sample_segments, sample_boundaries
    ):
        """Test creating a valid split proposal."""
        proposal = SplitProposal(
            pdf_path=test_pdf_path,
            total_pages=20,
            segments=sample_segments,
            detection_results=sample_boundaries,
        )

        assert proposal.pdf_path == test_pdf_path
        assert proposal.total_pages == 20
        assert len(proposal.segments) == 2
        assert proposal.segment_count == 2
        assert proposal.proposal_id  # Should have auto-generated ID
        assert proposal.created_at
        assert proposal.modified_at is None

    def test_nonexistent_pdf(self, tmp_path, sample_segments, sample_boundaries):
        """Test validation with non-existent PDF."""
        with pytest.raises(ValueError, match="PDF file not found"):
            SplitProposal(
                pdf_path=tmp_path / "nonexistent.pdf",
                total_pages=20,
                segments=sample_segments,
                detection_results=sample_boundaries,
            )

    def test_invalid_total_pages(
        self, test_pdf_path, sample_segments, sample_boundaries
    ):
        """Test validation with invalid total pages."""
        with pytest.raises(ValueError, match="total_pages must be > 0"):
            SplitProposal(
                pdf_path=test_pdf_path,
                total_pages=0,
                segments=sample_segments,
                detection_results=sample_boundaries,
            )

    def test_overlapping_segments(self, test_pdf_path, sample_boundaries):
        """Test validation with overlapping segments."""
        overlapping_segments = [
            DocumentSegment(
                start_page=0,
                end_page=5,
                document_type="Doc1",
                suggested_filename="doc1.pdf",
                confidence=0.9,
            ),
            DocumentSegment(
                start_page=4,  # Overlaps with previous
                end_page=8,
                document_type="Doc2",
                suggested_filename="doc2.pdf",
                confidence=0.9,
            ),
        ]

        with pytest.raises(ValueError, match="Segments overlap"):
            SplitProposal(
                pdf_path=test_pdf_path,
                total_pages=20,
                segments=overlapping_segments,
                detection_results=sample_boundaries,
            )

    def test_segment_exceeds_total_pages(self, test_pdf_path, sample_boundaries):
        """Test validation when segment exceeds total pages."""
        invalid_segments = [
            DocumentSegment(
                start_page=0,
                end_page=25,  # Exceeds total_pages=20
                document_type="Doc",
                suggested_filename="doc.pdf",
                confidence=0.9,
            ),
        ]

        with pytest.raises(ValueError, match="exceeds total pages"):
            SplitProposal(
                pdf_path=test_pdf_path,
                total_pages=20,
                segments=invalid_segments,
                detection_results=sample_boundaries,
            )

    def test_get_segment(self, test_pdf_path, sample_segments, sample_boundaries):
        """Test getting segment by ID."""
        proposal = SplitProposal(
            pdf_path=test_pdf_path,
            total_pages=20,
            segments=sample_segments,
            detection_results=sample_boundaries,
        )

        segment = proposal.get_segment(sample_segments[0].segment_id)
        assert segment == sample_segments[0]

        # Non-existent ID
        assert proposal.get_segment("non-existent-id") is None

    def test_add_segment(self, test_pdf_path, sample_boundaries):
        """Test adding a new segment."""
        proposal = SplitProposal(
            pdf_path=test_pdf_path,
            total_pages=20,
            segments=[],
            detection_results=sample_boundaries,
        )

        new_segment = DocumentSegment(
            start_page=0,
            end_page=5,
            document_type="New",
            suggested_filename="new.pdf",
            confidence=0.9,
        )

        proposal.add_segment(new_segment)
        assert proposal.segment_count == 1
        assert proposal.modified_at is not None

    def test_remove_segment(self, test_pdf_path, sample_segments, sample_boundaries):
        """Test removing a segment."""
        proposal = SplitProposal(
            pdf_path=test_pdf_path,
            total_pages=20,
            segments=sample_segments,
            detection_results=sample_boundaries,
        )

        removed = proposal.remove_segment(sample_segments[0].segment_id)
        assert removed
        assert proposal.segment_count == 1
        assert proposal.modified_at is not None

        # Try removing non-existent
        assert not proposal.remove_segment("non-existent-id")

    def test_update_segment(self, test_pdf_path, sample_segments, sample_boundaries):
        """Test updating a segment."""
        proposal = SplitProposal(
            pdf_path=test_pdf_path,
            total_pages=20,
            segments=sample_segments,
            detection_results=sample_boundaries,
        )

        updated = proposal.update_segment(
            sample_segments[0].segment_id,
            suggested_filename="updated_invoice.pdf",
            document_type="Updated Invoice",
        )

        assert updated
        assert proposal.segments[0].suggested_filename == "updated_invoice.pdf"
        assert proposal.segments[0].document_type == "Updated Invoice"
        assert proposal.modified_at is not None


class TestSplitSession:
    """Test SplitSession model."""

    @pytest.fixture
    def test_pdf_path(self, tmp_path):
        """Create a temporary PDF file."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")
        return pdf_path

    @pytest.fixture
    def sample_segments(self):
        """Create sample segments."""
        return [
            DocumentSegment(
                start_page=0,
                end_page=5,
                document_type="Invoice",
                suggested_filename="invoice.pdf",
                confidence=0.95,
            ),
            DocumentSegment(
                start_page=6,
                end_page=10,
                document_type="Email",
                suggested_filename="email.pdf",
                confidence=0.85,
            ),
        ]

    @pytest.fixture
    def sample_boundaries(self):
        """Create sample boundary results."""
        from pdf_splitter.detection.base_detector import BoundaryType, DetectorType

        return [
            BoundaryResult(
                page_number=0,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.95,
                detector_type=DetectorType.EMBEDDINGS,
            ),
            BoundaryResult(
                page_number=6,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.85,
                detector_type=DetectorType.EMBEDDINGS,
            ),
        ]

    @pytest.fixture
    def sample_proposal(self, test_pdf_path, sample_segments, sample_boundaries):
        """Create a sample proposal."""
        return SplitProposal(
            pdf_path=test_pdf_path,
            total_pages=20,
            segments=sample_segments,
            detection_results=sample_boundaries,
        )

    def test_session_creation(self, sample_proposal):
        """Test creating a split session."""
        session = SplitSession(session_id=str(uuid4()), proposal=sample_proposal)

        assert session.session_id
        assert session.proposal == sample_proposal
        assert session.status == "pending"
        assert session.user_modifications == []
        assert session.is_active
        assert not session.is_expired

    def test_add_modification(self, sample_proposal):
        """Test adding user modifications."""
        session = SplitSession(session_id=str(uuid4()), proposal=sample_proposal)

        mod = UserModification(
            modification_type="rename",
            segment_id=sample_proposal.segments[0].segment_id,
            details={"new_filename": "custom_name.pdf"},
        )

        session.add_modification(mod)
        assert len(session.user_modifications) == 1
        assert session.status == "modified"
        assert session.updated_at > session.created_at

    def test_confirm_session(self, sample_proposal, tmp_path):
        """Test confirming a session."""
        session = SplitSession(session_id=str(uuid4()), proposal=sample_proposal)

        output_dir = tmp_path / "output"
        session.confirm(output_dir)

        assert session.status == "confirmed"
        assert session.output_directory == output_dir

    def test_complete_session(self, sample_proposal):
        """Test completing a session."""
        session = SplitSession(session_id=str(uuid4()), proposal=sample_proposal)

        session.complete()
        assert session.status == "completed"
        assert not session.is_active

    def test_cancel_session(self, sample_proposal):
        """Test cancelling a session."""
        session = SplitSession(session_id=str(uuid4()), proposal=sample_proposal)

        session.cancel()
        assert session.status == "cancelled"
        assert not session.is_active

    def test_expired_session(self, sample_proposal):
        """Test session expiration."""
        session = SplitSession(
            session_id=str(uuid4()),
            proposal=sample_proposal,
            expires_at=datetime.now() - timedelta(minutes=1),
        )

        assert session.is_expired
        assert not session.is_active


class TestSplitResult:
    """Test SplitResult model."""

    def test_split_result_creation(self, tmp_path):
        """Test creating a split result."""
        output_files = [
            tmp_path / "doc1.pdf",
            tmp_path / "doc2.pdf",
        ]

        segments = [
            DocumentSegment(
                start_page=0,
                end_page=5,
                document_type="Doc1",
                suggested_filename="doc1.pdf",
                confidence=0.9,
            ),
            DocumentSegment(
                start_page=6,
                end_page=10,
                document_type="Doc2",
                suggested_filename="doc2.pdf",
                confidence=0.9,
            ),
        ]

        result = SplitResult(
            session_id=str(uuid4()),
            input_pdf=tmp_path / "input.pdf",
            output_files=output_files,
            segments=segments,
            duration_seconds=2.5,
        )

        assert result.session_id
        assert result.input_pdf == tmp_path / "input.pdf"
        assert len(result.output_files) == 2
        assert result.file_count == 2
        assert result.duration_seconds == 2.5
        assert result.completed_at
