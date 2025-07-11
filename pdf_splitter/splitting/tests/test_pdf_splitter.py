"""Tests for PDFSplitter service."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf_splitter.detection.base_detector import (
    BoundaryResult,
    BoundaryType,
    DetectorType,
    ProcessedPage,
)
from pdf_splitter.splitting.exceptions import PDFSplitError
from pdf_splitter.splitting.models import DocumentSegment, SplitProposal
from pdf_splitter.splitting.pdf_splitter import PDFSplitter


class TestPDFSplitter:
    """Test PDFSplitter functionality."""

    @pytest.fixture
    def splitter(self):
        """Create PDFSplitter instance."""
        return PDFSplitter()

    @pytest.fixture
    def sample_pages(self):
        """Create sample processed pages."""
        return [
            ProcessedPage(
                page_number=0,
                text="Invoice #12345\nDate: 2024-01-15\nBill To: ACME Corp",
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=1,
                text="Item details...\nTotal Amount: $1,234.56",
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=2,
                text="From: sender@email.com\nTo: recipient@email.com\nSubject: Meeting",
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=3,
                text="Dear John,\nI hope this email finds you well...",
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=4,
                text="Drawing #A-101\nScale: 1:100\nProject: New Building",
                page_type="SEARCHABLE",
            ),
        ]

    @pytest.fixture
    def sample_boundaries(self):
        """Create sample boundary results."""
        return [
            BoundaryResult(
                page_number=0,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.95,
                detector_type=DetectorType.EMBEDDINGS,
            ),
            BoundaryResult(
                page_number=2,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.88,
                detector_type=DetectorType.EMBEDDINGS,
            ),
            BoundaryResult(
                page_number=4,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.92,
                detector_type=DetectorType.EMBEDDINGS,
            ),
        ]

    def test_document_type_detection(self, splitter, sample_pages):
        """Test document type detection."""
        # Test invoice detection
        invoice_pages = sample_pages[:2]
        doc_type = splitter._detect_document_type(invoice_pages)
        assert doc_type == "Invoice"

        # Test email detection
        email_pages = sample_pages[2:4]
        doc_type = splitter._detect_document_type(email_pages)
        assert doc_type == "Email"

        # Test plans detection
        plans_pages = sample_pages[4:5]
        doc_type = splitter._detect_document_type(plans_pages)
        assert doc_type == "Plans"

        # Test unknown type
        unknown_pages = [ProcessedPage(page_number=0, text="Random text here")]
        doc_type = splitter._detect_document_type(unknown_pages)
        assert doc_type == "Document"

    def test_filename_suggestion(self, splitter, sample_pages):
        """Test filename suggestion generation."""
        # Test invoice filename
        invoice_pages = sample_pages[:2]
        filename = splitter._suggest_filename(invoice_pages, "Invoice", 1)
        assert "invoice" in filename.lower()
        assert "2024-01-15" in filename  # Date should be included
        assert "12345" in filename  # Invoice number should be included
        assert filename.endswith(".pdf")

        # Test email filename
        email_pages = sample_pages[2:4]
        filename = splitter._suggest_filename(email_pages, "Email", 2)
        assert "email" in filename.lower()
        assert filename.endswith(".pdf")

        # Test with no identifiable info
        generic_pages = [ProcessedPage(page_number=0, text="Some text")]
        filename = splitter._suggest_filename(generic_pages, "Document", 3)
        assert "document" in filename.lower()
        assert "003" in filename  # Segment number

    def test_date_extraction(self, splitter):
        """Test date extraction from text."""
        # Test MM/DD/YYYY format
        pages = [ProcessedPage(page_number=0, text="Date: 01/15/2024")]
        date = splitter._extract_date(pages)
        assert date == "2024-01-15"

        # Test YYYY-MM-DD format
        pages = [ProcessedPage(page_number=0, text="Date: 2024-01-15")]
        date = splitter._extract_date(pages)
        assert date == "2024-01-15"

        # Test no date
        pages = [ProcessedPage(page_number=0, text="No date here")]
        date = splitter._extract_date(pages)
        assert date is None

    def test_identifier_extraction(self, splitter):
        """Test identifier extraction."""
        # Test invoice number
        pages = [ProcessedPage(page_number=0, text="Invoice #12345")]
        identifier = splitter._extract_identifier(pages, "Invoice")
        assert identifier == "12345"

        # Test drawing number
        pages = [ProcessedPage(page_number=0, text="Drawing #A-101")]
        identifier = splitter._extract_identifier(pages, "Plans")
        assert identifier == "a-101"

        # Test no identifier
        pages = [ProcessedPage(page_number=0, text="No ID here")]
        identifier = splitter._extract_identifier(pages, "Document")
        assert identifier is None

    def test_summary_extraction(self, splitter, sample_pages):
        """Test summary extraction."""
        # Test normal summary
        summary = splitter._extract_summary(sample_pages[:1])
        assert "Invoice #12345" in summary
        assert len(summary) <= 200

        # Test empty pages
        summary = splitter._extract_summary([])
        assert summary == ""

        # Test long text truncation
        long_page = ProcessedPage(page_number=0, text="A" * 300)  # Very long text
        summary = splitter._extract_summary([long_page])
        assert len(summary) == 200
        assert summary.endswith("...")

    def test_filename_sanitization(self, splitter):
        """Test filename sanitization."""
        # Test invalid characters
        filename = splitter._sanitize_filename('test<>:"/\\|?*.pdf')
        assert not any(c in filename for c in '<>:"/\\|?*')
        assert filename == "test_.pdf"  # Multiple underscores are collapsed

        # Test multiple underscores
        filename = splitter._sanitize_filename("test___file.pdf")
        assert filename == "test_file.pdf"

        # Test long filename
        long_name = "a" * 300 + ".pdf"
        filename = splitter._sanitize_filename(long_name)
        assert len(filename) <= 255

    def test_generate_proposal(
        self, splitter, sample_pages, sample_boundaries, tmp_path
    ):
        """Test proposal generation."""
        # Create test PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf")

        proposal = splitter.generate_proposal(
            boundaries=sample_boundaries, pages=sample_pages, pdf_path=pdf_path
        )

        assert isinstance(proposal, SplitProposal)
        assert proposal.pdf_path == pdf_path
        assert proposal.total_pages == 5
        assert len(proposal.segments) == 3

        # Check first segment (invoice)
        seg1 = proposal.segments[0]
        assert seg1.start_page == 0
        assert seg1.end_page == 1
        assert seg1.document_type == "Invoice"
        assert "invoice" in seg1.suggested_filename.lower()

        # Check second segment (email)
        seg2 = proposal.segments[1]
        assert seg2.start_page == 2
        assert seg2.end_page == 3
        assert seg2.document_type == "Email"

        # Check third segment (plans)
        seg3 = proposal.segments[2]
        assert seg3.start_page == 4
        assert seg3.end_page == 4
        assert seg3.document_type == "Plans"

    def test_generate_proposal_nonexistent_pdf(
        self, splitter, sample_pages, sample_boundaries
    ):
        """Test proposal generation with non-existent PDF."""
        with pytest.raises(PDFSplitError, match="PDF file not found"):
            splitter.generate_proposal(
                boundaries=sample_boundaries,
                pages=sample_pages,
                pdf_path=Path("/nonexistent.pdf"),
            )

    @patch("pdf_splitter.splitting.pdf_splitter.pikepdf")
    def test_split_pdf(self, mock_pikepdf, splitter, tmp_path):
        """Test PDF splitting execution."""
        # Setup mocks
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock() for _ in range(5)]
        mock_pdf.metadata = {}
        mock_pikepdf.open.return_value.__enter__.return_value = mock_pdf
        mock_pikepdf.new.return_value = MagicMock()

        # Create proposal
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy")

        segments = [
            DocumentSegment(
                start_page=0,
                end_page=1,
                document_type="Invoice",
                suggested_filename="invoice_001.pdf",
                confidence=0.95,
            ),
            DocumentSegment(
                start_page=2,
                end_page=3,
                document_type="Email",
                suggested_filename="email_002.pdf",
                confidence=0.88,
            ),
        ]

        proposal = SplitProposal(
            pdf_path=pdf_path, total_pages=5, segments=segments, detection_results=[]
        )

        # Execute split
        output_dir = tmp_path / "output"
        result = splitter.split_pdf(proposal, output_dir)

        assert result.session_id == proposal.proposal_id
        assert result.input_pdf == pdf_path
        assert len(result.output_files) == 2
        assert result.file_count == 2
        assert result.duration_seconds >= 0

        # Verify pikepdf was called correctly
        assert mock_pikepdf.open.called
        assert mock_pikepdf.new.call_count == 2

    @patch("pdf_splitter.splitting.pdf_splitter.pikepdf")
    def test_split_pdf_with_custom_names(self, mock_pikepdf, splitter, tmp_path):
        """Test PDF splitting with custom filenames."""
        # Setup mocks
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock() for _ in range(5)]
        mock_pdf.metadata = {}
        mock_pikepdf.open.return_value.__enter__.return_value = mock_pdf
        mock_pikepdf.new.return_value = MagicMock()

        # Create proposal
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy")

        segment = DocumentSegment(
            start_page=0,
            end_page=1,
            document_type="Invoice",
            suggested_filename="invoice_001.pdf",
            confidence=0.95,
        )

        proposal = SplitProposal(
            pdf_path=pdf_path, total_pages=5, segments=[segment], detection_results=[]
        )

        # Execute with custom name
        output_dir = tmp_path / "output"
        custom_names = {segment.segment_id: "custom_name.pdf"}

        result = splitter.split_pdf(proposal, output_dir, custom_names)

        # Check custom name was used
        assert any("custom_name.pdf" in str(f) for f in result.output_files)

    def test_ensure_unique_path(self, splitter, tmp_path):
        """Test unique path generation."""
        # Test non-existent path
        path = tmp_path / "test.pdf"
        unique = splitter._ensure_unique_path(path)
        assert unique == path

        # Create file and test collision
        path.write_bytes(b"exists")
        unique = splitter._ensure_unique_path(path)
        assert unique != path
        assert unique.name == "test_1.pdf"

        # Create another collision
        unique.write_bytes(b"exists")
        unique2 = splitter._ensure_unique_path(path)
        assert unique2.name == "test_2.pdf"

    @patch("pdf_splitter.splitting.pdf_splitter.pikepdf")
    def test_generate_preview(self, mock_pikepdf, splitter, tmp_path):
        """Test preview generation."""
        # Setup mocks
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock() for _ in range(10)]
        mock_pikepdf.open.return_value.__enter__.return_value = mock_pdf

        mock_preview = MagicMock()
        mock_pikepdf.new.return_value = mock_preview

        # Create test data
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy")

        segment = DocumentSegment(
            start_page=2,
            end_page=8,
            document_type="Report",
            suggested_filename="report.pdf",
            confidence=0.9,
        )

        # Generate preview
        splitter.generate_preview(pdf_path, segment, max_pages=3)

        # Verify behavior
        assert mock_pikepdf.open.called
        assert mock_pikepdf.new.called
        # Should only copy 3 pages (2, 3, 4)
        assert len(mock_preview.pages.append.call_args_list) == 3
