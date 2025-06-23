"""
Comprehensive unit tests for PDFHandler class.

Tests cover PDF loading, validation, page rendering, text extraction,
and streaming capabilities.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.core.exceptions import (
    PDFHandlerError,
    PDFRenderError,
    PDFTextExtractionError,
    PDFValidationError,
)
from pdf_splitter.preprocessing.pdf_handler import (
    PageInfo,
    PageText,
    PageType,
    PDFHandler,
    ProcessingEstimate,
)


class TestPDFHandler:
    """Test suite for PDFHandler class."""

    @pytest.fixture
    def pdf_config(self):
        """Create a test PDF configuration."""
        return PDFConfig(
            default_dpi=150,
            max_dpi=300,
            max_file_size_mb=100,
            max_pages=1000,
            page_cache_size=5,
            stream_batch_size=3,
        )

    @pytest.fixture
    def pdf_handler(self, pdf_config):
        """Create a PDFHandler instance for testing."""
        return PDFHandler(config=pdf_config)

    @pytest.fixture
    def mock_pdf_path(self, tmp_path):
        """Create a temporary PDF path."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")
        return pdf_path

    def test_initialization(self, pdf_config):
        """Test PDFHandler initialization."""
        handler = PDFHandler(config=pdf_config)
        assert handler.config == pdf_config
        assert not handler.is_loaded
        assert handler.page_count == 0
        assert handler._document is None
        assert handler._pdf_path is None
        assert len(handler._page_cache) == 0
        assert handler._metadata is None

    def test_initialization_without_config(self):
        """Test PDFHandler initialization with default config."""
        handler = PDFHandler()
        assert isinstance(handler.config, PDFConfig)
        assert handler.config.default_dpi == 150

    def test_validate_pdf_file_not_exists(self, pdf_handler, tmp_path):
        """Test validation when file doesn't exist."""
        non_existent = tmp_path / "missing.pdf"
        result = pdf_handler.validate_pdf(non_existent)

        assert not result.is_valid
        assert result.page_count == 0
        assert "File does not exist" in result.errors[0]

    def test_validate_pdf_not_a_file(self, pdf_handler, tmp_path):
        """Test validation when path is a directory."""
        result = pdf_handler.validate_pdf(tmp_path)

        assert not result.is_valid
        assert "Path is not a file" in result.errors[0]

    @patch("fitz.open")
    def test_validate_pdf_success(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test successful PDF validation."""
        # Mock document
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=10)
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.metadata = {
            "format": "PDF-1.7",
            "producer": "Test Producer",
            "creator": "Test Creator",
        }
        mock_fitz_open.return_value.__enter__.return_value = mock_doc
        mock_fitz_open.return_value.__exit__.return_value = None

        result = pdf_handler.validate_pdf(mock_pdf_path)

        assert result.is_valid
        assert result.page_count == 10
        assert result.file_size_mb > 0
        assert not result.is_encrypted
        assert not result.is_damaged
        assert result.pdf_version == "PDF-1.7"
        assert len(result.errors) == 0

    @patch("fitz.open")
    def test_validate_pdf_encrypted(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test validation of encrypted PDF."""
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=5)
        mock_doc.is_encrypted = True
        mock_doc.authenticate = Mock(return_value=False)
        mock_fitz_open.return_value.__enter__.return_value = mock_doc
        mock_fitz_open.return_value.__exit__.return_value = None

        result = pdf_handler.validate_pdf(mock_pdf_path)

        assert not result.is_valid
        assert result.is_encrypted
        assert "PDF is password protected" in result.errors[0]

    @patch("fitz.open")
    def test_validate_pdf_damaged(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test validation of damaged PDF."""
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=5)
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = True
        mock_doc.metadata = {}
        mock_fitz_open.return_value.__enter__.return_value = mock_doc
        mock_fitz_open.return_value.__exit__.return_value = None

        result = pdf_handler.validate_pdf(mock_pdf_path)

        assert result.is_valid  # Damaged but repaired PDFs are valid
        assert result.is_damaged
        assert "PDF was damaged and repaired" in result.warnings[0]

    @patch("fitz.open")
    def test_validate_pdf_too_large(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test validation of PDF exceeding size limit."""
        # Make file appear larger than limit
        mock_stat = Mock()
        mock_stat.st_size = 200 * 1024 * 1024  # 200 MB

        with patch.object(mock_pdf_path, "stat", return_value=mock_stat):
            mock_doc = Mock()
            mock_doc.__len__ = Mock(return_value=10)
            mock_doc.is_encrypted = False
            mock_doc.is_repaired = False
            mock_doc.metadata = {}
            mock_fitz_open.return_value.__enter__.return_value = mock_doc
            mock_fitz_open.return_value.__exit__.return_value = None

            result = pdf_handler.validate_pdf(mock_pdf_path)

            assert not result.is_valid
            assert "exceeds maximum" in result.errors[0]

    @patch("fitz.open")
    def test_load_pdf_success(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test successful PDF loading."""
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=5)
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.is_fast = True
        mock_doc.metadata = {"title": "Test PDF", "author": "Test Author"}
        mock_doc.close = Mock()
        mock_fitz_open.return_value = mock_doc

        with pdf_handler.load_pdf(mock_pdf_path, validate=False) as handler:
            assert handler.is_loaded
            assert handler.page_count == 5
            assert handler._pdf_path == mock_pdf_path
            assert handler._metadata is not None
            assert handler._metadata.title == "Test PDF"

        # Check cleanup after context exit
        assert not pdf_handler.is_loaded
        assert pdf_handler._document is None
        mock_doc.close.assert_called_once()

    def test_load_pdf_validation_failure(self, pdf_handler, tmp_path):
        """Test PDF loading with validation failure."""
        non_existent = tmp_path / "missing.pdf"

        with pytest.raises(PDFValidationError, match="PDF validation failed"):
            with pdf_handler.load_pdf(non_existent):
                pass

    @patch("fitz.open")
    def test_load_pdf_open_failure(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test PDF loading when fitz.open fails."""
        mock_fitz_open.side_effect = Exception("Failed to open")

        with pytest.raises(PDFHandlerError, match="Failed to load PDF"):
            with pdf_handler.load_pdf(mock_pdf_path, validate=False):
                pass

    @patch("fitz.open")
    def test_get_page_type_searchable(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test page type detection for searchable pages."""
        # Setup mock document and page
        mock_page = Mock()
        mock_page.get_text.return_value = (
            "This is a test page with lots of text content."
        )
        mock_page.get_images.return_value = []
        mock_page.rect = Mock(width=612, height=792)
        mock_page.get_text.return_value = "Test text"
        mock_page.rotation = 0
        mock_page.annots.return_value = []

        # Mock text blocks for coverage calculation
        text_blocks = [
            (0, 0, 600, 50, "Header text", 0, 0),
            (0, 100, 600, 700, "Main content with lots of text", 0, 0),
        ]
        mock_page.get_text = Mock(
            side_effect=lambda fmt="text": text_blocks
            if fmt == "blocks"
            else "Test text content"
        )

        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.metadata = {}
        mock_fitz_open.return_value = mock_doc

        with pdf_handler.load_pdf(mock_pdf_path, validate=False):
            page_type = pdf_handler.get_page_type(0)

        assert page_type == PageType.SEARCHABLE

    @patch("fitz.open")
    def test_get_page_type_image_based(
        self, mock_fitz_open, pdf_handler, mock_pdf_path
    ):
        """Test page type detection for image-based pages."""
        mock_page = Mock()
        mock_page.get_text.return_value = ""  # No text
        mock_page.get_images.return_value = [("img1",), ("img2",)]  # Has images
        mock_page.rect = Mock(width=612, height=792)
        mock_page.rotation = 0
        mock_page.annots.return_value = []
        mock_page.get_text = Mock(
            side_effect=lambda fmt="text": [] if fmt == "blocks" else ""
        )

        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.metadata = {}
        mock_fitz_open.return_value = mock_doc

        with pdf_handler.load_pdf(mock_pdf_path, validate=False):
            page_type = pdf_handler.get_page_type(0)

        assert page_type == PageType.IMAGE_BASED

    @patch("fitz.open")
    def test_get_page_type_empty(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test page type detection for empty pages."""
        mock_page = Mock()
        mock_page.get_text.return_value = ""  # No text
        mock_page.get_images.return_value = []  # No images
        mock_page.rect = Mock(width=612, height=792)
        mock_page.rotation = 0
        mock_page.annots.return_value = []
        mock_page.get_text = Mock(
            side_effect=lambda fmt="text": [] if fmt == "blocks" else ""
        )

        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.metadata = {}
        mock_fitz_open.return_value = mock_doc

        with pdf_handler.load_pdf(mock_pdf_path, validate=False):
            page_type = pdf_handler.get_page_type(0)

        assert page_type == PageType.EMPTY

    def test_get_page_type_no_pdf_loaded(self, pdf_handler):
        """Test page type detection when no PDF is loaded."""
        with pytest.raises(PDFHandlerError, match="No PDF loaded"):
            pdf_handler.get_page_type(0)

    def test_get_page_type_invalid_page_number(self, pdf_handler):
        """Test page type detection with invalid page number."""
        pdf_handler._document = Mock()
        pdf_handler._document.__len__ = Mock(return_value=5)

        with pytest.raises(PDFHandlerError, match="Invalid page number"):
            pdf_handler.get_page_type(10)

    @patch("fitz.open")
    def test_render_page_success(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test successful page rendering."""
        # Create mock pixmap
        mock_pix = Mock()
        mock_pix.samples = np.zeros(612 * 792 * 3, dtype=np.uint8).tobytes()
        mock_pix.height = 792
        mock_pix.width = 612

        mock_page = Mock()
        mock_page.get_pixmap = Mock(return_value=mock_pix)

        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.metadata = {}
        mock_fitz_open.return_value = mock_doc

        with pdf_handler.load_pdf(mock_pdf_path, validate=False):
            img_array = pdf_handler.render_page(0)

        assert isinstance(img_array, np.ndarray)
        assert img_array.shape == (792, 612, 3)
        assert img_array.dtype == np.uint8

    @patch("fitz.open")
    def test_render_page_with_custom_dpi(
        self, mock_fitz_open, pdf_handler, mock_pdf_path
    ):
        """Test page rendering with custom DPI."""
        mock_pix = Mock()
        # Higher DPI = larger image
        mock_pix.samples = np.zeros(1224 * 1584 * 3, dtype=np.uint8).tobytes()
        mock_pix.height = 1584
        mock_pix.width = 1224

        mock_page = Mock()
        mock_page.get_pixmap = Mock(return_value=mock_pix)

        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.metadata = {}
        mock_fitz_open.return_value = mock_doc

        with pdf_handler.load_pdf(mock_pdf_path, validate=False):
            pdf_handler.render_page(0, dpi=300)

        # Check that matrix was calculated correctly
        mock_page.get_pixmap.assert_called_once()
        call_args = mock_page.get_pixmap.call_args
        assert "matrix" in call_args[1]
        assert call_args[1]["alpha"] is False

    @patch("fitz.open")
    def test_render_page_caching(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test that rendered pages are cached."""
        mock_pix = Mock()
        mock_pix.samples = np.zeros(612 * 792 * 3, dtype=np.uint8).tobytes()
        mock_pix.height = 792
        mock_pix.width = 612

        mock_page = Mock()
        mock_page.get_pixmap = Mock(return_value=mock_pix)

        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.metadata = {}
        mock_fitz_open.return_value = mock_doc

        with pdf_handler.load_pdf(mock_pdf_path, validate=False):
            # First render
            img1 = pdf_handler.render_page(0)
            # Second render should use cache
            img2 = pdf_handler.render_page(0)

        # get_pixmap should only be called once due to caching
        mock_page.get_pixmap.assert_called_once()
        # Arrays should be equal but not the same object (copy returned)
        assert np.array_equal(img1, img2)
        assert img1 is not img2

    @patch("fitz.open")
    def test_render_page_error(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test page rendering error handling."""
        mock_page = Mock()
        mock_page.get_pixmap.side_effect = Exception("Render failed")

        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.metadata = {}
        mock_fitz_open.return_value = mock_doc

        with pdf_handler.load_pdf(mock_pdf_path, validate=False):
            with pytest.raises(PDFRenderError, match="Failed to render page"):
                pdf_handler.render_page(0)

    @patch("fitz.open")
    def test_extract_text_success(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test successful text extraction."""
        mock_page = Mock()
        mock_page.get_text.return_value = "This is test text content."
        mock_page.get_images.return_value = [("img1",)]
        mock_page.find_tables.return_value = [Mock()]  # One table

        # Mock text blocks
        text_blocks = [
            (0, 0, 100, 20, "Header", 0, 0),
            (0, 30, 100, 50, "Content", 0, 0),
        ]
        mock_page.get_text = Mock(
            side_effect=lambda fmt="text": text_blocks
            if fmt == "blocks"
            else "This is test text content."
        )

        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.metadata = {}
        mock_fitz_open.return_value = mock_doc

        with pdf_handler.load_pdf(mock_pdf_path, validate=False):
            page_text = pdf_handler.extract_text(0)

        assert isinstance(page_text, PageText)
        assert page_text.text == "This is test text content."
        assert page_text.char_count == 26
        assert page_text.word_count == 5
        assert page_text.has_tables
        assert page_text.has_images
        assert page_text.extraction_method == "pdfplumber_direct"
        assert 0 <= page_text.confidence <= 1

    @patch("fitz.open")
    def test_extract_text_error(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test text extraction error handling."""
        mock_page = Mock()
        mock_page.get_text.side_effect = Exception("Extraction failed")

        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.metadata = {}
        mock_fitz_open.return_value = mock_doc

        with pdf_handler.load_pdf(mock_pdf_path, validate=False):
            with pytest.raises(PDFTextExtractionError, match="Failed to extract text"):
                pdf_handler.extract_text(0)

    def test_estimate_text_confidence(self, pdf_handler):
        """Test text confidence estimation."""
        # Good text
        good_text = "This is normal text with proper words."
        blocks = [(0, 0, 100, 20, "text", 0, 0)] * 10
        confidence = pdf_handler._estimate_text_confidence(good_text, blocks)
        assert confidence > 0.8

        # Text with many special characters
        bad_text = "@#$%^&*()_+{}|:<>?"
        confidence = pdf_handler._estimate_text_confidence(bad_text, [])
        assert confidence < 0.8

        # Empty text
        confidence = pdf_handler._estimate_text_confidence("", [])
        assert confidence == 0.0

    @patch("fitz.open")
    def test_analyze_all_pages(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test analyzing all pages in parallel."""
        # Create 3 mock pages with different types
        pages = []
        for i in range(3):
            mock_page = Mock()
            if i == 0:  # Searchable
                mock_page.get_text.return_value = "Text content"
                mock_page.get_images.return_value = []
            elif i == 1:  # Image-based
                mock_page.get_text.return_value = ""
                mock_page.get_images.return_value = [("img",)]
            else:  # Empty
                mock_page.get_text.return_value = ""
                mock_page.get_images.return_value = []

            mock_page.rect = Mock(width=612, height=792)
            mock_page.rotation = 0
            mock_page.annots.return_value = []
            mock_page.get_text = Mock(
                side_effect=lambda fmt="text": []
                if fmt == "blocks"
                else mock_page.get_text.return_value
            )
            pages.append(mock_page)

        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=3)
        mock_doc.__getitem__ = Mock(side_effect=lambda i: pages[i])
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.metadata = {"title": "Test"}
        mock_fitz_open.return_value = mock_doc

        with pdf_handler.load_pdf(mock_pdf_path, validate=False):
            page_infos = pdf_handler.analyze_all_pages(max_workers=2)

        assert len(page_infos) == 3
        assert all(isinstance(info, PageInfo) for info in page_infos)

        # Check metadata was updated with summary
        metadata = pdf_handler.get_metadata()
        assert PageType.SEARCHABLE in metadata.page_info_summary
        assert PageType.IMAGE_BASED in metadata.page_info_summary
        assert PageType.EMPTY in metadata.page_info_summary

    @patch("fitz.open")
    def test_estimate_processing_time(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test processing time estimation."""
        # Setup mock pages
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=10)
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.metadata = {}
        mock_fitz_open.return_value = mock_doc

        with pdf_handler.load_pdf(mock_pdf_path, validate=False):
            # Manually populate page info cache
            pdf_handler._page_info_cache = {
                0: PageInfo(
                    page_num=0,
                    width=612,
                    height=792,
                    rotation=0,
                    page_type=PageType.SEARCHABLE,
                    text_percentage=80,
                ),
                1: PageInfo(
                    page_num=1,
                    width=612,
                    height=792,
                    rotation=0,
                    page_type=PageType.IMAGE_BASED,
                    text_percentage=0,
                ),
                2: PageInfo(
                    page_num=2,
                    width=612,
                    height=792,
                    rotation=0,
                    page_type=PageType.MIXED,
                    text_percentage=30,
                ),
                3: PageInfo(
                    page_num=3,
                    width=612,
                    height=792,
                    rotation=0,
                    page_type=PageType.EMPTY,
                    text_percentage=0,
                ),
            }
            # Add more pages
            for i in range(4, 10):
                pdf_handler._page_info_cache[i] = PageInfo(
                    page_num=i,
                    width=612,
                    height=792,
                    rotation=0,
                    page_type=PageType.SEARCHABLE,
                    text_percentage=90,
                )

            estimate = pdf_handler.estimate_processing_time()

        assert isinstance(estimate, ProcessingEstimate)
        assert estimate.total_pages == 10
        assert estimate.searchable_pages == 7
        assert estimate.requires_ocr_pages == 2  # IMAGE_BASED + MIXED
        assert estimate.mixed_pages == 1
        assert estimate.empty_pages == 1
        assert estimate.estimated_seconds > 0
        assert estimate.estimated_memory_mb > 0

    @patch("fitz.open")
    def test_stream_pages(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test streaming pages in batches."""
        # Create mock pages
        pages = []
        for i in range(5):
            mock_page = Mock()
            mock_page.get_text.return_value = f"Page {i} text"
            mock_page.get_images.return_value = []
            mock_page.rect = Mock(width=612, height=792)
            mock_page.rotation = 0
            mock_page.annots.return_value = []
            mock_page.find_tables.return_value = []

            # Mock for rendering
            mock_pix = Mock()
            mock_pix.samples = np.zeros(612 * 792 * 3, dtype=np.uint8).tobytes()
            mock_pix.height = 792
            mock_pix.width = 612
            mock_page.get_pixmap = Mock(return_value=mock_pix)

            # Mock for text blocks
            mock_page.get_text = Mock(
                side_effect=lambda fmt="text": [(0, 0, 100, 20, f"Page {i} text", 0, 0)]
                if fmt == "blocks"
                else f"Page {i} text"
            )

            pages.append(mock_page)

        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=5)
        mock_doc.__getitem__ = Mock(side_effect=lambda i: pages[i])
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.metadata = {}
        mock_fitz_open.return_value = mock_doc

        with pdf_handler.load_pdf(mock_pdf_path, validate=False):
            batches = list(pdf_handler.stream_pages(batch_size=2))

        assert len(batches) == 3  # 5 pages / 2 per batch = 3 batches

        # Check first batch
        assert batches[0].start_idx == 0
        assert batches[0].end_idx == 2
        assert batches[0].batch_size == 2
        assert len(batches[0].pages) == 2

        # Check last batch (partial)
        assert batches[2].start_idx == 4
        assert batches[2].end_idx == 5
        assert batches[2].batch_size == 1

        # Check page data
        first_page = batches[0].pages[0]
        assert first_page["page_num"] == 0
        assert first_page["page_type"] == PageType.SEARCHABLE
        assert isinstance(first_page["image"], np.ndarray)
        assert isinstance(first_page["text"], PageText)

    @patch("fitz.open")
    def test_stream_pages_with_error(self, mock_fitz_open, pdf_handler, mock_pdf_path):
        """Test streaming pages with error handling."""
        mock_page = Mock()
        mock_page.get_text.side_effect = Exception("Page error")

        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.metadata = {}
        mock_fitz_open.return_value = mock_doc

        with pdf_handler.load_pdf(mock_pdf_path, validate=False):
            batches = list(pdf_handler.stream_pages())

        assert len(batches) == 1
        assert "error" in batches[0].pages[0]
        assert batches[0].pages[0]["page_num"] == 0

    @patch("fitz.open")
    def test_save_page_image(
        self, mock_fitz_open, pdf_handler, mock_pdf_path, tmp_path
    ):
        """Test saving a rendered page as an image."""
        # Setup rendering
        mock_pix = Mock()
        mock_pix.samples = np.zeros(612 * 792 * 3, dtype=np.uint8).tobytes()
        mock_pix.height = 792
        mock_pix.width = 612

        mock_page = Mock()
        mock_page.get_pixmap = Mock(return_value=mock_pix)

        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.is_encrypted = False
        mock_doc.is_repaired = False
        mock_doc.metadata = {}
        mock_fitz_open.return_value = mock_doc

        # Mock PIL Image
        with patch("pdf_splitter.preprocessing.pdf_handler.Image") as mock_image:
            mock_img = Mock()
            mock_image.fromarray = Mock(return_value=mock_img)

            output_path = tmp_path / "page_0.png"

            with pdf_handler.load_pdf(mock_pdf_path, validate=False):
                pdf_handler.save_page_image(0, output_path)

            mock_image.fromarray.assert_called_once()
            mock_img.save.assert_called_once_with(output_path, format="PNG")

    def test_close_cleanup(self, pdf_handler):
        """Test that close properly cleans up resources."""
        # Setup some state
        pdf_handler._document = Mock()
        pdf_handler._pdf_path = Path("/test/path.pdf")
        pdf_handler._page_cache = {0: np.zeros((100, 100, 3))}
        pdf_handler._page_info_cache = {0: Mock()}
        pdf_handler._metadata = Mock()

        pdf_handler.close()

        assert pdf_handler._document is None
        assert pdf_handler._pdf_path is None
        assert len(pdf_handler._page_cache) == 0
        assert len(pdf_handler._page_info_cache) == 0
        assert pdf_handler._metadata is None
