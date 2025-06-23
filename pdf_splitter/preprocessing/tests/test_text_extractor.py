"""
Tests for the text extraction module.

These tests validate the TextExtractor's ability to extract high-quality
text from searchable PDFs, assess extraction quality, and handle various
document types.
"""

import json
from pathlib import Path

import pytest

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.core.exceptions import PDFTextExtractionError
from pdf_splitter.preprocessing.pdf_handler import PageType, PDFHandler
from pdf_splitter.preprocessing.text_extractor import (
    ExtractedPage,
    TextBlock,
    TextExtractor,
)


class TestTextExtractor:
    """Test suite for TextExtractor class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PDFConfig()

    @pytest.fixture
    def pdf_handler(self, config):
        """Create PDF handler instance."""
        return PDFHandler(config)

    @pytest.fixture
    def text_extractor(self, pdf_handler):
        """Create text extractor instance."""
        return TextExtractor(pdf_handler)

    def test_init(self, text_extractor, pdf_handler):
        """Test TextExtractor initialization."""
        assert text_extractor.pdf_handler == pdf_handler
        assert isinstance(text_extractor._font_cache, dict)

    def test_extract_without_loaded_pdf(self, text_extractor):
        """Test extraction fails when no PDF is loaded."""
        with pytest.raises(PDFTextExtractionError) as exc_info:
            text_extractor.extract_all_pages()
        assert "No PDF loaded" in str(exc_info.value)

    def test_extract_invalid_page(self, text_extractor, pdf_handler):
        """Test extraction with invalid page number."""
        test_pdf = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_2_ocr.pdf"
        )

        if test_pdf.exists():
            with pdf_handler.load_pdf(test_pdf):
                with pytest.raises(PDFTextExtractionError) as exc_info:
                    text_extractor.extract_page(-1)
                assert "Invalid page number" in str(exc_info.value)

                with pytest.raises(PDFTextExtractionError) as exc_info:
                    text_extractor.extract_page(1000)
                assert "Invalid page number" in str(exc_info.value)

    def test_extract_single_page(self, text_extractor, pdf_handler):
        """Test extracting text from a single page."""
        test_pdf = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_2_ocr.pdf"
        )

        if test_pdf.exists():
            with pdf_handler.load_pdf(test_pdf) as handler:  # noqa: F841
                # Extract first page
                extracted = text_extractor.extract_page(0)

                assert isinstance(extracted, ExtractedPage)
                assert extracted.page_num == 0
                assert extracted.text
                assert extracted.word_count > 0
                assert extracted.char_count > 0
                assert 0.0 <= extracted.quality_score <= 1.0
                assert 0.0 <= extracted.reading_order_confidence <= 1.0

                # Check for blocks
                assert isinstance(extracted.blocks, list)
                if extracted.blocks:
                    first_block = extracted.blocks[0]
                    assert isinstance(first_block, TextBlock)
                    assert first_block.text
                    assert len(first_block.bbox) == 4

    def test_extract_all_pages(self, text_extractor, pdf_handler):
        """Test extracting text from all pages."""
        test_pdf = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_2_ocr.pdf"
        )

        if test_pdf.exists():
            with pdf_handler.load_pdf(test_pdf) as handler:  # noqa: F841
                extracted_pages = text_extractor.extract_all_pages()

                assert isinstance(extracted_pages, list)
                assert len(extracted_pages) > 0

                # Should skip image-based pages
                assert len(extracted_pages) <= handler.page_count

                # Check each extracted page
                for page in extracted_pages:
                    assert isinstance(page, ExtractedPage)
                    assert page.text
                    assert page.quality_score > 0

    def test_page_text_compatibility(self, text_extractor, pdf_handler):
        """Test PageText model compatibility."""
        test_pdf = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_2_ocr.pdf"
        )

        if test_pdf.exists():
            with pdf_handler.load_pdf(test_pdf) as handler:  # noqa: F841
                # Find a searchable page
                for page_num in range(min(5, handler.page_count)):
                    if handler.get_page_type(page_num) != PageType.IMAGE_BASED:
                        page_text = text_extractor.extract_page_text(page_num)

                        assert page_text.text
                        assert page_text.extraction_method == "pymupdf_advanced"
                        assert page_text.char_count > 0
                        assert page_text.word_count > 0
                        assert 0.0 <= page_text.confidence <= 1.0
                        break

    def test_font_analysis(self, text_extractor, pdf_handler):
        """Test font analysis functionality."""
        test_pdf = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_2_ocr.pdf"
        )

        if test_pdf.exists():
            with pdf_handler.load_pdf(test_pdf) as handler:  # noqa: F841
                # Extract a page with text
                for page_num in range(min(5, handler.page_count)):
                    if handler.get_page_type(page_num) != PageType.IMAGE_BASED:
                        extracted = text_extractor.extract_page(page_num)

                        # Should have font information
                        if extracted.avg_font_size > 0:
                            assert extracted.dominant_font  # May be empty for some PDFs
                        break

    def test_header_footer_detection(self, text_extractor, pdf_handler):
        """Test header and footer detection."""
        test_pdf = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_2_ocr.pdf"
        )

        if test_pdf.exists():
            with pdf_handler.load_pdf(test_pdf) as handler:  # noqa: F841
                # Test a few pages
                for page_num in range(min(5, handler.page_count)):
                    if handler.get_page_type(page_num) != PageType.IMAGE_BASED:
                        extracted = text_extractor.extract_page(page_num)

                        # Headers and footers are boolean
                        assert isinstance(extracted.has_headers, bool)
                        assert isinstance(extracted.has_footers, bool)

    def test_document_segments_extraction(self, text_extractor, pdf_handler):
        """Test extracting specific document segments."""
        test_pdf = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_2_ocr.pdf"
        )
        ground_truth_file = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_Ground_Truth.json"
        )

        if test_pdf.exists() and ground_truth_file.exists():
            # Load ground truth
            with open(ground_truth_file, "r") as f:
                ground_truth = json.load(f)

            # Extract first few documents
            page_ranges = []
            for doc in ground_truth["documents"][:3]:
                pages = doc["pages"]
                if "-" in pages:
                    start, end = map(int, pages.split("-"))
                    page_ranges.append((start, end))
                else:
                    page = int(pages)
                    page_ranges.append((page, page))

            with pdf_handler.load_pdf(test_pdf) as handler:  # noqa: F841
                segments = text_extractor.extract_document_segments(page_ranges)

                assert len(segments) == len(page_ranges)

                for i, segment in enumerate(segments):
                    assert segment["document_index"] == i
                    assert segment["full_text"]
                    assert segment["page_count"] > 0
                    assert segment["avg_quality_score"] > 0
                    assert segment["total_word_count"] > 0

    def test_quality_score_calculation(self, text_extractor, pdf_handler):
        """Test quality score calculation for various page types."""
        test_pdf = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_2_ocr.pdf"
        )

        if test_pdf.exists():
            with pdf_handler.load_pdf(test_pdf) as handler:  # noqa: F841
                quality_scores = []

                # Extract several pages to test quality scoring
                for page_num in range(min(10, handler.page_count)):
                    if handler.get_page_type(page_num) != PageType.IMAGE_BASED:
                        extracted = text_extractor.extract_page(page_num)
                        quality_scores.append(extracted.quality_score)

                # Should have reasonable quality scores
                assert all(0.0 <= score <= 1.0 for score in quality_scores)
                assert any(
                    score > 0.5 for score in quality_scores
                )  # At least some good quality

    def test_table_detection(self, text_extractor, pdf_handler):
        """Test table detection functionality."""
        test_pdf = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_2_ocr.pdf"
        )

        if test_pdf.exists():
            with pdf_handler.load_pdf(test_pdf) as handler:  # noqa: F841
                # Look for pages that might contain tables
                # (Based on ground truth, pages 9-12 have Schedule of Values
                # which might be tabular)
                for page_num in range(8, min(12, handler.page_count)):
                    if handler.get_page_type(page_num) != PageType.IMAGE_BASED:
                        extracted = text_extractor.extract_page(page_num)

                        # Tables list should be present (even if empty)
                        assert isinstance(extracted.tables, list)

                        # If tables detected, check structure
                        if extracted.tables:
                            for table in extracted.tables:
                                assert "rows" in table
                                assert "avg_columns" in table
                                assert "confidence" in table

    def test_reading_order_assessment(self, text_extractor, pdf_handler):
        """Test reading order confidence assessment."""
        test_pdf = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_2_ocr.pdf"
        )

        if test_pdf.exists():
            with pdf_handler.load_pdf(test_pdf) as handler:  # noqa: F841
                # Test several pages
                for page_num in range(min(5, handler.page_count)):
                    if handler.get_page_type(page_num) != PageType.IMAGE_BASED:
                        extracted = text_extractor.extract_page(page_num)

                        # Reading order confidence should be between 0 and 1
                        assert 0.0 <= extracted.reading_order_confidence <= 1.0

                        # Most well-formatted PDFs should have good reading order
                        if len(extracted.blocks) > 5:
                            assert extracted.reading_order_confidence > 0.5

    def test_empty_page_handling(self, text_extractor, pdf_handler):
        """Test handling of empty or nearly empty pages."""
        test_pdf = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_2_ocr.pdf"
        )

        if test_pdf.exists():
            with pdf_handler.load_pdf(test_pdf) as handler:  # noqa: F841
                # Find an empty or nearly empty page
                for page_num in range(handler.page_count):
                    page_type = handler.get_page_type(page_num)
                    if page_type == PageType.EMPTY:
                        # Empty pages might be skipped or return minimal content
                        try:
                            extracted = text_extractor.extract_page(page_num)
                            assert (
                                extracted.word_count == 0 or extracted.word_count < 10
                            )
                            assert extracted.quality_score < 0.5
                        except PDFTextExtractionError:
                            # It's ok if extraction fails on empty pages
                            pass
                        break


class TestTextExtractionIntegration:
    """Integration tests for text extraction with real PDFs."""

    def test_ocr_vs_non_ocr_comparison(self):
        """Compare text extraction between OCR and non-OCR versions."""
        ocr_pdf = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_2_ocr.pdf"
        )
        non_ocr_pdf = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_1.pdf"
        )

        if ocr_pdf.exists() and non_ocr_pdf.exists():
            config = PDFConfig()

            # Extract from OCR version
            with PDFHandler(config).load_pdf(ocr_pdf) as handler:
                extractor = TextExtractor(handler)
                ocr_pages = extractor.extract_all_pages()

            # Try to extract from non-OCR version (should skip most pages)
            with PDFHandler(config).load_pdf(non_ocr_pdf) as handler:
                extractor = TextExtractor(handler)
                non_ocr_pages = extractor.extract_all_pages()

            # OCR version should have much more extractable text
            assert len(ocr_pages) > len(non_ocr_pages)

            # OCR version should have actual text content
            total_words_ocr = sum(p.word_count for p in ocr_pages)
            total_words_non_ocr = sum(p.word_count for p in non_ocr_pages)
            assert total_words_ocr > total_words_non_ocr

    def test_ground_truth_alignment(self):
        """Test that extracted text aligns with ground truth document boundaries."""
        test_pdf = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_2_ocr.pdf"
        )
        ground_truth_file = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_files"
            / "Test_PDF_Set_Ground_Truth.json"
        )

        if test_pdf.exists() and ground_truth_file.exists():
            with open(ground_truth_file, "r") as f:
                ground_truth = json.load(f)

            config = PDFConfig()
            with PDFHandler(config).load_pdf(test_pdf) as handler:
                extractor = TextExtractor(handler)

                # Test first document (Email Chain, pages 1-4)
                first_doc = ground_truth["documents"][0]
                assert first_doc["pages"] == "1-4"
                assert first_doc["type"] == "Email Chain"

                # Extract these pages
                segments = extractor.extract_document_segments([(1, 4)])
                assert len(segments) == 1

                segment = segments[0]
                assert segment["page_count"] == 4

                # Check if content matches expected summary
                text = segment["full_text"].lower()
                assert "email" in text or "from:" in text or "to:" in text
                assert "february" in text  # From the summary
