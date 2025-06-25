"""Tests for exception classes."""

import pytest

from pdf_splitter.core.exceptions import (
    ConfigurationError,
    DetectionError,
    OCREngineError,
    OCRError,
    PDFHandlerError,
    PDFProcessingError,
    PDFRenderError,
    PDFSplitterError,
    PDFTextExtractionError,
    PDFValidationError,
    SplittingError,
)


class TestExceptionHierarchy:
    """Test exception class hierarchy."""

    def test_base_exception(self):
        """Test base exception class."""
        exc = PDFSplitterError("Test error")
        assert str(exc) == "Test error"
        assert isinstance(exc, Exception)

    def test_pdf_handler_exceptions(self):
        """Test PDF handler exceptions."""
        # PDFHandlerError
        exc = PDFHandlerError("Handler error")
        assert isinstance(exc, PDFSplitterError)
        assert str(exc) == "Handler error"

        # PDFValidationError - note: it inherits from PDFProcessingError,
        # not PDFHandlerError
        exc = PDFValidationError("Invalid PDF")
        assert isinstance(exc, PDFProcessingError)
        assert str(exc) == "Invalid PDF"

        # PDFRenderError
        exc = PDFRenderError("Render failed")
        assert isinstance(exc, PDFHandlerError)
        assert str(exc) == "Render failed"

    def test_ocr_exceptions(self):
        """Test OCR exceptions."""
        # OCRError
        exc = OCRError("OCR failed")
        assert isinstance(exc, PDFSplitterError)
        assert str(exc) == "OCR failed"

        # OCREngineError
        exc = OCREngineError("Engine failed")
        assert isinstance(exc, OCRError)
        assert str(exc) == "Engine failed"

    def test_other_exceptions(self):
        """Test other exception types."""
        # PDFTextExtractionError
        exc = PDFTextExtractionError("Extraction failed")
        assert isinstance(exc, PDFSplitterError)
        assert str(exc) == "Extraction failed"

        # DetectionError
        exc = DetectionError("Detection failed")
        assert isinstance(exc, PDFSplitterError)
        assert str(exc) == "Detection failed"

        # SplittingError
        exc = SplittingError("Splitting failed")
        assert isinstance(exc, PDFSplitterError)
        assert str(exc) == "Splitting failed"

        # ConfigurationError
        exc = ConfigurationError("Config error")
        assert isinstance(exc, PDFSplitterError)
        assert str(exc) == "Config error"


class TestExceptionUsage:
    """Test exception usage patterns."""

    def test_exception_with_context(self):
        """Test exceptions with additional context."""
        try:
            raise PDFHandlerError("Failed to load PDF") from ValueError("Invalid path")
        except PDFHandlerError as e:
            assert str(e) == "Failed to load PDF"
            assert isinstance(e.__cause__, ValueError)

    def test_exception_chaining(self):
        """Test exception chaining."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as ve:
                raise PDFValidationError("PDF validation failed") from ve
        except PDFValidationError as e:
            assert str(e) == "PDF validation failed"
            assert str(e.__cause__) == "Original error"

    def test_exception_attributes(self):
        """Test custom exception attributes."""
        # Test that exceptions can carry additional data
        exc = PDFHandlerError("Error with data")
        exc.page_num = 5
        exc.pdf_path = "/path/to/pdf"

        assert exc.page_num == 5
        assert exc.pdf_path == "/path/to/pdf"

    def test_exception_formatting(self):
        """Test exception message formatting."""
        page_num = 10
        error_msg = f"Failed to render page {page_num}"
        exc = PDFRenderError(error_msg)

        assert str(exc) == "Failed to render page 10"


class TestExceptionHandling:
    """Test exception handling patterns."""

    def test_catch_specific_exception(self):
        """Test catching specific exceptions."""

        def risky_operation():
            raise OCREngineError("Engine not available")

        with pytest.raises(OCREngineError) as exc_info:
            risky_operation()

        assert "Engine not available" in str(exc_info.value)

    def test_catch_parent_exception(self):
        """Test catching parent exceptions."""

        def risky_operation():
            raise PDFValidationError("Invalid PDF format")

        # Should be caught by PDFProcessingError (parent of PDFValidationError)
        with pytest.raises(PDFProcessingError):
            risky_operation()

        # Should also be caught by PDFSplitterError
        with pytest.raises(PDFSplitterError):
            risky_operation()

    def test_exception_propagation(self):
        """Test exception propagation through layers."""

        def low_level():
            raise ValueError("Low level error")

        def mid_level():
            try:
                low_level()
            except ValueError as e:
                raise OCRError("OCR processing failed") from e

        def high_level():
            try:
                mid_level()
            except OCRError as e:
                raise PDFSplitterError("PDF processing failed") from e

        with pytest.raises(PDFSplitterError) as exc_info:
            high_level()

        # Check the exception chain
        exc = exc_info.value
        assert str(exc) == "PDF processing failed"
        assert isinstance(exc.__cause__, OCRError)
        assert isinstance(exc.__cause__.__cause__, ValueError)
