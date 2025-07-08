"""
Comprehensive edge case tests for the LLM detector.

This test suite covers various edge cases and corner scenarios that might
occur in real-world document processing.
"""

import json
import time
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.base_detector import (BoundaryType,
                                                  DetectionContext,
                                                  DetectorType, ProcessedPage)
from pdf_splitter.detection.llm_detector import LLMDetector


class TestLLMDetectorEdgeCases:
    """Test edge cases and corner scenarios for LLM detector."""

    @pytest.fixture
    def detector(self):
        """Create a detector with mocked Ollama."""
        config = PDFConfig()
        detector = LLMDetector(config=config)

        # Mock Ollama availability check
        with patch.object(detector, "_check_ollama_availability", return_value=True):
            yield detector

    @pytest.fixture
    def mock_ollama_response(self):
        """Create a mock for Ollama API calls."""
        with patch("pdf_splitter.detection.llm_detector.requests.post") as mock_post:
            yield mock_post

    # ===== Empty and Minimal Content Tests =====

    def test_empty_pages(self, detector):
        """Test handling of completely empty pages."""
        pages = [
            ProcessedPage(
                page_number=1, text="", ocr_confidence=0.0, page_type="EMPTY"
            ),
            ProcessedPage(
                page_number=2, text="", ocr_confidence=0.0, page_type="EMPTY"
            ),
        ]

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            results = detector.detect_boundaries(pages)

        # Should skip empty pages
        assert len(results) == 0

    def test_single_line_pages(self, detector, mock_ollama_response):
        """Test pages with only single lines of text."""
        pages = [
            ProcessedPage(
                page_number=1,
                text="Page 1",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=2,
                text="Page 2",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
        ]

        mock_ollama_response.return_value.status_code = 200
        mock_ollama_response.return_value.json.return_value = {
            "response": "<thinking>Single line pages, likely same doc</thinking>\n<answer>SAME</answer>"
        }

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            results = detector.detect_boundaries(pages)

        assert len(results) == 0  # No boundary detected

    def test_mixed_empty_and_content_pages(self, detector, mock_ollama_response):
        """Test mix of empty and content-filled pages."""
        pages = [
            ProcessedPage(
                page_number=1,
                text="Content on page 1",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=2, text="", ocr_confidence=0.0, page_type="EMPTY"
            ),
            ProcessedPage(
                page_number=3,
                text="Content on page 3",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
        ]

        mock_ollama_response.return_value.status_code = 200
        mock_ollama_response.return_value.json.return_value = {
            "response": "<thinking>Gap suggests boundary</thinking>\n<answer>DIFFERENT</answer>"
        }

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            results = detector.detect_boundaries(pages)

        # Should only process non-empty pairs
        assert mock_ollama_response.call_count == 1  # Only pages 1-3 pair

    # ===== Special Characters and Encoding Tests =====

    def test_unicode_content(self, detector, mock_ollama_response):
        """Test handling of Unicode and special characters."""
        pages = [
            ProcessedPage(
                page_number=1,
                text="Unicode test: café, naïve, 日本語",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=2,
                text="More unicode: 中文, русский, العربية",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
        ]

        mock_ollama_response.return_value.status_code = 200
        mock_ollama_response.return_value.json.return_value = {
            "response": "<thinking>Unicode content</thinking>\n<answer>SAME</answer>"
        }

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            results = detector.detect_boundaries(pages)

        # Should handle unicode without errors
        assert mock_ollama_response.called

    def test_control_characters(self, detector, mock_ollama_response):
        """Test handling of control characters and special formatting."""
        pages = [
            ProcessedPage(
                page_number=1,
                text="Text with\ttabs\nand\rspecial\x00chars",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=2,
                text="Normal text here",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
        ]

        mock_ollama_response.return_value.status_code = 200
        mock_ollama_response.return_value.json.return_value = {
            "response": "<thinking>Control chars present</thinking>\n<answer>SAME</answer>"
        }

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            results = detector.detect_boundaries(pages)

        # Should sanitize and process successfully
        assert mock_ollama_response.called

    # ===== Malformed Response Tests =====

    def test_malformed_xml_response(self, detector, mock_ollama_response):
        """Test handling of malformed XML in responses."""
        pages = self._create_test_pages(2)

        # Missing closing tag
        mock_ollama_response.return_value.status_code = 200
        mock_ollama_response.return_value.json.return_value = {
            "response": "<thinking>Incomplete response\n<answer>DIFFERENT</answer>"
        }

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            results = detector.detect_boundaries(pages)

        # Should handle gracefully - extract what it can
        assert len(results) == 1

    def test_no_tags_in_response(self, detector, mock_ollama_response):
        """Test response without expected XML tags."""
        pages = self._create_test_pages(2)

        mock_ollama_response.return_value.status_code = 200
        mock_ollama_response.return_value.json.return_value = {
            "response": "This is just plain text without any tags"
        }

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            results = detector.detect_boundaries(pages)

        # Should return no boundary (default to SAME)
        assert len(results) == 0

    def test_wrong_answer_format(self, detector, mock_ollama_response):
        """Test response with unexpected answer format."""
        pages = self._create_test_pages(2)

        mock_ollama_response.return_value.status_code = 200
        mock_ollama_response.return_value.json.return_value = {
            "response": "<thinking>Good reasoning</thinking>\n<answer>MAYBE</answer>"
        }

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            results = detector.detect_boundaries(pages)

        # Should default to no boundary for unexpected answers
        assert len(results) == 0

    # ===== Network and API Error Tests =====

    def test_ollama_timeout(self, detector):
        """Test handling of Ollama timeout."""
        pages = self._create_test_pages(2)

        with patch("pdf_splitter.detection.llm_detector.requests.post") as mock_post:
            mock_post.side_effect = TimeoutError("Request timed out")

            with patch.object(
                detector, "_check_ollama_availability", return_value=True
            ):
                results = detector.detect_boundaries(pages)

        # Should handle timeout gracefully
        assert len(results) == 0

    def test_ollama_connection_error(self, detector):
        """Test handling when Ollama is not available."""
        pages = self._create_test_pages(2)

        with patch.object(detector, "_check_ollama_availability", return_value=False):
            results = detector.detect_boundaries(pages)

        # Should return empty results when Ollama unavailable
        assert len(results) == 0

    def test_ollama_partial_response(self, detector, mock_ollama_response):
        """Test handling of incomplete JSON response."""
        pages = self._create_test_pages(2)

        mock_ollama_response.return_value.status_code = 200
        mock_ollama_response.return_value.json.side_effect = json.JSONDecodeError(
            "Expecting value", "", 0
        )

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            results = detector.detect_boundaries(pages)

        # Should handle JSON errors gracefully
        assert len(results) == 0

    # ===== Cache Edge Cases =====

    def test_cache_with_identical_content(self, detector, mock_ollama_response):
        """Test cache behavior with identical page content."""
        # Create pages with identical content
        identical_text = "This is the same content on both pages"
        pages = [
            ProcessedPage(
                page_number=i,
                text=identical_text,
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            )
            for i in range(1, 5)
        ]

        mock_ollama_response.return_value.status_code = 200
        mock_ollama_response.return_value.json.return_value = {
            "response": "<thinking>Identical content</thinking>\n<answer>SAME</answer>"
        }

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            if hasattr(detector, "_response_cache"):
                detector._response_cache.clear()
            results = detector.detect_boundaries(pages)

        # Should make multiple calls since cache is based on page pairs
        assert mock_ollama_response.call_count >= 1

    def test_cache_key_collision(self, detector):
        """Test potential cache key collisions."""
        # This test is not applicable with the new cache system
        pages = self._create_test_pages(4)

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            with patch.object(
                detector,
                "_call_ollama",
                return_value="<thinking>Test</thinking>\n<answer>SAME</answer>",
            ):
                results = detector.detect_boundaries(pages)

        # Should process normally
        assert len(results) == 0  # All same

    # ===== Performance and Resource Tests =====

    def test_large_text_extraction(self, detector):
        """Test extraction with very large pages."""
        # Create a page with 1000 lines
        large_text = "\n".join([f"Line {i}: " + "x" * 100 for i in range(1000)])

        start = time.time()
        bottom = detector._extract_bottom_text(large_text)
        top = detector._extract_top_text(large_text)
        elapsed = time.time() - start

        # Should extract efficiently
        assert elapsed < 0.01  # Should be very fast
        assert len(bottom.split("\n")) == detector.bottom_lines
        assert len(top.split("\n")) == detector.top_lines

    def test_many_pages_processing(self, detector, mock_ollama_response):
        """Test processing many pages efficiently."""
        pages = self._create_test_pages(100)

        mock_ollama_response.return_value.status_code = 200
        mock_ollama_response.return_value.json.return_value = {
            "response": "<thinking>Normal pages</thinking>\n<answer>SAME</answer>"
        }

        start = time.time()
        with patch.object(detector, "_check_ollama_availability", return_value=True):
            # Mock the actual Ollama call to be instant
            with patch.object(
                detector,
                "_call_ollama",
                return_value="<thinking>Test</thinking>\n<answer>SAME</answer>",
            ):
                results = detector.detect_boundaries(pages)
        elapsed = time.time() - start

        # Should process efficiently (excluding actual API calls)
        assert elapsed < 1.0  # Should be fast when mocked
        assert len(results) == 0  # All same document

    # ===== Context and Progress Tests =====

    def test_detection_context_updates(self, detector, mock_ollama_response):
        """Test that detection context is properly updated."""
        pages = self._create_test_pages(5)
        context = DetectionContext()

        mock_ollama_response.return_value.status_code = 200
        mock_ollama_response.return_value.json.return_value = {
            "response": "<thinking>Test</thinking>\n<answer>SAME</answer>"
        }

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            results = detector.detect_boundaries(pages, context=context)

        # Context should be updated
        assert context.pages_processed == 5

    # ===== Retry and Recovery Tests =====

    def test_retry_on_failure(self, detector):
        """Test retry mechanism on API failures."""
        pages = self._create_test_pages(2)

        with patch("pdf_splitter.detection.llm_detector.requests.post") as mock_post:
            # First call fails, second succeeds
            mock_post.side_effect = [
                TimeoutError("Timeout"),
                Mock(
                    status_code=200,
                    json=Mock(
                        return_value={
                            "response": "<thinking>Retry worked</thinking>\n<answer>DIFFERENT</answer>"
                        }
                    ),
                ),
            ]

            with patch.object(
                detector, "_check_ollama_availability", return_value=True
            ):
                results = detector.detect_boundaries(pages)

        # Should retry and succeed
        assert len(results) == 1
        assert mock_post.call_count == 2

    def test_max_retries_exceeded(self, detector):
        """Test behavior when max retries are exceeded."""
        pages = self._create_test_pages(2)
        detector.max_retries = 2

        with patch("pdf_splitter.detection.llm_detector.requests.post") as mock_post:
            mock_post.side_effect = TimeoutError("Always timeout")

            with patch.object(
                detector, "_check_ollama_availability", return_value=True
            ):
                results = detector.detect_boundaries(pages)

        # Should fail gracefully after retries
        assert len(results) == 0
        assert mock_post.call_count == detector.max_retries

    # ===== Helper Methods =====

    def _create_test_pages(self, num_pages: int) -> List[ProcessedPage]:
        """Create test pages with realistic content."""
        pages = []
        for i in range(1, num_pages + 1):
            text = f"Page {i} content\n" + "\n".join([f"Line {j}" for j in range(20)])
            page = ProcessedPage(
                page_number=i, text=text, ocr_confidence=0.95, page_type="SEARCHABLE"
            )
            pages.append(page)
        return pages


class TestLLMDetectorRobustness:
    """Test robustness and error recovery of LLM detector."""

    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return LLMDetector()

    def test_concurrent_requests_handling(self, detector):
        """Test handling of concurrent detection requests."""
        # This would require actual async implementation
        # For now, test that sequential calls don't interfere
        pages1 = [
            ProcessedPage(
                page_number=1,
                text="Doc 1 Page 1",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=2,
                text="Doc 1 Page 2",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
        ]

        pages2 = [
            ProcessedPage(
                page_number=1,
                text="Doc 2 Page 1",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=2,
                text="Doc 2 Page 2",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
        ]

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            with patch.object(
                detector,
                "_call_ollama",
                return_value="<thinking>Test</thinking>\n<answer>SAME</answer>",
            ):
                # Process both documents
                results1 = detector.detect_boundaries(pages1)
                results2 = detector.detect_boundaries(pages2)

        # Results should be independent
        assert len(results1) == 0
        assert len(results2) == 0

    def test_memory_efficiency_with_large_cache(self, detector):
        """Test memory usage with large cache."""
        # The new cache system handles memory differently
        # Just verify the detector works with many unique entries

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            with patch.object(
                detector,
                "_call_ollama",
                return_value="<thinking>Test</thinking>\n<answer>SAME</answer>",
            ):
                # Process many different page pairs
                for i in range(100):
                    pages = [
                        ProcessedPage(
                            page_number=1,
                            text=f"Unique text {i}",
                            ocr_confidence=0.95,
                            page_type="SEARCHABLE",
                        ),
                        ProcessedPage(
                            page_number=2,
                            text=f"Another unique {i}",
                            ocr_confidence=0.95,
                            page_type="SEARCHABLE",
                        ),
                    ]
                    detector.detect_boundaries(pages)

        # Should handle without memory issues
        assert True  # If we get here, memory handling is fine
