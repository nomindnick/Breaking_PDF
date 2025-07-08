"""
Tests for the LLM-based document boundary detector.

This module tests the LLMDetector class functionality including:
- Boundary detection between pages
- Response parsing
- Caching behavior
- Error handling
- Ollama integration
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.base_detector import (
    BoundaryType,
    DetectorType,
    ProcessedPage,
)
from pdf_splitter.detection.llm_detector import LLMDetector


@pytest.fixture
def mock_config():
    """Create a mock PDF configuration."""
    return PDFConfig()


@pytest.fixture
def detector(mock_config):
    """Create an LLMDetector instance."""
    return LLMDetector(config=mock_config)


@pytest.fixture
def sample_pages():
    """Create sample processed pages for testing."""
    return [
        ProcessedPage(
            page_number=1,
            text="This is the end of a document.\nSincerely,\nJohn Doe",
            page_type="SEARCHABLE",
        ),
        ProcessedPage(
            page_number=2,
            text="INVOICE\nNumber: INV-2025-001\nDate: January 5, 2025",
            page_type="SEARCHABLE",
        ),
        ProcessedPage(
            page_number=3,
            text="Bill To:\nAcme Corporation\n123 Main Street",
            page_type="SEARCHABLE",
        ),
    ]


@pytest.fixture
def continuation_pages():
    """Create pages that continue from one to the next."""
    return [
        ProcessedPage(
            page_number=1,
            text="The analysis shows that performance improved by 25%.",
            page_type="SEARCHABLE",
        ),
        ProcessedPage(
            page_number=2,
            text="This improvement is primarily due to optimization efforts.",
            page_type="SEARCHABLE",
        ),
    ]


class TestLLMDetector:
    """Test suite for LLMDetector class."""

    def test_initialization(self, mock_config):
        """Test detector initialization."""
        detector = LLMDetector(
            config=mock_config,
            model_name="test-model",
            ollama_url="http://test:11434",
            cache_responses=False,
        )

        assert detector.model_name == "test-model"
        assert detector.ollama_url == "http://test:11434"
        assert detector.cache_responses is False
        assert detector.get_detector_type() == DetectorType.LLM
        assert detector.get_confidence_threshold() == 0.8

    def test_empty_pages_handling(self, detector):
        """Test handling of empty pages."""
        pages = [
            ProcessedPage(page_number=1, text="", page_type="SEARCHABLE"),
            ProcessedPage(page_number=2, text="Some content", page_type="SEARCHABLE"),
        ]

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            results = detector.detect_boundaries(pages)

        assert len(results) == 0  # Should skip empty pages

    def test_single_page_handling(self, detector):
        """Test handling of single page input."""
        pages = [ProcessedPage(page_number=1, text="Content", page_type="SEARCHABLE")]

        results = detector.detect_boundaries(pages)
        assert len(results) == 0  # Need at least 2 pages

    @patch("requests.get")
    def test_ollama_availability_check_success(self, mock_get, detector):
        """Test successful Ollama availability check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "gemma3:latest"}, {"name": "llama2:latest"}]
        }
        mock_get.return_value = mock_response

        assert detector._check_ollama_availability() is True

    @patch("requests.get")
    def test_ollama_availability_check_failure(self, mock_get, detector):
        """Test failed Ollama availability check."""
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        assert detector._check_ollama_availability() is False

    @patch("requests.get")
    def test_ollama_model_not_available(self, mock_get, detector):
        """Test when specific model is not available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama2:latest"}]  # gemma3 not in list
        }
        mock_get.return_value = mock_response

        assert detector._check_ollama_availability() is False

    def test_parse_llm_response_success(self, detector):
        """Test successful parsing of LLM response."""
        response = """<thinking>Page 1 ends with a signature. Page 2 starts with an invoice header. These are different documents.</thinking>
<answer>DIFFERENT</answer>"""

        is_boundary, confidence, reasoning = detector._parse_llm_response(response)

        assert is_boundary is True
        assert confidence == 0.95
        assert "signature" in reasoning and "invoice" in reasoning

    def test_parse_llm_response_same_document(self, detector):
        """Test parsing response for same document."""
        response = """<thinking>Page 2 continues the topic from page 1.</thinking>
<answer>SAME</answer>"""

        is_boundary, confidence, reasoning = detector._parse_llm_response(response)

        assert is_boundary is False
        assert confidence == 0.85
        assert "continues" in reasoning

    def test_parse_llm_response_invalid_format(self, detector):
        """Test parsing of invalid response format."""
        response = "This response doesn't have the required tags"

        is_boundary, confidence, reasoning = detector._parse_llm_response(response)

        assert is_boundary is False
        assert confidence == 0.0
        assert reasoning == "Invalid response format"

    def test_parse_llm_response_empty(self, detector):
        """Test parsing of empty response."""
        is_boundary, confidence, reasoning = detector._parse_llm_response("")

        assert is_boundary is False
        assert confidence == 0.0
        assert reasoning == "No response from LLM"

    def test_text_extraction_bottom(self, detector):
        """Test extraction of bottom text from page."""
        text = "\n".join([f"Line {i}" for i in range(30)])
        extracted = detector._extract_bottom_text(text)

        lines = extracted.split("\n")
        assert len(lines) == 15
        assert lines[0] == "Line 15"
        assert lines[-1] == "Line 29"

    def test_text_extraction_top(self, detector):
        """Test extraction of top text from page."""
        text = "\n".join([f"Line {i}" for i in range(30)])
        extracted = detector._extract_top_text(text)

        lines = extracted.split("\n")
        assert len(lines) == 15
        assert lines[0] == "Line 0"
        assert lines[-1] == "Line 14"

    def test_text_extraction_short_pages(self, detector):
        """Test text extraction with pages shorter than extraction limit."""
        short_text = "Line 1\nLine 2\nLine 3"

        bottom = detector._extract_bottom_text(short_text)
        top = detector._extract_top_text(short_text)

        assert bottom == short_text.strip()
        assert top == short_text.strip()

    @patch("requests.post")
    def test_call_ollama_success(self, mock_post, detector):
        """Test successful Ollama API call."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Model response"}
        mock_post.return_value = mock_response

        response = detector._call_ollama("test prompt")

        assert response == "Model response"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/generate"
        assert call_args[1]["json"]["model"] == "gemma3:latest"
        assert call_args[1]["json"]["prompt"] == "test prompt"

    @patch("requests.post")
    def test_call_ollama_timeout_retry(self, mock_post, detector):
        """Test Ollama call with timeout and retry."""
        mock_post.side_effect = [
            requests.Timeout("Timeout"),
            Mock(status_code=200, json=lambda: {"response": "Success after retry"}),
        ]

        response = detector._call_ollama("test prompt")

        assert response == "Success after retry"
        assert mock_post.call_count == 2

    @patch("requests.post")
    def test_call_ollama_failure(self, mock_post, detector):
        """Test Ollama call failure after retries."""
        mock_post.side_effect = requests.Timeout("Timeout")

        response = detector._call_ollama("test prompt")

        assert response == ""
        assert mock_post.call_count == 2  # Initial + 1 retry

    def test_cache_key_generation(self, detector):
        """Test cache key generation."""
        key1 = detector._get_cache_key("text1", "text2")
        key2 = detector._get_cache_key("text1", "text2")
        key3 = detector._get_cache_key("different", "text")

        assert key1 == key2  # Same input should give same key
        assert key1 != key3  # Different input should give different key

    @patch("requests.get")
    @patch("requests.post")
    def test_detect_boundaries_with_cache(
        self, mock_post, mock_get, detector, sample_pages
    ):
        """Test boundary detection with caching enabled."""
        # Mock Ollama availability
        mock_get.return_value = Mock(
            status_code=200, json=lambda: {"models": [{"name": "gemma3:latest"}]}
        )

        # Mock LLM response
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {
                "response": "<thinking>Different documents</thinking>\n<answer>DIFFERENT</answer>"
            },
        )

        # First detection
        results1 = detector.detect_boundaries(sample_pages[:2])

        # Second detection (should use cache)
        results2 = detector.detect_boundaries(sample_pages[:2])

        assert len(results1) == 1
        assert len(results2) == 1
        assert mock_post.call_count == 1  # Only one API call due to cache

    @patch("requests.get")
    @patch("requests.post")
    def test_detect_boundaries_full_workflow(
        self, mock_post, mock_get, detector, sample_pages
    ):
        """Test complete boundary detection workflow."""
        # Mock Ollama availability
        mock_get.return_value = Mock(
            status_code=200, json=lambda: {"models": [{"name": "gemma3:latest"}]}
        )

        # Mock different responses for different page pairs
        responses = [
            "<thinking>Letter ending and invoice start</thinking>\n<answer>DIFFERENT</answer>",
            "<thinking>Invoice continues</thinking>\n<answer>SAME</answer>",
        ]
        mock_post.side_effect = [
            Mock(status_code=200, json=lambda r=r: {"response": r}) for r in responses
        ]

        results = detector.detect_boundaries(sample_pages)

        assert len(results) == 1  # Only one boundary detected
        boundary = results[0]
        assert boundary.page_number == 1
        assert boundary.boundary_type == BoundaryType.DOCUMENT_END
        assert boundary.confidence == 0.95
        assert boundary.is_between_pages is True
        assert boundary.next_page_number == 2

    def test_validate_configuration(self, detector):
        """Test configuration validation."""
        with patch("requests.get") as mock_get:
            # Mock successful validation
            mock_get.return_value = Mock(
                status_code=200, json=lambda: {"models": [{"name": "gemma3:latest"}]}
            )

            validation = detector.validate_configuration()

            assert validation["ollama_available"] is True
            assert validation["model_available"] is True
            assert validation["prompt_loaded"] is True
            assert validation["cache_enabled"] is True

    def test_get_detection_stats(self, detector):
        """Test detection statistics tracking."""
        # Initially empty
        stats = detector.get_detection_stats()
        assert stats["detections"] == 0
        assert stats["avg_confidence"] == 0.0

        # Add some detection history
        from pdf_splitter.detection.base_detector import BoundaryResult

        detector._detection_history = [
            BoundaryResult(
                page_number=1,
                boundary_type=BoundaryType.DOCUMENT_END,
                confidence=0.95,
                detector_type=DetectorType.LLM,
            ),
            BoundaryResult(
                page_number=3,
                boundary_type=BoundaryType.DOCUMENT_END,
                confidence=0.85,
                detector_type=DetectorType.LLM,
            ),
        ]

        stats = detector.get_detection_stats()
        assert stats["detections"] == 2
        assert stats["avg_confidence"] == 0.9
        assert stats["min_confidence"] == 0.85
        assert stats["max_confidence"] == 0.95

    def test_prompt_template_loading(self, mock_config):
        """Test prompt template loading with fallback."""
        with patch("pathlib.Path.exists", return_value=False):
            detector = LLMDetector(config=mock_config)

            # Should use embedded prompt
            assert "<start_of_turn>user" in detector.prompt_template
            assert "{page1_bottom}" in detector.prompt_template
            assert "{page2_top}" in detector.prompt_template
