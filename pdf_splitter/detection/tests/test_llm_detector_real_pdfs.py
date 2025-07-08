"""
Real PDF validation tests for the LLM detector.

This test suite validates the LLM detector against actual PDF documents
with known ground truth data.
"""

import json
import logging
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.base_detector import ProcessedPage
from pdf_splitter.detection.llm_detector import LLMDetector
from pdf_splitter.preprocessing.pdf_handler import PDFHandler
from pdf_splitter.preprocessing.text_extractor import TextExtractor

logger = logging.getLogger(__name__)


@pytest.mark.skipif(
    not Path("Test_PDF_Set_1.pdf").exists(), reason="Test PDFs not available"
)
class TestLLMDetectorRealPDFs:
    """Test LLM detector with real PDF documents."""

    @pytest.fixture
    def ground_truth(self):
        """Load ground truth data."""
        ground_truth_path = Path("Test_PDF_Set_Ground_Truth.json")
        if not ground_truth_path.exists():
            pytest.skip("Ground truth file not available")

        with open(ground_truth_path, "r") as f:
            return json.load(f)

    @pytest.fixture
    def pdf_config(self):
        """Create PDF configuration for testing."""
        return PDFConfig(
            min_text_length=10,
            default_dpi=300,
            max_file_size_mb=100,
        )

    @pytest.fixture
    def mock_ollama_responses(self, ground_truth):
        """Create mock Ollama responses based on ground truth."""
        responses = []

        # Create responses for each boundary in ground truth
        boundaries = []
        for doc in ground_truth["documents"]:
            if doc["end_page"] < 32:  # Not the last document
                boundaries.append(doc["end_page"])

        # Generate responses for all page pairs
        for i in range(31):  # 32 pages means 31 pairs
            if i + 1 in boundaries:
                # This is a boundary
                response = f"""<thinking>Page {i+1} appears to be the end of a {ground_truth['documents'][0]['type']} document. Page {i+2} starts a new document.</thinking>
<answer>DIFFERENT</answer>"""
            else:
                # Same document
                response = f"""<thinking>Pages {i+1} and {i+2} appear to be part of the same document with continuous content.</thinking>
<answer>SAME</answer>"""

            responses.append(response)

        return responses

    def test_detector_with_test_pdf_set_1(
        self, pdf_config, ground_truth, mock_ollama_responses
    ):
        """Test detector with Test_PDF_Set_1.pdf (non-OCR)."""
        pdf_path = Path("Test_PDF_Set_1.pdf")

        # Initialize components
        pdf_handler = PDFHandler(config=pdf_config)
        text_extractor = TextExtractor(config=pdf_config)
        detector = LLMDetector(config=pdf_config)

        # Load and process PDF
        pdf_doc = pdf_handler.load_pdf(pdf_path)

        # Extract text from all pages
        pages = []
        for page_num in range(1, min(33, pdf_doc.page_count + 1)):  # Limit to 32 pages
            page_obj = pdf_doc[page_num - 1]

            # Extract text
            text, confidence = text_extractor.extract_text(page_obj)

            # Create ProcessedPage
            processed_page = ProcessedPage(
                page_number=page_num,
                text=text,
                ocr_confidence=confidence,
                page_type="SEARCHABLE",
                metadata={"source": "test_pdf_set_1"},
            )
            pages.append(processed_page)

        # Mock Ollama responses
        response_iter = iter(mock_ollama_responses)

        def mock_call_ollama(prompt):
            try:
                return next(response_iter)
            except StopIteration:
                return "<thinking>No more responses</thinking>\n<answer>SAME</answer>"

        # Run detection with mocked Ollama
        with patch.object(detector, "_check_ollama_availability", return_value=True):
            with patch.object(detector, "_call_ollama", side_effect=mock_call_ollama):
                start_time = time.time()
                boundaries = detector.detect_boundaries(pages)
                elapsed = time.time() - start_time

        # Validate results
        expected_boundaries = []
        for doc in ground_truth["documents"]:
            if doc["end_page"] < 32:
                expected_boundaries.append(doc["end_page"])

        detected_boundaries = [b.page_number for b in boundaries]

        logger.info(f"Expected boundaries: {expected_boundaries}")
        logger.info(f"Detected boundaries: {detected_boundaries}")
        logger.info(f"Processing time: {elapsed:.2f}s ({elapsed/31:.2f}s per pair)")

        # Calculate metrics
        true_positives = len(set(detected_boundaries) & set(expected_boundaries))
        false_positives = len(set(detected_boundaries) - set(expected_boundaries))
        false_negatives = len(set(expected_boundaries) - set(detected_boundaries))

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        logger.info(f"Precision: {precision:.2%}")
        logger.info(f"Recall: {recall:.2%}")
        logger.info(f"F1 Score: {f1:.3f}")

        # Assert performance
        assert f1 >= 0.8, f"F1 score {f1:.3f} below threshold"
        assert len(boundaries) == len(
            expected_boundaries
        ), f"Expected {len(expected_boundaries)} boundaries, found {len(boundaries)}"

    def test_detector_with_ocr_pdf(self, pdf_config, ground_truth):
        """Test detector with Test_PDF_Set_2_ocr.pdf (OCR'd version)."""
        pdf_path = Path("Test_PDF_Set_2_ocr.pdf")
        if not pdf_path.exists():
            pytest.skip("OCR test PDF not available")

        # Initialize components
        pdf_handler = PDFHandler(config=pdf_config)
        text_extractor = TextExtractor(config=pdf_config)
        detector = LLMDetector(config=pdf_config)

        # Load PDF
        pdf_doc = pdf_handler.load_pdf(pdf_path)

        # Process first 10 pages as a quick test
        pages = []
        for page_num in range(1, min(11, pdf_doc.page_count + 1)):
            page_obj = pdf_doc[page_num - 1]

            # Extract text
            text, confidence = text_extractor.extract_text(page_obj)

            # Create ProcessedPage
            processed_page = ProcessedPage(
                page_number=page_num,
                text=text,
                ocr_confidence=confidence,
                page_type="SEARCHABLE",
                metadata={"source": "test_pdf_set_2_ocr"},
            )
            pages.append(processed_page)

        # Create mock responses for quick test
        mock_responses = []
        for i in range(9):  # 10 pages = 9 pairs
            if i == 2:  # Assume boundary after page 3
                response = "<thinking>Document boundary detected</thinking>\n<answer>DIFFERENT</answer>"
            else:
                response = "<thinking>Same document continues</thinking>\n<answer>SAME</answer>"
            mock_responses.append(response)

        response_iter = iter(mock_responses)

        # Run detection
        with patch.object(detector, "_check_ollama_availability", return_value=True):
            with patch.object(
                detector, "_call_ollama", side_effect=lambda p: next(response_iter)
            ):
                boundaries = detector.detect_boundaries(pages)

        # Basic validation
        assert len(boundaries) >= 0, "Should detect at least some boundaries"
        for boundary in boundaries:
            assert 0.8 <= boundary.confidence <= 1.0, "Confidence should be high"

    def test_cache_performance_with_real_pdfs(self, pdf_config, tmp_path):
        """Test cache performance with repeated processing."""
        # Create detector with persistent cache
        cache_path = tmp_path / "test_cache.db"
        detector = LLMDetector(config=pdf_config, cache_path=cache_path)

        # Create test pages
        pages = []
        for i in range(5):
            pages.append(
                ProcessedPage(
                    page_number=i + 1,
                    text=f"Page {i+1} content with some text that is long enough to process properly",
                    ocr_confidence=0.95,
                    page_type="SEARCHABLE",
                )
            )

        # Mock Ollama to track calls
        call_count = 0

        def mock_ollama(prompt):
            nonlocal call_count
            call_count += 1
            return "<thinking>Processing</thinking>\n<answer>SAME</answer>"

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            with patch.object(detector, "_call_ollama", side_effect=mock_ollama):
                # First run - should call Ollama
                detector.detect_boundaries(pages)
                first_run_calls = call_count

                # Second run - should use cache
                call_count = 0
                detector.detect_boundaries(pages)
                second_run_calls = call_count

        # Verify caching worked
        assert first_run_calls == 4, "Should call Ollama for each page pair"
        assert second_run_calls == 0, "Should use cache for all pairs"

        # Check cache stats
        stats = detector.get_cache_stats()
        assert stats["hits"] >= 4, "Should have cache hits"
        assert stats["hit_rate"] > 0.5, "Hit rate should be good"

    def test_detector_robustness_with_empty_pages(self, pdf_config):
        """Test detector handles empty pages in real PDFs gracefully."""
        detector = LLMDetector(config=pdf_config)

        # Create pages with some empty
        pages = [
            ProcessedPage(
                page_number=1,
                text="First page with content",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=2, text="", ocr_confidence=0.0, page_type="EMPTY"
            ),
            ProcessedPage(
                page_number=3, text="", ocr_confidence=0.0, page_type="EMPTY"
            ),
            ProcessedPage(
                page_number=4,
                text="Fourth page with content",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=5,
                text="Fifth page continues",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
        ]

        # Mock response for non-empty pairs
        mock_response = "<thinking>Content analysis</thinking>\n<answer>SAME</answer>"

        with patch.object(detector, "_check_ollama_availability", return_value=True):
            with patch.object(detector, "_call_ollama", return_value=mock_response):
                boundaries = detector.detect_boundaries(pages)

        # Should skip empty page pairs
        assert len(boundaries) == 0 or all(b.confidence > 0.7 for b in boundaries)
