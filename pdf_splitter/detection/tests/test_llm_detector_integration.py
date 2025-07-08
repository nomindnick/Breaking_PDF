"""
Integration tests for LLM detector with live Ollama instance.

These tests require a running Ollama instance with the gemma3:latest model.
Run with: RUN_INTEGRATION_TESTS=true pytest test_llm_detector_integration.py
"""

import logging
import os
import time
from pathlib import Path
from typing import List

import pytest
import requests

from pdf_splitter.core.config import PDFConfig
# BatchedLLMDetector removed - batching can be added to main detector if needed
from pdf_splitter.detection.base_detector import ProcessedPage
from pdf_splitter.detection.llm_detector import LLMDetector

logger = logging.getLogger(__name__)


def check_ollama_available():
    """Check if Ollama is running and gemma3:latest is available."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            return False

        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        return "gemma3:latest" in model_names
    except Exception:
        return False


@pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS") or not check_ollama_available(),
    reason="Integration tests require RUN_INTEGRATION_TESTS=true and live Ollama with gemma3:latest",
)
class TestLLMDetectorIntegration:
    """Integration tests with live Ollama instance."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PDFConfig()

    @pytest.fixture
    def test_pages(self) -> List[ProcessedPage]:
        """Create realistic test pages for integration testing."""
        pages = [
            # Document 1: Letter
            ProcessedPage(
                page_number=1,
                text="""
                ACME Corporation
                123 Business Street
                New York, NY 10001

                January 15, 2024

                Dear Mr. Johnson,

                I am writing to inform you about the recent changes to our service agreement.
                As discussed in our meeting last week, we will be implementing new pricing
                structures effective February 1st.

                The main changes include:
                - Basic service tier increase of 10%
                - Premium features now included in standard package
                - Volume discounts for orders over $10,000

                We value your continued partnership and believe these changes will better
                serve your growing needs. Please review the attached documentation for
                complete details about the new pricing structure.
                """,
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=2,
                text="""
                If you have any questions or concerns about these changes, please don't
                hesitate to contact your account manager, Sarah Williams, at
                swilliams@acmecorp.com or (555) 123-4567.

                We appreciate your understanding and look forward to continuing our
                successful partnership.

                Sincerely,

                Robert Chen
                Vice President of Sales
                ACME Corporation
                """,
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
            # Document 2: Invoice
            ProcessedPage(
                page_number=3,
                text="""
                INVOICE

                Invoice Number: INV-2024-0892
                Date: January 15, 2024
                Due Date: February 15, 2024

                Bill To:
                Johnson Enterprises
                456 Commerce Drive
                Chicago, IL 60601

                Description                    Quantity    Unit Price    Total
                ----------------------------------------------------------------
                Professional Services              40         $150      $6,000
                Software License (Annual)           1       $2,400      $2,400
                Training Sessions                   3         $500      $1,500

                                                        Subtotal:      $9,900
                                                        Tax (8%):        $792
                                                        Total Due:    $10,692
                """,
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
            # Document 3: Report (starts mid-page 3, continues to 4)
            ProcessedPage(
                page_number=4,
                text="""
                QUARTERLY PERFORMANCE REPORT
                Q4 2023

                Executive Summary

                The fourth quarter of 2023 showed significant improvement across all
                key performance indicators. Revenue increased by 23% compared to Q3,
                while operational costs decreased by 8% due to efficiency improvements.

                Key Highlights:
                • Total Revenue: $4.2M (up from $3.4M in Q3)
                • New Customer Acquisitions: 127 (15% above target)
                • Customer Retention Rate: 94% (industry average: 85%)
                • Employee Satisfaction Score: 8.7/10

                Market Analysis:
                The competitive landscape continues to evolve rapidly. Our main
                competitors have introduced new pricing models, but our value
                proposition remains strong due to superior customer service and
                product reliability.
                """,
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
        ]

        return pages

    def test_basic_ollama_connection(self, config):
        """Test basic connection to Ollama."""
        detector = LLMDetector(config=config)

        # Validate configuration
        validation = detector.validate_configuration()

        assert validation["ollama_available"] is True
        assert validation["model_available"] is True
        assert "gemma3:latest" in validation.get("available_models", [])

    def test_single_boundary_detection(self, config, test_pages):
        """Test detecting a single boundary with live Ollama."""
        detector = LLMDetector(config=config, cache_responses=False)

        # Test pages 2-3 (letter end to invoice start)
        pages = test_pages[1:3]

        start_time = time.time()
        boundaries = detector.detect_boundaries(pages)
        elapsed = time.time() - start_time

        logger.info(f"Single boundary detection took {elapsed:.2f}s")

        # Should detect boundary between letter and invoice
        assert len(boundaries) == 1
        assert boundaries[0].page_number == 2
        assert boundaries[0].confidence >= 0.8
        assert boundaries[0].reasoning is not None

        # Performance check
        assert elapsed < 60, f"Detection took too long: {elapsed}s"

    def test_multiple_boundaries(self, config, test_pages):
        """Test detecting multiple boundaries with live Ollama."""
        detector = LLMDetector(config=config)

        start_time = time.time()
        boundaries = detector.detect_boundaries(test_pages)
        elapsed = time.time() - start_time

        logger.info(f"Detected {len(boundaries)} boundaries in {elapsed:.2f}s")

        # Should detect at least the letter->invoice boundary
        assert len(boundaries) >= 1

        # Check first boundary (letter to invoice)
        letter_invoice_boundary = next(
            (b for b in boundaries if b.page_number == 2), None
        )
        assert letter_invoice_boundary is not None
        assert letter_invoice_boundary.confidence >= 0.8

        # Log all detected boundaries
        for b in boundaries:
            logger.info(
                f"Boundary after page {b.page_number}: "
                f"confidence={b.confidence:.2f}, reasoning='{b.reasoning}'"
            )

    def test_cache_effectiveness(self, config, test_pages, tmp_path):
        """Test cache performance with live Ollama."""
        cache_path = tmp_path / "test_cache.db"
        detector = LLMDetector(config=config, cache_path=cache_path)

        # First run - no cache
        start1 = time.time()
        boundaries1 = detector.detect_boundaries(test_pages)
        time1 = time.time() - start1

        # Get cache stats after first run
        stats1 = detector.get_cache_stats()

        # Second run - with cache
        start2 = time.time()
        boundaries2 = detector.detect_boundaries(test_pages)
        time2 = time.time() - start2

        # Get cache stats after second run
        stats2 = detector.get_cache_stats()

        logger.info(f"First run: {time1:.2f}s, Second run: {time2:.2f}s")
        logger.info(f"Cache stats: {stats2}")

        # Verify cache is working
        assert time2 < time1 * 0.1, "Cache should provide significant speedup"
        assert stats2["hits"] > stats1["hits"], "Should have cache hits on second run"
        assert len(boundaries1) == len(boundaries2), "Results should be consistent"

    @pytest.mark.skip(
        "Batched detector removed - could be added to main detector if needed"
    )
    def test_batched_detector(self, config, test_pages):
        """Test batched detector with live Ollama."""
        pass  # Functionality could be added to main LLMDetector

    def test_error_recovery(self, config):
        """Test error handling with invalid content."""
        detector = LLMDetector(config=config)

        # Create pages with problematic content
        pages = [
            ProcessedPage(
                page_number=1,
                text="🚀 Unicode emoji test 你好世界 مرحبا بالعالم",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=2,
                text="Normal text after unicode",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
        ]

        # Should handle without crashing
        boundaries = detector.detect_boundaries(pages)
        assert isinstance(boundaries, list)

    def test_model_response_quality(self, config):
        """Test quality of model responses with known boundary."""
        detector = LLMDetector(config=config, cache_responses=False)

        # Clear boundary case
        pages = [
            ProcessedPage(
                page_number=1,
                text="Thank you for your business.\n\nSincerely,\nJohn Doe\nCEO",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
            ProcessedPage(
                page_number=2,
                text="INVOICE #12345\n\nDate: 2024-01-15\nAmount Due: $1,000",
                ocr_confidence=0.95,
                page_type="SEARCHABLE",
            ),
        ]

        boundaries = detector.detect_boundaries(pages)

        # Should confidently detect this clear boundary
        assert len(boundaries) == 1
        assert boundaries[0].confidence >= 0.9
        assert (
            "invoice" in boundaries[0].reasoning.lower()
            or "different" in boundaries[0].reasoning.lower()
        )

    def test_performance_monitoring(self, config, test_pages):
        """Test performance metrics collection."""
        detector = LLMDetector(config=config)

        # Process pages multiple times
        for i in range(3):
            boundaries = detector.detect_boundaries(test_pages)

            # Check that timing is recorded
            for boundary in boundaries:
                assert "processing_time" in boundary.evidence
                assert boundary.evidence["processing_time"] > 0

        # Get final stats
        stats = detector.get_detection_stats()
        assert stats["detections_performed"] == 3
        assert stats["total_boundaries_found"] >= 3
