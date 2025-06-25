"""Integration tests for caching with PDFHandler and TextExtractor."""

import time
from pathlib import Path

import pytest

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.preprocessing.pdf_handler import PDFHandler
from pdf_splitter.preprocessing.text_extractor import TextExtractor


class TestCacheIntegration:
    """Test caching integration with real PDF processing."""

    @pytest.fixture
    def test_pdf_path(self):
        """Get path to test PDF."""
        # Try to find test PDF
        test_paths = [
            Path("test_files/Test_PDF_Set_2_ocr.pdf"),
            Path("../../../test_files/Test_PDF_Set_2_ocr.pdf"),
            Path("Test_PDF_Set_2_ocr.pdf"),
        ]

        for path in test_paths:
            if path.exists():
                return path

        pytest.skip("Test PDF not found")

    @pytest.fixture
    def pdf_handler_with_cache(self):
        """Create PDFHandler with caching enabled."""
        config = PDFConfig(
            enable_cache_metrics=True,
            render_cache_memory_mb=300,  # Increased to hold ~3 pages at 300 DPI
            text_cache_memory_mb=25,
            cache_warmup_pages=5,
        )
        return PDFHandler(config)

    def test_render_cache_performance(self, pdf_handler_with_cache, test_pdf_path):
        """Test render cache improves performance."""
        handler = pdf_handler_with_cache

        with handler.load_pdf(test_pdf_path):
            # First render - cache miss
            start_time = time.time()
            img1 = handler.render_page(0)  # noqa: F841
            first_render_time = time.time() - start_time

            # Second render - cache hit
            start_time = time.time()
            img2 = handler.render_page(0)  # noqa: F841
            cached_render_time = time.time() - start_time

            # Cache should be significantly faster
            assert cached_render_time < first_render_time * 0.1

            # Verify cache stats
            stats = handler.get_cache_stats()
            assert stats["render_cache"]["hits"] == 1
            assert stats["render_cache"]["misses"] == 1
            assert stats["render_cache"]["hit_rate"] == "50.0%"

    def test_text_cache_performance(self, pdf_handler_with_cache, test_pdf_path):
        """Test text extraction cache improves performance."""
        handler = pdf_handler_with_cache

        with handler.load_pdf(test_pdf_path):
            extractor = TextExtractor(handler)

            # First extraction - cache miss
            start_time = time.time()
            page1 = extractor.extract_page(0)
            first_extract_time = time.time() - start_time

            # Second extraction - cache hit
            start_time = time.time()
            page2 = extractor.extract_page(0)
            cached_extract_time = time.time() - start_time

            # Cache should be significantly faster
            assert cached_extract_time < first_extract_time * 0.1

            # Verify content is identical
            assert page1.text == page2.text
            assert page1.quality_score == page2.quality_score

    def test_cache_warmup(self, pdf_handler_with_cache, test_pdf_path):
        """Test cache warmup functionality."""
        handler = pdf_handler_with_cache

        with handler.load_pdf(test_pdf_path):
            # Warmup cache
            handler.warmup_cache(range(3))

            # Check that pages are pre-cached
            stats = handler.get_cache_stats()
            assert stats["render_cache"]["size"] >= 3

            # Accessing warmed pages should be hits
            initial_hits = stats["render_cache"]["hits"]
            handler.render_page(0)
            handler.render_page(1)
            handler.render_page(2)

            stats = handler.get_cache_stats()
            assert stats["render_cache"]["hits"] >= initial_hits + 3

    def test_memory_limits_respected(self, pdf_handler_with_cache, test_pdf_path):
        """Test that cache respects memory limits."""
        # Create new handler with small memory limit
        config = PDFConfig(
            render_cache_memory_mb=10,
            enable_cache_metrics=True,
            default_dpi=72,  # Lower DPI to test memory limits with smaller images
        )
        handler = PDFHandler(config)

        with handler.load_pdf(test_pdf_path):
            # Render many pages at low DPI
            for i in range(20):
                handler.render_page(i % 10, dpi=72)

            # Check memory usage stays within limits (with small tolerance)
            stats = handler.get_cache_stats()
            assert stats["render_cache"]["memory_mb"] <= 12  # Allow small overhead

    def test_cache_across_pdf_reloads(self, test_pdf_path):
        """Test cache behavior when PDFs are reloaded."""
        # Create fresh handler for this test
        config = PDFConfig(enable_cache_metrics=True)
        handler = PDFHandler(config)

        # First load
        with handler.load_pdf(test_pdf_path):
            handler.render_page(0)
            extractor = TextExtractor(handler)
            extractor.extract_page(0)

        # Cache should be cleared after close
        stats = handler.get_cache_stats()
        assert stats["render_cache"]["size"] == 0
        assert stats["text_cache"]["size"] == 0

        # Metrics should persist but cache should be empty
        # So next access will be a miss
        with handler.load_pdf(test_pdf_path):
            handler.render_page(0)
            stats = handler.get_cache_stats()
            # Should have accumulated misses from both loads
            assert stats["render_cache"]["size"] == 1

    def test_cache_performance_logging(
        self, pdf_handler_with_cache, test_pdf_path, caplog
    ):
        """Test cache performance logging."""
        import logging

        caplog.set_level(logging.INFO)

        handler = pdf_handler_with_cache

        with handler.load_pdf(test_pdf_path):
            # Generate cache activity
            for i in range(5):
                handler.render_page(0)  # Repeated access

            # Log performance
            handler.log_cache_performance()

            # Check logs contain performance info
            assert "Cache Performance" in caplog.text or "Cache Summary" in caplog.text

    def test_different_dpi_caching(self, pdf_handler_with_cache, test_pdf_path):
        """Test that different DPIs are cached separately."""
        handler = pdf_handler_with_cache

        with handler.load_pdf(test_pdf_path):
            # Render at different DPIs
            img_150 = handler.render_page(0, dpi=150)  # noqa: F841
            img_300 = handler.render_page(0, dpi=300)  # noqa: F841

            # Both should be cache misses
            stats = handler.get_cache_stats()
            assert stats["render_cache"]["misses"] == 2

            # Access again - both should hit
            img_150_2 = handler.render_page(0, dpi=150)  # noqa: F841
            img_300_2 = handler.render_page(0, dpi=300)  # noqa: F841

            stats = handler.get_cache_stats()
            assert stats["render_cache"]["hits"] == 2

    def test_concurrent_access(self, test_pdf_path):
        """Test cache with concurrent access patterns."""
        # Create fresh handler
        config = PDFConfig(enable_cache_metrics=True)
        handler = PDFHandler(config)

        with handler.load_pdf(test_pdf_path):
            extractor = TextExtractor(handler)

            # Simulate boundary detection access pattern
            # (multiple detectors accessing overlapping pages)

            # First, extract text from pages 0-2
            extractor.extract_page(0)
            extractor.extract_page(1)
            extractor.extract_page(2)

            # Now extract same pages again (should hit cache)
            extractor.extract_page(0)  # Cache hit
            extractor.extract_page(1)  # Cache hit

            # Render pages (separate cache)
            handler.render_page(0)  # Cache miss
            handler.render_page(1)  # Cache miss
            handler.render_page(0)  # Cache hit

            stats = handler.get_cache_stats()

            # Verify we have hits in both caches
            assert stats["text_cache"]["hits"] >= 2  # Re-accessed pages 0,1
            assert stats["render_cache"]["hits"] >= 1  # Re-accessed page 0
