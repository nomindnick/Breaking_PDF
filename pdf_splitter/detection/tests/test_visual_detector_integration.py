"""Integration tests for visual detector with signal combiner."""

import io
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.base_detector import (
    BoundaryResult,
    DetectionContext,
    DetectorType,
    ProcessedPage,
)
from pdf_splitter.detection.visual_detector import VisualDetector


@pytest.mark.skip(
    reason="SignalCombiner not implemented - visual detector integration tests disabled"
)
class TestVisualDetectorIntegration:
    """Test visual detector integration with signal combiner."""

    def create_test_image(
        self, color: tuple = (255, 255, 255), text: str = ""
    ) -> bytes:
        """Create a test image with specified color and optional text."""
        from PIL import ImageDraw, ImageFont

        # Create image
        img = Image.new("RGB", (800, 600), color=color)

        if text:
            # Add text to make pages different
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.load_default()
            except:
                font = None
            draw.text((50, 50), text, fill=(0, 0, 0), font=font)

        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    def create_pages_with_images(self, num_pages: int = 5) -> list[ProcessedPage]:
        """Create test pages with pre-rendered images."""
        pages = []

        for i in range(num_pages):
            # Create very different images for different "documents"
            if i < 2:
                # Document 1: White pages with text at top
                image_bytes = self.create_test_image(
                    (255, 255, 255), f"DOCUMENT 1 - HEADER\n\n\nContent of page {i+1}"
                )
            elif i < 4:
                # Document 2: Dark gray pages with different layout
                image_bytes = self.create_test_image(
                    (100, 100, 100),
                    f"===== DOCUMENT 2 =====\n\n\n\n\nPage {i+1} content here",
                )
            else:
                # Document 3: Black pages with minimal text
                image_bytes = self.create_test_image((0, 0, 0), f"Doc 3 / Pg {i+1}")

            page = ProcessedPage(
                page_number=i + 1,
                text=f"Page {i + 1} content",
                ocr_confidence=0.95,
                rendered_image=image_bytes,
            )
            pages.append(page)

        return pages

    def test_visual_detector_with_pre_rendered_images(self):
        """Test that visual detector works with pre-rendered images."""
        # Create detector without PDF handler, with lower thresholds for testing
        detector = VisualDetector(
            pdf_handler=None,
            voting_threshold=1,  # Only need 1 algorithm to vote
            phash_threshold=5,  # Lower thresholds for test images
            ahash_threshold=5,
            dhash_threshold=5,
        )

        # Create pages with pre-rendered images
        pages = self.create_pages_with_images(5)

        # Detect boundaries
        boundaries = detector.detect_boundaries(pages)

        # Should detect boundaries between different colored documents
        assert len(boundaries) >= 2

        # Check that boundaries are at expected positions (between docs)
        boundary_pages = [b.page_number for b in boundaries]
        # Boundaries should be after the last page of each doc (except the last)
        assert any(p in [1, 2] for p in boundary_pages)  # Between doc1 and doc2
        assert any(p in [3, 4] for p in boundary_pages)  # Between doc2 and doc3

    def test_visual_detector_with_signal_combiner(self):
        """Test visual detector integration with signal combiner."""
        # Create visual detector with lower thresholds
        visual_detector = VisualDetector(
            pdf_handler=None,
            voting_threshold=1,
            phash_threshold=5,
            ahash_threshold=5,
            dhash_threshold=5,
        )

        # Create signal combiner with only visual detector
        detectors = {DetectorType.VISUAL: visual_detector}
        config = SignalCombinerConfig(
            combination_strategy="weighted_voting",
            detector_weights={DetectorType.VISUAL: 1.0},
            add_implicit_start_boundary=False,  # Don't add automatic boundary at start
        )
        combiner = SignalCombiner(detectors, config)

        # Create test pages
        pages = self.create_pages_with_images(5)

        # Run detection through combiner
        boundaries = combiner.detect_boundaries(pages)

        # Should get boundaries
        assert len(boundaries) >= 2

        # Verify boundaries have correct type
        for boundary in boundaries:
            assert boundary.detector_type in [
                DetectorType.VISUAL,
                DetectorType.COMBINED,
            ]

    def test_visual_detector_fallback_to_pdf(self):
        """Test visual detector falls back to PDF rendering when no pre-rendered images."""
        # Mock PDF handler
        mock_pdf_handler = Mock()
        mock_pdf_handler.render_page.return_value = np.zeros(
            (600, 800, 3), dtype=np.uint8
        )

        detector = VisualDetector(pdf_handler=mock_pdf_handler)

        # Create pages without rendered images
        pages = []
        for i in range(3):
            page = ProcessedPage(
                page_number=i + 1,
                text=f"Page {i + 1} content",
                ocr_confidence=0.95,
                rendered_image=None,  # No pre-rendered image
            )
            pages.append(page)

        # Detect boundaries
        boundaries = detector.detect_boundaries(pages)

        # Should have called render_page for each page
        assert mock_pdf_handler.render_page.call_count >= 2

    def test_visual_detector_error_handling(self):
        """Test visual detector handles errors gracefully."""
        # Create detector without PDF handler
        detector = VisualDetector(pdf_handler=None)

        # Create pages with invalid image data
        pages = []
        for i in range(3):
            page = ProcessedPage(
                page_number=i + 1,
                text=f"Page {i + 1} content",
                ocr_confidence=0.95,
                rendered_image=b"invalid image data"
                if i == 1
                else self.create_test_image(),
            )
            pages.append(page)

        # Should handle the error and continue
        boundaries = detector.detect_boundaries(pages)

        # Should skip the problematic page pair but process others
        assert isinstance(boundaries, list)

    def test_visual_detector_caching(self):
        """Test that visual detector properly caches images."""
        detector = VisualDetector(pdf_handler=None)

        # Create pages
        pages = self.create_pages_with_images(3)

        # First detection - should cache images
        boundaries1 = detector.detect_boundaries(pages)

        # Check cache has entries
        assert len(detector._page_cache) > 0
        initial_cache_size = len(detector._page_cache)

        # Clear the spy count by creating a new detector with the same cache
        detector2 = VisualDetector(pdf_handler=None)
        detector2._page_cache = detector._page_cache.copy()

        # Second detection with cached images
        boundaries2 = detector2.detect_boundaries(pages)

        # Cache size should remain the same (no new entries)
        assert len(detector2._page_cache) == initial_cache_size

    def test_mixed_rendered_and_pdf_pages(self):
        """Test handling mix of pre-rendered and PDF-rendered pages."""
        # Mock PDF handler for fallback
        mock_pdf_handler = Mock()
        mock_pdf_handler.render_page.return_value = (
            np.ones((600, 800, 3), dtype=np.uint8) * 255
        )

        detector = VisualDetector(pdf_handler=mock_pdf_handler)

        # Create mixed pages
        pages = []
        for i in range(4):
            # Alternate between pre-rendered and no image
            has_image = i % 2 == 0
            page = ProcessedPage(
                page_number=i + 1,
                text=f"Page {i + 1} content",
                ocr_confidence=0.95,
                rendered_image=self.create_test_image() if has_image else None,
            )
            pages.append(page)

        # Should handle mixed scenario
        boundaries = detector.detect_boundaries(pages)

        # PDF handler should be called for pages without images
        assert mock_pdf_handler.render_page.call_count == 2  # Pages 2 and 4
