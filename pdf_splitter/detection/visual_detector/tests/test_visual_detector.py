"""Tests for the VisualDetector class."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.base_detector import (
    BoundaryType,
    DetectionContext,
    DetectorType,
    ProcessedPage,
)
from pdf_splitter.detection.visual_detector import VisualDetector

# from PIL import Image  # Not used in this file


class TestVisualDetector:
    """Test cases for VisualDetector."""

    @pytest.fixture
    def detector(self):
        """Create a VisualDetector instance with mocked PDF handler."""
        mock_handler = Mock()
        detector = VisualDetector(
            pdf_handler=mock_handler,
            voting_threshold=1,  # Sensitive threshold for testing
        )
        return detector

    @pytest.fixture
    def processed_pages(self):
        """Create sample processed pages."""
        pages = []
        for i in range(1, 5):
            page = ProcessedPage(
                page_number=i,
                text=f"Page {i} content",
                page_type="SEARCHABLE",
            )
            pages.append(page)
        return pages

    def test_initialization(self):
        """Test detector initialization."""
        detector = VisualDetector()
        assert detector.get_detector_type() == DetectorType.VISUAL
        assert detector.hash_size == 8
        assert detector.voting_threshold == 1
        assert detector.get_confidence_threshold() == 0.5

    def test_custom_initialization(self):
        """Test detector with custom parameters."""
        detector = VisualDetector(
            hash_size=16,
            voting_threshold=2,
            phash_threshold=15,
            ahash_threshold=18,
            dhash_threshold=20,
        )
        assert detector.hash_size == 16
        assert detector.voting_threshold == 2
        assert detector.phash_threshold == 15
        assert detector.ahash_threshold == 18
        assert detector.dhash_threshold == 20

    def test_empty_pages(self, detector):
        """Test detection with empty page list."""
        results = detector.detect_boundaries([])
        assert results == []

    def test_single_page(self, detector, processed_pages):
        """Test detection with single page."""
        results = detector.detect_boundaries([processed_pages[0]])
        assert results == []

    @patch("pdf_splitter.detection.visual_detector.visual_detector.imagehash")
    def test_similar_pages_no_boundary(self, mock_imagehash, detector, processed_pages):
        """Test detection between similar pages (no boundary)."""
        # Mock similar hashes (small distances)
        mock_hash1 = MagicMock()
        mock_hash2 = MagicMock()
        mock_hash1.__sub__.return_value = 2  # Small distance
        mock_hash2.__sub__.return_value = 3

        mock_imagehash.phash.side_effect = [mock_hash1, mock_hash1]
        mock_imagehash.average_hash.side_effect = [mock_hash2, mock_hash2]
        mock_imagehash.dhash.side_effect = [mock_hash2, mock_hash2]

        # Mock page rendering - render_page returns numpy array
        mock_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
        detector.pdf_handler.render_page.return_value = mock_array

        results = detector.detect_boundaries(processed_pages[:2])

        # Should not detect boundary for similar pages
        assert len(results) == 0

    @patch("pdf_splitter.detection.visual_detector.visual_detector.imagehash")
    def test_dissimilar_pages_boundary_detected(
        self, mock_imagehash, detector, processed_pages
    ):
        """Test detection between dissimilar pages (boundary detected)."""
        # Mock dissimilar hashes (large distances)
        mock_hash1 = MagicMock()
        mock_hash2 = MagicMock()
        mock_hash3 = MagicMock()
        mock_hash4 = MagicMock()

        # Large distances indicating dissimilarity
        mock_hash1.__sub__.return_value = 25  # pHash distance > 10
        mock_hash2.__sub__.return_value = 20  # aHash distance > 12
        mock_hash3.__sub__.return_value = 18  # dHash distance > 12

        mock_imagehash.phash.side_effect = [mock_hash1, mock_hash4]
        mock_imagehash.average_hash.side_effect = [mock_hash2, mock_hash4]
        mock_imagehash.dhash.side_effect = [mock_hash3, mock_hash4]

        # Mock page rendering - render_page returns numpy array
        mock_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
        detector.pdf_handler.render_page.return_value = mock_array

        results = detector.detect_boundaries(processed_pages[:2])

        # Should detect boundary
        assert len(results) == 1
        boundary = results[0]
        assert boundary.page_number == 1
        assert boundary.boundary_type == BoundaryType.DOCUMENT_END
        assert boundary.is_between_pages is True
        assert boundary.next_page_number == 2
        assert boundary.detector_type == DetectorType.VISUAL
        assert 0.5 <= boundary.confidence <= 1.0

        # Check evidence
        assert boundary.evidence["votes"] == 3
        assert boundary.evidence["phash_distance"] == 25
        assert boundary.evidence["ahash_distance"] == 20
        assert boundary.evidence["dhash_distance"] == 18

    @patch("pdf_splitter.detection.visual_detector.visual_detector.imagehash")
    def test_voting_threshold(self, mock_imagehash, detector, processed_pages):
        """Test voting threshold behavior."""
        # Set threshold to require 2 votes
        detector.voting_threshold = 2

        # Mock hashes where only 1 algorithm votes for boundary
        mock_hash1 = MagicMock()
        mock_hash2 = MagicMock()

        # Only pHash shows dissimilarity
        mock_hash1.__sub__.side_effect = [15, 4, 5]  # pHash, aHash, dHash distances

        mock_imagehash.phash.side_effect = [mock_hash1, mock_hash2]
        mock_imagehash.average_hash.side_effect = [mock_hash1, mock_hash2]
        mock_imagehash.dhash.side_effect = [mock_hash1, mock_hash2]

        # Mock page rendering - render_page returns numpy array
        mock_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
        detector.pdf_handler.render_page.return_value = mock_array

        results = detector.detect_boundaries(processed_pages[:2])

        # Should not detect boundary (only 1 vote, need 2)
        assert len(results) == 0

    def test_confidence_calculation(self, detector):
        """Test confidence score calculation."""
        # Test with different vote counts
        conf1 = detector._calculate_confidence(votes=1, similarity=0.7)
        conf2 = detector._calculate_confidence(votes=2, similarity=0.5)
        conf3 = detector._calculate_confidence(votes=3, similarity=0.2)

        # Higher votes should give higher confidence
        assert conf1 < conf2 < conf3

        # All should be in valid range
        assert 0.5 <= conf1 <= 1.0
        assert 0.5 <= conf2 <= 1.0
        assert 0.5 <= conf3 <= 1.0

    def test_cache_management(self, detector):
        """Test page image cache management."""
        # Mock page rendering - render_page returns numpy arrays
        mock_arrays = []
        for i in range(25):
            arr = np.ones((100, 100, 3), dtype=np.uint8) * 255
            mock_arrays.append(arr)

        detector.pdf_handler.render_page.side_effect = mock_arrays

        # Access many pages
        for i in range(1, 26):
            detector._get_page_image(i)

        # Cache should be limited to 20 entries
        assert len(detector._page_cache) <= 20

        # Clear cache
        detector.clear_cache()
        assert len(detector._page_cache) == 0

    @patch("pdf_splitter.detection.visual_detector.visual_detector.imagehash")
    def test_error_handling(self, mock_imagehash, detector, processed_pages):
        """Test error handling during detection."""
        # Make hash calculation raise an error
        mock_imagehash.phash.side_effect = Exception("Hash calculation failed")

        # Mock page rendering - render_page returns numpy array
        mock_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
        detector.pdf_handler.render_page.return_value = mock_array

        # Should handle error gracefully
        results = detector.detect_boundaries(processed_pages[:2])
        assert results == []

    def test_detection_with_context(self, detector, processed_pages):
        """Test detection with context tracking."""
        context = DetectionContext(
            config=PDFConfig(),
            total_pages=4,
        )

        # Mock successful detection
        with patch.object(detector, "_calculate_similarity") as mock_calc:
            mock_calc.return_value = (0.3, 2, {"votes": 2})

            detector.detect_boundaries(processed_pages, context)

            # Context should be updated
            assert context.pages_processed == 4
