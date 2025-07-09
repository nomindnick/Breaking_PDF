"""Tests for visual boundary detection techniques."""

import numpy as np
import pytest

from pdf_splitter.detection.visual_detector.experiments.visual_techniques import (
    HistogramComparison,
    PerceptualHash,
    StructuralSimilarity,
    VisualComparison,
    create_technique,
)


class TestVisualTechniques:
    """Test visual detection techniques."""

    @pytest.fixture
    def sample_image1(self):
        """Create a simple test image - white background with black text."""
        img = np.ones((100, 100), dtype=np.uint8) * 255
        # Add some "text" (black rectangles)
        img[20:30, 20:80] = 0
        img[40:50, 20:80] = 0
        img[60:70, 20:80] = 0
        return img

    @pytest.fixture
    def sample_image2(self):
        """Create a similar test image with slight differences."""
        img = np.ones((100, 100), dtype=np.uint8) * 255
        # Similar but different "text"
        img[25:35, 20:80] = 0
        img[45:55, 20:80] = 0
        img[65:75, 20:80] = 0
        return img

    @pytest.fixture
    def different_image(self):
        """Create a very different test image."""
        img = np.ones((100, 100), dtype=np.uint8) * 128  # Gray background
        # Different pattern
        img[10:90, 45:55] = 0  # Vertical line
        img[45:55, 10:90] = 255  # Horizontal line
        return img

    def test_histogram_comparison_similar(self, sample_image1, sample_image2):
        """Test histogram comparison on similar images."""
        technique = HistogramComparison(threshold=0.8)

        result = technique.detect_boundary(sample_image1, sample_image2, 1, 2)

        assert isinstance(result, VisualComparison)
        assert result.similarity_score > 0.8
        assert not result.is_boundary
        assert result.technique_name == "HistogramComparison"

    def test_histogram_comparison_different(self, sample_image1, different_image):
        """Test histogram comparison on different images."""
        technique = HistogramComparison(threshold=0.8)

        result = technique.detect_boundary(sample_image1, different_image, 1, 2)

        assert result.similarity_score < 0.8
        assert result.is_boundary

    def test_ssim_similar(self, sample_image1, sample_image2):
        """Test SSIM on similar images."""
        technique = StructuralSimilarity(threshold=0.7)

        result = technique.detect_boundary(sample_image1, sample_image2, 1, 2)

        assert isinstance(result, VisualComparison)
        assert result.similarity_score > 0.5  # SSIM is sensitive
        assert result.technique_name == "StructuralSimilarity"

    def test_ssim_different(self, sample_image1, different_image):
        """Test SSIM on different images."""
        technique = StructuralSimilarity(threshold=0.7)

        result = technique.detect_boundary(sample_image1, different_image, 1, 2)

        assert result.similarity_score < 0.7
        assert result.is_boundary

    def test_perceptual_hash_similar(self, sample_image1, sample_image2):
        """Test perceptual hash on similar images."""
        technique = PerceptualHash(threshold=10)  # Hamming distance

        result = technique.detect_boundary(sample_image1, sample_image2, 1, 2)

        assert isinstance(result, VisualComparison)
        assert result.metadata["hamming_distance"] < 10
        assert not result.is_boundary
        assert result.technique_name == "PerceptualHash"

    def test_perceptual_hash_different(self, sample_image1, different_image):
        """Test perceptual hash on different images."""
        technique = PerceptualHash(threshold=10)

        result = technique.detect_boundary(sample_image1, different_image, 1, 2)

        assert result.metadata["hamming_distance"] > 10
        assert result.is_boundary

    def test_create_technique(self):
        """Test technique factory function."""
        hist = create_technique("histogram", bins=128)
        assert isinstance(hist, HistogramComparison)
        assert hist.bins == 128

        ssim = create_technique("ssim", window_size=7)
        assert isinstance(ssim, StructuralSimilarity)
        assert ssim.window_size == 7

        phash = create_technique("phash", hash_size=16)
        assert isinstance(phash, PerceptualHash)
        assert phash.hash_size == 16

    def test_preprocessing(self):
        """Test image preprocessing."""
        # Create RGB image
        rgb_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        technique = HistogramComparison()
        gray_image = technique.preprocess_image(rgb_image)

        assert len(gray_image.shape) == 2  # Should be grayscale
        assert gray_image.shape == (100, 100)

    def test_visual_comparison_dataclass(self):
        """Test VisualComparison dataclass."""
        comp = VisualComparison(
            page1_num=1,
            page2_num=2,
            similarity_score=0.85,
            technique_name="TestTechnique",
            processing_time=0.05,
            metadata={"is_boundary": False},
        )

        assert comp.page1_num == 1
        assert comp.page2_num == 2
        assert comp.similarity_score == 0.85
        assert not comp.is_boundary

    def test_technique_metadata(self, sample_image1, sample_image2):
        """Test that techniques include proper metadata."""
        # Test histogram metadata
        hist = HistogramComparison()
        _, hist_meta = hist.compute_similarity(sample_image1, sample_image2)
        assert "chi_squared" in hist_meta
        assert "intersection" in hist_meta

        # Test SSIM metadata
        ssim = StructuralSimilarity()
        _, ssim_meta = ssim.compute_similarity(sample_image1, sample_image2)
        assert "ssim_mean" in ssim_meta
        assert "ssim_std" in ssim_meta

        # Test hash metadata
        phash = PerceptualHash()
        _, hash_meta = phash.compute_similarity(sample_image1, sample_image2)
        assert "hamming_distance" in hash_meta
        assert "hash1" in hash_meta
        assert "hash2" in hash_meta
