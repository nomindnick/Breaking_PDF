"""
Visual boundary detection techniques for experimentation.

This module implements various computer vision techniques for detecting
document boundaries based on visual similarity between pages.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


@dataclass
class VisualComparison:
    """Result of visual comparison between two pages."""

    page1_num: int
    page2_num: int
    similarity_score: float
    technique_name: str
    processing_time: float
    metadata: Dict[str, Any]

    @property
    def is_boundary(self) -> bool:
        """Check if this comparison indicates a boundary."""
        # This will be determined by threshold in each technique
        return self.metadata.get("is_boundary", False)


class BaseVisualTechnique(ABC):
    """
    Abstract base class for visual boundary detection techniques.

    Each technique implements a different approach to comparing
    visual similarity between adjacent pages.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize the visual technique.

        Args:
            threshold: Similarity threshold for boundary detection
        """
        self.threshold = threshold
        self.name = self.__class__.__name__

    @abstractmethod
    def compute_similarity(
        self, image1: np.ndarray, image2: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute similarity between two page images.

        Args:
            image1: First page image as numpy array
            image2: Second page image as numpy array

        Returns:
            Tuple of (similarity_score, metadata)
        """
        pass

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image before comparison.

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        # Default: convert to grayscale if needed
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def detect_boundary(
        self, image1: np.ndarray, image2: np.ndarray, page1_num: int, page2_num: int
    ) -> VisualComparison:
        """
        Detect if there's a boundary between two pages.

        Args:
            image1: First page image
            image2: Second page image
            page1_num: Page number of first image
            page2_num: Page number of second image

        Returns:
            VisualComparison result
        """
        import time

        start_time = time.time()

        # Preprocess images
        img1_processed = self.preprocess_image(image1)
        img2_processed = self.preprocess_image(image2)

        # Compute similarity
        similarity, metadata = self.compute_similarity(img1_processed, img2_processed)

        # Determine if boundary based on threshold
        is_boundary = similarity < self.threshold
        metadata["is_boundary"] = is_boundary
        metadata["threshold"] = self.threshold

        processing_time = time.time() - start_time

        return VisualComparison(
            page1_num=page1_num,
            page2_num=page2_num,
            similarity_score=similarity,
            technique_name=self.name,
            processing_time=processing_time,
            metadata=metadata,
        )


class HistogramComparison(BaseVisualTechnique):
    """
    Document boundary detection using histogram comparison.

    Compares grayscale intensity distributions between pages.
    Fast but less accurate for subtle changes.
    """

    def __init__(self, threshold: float = 0.8, bins: int = 256):
        """
        Initialize histogram comparison technique.

        Args:
            threshold: Similarity threshold (0-1, higher means more similar)
            bins: Number of histogram bins
        """
        super().__init__(threshold)
        self.bins = bins

    def compute_similarity(
        self, image1: np.ndarray, image2: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute histogram similarity using correlation coefficient."""
        # Calculate histograms
        hist1 = cv2.calcHist([image1], [0], None, [self.bins], [0, 256])
        hist2 = cv2.calcHist([image2], [0], None, [self.bins], [0, 256])

        # Normalize histograms
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        # Compute correlation coefficient
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Also compute other metrics for analysis
        chi_squared = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)

        metadata = {
            "method": "correlation",
            "bins": self.bins,
            "chi_squared": float(chi_squared),
            "intersection": float(intersection),
        }

        return float(similarity), metadata


class StructuralSimilarity(BaseVisualTechnique):
    """
    Document boundary detection using Structural Similarity Index (SSIM).

    More sophisticated than histogram comparison, considers luminance,
    contrast, and structure.
    """

    def __init__(self, threshold: float = 0.7, window_size: int = 11):
        """
        Initialize SSIM technique.

        Args:
            threshold: SSIM threshold (0-1, higher means more similar)
            window_size: Size of the sliding window for SSIM
        """
        super().__init__(threshold)
        self.window_size = window_size

    def compute_similarity(
        self, image1: np.ndarray, image2: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute SSIM between two images."""
        # Ensure images have the same dimensions
        if image1.shape != image2.shape:
            # Resize to match dimensions
            height = min(image1.shape[0], image2.shape[0])
            width = min(image1.shape[1], image2.shape[1])
            image1 = cv2.resize(image1, (width, height))
            image2 = cv2.resize(image2, (width, height))

        # Compute SSIM
        similarity, ssim_image = ssim(
            image1, image2, win_size=self.window_size, full=True
        )

        # Compute mean and std of SSIM image for additional insights
        ssim_mean = np.mean(ssim_image)
        ssim_std = np.std(ssim_image)

        metadata = {
            "window_size": self.window_size,
            "ssim_mean": float(ssim_mean),
            "ssim_std": float(ssim_std),
            "image_shape": image1.shape,
        }

        return float(similarity), metadata


class PerceptualHash(BaseVisualTechnique):
    """
    Document boundary detection using perceptual hashing.

    Fast and robust to minor variations, good baseline approach.
    """

    def __init__(self, threshold: float = 10, hash_size: int = 8):
        """
        Initialize perceptual hash technique.

        Args:
            threshold: Hamming distance threshold (lower means more similar)
            hash_size: Size of the hash (8 means 64-bit hash)
        """
        # Note: For hashing, lower distance means more similar
        # So we use threshold differently
        super().__init__(threshold)
        self.hash_size = hash_size

    def compute_similarity(
        self, image1: np.ndarray, image2: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute perceptual hash similarity."""
        # Convert numpy arrays to PIL Images
        pil_img1 = Image.fromarray(image1)
        pil_img2 = Image.fromarray(image2)

        # Compute perceptual hashes
        hash1 = imagehash.phash(pil_img1, hash_size=self.hash_size)
        hash2 = imagehash.phash(pil_img2, hash_size=self.hash_size)

        # Compute Hamming distance
        distance = hash1 - hash2

        # Also try other hash algorithms for comparison
        ahash_dist = imagehash.average_hash(pil_img1) - imagehash.average_hash(pil_img2)
        dhash_dist = imagehash.dhash(pil_img1) - imagehash.dhash(pil_img2)

        # Convert distance to similarity (inverse relationship)
        # Max possible distance is hash_size^2
        max_distance = self.hash_size * self.hash_size
        similarity = 1.0 - (distance / max_distance)

        metadata = {
            "hash_algorithm": "phash",
            "hash_size": self.hash_size,
            "hamming_distance": int(distance),
            "ahash_distance": int(ahash_dist),
            "dhash_distance": int(dhash_dist),
            "hash1": str(hash1),
            "hash2": str(hash2),
        }

        return float(similarity), metadata

    def detect_boundary(
        self, image1: np.ndarray, image2: np.ndarray, page1_num: int, page2_num: int
    ) -> VisualComparison:
        """Override to handle distance-based threshold."""
        import time

        start_time = time.time()

        # Preprocess images
        img1_processed = self.preprocess_image(image1)
        img2_processed = self.preprocess_image(image2)

        # Compute similarity
        similarity, metadata = self.compute_similarity(img1_processed, img2_processed)

        # For hashing, we check if distance exceeds threshold
        # The threshold is in terms of Hamming distance
        hamming_distance = metadata["hamming_distance"]
        is_boundary = hamming_distance > self.threshold
        metadata["is_boundary"] = is_boundary
        metadata["threshold"] = self.threshold

        processing_time = time.time() - start_time

        return VisualComparison(
            page1_num=page1_num,
            page2_num=page2_num,
            similarity_score=similarity,
            technique_name=self.name,
            processing_time=processing_time,
            metadata=metadata,
        )


# Factory function for creating techniques
def create_technique(name: str, **kwargs) -> BaseVisualTechnique:
    """
    Create a visual technique by name.

    Args:
        name: Name of the technique
        **kwargs: Arguments for the technique

    Returns:
        Instance of the requested technique
    """
    # Import here to avoid circular import
    from .combined_hash import CombinedHash

    techniques = {
        "histogram": HistogramComparison,
        "ssim": StructuralSimilarity,
        "phash": PerceptualHash,
        "perceptual_hash": PerceptualHash,
        "combined": CombinedHash,
        "combined_hash": CombinedHash,
    }

    if name.lower() not in techniques:
        raise ValueError(f"Unknown technique: {name}")

    return techniques[name.lower()](**kwargs)
