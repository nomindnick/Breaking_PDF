"""
Visual boundary detector using perceptual hashing techniques.

This module implements document boundary detection based on visual similarity
between adjacent pages using a combined hash voting approach.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import imagehash

# import numpy as np  # Not directly used
from PIL import Image

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.base_detector import (
    BaseDetector,
    BoundaryResult,
    BoundaryType,
    DetectionContext,
    DetectorType,
    ProcessedPage,
)
from pdf_splitter.preprocessing.pdf_handler import PDFHandler

logger = logging.getLogger(__name__)


class VisualDetector(BaseDetector):
    """
    Detects document boundaries using visual similarity analysis.

    This detector uses a combined perceptual hashing approach with voting
    to identify when adjacent pages are visually dissimilar enough to
    indicate a document boundary.
    """

    def __init__(
        self,
        config: Optional[PDFConfig] = None,
        pdf_handler: Optional[PDFHandler] = None,
        hash_size: int = 8,
        voting_threshold: int = 1,
        phash_threshold: int = 10,
        ahash_threshold: int = 12,
        dhash_threshold: int = 12,
    ):
        """
        Initialize the visual detector.

        Args:
            config: PDF processing configuration
            pdf_handler: PDF handler for rendering pages (will create if not provided)
            hash_size: Size of the hash (default: 8 for 64-bit hashes)
            voting_threshold: Number of hash algorithms that must vote for boundary (1-3)
            phash_threshold: Hamming distance threshold for pHash
            ahash_threshold: Hamming distance threshold for aHash
            dhash_threshold: Hamming distance threshold for dHash
        """
        super().__init__(config)
        self.pdf_handler = pdf_handler or PDFHandler(config)

        # Hash configuration
        self.hash_size = hash_size
        self.voting_threshold = voting_threshold
        self.phash_threshold = phash_threshold
        self.ahash_threshold = ahash_threshold
        self.dhash_threshold = dhash_threshold

        # Cache for rendered pages
        self._page_cache: Dict[int, Image.Image] = {}

        logger.info(
            f"Initialized VisualDetector with hash_size={hash_size}, "
            f"voting_threshold={voting_threshold}, thresholds=(p:{phash_threshold}, "
            f"a:{ahash_threshold}, d:{dhash_threshold})"
        )

    def get_detector_type(self) -> DetectorType:
        """Return the detector type."""
        return DetectorType.VISUAL

    def get_confidence_threshold(self) -> float:
        """
        Get minimum confidence threshold for visual detection.

        Visual detection is used as supplementary signal, so we use
        a lower threshold to catch more potential boundaries.
        """
        return 0.5

    def detect_boundaries(
        self,
        pages: List[ProcessedPage],
        context: Optional[DetectionContext] = None,
    ) -> List[BoundaryResult]:
        """
        Detect document boundaries using visual similarity analysis.

        Args:
            pages: List of processed pages to analyze
            context: Optional detection context

        Returns:
            List of detected boundaries
        """
        if not pages:
            return []

        if len(pages) == 1:
            logger.debug("Only one page provided, no boundaries to detect")
            return []

        boundaries = []

        # Process adjacent page pairs
        for i in range(len(pages) - 1):
            page1 = pages[i]
            page2 = pages[i + 1]

            try:
                # Calculate visual similarity
                similarity, votes, evidence = self._calculate_similarity(
                    page1.page_number, page2.page_number
                )

                # Determine if boundary exists based on voting
                is_boundary = votes >= self.voting_threshold

                if is_boundary:
                    # Calculate confidence based on number of votes
                    confidence = self._calculate_confidence(votes, similarity)

                    boundary = BoundaryResult(
                        page_number=page1.page_number,
                        boundary_type=BoundaryType.DOCUMENT_END,
                        confidence=confidence,
                        detector_type=self.get_detector_type(),
                        is_between_pages=True,
                        next_page_number=page2.page_number,
                        evidence=evidence,
                        reasoning=f"Visual dissimilarity detected ({votes}/3 algorithms voted for boundary)",
                    )

                    boundaries.append(boundary)
                    self._detection_history.append(boundary)

                    logger.debug(
                        f"Boundary detected between pages {page1.page_number} and "
                        f"{page2.page_number} (votes={votes}, confidence={confidence:.3f})"
                    )

            except Exception as e:
                logger.warning(
                    f"Error detecting boundary between pages {page1.page_number} "
                    f"and {page2.page_number}: {e}"
                )
                continue

        # Update context if provided
        if context:
            context.update_progress(len(pages))

        return boundaries

    def _calculate_similarity(
        self, page_num1: int, page_num2: int
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Calculate visual similarity between two pages using combined hashing.

        Args:
            page_num1: First page number
            page_num2: Second page number

        Returns:
            Tuple of (combined similarity score, number of votes, evidence dict)
        """
        # Render pages if not cached
        img1 = self._get_page_image(page_num1)
        img2 = self._get_page_image(page_num2)

        # Calculate hashes
        phash1 = imagehash.phash(img1, hash_size=self.hash_size)
        phash2 = imagehash.phash(img2, hash_size=self.hash_size)
        phash_distance = phash1 - phash2

        ahash1 = imagehash.average_hash(img1, hash_size=self.hash_size)
        ahash2 = imagehash.average_hash(img2, hash_size=self.hash_size)
        ahash_distance = ahash1 - ahash2

        dhash1 = imagehash.dhash(img1, hash_size=self.hash_size)
        dhash2 = imagehash.dhash(img2, hash_size=self.hash_size)
        dhash_distance = dhash1 - dhash2

        # Convert distances to similarities (0-1 range)
        max_distance = self.hash_size * self.hash_size
        phash_similarity = 1.0 - (phash_distance / max_distance)
        ahash_similarity = 1.0 - (ahash_distance / max_distance)
        dhash_similarity = 1.0 - (dhash_distance / max_distance)

        # Count votes for boundary
        votes = 0
        if phash_distance > self.phash_threshold:
            votes += 1
        if ahash_distance > self.ahash_threshold:
            votes += 1
        if dhash_distance > self.dhash_threshold:
            votes += 1

        # Combined similarity (weighted average)
        combined_similarity = (
            0.5 * phash_similarity + 0.25 * ahash_similarity + 0.25 * dhash_similarity
        )

        evidence = {
            "phash_distance": int(phash_distance),
            "ahash_distance": int(ahash_distance),
            "dhash_distance": int(dhash_distance),
            "phash_similarity": round(phash_similarity, 3),
            "ahash_similarity": round(ahash_similarity, 3),
            "dhash_similarity": round(dhash_similarity, 3),
            "combined_similarity": round(combined_similarity, 3),
            "votes": votes,
            "voting_threshold": self.voting_threshold,
        }

        return combined_similarity, votes, evidence

    def _calculate_confidence(self, votes: int, similarity: float) -> float:
        """
        Calculate confidence score based on votes and similarity.

        Args:
            votes: Number of algorithms voting for boundary (0-3)
            similarity: Combined similarity score (0-1)

        Returns:
            Confidence score between 0.5 and 1.0
        """
        # Base confidence from votes (0.5 to 0.8)
        vote_confidence = 0.5 + (votes / 3) * 0.3

        # Boost confidence based on dissimilarity
        dissimilarity_boost = (1 - similarity) * 0.2

        confidence = min(1.0, vote_confidence + dissimilarity_boost)

        return confidence

    def _get_page_image(self, page_num: int) -> Image.Image:
        """
        Get rendered image for a page, using cache if available.

        Args:
            page_num: Page number to render

        Returns:
            PIL Image object
        """
        if page_num not in self._page_cache:
            # Render page at moderate DPI for efficiency
            numpy_image = self.pdf_handler.render_page(page_num - 1, dpi=150)

            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(numpy_image)
            self._page_cache[page_num] = pil_image

            # Limit cache size to prevent memory issues
            if len(self._page_cache) > 20:
                # Remove oldest entries
                oldest_pages = sorted(self._page_cache.keys())[:10]
                for page in oldest_pages:
                    del self._page_cache[page]

        return self._page_cache[page_num]

    def clear_cache(self) -> None:
        """Clear the page image cache."""
        self._page_cache.clear()
        logger.debug("Cleared visual detector page cache")
