"""
Combined hash approach for improved boundary detection.

This module implements a technique that combines multiple hash algorithms
to improve precision while maintaining good recall.
"""

import logging
from typing import Any, Dict, Tuple

import imagehash
import numpy as np
from PIL import Image

from .visual_techniques import BaseVisualTechnique

logger = logging.getLogger(__name__)


class CombinedHash(BaseVisualTechnique):
    """
    Document boundary detection using combined hash algorithms.

    Combines pHash, aHash, and dHash with voting or weighted scoring
    to reduce false positives while maintaining sensitivity.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        phash_weight: float = 0.5,
        ahash_weight: float = 0.25,
        dhash_weight: float = 0.25,
        hash_size: int = 8,
        voting_mode: bool = False,
    ):
        """
        Initialize combined hash technique.

        Args:
            threshold: Combined score threshold (0-1) or vote threshold (0-3)
            phash_weight: Weight for perceptual hash
            ahash_weight: Weight for average hash
            dhash_weight: Weight for difference hash
            hash_size: Size of the hash (8 means 64-bit hash)
            voting_mode: If True, use voting instead of weighted average
        """
        super().__init__(threshold)
        self.phash_weight = phash_weight
        self.ahash_weight = ahash_weight
        self.dhash_weight = dhash_weight
        self.hash_size = hash_size
        self.voting_mode = voting_mode

        # Normalize weights
        total_weight = phash_weight + ahash_weight + dhash_weight
        self.phash_weight /= total_weight
        self.ahash_weight /= total_weight
        self.dhash_weight /= total_weight

        # Individual thresholds for voting mode
        self.phash_threshold = 10  # Hamming distance
        self.ahash_threshold = 12
        self.dhash_threshold = 12

    def compute_similarity(
        self, image1: np.ndarray, image2: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute combined hash similarity."""
        # Convert numpy arrays to PIL Images
        pil_img1 = Image.fromarray(image1)
        pil_img2 = Image.fromarray(image2)

        # Compute all three hashes
        phash1 = imagehash.phash(pil_img1, hash_size=self.hash_size)
        phash2 = imagehash.phash(pil_img2, hash_size=self.hash_size)
        phash_dist = phash1 - phash2

        ahash1 = imagehash.average_hash(pil_img1, hash_size=self.hash_size)
        ahash2 = imagehash.average_hash(pil_img2, hash_size=self.hash_size)
        ahash_dist = ahash1 - ahash2

        dhash1 = imagehash.dhash(pil_img1, hash_size=self.hash_size)
        dhash2 = imagehash.dhash(pil_img2, hash_size=self.hash_size)
        dhash_dist = dhash1 - dhash2

        # Convert distances to similarities (0-1 range)
        max_distance = self.hash_size * self.hash_size
        phash_sim = 1.0 - (phash_dist / max_distance)
        ahash_sim = 1.0 - (ahash_dist / max_distance)
        dhash_sim = 1.0 - (dhash_dist / max_distance)

        if self.voting_mode:
            # Voting approach: each hash votes for boundary/non-boundary
            votes = 0
            if phash_dist > self.phash_threshold:
                votes += 1
            if ahash_dist > self.ahash_threshold:
                votes += 1
            if dhash_dist > self.dhash_threshold:
                votes += 1

            # Similarity is inversely related to votes (0 votes = similar)
            similarity = 1.0 - (votes / 3.0)
        else:
            # Weighted average approach
            similarity = (
                self.phash_weight * phash_sim
                + self.ahash_weight * ahash_sim
                + self.dhash_weight * dhash_sim
            )

        metadata = {
            "hash_algorithm": "combined",
            "hash_size": self.hash_size,
            "phash_distance": int(phash_dist),
            "ahash_distance": int(ahash_dist),
            "dhash_distance": int(dhash_dist),
            "phash_similarity": float(phash_sim),
            "ahash_similarity": float(ahash_sim),
            "dhash_similarity": float(dhash_sim),
            "voting_mode": self.voting_mode,
            "votes": votes if self.voting_mode else None,
            "weights": {
                "phash": self.phash_weight,
                "ahash": self.ahash_weight,
                "dhash": self.dhash_weight,
            },
        }

        return float(similarity), metadata
