"""
Fixed embeddings detector that only returns actual boundaries.
"""

from typing import List, Optional
from ..embeddings_detector import EmbeddingsDetector
from ..base_detector import BoundaryResult, ProcessedPage, DetectionContext, BoundaryType


class FixedEmbeddingsDetector(EmbeddingsDetector):
    """
    Fixed version of embeddings detector that only returns boundaries,
    not continuation pages.
    """
    
    def detect_boundaries(
        self,
        pages: List[ProcessedPage],
        context: Optional[DetectionContext] = None
    ) -> List[BoundaryResult]:
        """
        Detect boundaries by analyzing semantic similarity between consecutive pages.
        
        This version only returns actual boundaries, not all pages.
        """
        # Get all results from parent
        all_results = super().detect_boundaries(pages, context)
        
        # Filter to only return boundaries
        boundaries = [
            result for result in all_results 
            if result.boundary_type == BoundaryType.DOCUMENT_START
        ]
        
        return boundaries