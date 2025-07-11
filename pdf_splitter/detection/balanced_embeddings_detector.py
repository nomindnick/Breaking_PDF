"""
Balanced embeddings detector with less aggressive post-processing.

This version aims for better generalization while still improving precision.
"""

from typing import List, Optional
from .embeddings_detector import EmbeddingsDetector
from .base_detector import BoundaryResult, ProcessedPage, DetectionContext, BoundaryType


class BalancedEmbeddingsDetector(EmbeddingsDetector):
    """
    Balanced embeddings detector with more conservative post-processing.
    
    This version is less likely to overfit to specific test data.
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.5,
        apply_post_processing: bool = True,
        **kwargs
    ):
        """Initialize with balanced settings."""
        super().__init__(
            model_name=model_name,
            similarity_threshold=similarity_threshold,
            **kwargs
        )
        self.apply_post_processing = apply_post_processing
    
    def detect_boundaries(
        self,
        pages: List[ProcessedPage],
        context: Optional[DetectionContext] = None
    ) -> List[BoundaryResult]:
        """
        Detect boundaries with balanced post-processing.
        """
        # Get all results from parent
        all_results = super().detect_boundaries(pages, context)
        
        # Filter to only boundaries (not continuations)
        boundaries = [
            result for result in all_results 
            if result.boundary_type == BoundaryType.DOCUMENT_START
        ]
        
        # Apply post-processing if enabled
        if self.apply_post_processing:
            boundaries = self._apply_balanced_filters(boundaries, pages)
        
        return boundaries
    
    def _apply_balanced_filters(
        self,
        boundaries: List[BoundaryResult],
        pages: List[ProcessedPage]
    ) -> List[BoundaryResult]:
        """
        Apply more conservative post-processing filters.
        
        Key changes from OptimizedEmbeddingsDetector:
        1. Less aggressive position filtering
        2. More nuanced content analysis
        3. Smarter minimum length handling
        """
        filtered = []
        
        # Sort boundaries by page number
        boundaries.sort(key=lambda b: b.page_number)
        
        for i, boundary in enumerate(boundaries):
            page_num = boundary.page_number
            
            # Filter 1: Very early pages (only page 0) with very low confidence
            # Less aggressive than original (was pages 0-2)
            if page_num == 0 and boundary.confidence < 0.6:
                continue
            
            # Filter 2: Content-based filtering with context
            if page_num + 1 < len(pages):
                curr_text = pages[page_num].text.strip()
                next_text = pages[page_num + 1].text.strip()
                
                # Check for explicit continuation patterns
                continuation_phrases = [
                    "continued from",
                    "continuation of",
                    "...continued",
                    "(continued)",
                ]
                
                # Strong continuation signal - skip unless very high confidence
                if any(phrase in next_text.lower()[:50] for phrase in continuation_phrases):
                    if boundary.confidence < 0.9:
                        continue
                
                # Lowercase start is only filtered if it looks like mid-sentence
                if next_text and next_text[0].islower():
                    # Check if previous page ends with punctuation
                    if curr_text and curr_text[-1] not in '.!?':
                        # Likely mid-sentence split
                        if boundary.confidence < 0.75:
                            continue
            
            # Filter 3: Adaptive minimum document length
            # Only enforce for low-confidence boundaries
            if filtered and boundary.confidence < 0.6:
                last_boundary = filtered[-1]
                doc_length = page_num - last_boundary.page_number
                
                # Only filter very short (1-page) documents with low confidence
                if doc_length == 1:
                    continue
            
            # Filter 4: Pattern-based confidence boost
            # If the boundary has strong patterns, keep it regardless
            if page_num + 1 < len(pages):
                next_text = pages[page_num + 1].text.strip()[:100]
                
                # Strong new document indicators
                strong_patterns = [
                    r'^From:.*@',  # Email start
                    r'^TO:',       # Memo start
                    r'^MEMORANDUM',
                    r'^Subject:',
                    r'^\d{1,2}/\d{1,2}/\d{4}',  # Date at start
                    r'^[A-Z][A-Z\s]{10,}$',  # All caps title
                ]
                
                import re
                for pattern in strong_patterns:
                    if re.match(pattern, next_text, re.IGNORECASE):
                        # This is very likely a new document
                        filtered.append(boundary)
                        break
                else:
                    # No strong pattern found, use regular filtering
                    if page_num not in [b.page_number for b in filtered]:
                        filtered.append(boundary)
            else:
                # Last boundary
                filtered.append(boundary)
        
        return filtered