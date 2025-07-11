"""
Mock LLM detector for integration testing.

This simulates LLM behavior based on known accuracy characteristics:
- True positive rate: 0.889
- False positive rate: 0.111
"""

import random
from typing import List, Optional
import logging

from pdf_splitter.detection.base_detector import (
    BaseDetector,
    BoundaryResult,
    BoundaryType,
    DetectionContext,
    DetectorType,
    ProcessedPage,
)

logger = logging.getLogger(__name__)


class MockLLMDetector(BaseDetector):
    """Mock LLM detector for testing without actual LLM calls."""
    
    def __init__(self, accuracy: float = 0.889):
        """Initialize mock detector with specified accuracy."""
        super().__init__()
        self.accuracy = accuracy
        self.false_positive_rate = 1.0 - accuracy
        
    def get_detector_type(self) -> DetectorType:
        """Return the detector type."""
        return DetectorType.LLM
        
    def detect_boundaries(
        self, 
        pages: List[ProcessedPage], 
        context: Optional[DetectionContext] = None
    ) -> List[BoundaryResult]:
        """Simulate LLM boundary detection."""
        if not pages or len(pages) < 2:
            return []
            
        # Extract target pages from context if provided
        target_pages = None
        if context and context.document_metadata:
            target_pages = context.document_metadata.get("target_pages", None)
            
        boundaries = []
        
        for i in range(len(pages) - 1):
            # Skip if not in target pages
            if target_pages is not None and i not in target_pages:
                continue
                
            # Simulate LLM analysis based on page content
            should_detect = self._simulate_detection(pages[i], pages[i + 1], i)
            
            if should_detect:
                boundaries.append(
                    BoundaryResult(
                        page_number=i,
                        boundary_type=BoundaryType.DOCUMENT_START,
                        confidence=0.95,  # LLMs typically return high confidence
                        detector_type=DetectorType.LLM,
                        evidence={
                            "mock": True,
                            "simulated_accuracy": self.accuracy,
                        },
                        reasoning="Mock LLM detected boundary based on content analysis",
                    )
                )
                
        return boundaries
        
    def _simulate_detection(
        self, 
        prev_page: ProcessedPage, 
        curr_page: ProcessedPage,
        position: int
    ) -> bool:
        """Simulate whether LLM would detect a boundary."""
        # Use position as random seed for consistent results
        random.seed(f"{position}_{prev_page.text[:10]}_{curr_page.text[:10]}")
        
        # Check for obvious boundaries that LLM would catch
        prev_text = prev_page.text.strip().lower()
        curr_text = curr_page.text.strip().lower()
        
        # Strong indicators LLM would recognize
        strong_indicators = [
            curr_text.startswith("from:") and "@" in curr_text,
            curr_text.startswith("invoice"),
            curr_text.startswith("memorandum"),
            curr_text.startswith("contract"),
            curr_text.startswith("report") and len(curr_text) > 20,
            prev_text == "" and curr_text != "",  # Empty to content
            "page 1" in curr_text and "page 2" not in prev_text,
        ]
        
        if any(strong_indicators):
            # LLM would likely detect this with high accuracy
            return random.random() < self.accuracy
        else:
            # Weak or no indicators - LLM might false positive
            return random.random() < self.false_positive_rate