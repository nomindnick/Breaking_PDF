"""
Calibrated heuristic detector with improved precision for short documents.

This module provides a calibrated version of the enhanced heuristic detector that:
1. Reduces false positives for short documents
2. Uses more sophisticated pattern matching
3. Provides better confidence calibration
"""

from typing import Dict, List, Optional, Set
import logging
import re

from pdf_splitter.detection.base_detector import (
    ProcessedPage, 
    BoundaryResult,
    BoundaryType,
    DetectorType
)
from pdf_splitter.detection.heuristic_detector import HeuristicDetector, HeuristicConfig

logger = logging.getLogger(__name__)


class CalibratedHeuristicDetector(HeuristicDetector):
    """Calibrated heuristic detector with better precision."""
    
    def __init__(self, config: HeuristicConfig):
        """Initialize calibrated detector."""
        super().__init__(config)
        
        # Calibration parameters
        self.min_content_threshold = 10
        self.very_short_threshold = 20  # Very short pages need special handling
        
        # More specific patterns for short documents
        self.strong_boundary_indicators = {
            "email_start": re.compile(r'^from:\s*\S+@\S+', re.IGNORECASE),
            "document_headers": re.compile(
                r'^(invoice|receipt|memo|memorandum|report|letter|contract|agreement|'
                r'purchase order|po number|bill|statement)\b', 
                re.IGNORECASE
            ),
            "page_numbers": re.compile(r'^page\s+\d+\s+of\s+\d+$', re.IGNORECASE),
            "section_headers": re.compile(r'^(chapter|section|part)\s+\d+', re.IGNORECASE),
            "date_headers": re.compile(r'^(date|dated):\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', re.IGNORECASE),
        }
        
        # Patterns that indicate continuation (not a boundary)
        self.continuation_patterns = {
            "continued": re.compile(r'(continued|cont\.?)\s*$', re.IGNORECASE),
            "page_x_of_y": re.compile(r'page\s+\d+\s+of\s+\d+', re.IGNORECASE),
            "partial_sentence": re.compile(r'^[a-z]'),  # Starts with lowercase
        }
        
        # Generic patterns that should be ignored for short pages
        self.generic_patterns = {
            "simple_page": re.compile(r'^page\s+\d+$', re.IGNORECASE),
            "single_word": re.compile(r'^\w+$'),
            "single_letter": re.compile(r'^[a-zA-Z]$'),
        }
    
    def detect_boundaries(
        self, pages: List[ProcessedPage], context: Optional[Dict] = None
    ) -> List[BoundaryResult]:
        """Detect boundaries with improved calibration."""
        if not pages or len(pages) < 2:
            return []
        
        boundaries = []
        
        for i in range(len(pages) - 1):
            result = self._detect_calibrated_boundary(pages, i)
            if result and result.confidence >= self.config.min_confidence:
                boundaries.append(result)
        
        return boundaries
    
    def _detect_calibrated_boundary(
        self, pages: List[ProcessedPage], position: int
    ) -> Optional[BoundaryResult]:
        """Detect boundary with calibrated confidence."""
        if position >= len(pages) - 1:
            return None
            
        prev_page = pages[position]
        curr_page = pages[position + 1]
        
        # Get text and check if pages are short
        prev_text = prev_page.text.strip()
        curr_text = curr_page.text.strip()
        
        prev_is_short = len(prev_text) < self.very_short_threshold
        curr_is_short = len(curr_text) < self.very_short_threshold
        
        signals = {}
        
        # 1. Check for strong boundary indicators
        if curr_text:
            for pattern_name, pattern in self.strong_boundary_indicators.items():
                if pattern.match(curr_text):
                    signals[f"strong_{pattern_name}"] = 0.9
                    break
        
        # 2. Check for continuation patterns (negative signal)
        has_continuation = False
        if prev_text:
            for pattern_name, pattern in self.continuation_patterns.items():
                if pattern.search(prev_text):
                    has_continuation = True
                    signals[f"continuation_{pattern_name}"] = -0.3
                    
        # 3. Empty page handling
        if not prev_text or not curr_text:
            # Empty page is only a boundary signal if the other page has content
            if bool(prev_text) != bool(curr_text):
                signals["empty_page_transition"] = 0.6
            else:
                # Both empty or very similar - likely not a boundary
                signals["both_empty"] = -0.5
        
        # 4. Check for generic patterns that shouldn't trigger boundaries
        is_generic_prev = any(p.match(prev_text) for p in self.generic_patterns.values())
        is_generic_curr = any(p.match(curr_text) for p in self.generic_patterns.values())
        
        if is_generic_prev or is_generic_curr:
            # Generic content reduces boundary likelihood
            signals["generic_content"] = -0.4
        
        # 5. Short page special handling
        if prev_is_short and curr_is_short:
            # Both pages are very short - need stronger evidence
            if not any(k.startswith("strong_") for k in signals):
                # No strong indicators - check semantic difference
                if prev_text.lower() != curr_text.lower():
                    # Different content but no strong indicators
                    if not (is_generic_prev and is_generic_curr):
                        signals["short_different"] = 0.3  # Low confidence
                else:
                    signals["short_similar"] = -0.5
        
        # 6. Length transition
        if prev_is_short != curr_is_short:
            # Transition from short to long or vice versa
            signals["length_transition"] = 0.4
        
        # 7. Standard heuristic detection for longer pages
        if not prev_is_short and not curr_is_short:
            # Use parent's detection for normal-length pages
            parent_result = super().detect_boundaries([prev_page, curr_page])
            if parent_result:
                return parent_result[0]
        
        # Calculate final confidence
        if not signals:
            return None
            
        # Separate positive and negative signals
        positive_signals = {k: v for k, v in signals.items() if v > 0}
        negative_signals = {k: v for k, v in signals.items() if v < 0}
        
        # Calculate base confidence from positive signals
        if positive_signals:
            base_confidence = max(positive_signals.values())
            
            # Boost if multiple positive signals
            if len(positive_signals) > 1:
                base_confidence = min(1.0, base_confidence + 0.1 * (len(positive_signals) - 1))
        else:
            base_confidence = 0.0
        
        # Apply negative signals as reduction
        if negative_signals:
            reduction = sum(abs(v) for v in negative_signals.values())
            base_confidence = max(0.0, base_confidence - reduction)
        
        # Apply calibration based on document characteristics
        final_confidence = self._calibrate_confidence(
            base_confidence, prev_text, curr_text, signals
        )
        
        if final_confidence > 0:
            return BoundaryResult(
                page_number=position,
                confidence=final_confidence,
                boundary_type=BoundaryType.DOCUMENT_START,
                evidence={
                    "detector": "calibrated_heuristic",
                    "signals": signals,
                    "prev_length": len(prev_text),
                    "curr_length": len(curr_text),
                    "calibrated": True,
                },
                detector_type=DetectorType.HEURISTIC
            )
        
        return None
    
    def _calibrate_confidence(
        self, 
        base_confidence: float, 
        prev_text: str, 
        curr_text: str,
        signals: Dict[str, float]
    ) -> float:
        """Apply confidence calibration based on context."""
        # Strong positive signals should maintain high confidence
        if any(k.startswith("strong_") for k in signals if signals.get(k, 0) > 0):
            return max(0.7, base_confidence)
        
        # Multiple weak signals on short pages should be reduced
        if len(prev_text) < 50 and len(curr_text) < 50:
            if base_confidence < 0.6:
                # Reduce confidence for weak signals on short pages
                return base_confidence * 0.7
        
        # Empty page transitions need careful handling
        if "empty_page_transition" in signals:
            # Check if it's likely a formatting artifact
            if len(prev_text) < 10 or len(curr_text) < 10:
                # Very short text before/after empty - likely not a real boundary
                return base_confidence * 0.8
        
        return base_confidence


def create_calibrated_config() -> HeuristicConfig:
    """Create configuration optimized for calibrated detection."""
    from pdf_splitter.detection.heuristic_detector import get_general_purpose_config
    
    # Start with general purpose config
    config = get_general_purpose_config()
    
    # Lower minimum confidence to allow calibration to work
    config.min_confidence = 0.35
    
    # Reduce weights for patterns that cause false positives
    if "whitespace_changes" in config.patterns:
        config.patterns["whitespace_changes"].weight = 0.2
    
    if "length_changes" in config.patterns:
        config.patterns["length_changes"].weight = 0.3
    
    return config