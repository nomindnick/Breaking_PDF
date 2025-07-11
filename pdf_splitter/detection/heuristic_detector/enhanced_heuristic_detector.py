"""
Enhanced heuristic detector with better support for short documents and edge cases.

This module extends the base heuristic detector with:
1. Minimum content threshold handling
2. Short document-specific patterns
3. Context-aware detection for better accuracy
"""

from typing import Dict, List, Optional
import logging

from pdf_splitter.detection.base_detector import (
    ProcessedPage, 
    BoundaryResult,
    BoundaryType,
    DetectorType
)
from pdf_splitter.detection.heuristic_detector import HeuristicDetector, HeuristicConfig

logger = logging.getLogger(__name__)


class EnhancedHeuristicDetector(HeuristicDetector):
    """Enhanced heuristic detector with better edge case handling."""
    
    def __init__(self, config: HeuristicConfig):
        """Initialize enhanced detector with additional parameters."""
        super().__init__(config)
        
        # Additional configuration for edge cases
        self.min_content_threshold = 10  # Minimum characters for normal processing
        self.short_doc_patterns = {
            "single_line_docs": [
                r"^invoice\s*#?\s*\d+$",
                r"^receipt\s*#?\s*\d+$",
                r"^page\s*\d+$",
                r"^end$",
                r"^blank\s*page$",
            ],
            "minimal_headers": [
                r"^(invoice|receipt|memo|note|report)$",
                r"^(confidential|draft|final)$",
            ]
        }
        
        # Compile patterns
        import re
        self._short_patterns = {}
        for pattern_type, patterns in self.short_doc_patterns.items():
            self._short_patterns[pattern_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def detect_boundaries(
        self, pages: List[ProcessedPage], context: Optional[Dict] = None
    ) -> List[BoundaryResult]:
        """Detect boundaries with enhanced handling for edge cases."""
        if not pages or len(pages) < 2:
            return []
        
        boundaries = []
        
        for i in range(len(pages) - 1):
            prev_page = pages[i]
            curr_page = pages[i + 1]
            
            # Check if we're dealing with short documents
            prev_is_short = self._is_short_document(prev_page)
            curr_is_short = self._is_short_document(curr_page)
            
            if prev_is_short or curr_is_short:
                # Use specialized detection for short documents
                result = self._detect_short_document_boundary(
                    prev_page, curr_page, i, prev_is_short, curr_is_short
                )
            else:
                # Use standard detection
                result = self._detect_boundary(prev_page, curr_page, i)
            
            if result and result.confidence >= self.config.min_confidence:
                boundaries.append(result)
        
        return boundaries
    
    def _is_short_document(self, page: ProcessedPage) -> bool:
        """Check if a page is a short document."""
        # Empty or nearly empty
        if not page.text or len(page.text.strip()) < self.min_content_threshold:
            return True
        
        # Single line
        lines = [l for l in page.text.split('\n') if l.strip()]
        if len(lines) <= 1:
            return True
        
        # Very few words
        words = page.text.split()
        if len(words) <= 5:
            return True
        
        return False
    
    def _detect_short_document_boundary(
        self,
        prev_page: ProcessedPage,
        curr_page: ProcessedPage,
        position: int,
        prev_is_short: bool,
        curr_is_short: bool
    ) -> Optional[BoundaryResult]:
        """Specialized detection for short documents."""
        signals = {}
        
        # Empty page handling
        if not prev_page.text.strip() or not curr_page.text.strip():
            # Empty pages often indicate document boundaries
            signals["empty_page"] = 0.7
        
        # Check for single-line document patterns
        if prev_is_short:
            prev_text = prev_page.text.strip().lower()
            for pattern in self._short_patterns["single_line_docs"]:
                if pattern.match(prev_text):
                    signals["single_line_doc"] = 0.8
                    break
        
        # Check for minimal headers
        if curr_is_short:
            curr_text = curr_page.text.strip().lower()
            for pattern in self._short_patterns["minimal_headers"]:
                if pattern.match(curr_text):
                    signals["minimal_header"] = 0.7
                    break
        
        # If both are short, likely a boundary
        if prev_is_short and curr_is_short:
            # Unless they're very similar
            if prev_page.text.strip() != curr_page.text.strip():
                signals["both_short"] = 0.6
        
        # Transition from normal to short or vice versa
        if prev_is_short != curr_is_short:
            signals["length_transition"] = 0.5
        
        # Combine signals
        if signals:
            confidence = max(signals.values())
            
            # Boost confidence if multiple signals present
            if len(signals) > 1:
                confidence = min(1.0, confidence + 0.1 * (len(signals) - 1))
            
            return BoundaryResult(
                page_number=position,
                confidence=confidence,
                boundary_type=BoundaryType.DOCUMENT_START,
                evidence={
                    "detector": "enhanced_heuristic",
                    "short_doc_detection": True,
                    "signals": signals,
                    "prev_is_short": prev_is_short,
                    "curr_is_short": curr_is_short,
                },
                detector_type=DetectorType.HEURISTIC
            )
        
        # Fall back to standard detection if no short doc patterns
        return self._detect_boundary(prev_page, curr_page, position)
    
    def _detect_boundary(
        self, prev_page: ProcessedPage, curr_page: ProcessedPage, position: int
    ) -> Optional[BoundaryResult]:
        """Enhanced boundary detection with context awareness."""
        # Use parent's detect_boundary_pair method
        results = super().detect_boundaries([prev_page, curr_page])
        result = results[0] if results else None
        
        # Apply additional enhancements
        if result:
            # Adjust confidence based on content length
            prev_len = len(prev_page.text.strip())
            curr_len = len(curr_page.text.strip())
            
            # Very short pages get a confidence boost
            if prev_len < 50 or curr_len < 50:
                result.confidence = min(1.0, result.confidence * 1.2)
            
            # Add context to evidence
            result.evidence["content_lengths"] = {
                "prev": prev_len,
                "curr": curr_len
            }
        
        return result


def create_enhanced_config() -> HeuristicConfig:
    """Create configuration optimized for diverse documents including edge cases."""
    from pdf_splitter.detection.heuristic_detector import get_general_purpose_config
    
    # Start with general purpose config
    config = get_general_purpose_config()
    
    # Adjust for better edge case handling
    config.min_confidence = 0.25  # Lower threshold for short docs
    
    # Add patterns for edge cases
    if "terminal_phrases" in config.patterns:
        config.patterns["terminal_phrases"].params["phrases"].extend([
            "end",
            "blank page",
            "this page intentionally left blank",
            "[end of document]",
        ])
    
    if "document_keywords" in config.patterns:
        config.patterns["document_keywords"].params["keywords"].extend([
            "RECEIPT",
            "STUB",
            "SLIP",
            "VOUCHER",
            "TICKET",
        ])
    
    return config