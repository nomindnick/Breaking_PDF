"""Heuristic-based document boundary detector.

This module implements a fast, pattern-based approach to detecting document boundaries
using multiple heuristic signals that can be combined and weighted for optimal results.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pdf_splitter.detection.base_detector import (
    BaseDetector,
    BoundaryResult,
    BoundaryType,
    DetectionContext,
    DetectorType,
    ProcessedPage,
)

logger = logging.getLogger(__name__)


@dataclass
class PatternConfig:
    """Configuration for a single heuristic pattern."""

    name: str
    enabled: bool = True
    weight: float = 1.0
    confidence_threshold: float = 0.5
    params: Dict[str, any] = field(default_factory=dict)


@dataclass
class HeuristicConfig:
    """Configuration for the heuristic detector."""

    # Pattern configurations
    patterns: Dict[str, PatternConfig] = field(default_factory=dict)

    # Global settings
    min_confidence_threshold: float = 0.3
    max_confidence_threshold: float = 0.9
    ensemble_threshold: float = 0.5

    # Performance settings
    max_text_length: int = 5000  # Limit text analysis for performance

    def __post_init__(self):
        """Initialize default patterns if none provided."""
        if not self.patterns:
            self.patterns = self._get_default_patterns()

    def _get_default_patterns(self) -> Dict[str, PatternConfig]:
        """Get default pattern configurations."""
        return {
            "date_pattern": PatternConfig(
                name="date_pattern",
                weight=0.7,
                confidence_threshold=0.6,
                params={
                    "formats": [
                        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # MM/DD/YYYY
                        r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",  # MM-DD-YYYY
                        r"\b\w+ \d{1,2}, \d{4}\b",  # Month DD, YYYY
                    ]
                },
            ),
            "document_keywords": PatternConfig(
                name="document_keywords",
                weight=0.8,
                confidence_threshold=0.7,
                params={
                    "keywords": [
                        "MEMORANDUM",
                        "INVOICE",
                        "CONTRACT",
                        "AGREEMENT",
                        "LETTER",
                        "REPORT",
                        "NOTICE",
                        "CERTIFICATE",
                        "STATEMENT",
                        "RECEIPT",
                        "FORM",
                        "APPLICATION",
                    ]
                },
            ),
            "page_numbering": PatternConfig(
                name="page_numbering",
                weight=0.9,
                confidence_threshold=0.8,
                params={
                    "patterns": [
                        r"Page \d+ of \d+",
                        r"Page \d+$",
                        r"^\d+$",  # Just a number
                    ]
                },
            ),
            "email_header": PatternConfig(
                name="email_header",
                weight=0.95,
                confidence_threshold=0.9,
                params={
                    "patterns": [
                        r"^From:\s*.*\nTo:\s*.*\nSubject:\s*",
                        r"^From:\s*.*\nSent:\s*.*\nTo:\s*",
                    ]
                },
            ),
            "terminal_phrases": PatternConfig(
                name="terminal_phrases",
                weight=0.6,
                confidence_threshold=0.5,
                params={
                    "phrases": [
                        "Sincerely",
                        "Very truly yours",
                        "Respectfully",
                        "Best regards",
                        "Thank you",
                        "Regards",
                        "Respectfully submitted",
                        "Yours truly",
                    ]
                },
            ),
            "whitespace_ratio": PatternConfig(
                name="whitespace_ratio",
                weight=0.5,
                confidence_threshold=0.4,
                params={
                    "end_threshold": 0.6,  # Page with >60% whitespace at end
                    "start_threshold": 0.3,  # Page with >30% whitespace at start
                },
            ),
            "header_footer_change": PatternConfig(
                name="header_footer_change",
                weight=0.7,
                confidence_threshold=0.6,
                params={
                    "top_lines": 3,
                    "bottom_lines": 3,
                },
            ),
        }


class HeuristicDetector(BaseDetector):
    """Fast heuristic-based document boundary detector."""

    def __init__(self, config: Optional[HeuristicConfig] = None):
        """Initialize the heuristic detector.

        Args:
            config: Configuration for the detector. Uses defaults if not provided.
        """
        super().__init__()
        self.config = config or HeuristicConfig()
        self._pattern_cache: Dict[str, List[re.Pattern]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        for pattern_name, pattern_config in self.config.patterns.items():
            if not pattern_config.enabled:
                continue

            if pattern_name in ["date_pattern", "page_numbering", "email_header"]:
                patterns = pattern_config.params.get(
                    "patterns", pattern_config.params.get("formats", [])
                )
                self._pattern_cache[pattern_name] = [
                    re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns
                ]

    def _extract_text_segments(self, page: ProcessedPage) -> Dict[str, str]:
        """Extract relevant text segments from a page for analysis.

        Args:
            page: The processed page to analyze

        Returns:
            Dictionary with text segments (full, top, bottom, etc.)
        """
        text = page.text[: self.config.max_text_length] if page.text else ""
        lines = text.split("\n")

        # Extract segments
        header_config = self.config.patterns.get("header_footer_change")
        top_lines = header_config.params.get("top_lines", 3) if header_config else 3
        bottom_lines = (
            header_config.params.get("bottom_lines", 3) if header_config else 3
        )

        return {
            "full": text,
            "top": "\n".join(lines[:top_lines]) if lines else "",
            "bottom": "\n".join(lines[-bottom_lines:]) if lines else "",
            "lines": lines,
        }

    def _detect_date_pattern(self, text_segments: Dict[str, str]) -> float:
        """Detect date patterns that might indicate document start."""
        if "date_pattern" not in self._pattern_cache:
            return 0.0

        text = text_segments["top"]  # Dates usually appear at top
        for pattern in self._pattern_cache["date_pattern"]:
            if pattern.search(text):
                return 0.8  # High confidence if date found at top
        return 0.0

    def _detect_document_keywords(self, text_segments: Dict[str, str]) -> float:
        """Detect document type keywords."""
        config = self.config.patterns.get("document_keywords")
        if not config:
            return 0.0
        keywords = config.params.get("keywords", [])

        text_upper = text_segments["top"].upper()

        # Check for keywords in top portion
        for keyword in keywords:
            if keyword in text_upper:
                # Higher confidence if it appears early and prominently
                if text_upper.strip().startswith(keyword):
                    return 0.9
                elif keyword in text_upper[:100]:
                    return 0.7
                else:
                    return 0.5
        return 0.0

    def _detect_page_numbering(
        self, prev_segments: Dict[str, str], curr_segments: Dict[str, str]
    ) -> float:
        """Detect page numbering resets or patterns."""
        if "page_numbering" not in self._pattern_cache:
            return 0.0

        # Check for "Page 1 of X" pattern in current page
        for pattern in self._pattern_cache["page_numbering"]:
            match = pattern.search(curr_segments["bottom"])
            if match and "Page 1 " in match.group():
                return 0.95  # Very high confidence for page 1

        # Check for page number discontinuity
        prev_nums = []
        curr_nums = []

        for pattern in self._pattern_cache["page_numbering"]:
            for match in pattern.finditer(prev_segments["bottom"]):
                try:
                    nums = [int(n) for n in re.findall(r"\d+", match.group())]
                    if nums:
                        prev_nums.extend(nums)
                except Exception:
                    pass

            for match in pattern.finditer(curr_segments["bottom"]):
                try:
                    nums = [int(n) for n in re.findall(r"\d+", match.group())]
                    if nums:
                        curr_nums.extend(nums)
                except Exception:
                    pass

        # Check for reset (e.g., page 10 -> page 1)
        if prev_nums and curr_nums:
            if max(prev_nums) > 5 and min(curr_nums) == 1:
                return 0.9

        return 0.0

    def _detect_email_header(self, text_segments: Dict[str, str]) -> float:
        """Detect email headers."""
        if "email_header" not in self._pattern_cache:
            return 0.0

        for pattern in self._pattern_cache["email_header"]:
            if pattern.search(text_segments["top"]):
                return 0.95  # Very high confidence for email headers
        return 0.0

    def _detect_terminal_phrases(self, text_segments: Dict[str, str]) -> float:
        """Detect phrases that typically end documents."""
        config = self.config.patterns.get("terminal_phrases")
        if not config:
            return 0.0
        phrases = config.params.get("phrases", [])

        bottom_text = text_segments["bottom"].lower()

        for phrase in phrases:
            if phrase.lower() in bottom_text:
                # Check if it's near the end and followed by minimal text
                idx = bottom_text.find(phrase.lower())
                remaining_text = bottom_text[idx + len(phrase) :].strip()
                if len(remaining_text) < 100:  # Not much text after
                    return 0.7
                else:
                    return 0.3
        return 0.0

    def _detect_whitespace_ratio(
        self, text_segments: Dict[str, str]
    ) -> Tuple[float, float]:
        """Detect significant whitespace at page boundaries.

        Returns:
            Tuple of (end_signal, start_signal) confidence scores
        """
        config = self.config.patterns.get("whitespace_ratio")
        if not config:
            return 0.0, 0.0

        lines = text_segments["lines"]
        if not lines:
            return 0.0, 0.0

        # Check whitespace at end of page
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            last_content_idx = len(lines) - 1
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip():
                    last_content_idx = i
                    break

            end_whitespace_ratio = (len(lines) - last_content_idx - 1) / len(lines)
            end_signal = (
                0.8 if end_whitespace_ratio > config.params["end_threshold"] else 0.0
            )
        else:
            end_signal = 0.0

        # Check whitespace at start of page
        first_content_idx = 0
        for i, line in enumerate(lines):
            if line.strip():
                first_content_idx = i
                break

        start_whitespace_ratio = first_content_idx / len(lines) if lines else 0
        start_signal = (
            0.6 if start_whitespace_ratio > config.params["start_threshold"] else 0.0
        )

        return end_signal, start_signal

    def _detect_header_footer_change(
        self, prev_segments: Dict[str, str], curr_segments: Dict[str, str]
    ) -> float:
        """Detect changes in headers or footers between pages."""
        # Simple similarity check
        if prev_segments["top"] != curr_segments["top"]:
            # Check if it's a significant change
            if not prev_segments["top"] or not curr_segments["top"]:
                return 0.7
            # Could implement more sophisticated similarity here
            return 0.5
        return 0.0

    def _combine_signals(self, signals: Dict[str, float]) -> float:
        """Combine multiple heuristic signals into a final confidence score.

        Args:
            signals: Dictionary of signal_name -> confidence pairs

        Returns:
            Combined confidence score
        """
        if not signals:
            return 0.0

        # Weighted average based on pattern configurations
        total_weight = 0.0
        weighted_sum = 0.0

        for signal_name, confidence in signals.items():
            if signal_name in self.config.patterns:
                pattern_config = self.config.patterns[signal_name]
                if pattern_config.enabled and confidence > 0:
                    weight = pattern_config.weight
                    total_weight += weight
                    weighted_sum += weight * confidence

        if total_weight > 0:
            combined = weighted_sum / total_weight

            # Apply ensemble threshold
            if combined >= self.config.ensemble_threshold:
                # Boost confidence if multiple signals agree
                active_signals = sum(1 for c in signals.values() if c > 0.3)
                if active_signals >= 3:
                    combined = min(combined * 1.2, 1.0)
                elif active_signals >= 2:
                    combined = min(combined * 1.1, 1.0)

            return combined
        else:
            return 0.0

    def detect_boundary(
        self,
        prev_page: ProcessedPage,
        current_page: ProcessedPage,
        context: Optional[DetectionContext] = None,
    ) -> BoundaryResult:
        """Detect if there's a document boundary between two pages.

        Args:
            prev_page: The previous page
            current_page: The current page
            context: Optional context about the detection task

        Returns:
            BoundaryResult with detection outcome
        """
        # Extract text segments for analysis
        prev_segments = self._extract_text_segments(prev_page)
        curr_segments = self._extract_text_segments(current_page)

        # Run all enabled pattern detectors
        signals = {}

        # Patterns that only look at current page
        if (
            self.config.patterns.get("date_pattern")
            and self.config.patterns["date_pattern"].enabled
        ):
            signals["date_pattern"] = self._detect_date_pattern(curr_segments)

        if (
            self.config.patterns.get("document_keywords")
            and self.config.patterns["document_keywords"].enabled
        ):
            signals["document_keywords"] = self._detect_document_keywords(curr_segments)

        if (
            self.config.patterns.get("email_header")
            and self.config.patterns["email_header"].enabled
        ):
            signals["email_header"] = self._detect_email_header(curr_segments)

        # Patterns that look at previous page
        if (
            self.config.patterns.get("terminal_phrases")
            and self.config.patterns["terminal_phrases"].enabled
        ):
            signals["terminal_phrases"] = self._detect_terminal_phrases(prev_segments)

        # Patterns that compare both pages
        if (
            self.config.patterns.get("page_numbering")
            and self.config.patterns["page_numbering"].enabled
        ):
            signals["page_numbering"] = self._detect_page_numbering(
                prev_segments, curr_segments
            )

        if (
            self.config.patterns.get("header_footer_change")
            and self.config.patterns["header_footer_change"].enabled
        ):
            signals["header_footer_change"] = self._detect_header_footer_change(
                prev_segments, curr_segments
            )

        # Whitespace patterns
        if (
            self.config.patterns.get("whitespace_ratio")
            and self.config.patterns["whitespace_ratio"].enabled
        ):
            end_signal, start_signal = self._detect_whitespace_ratio(prev_segments)
            if end_signal > 0:
                signals["whitespace_end"] = end_signal
            if start_signal > 0:
                signals["whitespace_start"] = start_signal

        # Combine signals
        confidence = self._combine_signals(signals)

        # Log detailed results for debugging
        if signals:
            active_signals = {k: v for k, v in signals.items() if v > 0}
            if active_signals:
                logger.debug(
                    f"Heuristic signals for pages {prev_page.page_number}->{current_page.page_number}: "
                    f"{active_signals}, combined confidence: {confidence:.3f}"
                )

        # Make decision based on confidence
        is_boundary = confidence >= self.config.min_confidence_threshold

        # Create detailed metadata
        metadata = {
            "signals": signals,
            "active_patterns": [k for k, v in signals.items() if v > 0],
            "combined_confidence": confidence,
            "threshold": self.config.min_confidence_threshold,
        }

        # Determine boundary type based on signals
        if is_boundary:
            if "email_header" in signals and signals["email_header"] > 0.5:
                boundary_type = BoundaryType.DOCUMENT_START
            elif "terminal_phrases" in signals and signals["terminal_phrases"] > 0.5:
                boundary_type = BoundaryType.DOCUMENT_END
            elif "page_numbering" in signals and signals["page_numbering"] > 0.8:
                boundary_type = BoundaryType.DOCUMENT_START
            else:
                boundary_type = BoundaryType.SECTION_BREAK
        else:
            boundary_type = BoundaryType.PAGE_CONTINUATION

        return BoundaryResult(
            page_number=current_page.page_number,
            boundary_type=boundary_type,
            confidence=confidence,
            detector_type=DetectorType.HEURISTIC,
            evidence=metadata,
            reasoning=f"Detected {len(active_signals)} active patterns: {', '.join(active_signals.keys())}"
            if active_signals
            else "No strong patterns detected",
            is_between_pages=True,
            next_page_number=current_page.page_number + 1,
        )

    def detect_all_boundaries(
        self, pages: List[ProcessedPage], context: Optional[DetectionContext] = None
    ) -> List[BoundaryResult]:
        """Detect boundaries across all consecutive page pairs.

        Args:
            pages: List of processed pages
            context: Optional context about the detection task

        Returns:
            List of boundary results
        """
        if len(pages) < 2:
            return []

        results = []
        for i in range(len(pages) - 1):
            result = self.detect_boundary(pages[i], pages[i + 1], context)
            results.append(result)

        return results

    def detect_boundaries(
        self, pages: List[ProcessedPage], context: Optional[DetectionContext] = None
    ) -> List[BoundaryResult]:
        """Detect document boundaries in a list of processed pages.

        This is the required abstract method from BaseDetector.

        Args:
            pages: List of processed pages to analyze
            context: Optional context information for detection

        Returns:
            List of detected boundaries with confidence scores
        """
        return self.detect_all_boundaries(pages, context)

    def get_detector_type(self) -> DetectorType:
        """Return the type of this detector."""
        return DetectorType.HEURISTIC
