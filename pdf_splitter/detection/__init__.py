"""Document boundary detection module using multi-signal analysis."""

from pdf_splitter.detection.base_detector import (
    BaseDetector,
    BoundaryResult,
    BoundaryType,
    DetectionContext,
    DetectorType,
    ProcessedPage,
)
from pdf_splitter.detection.heuristic_detector import (
    HeuristicDetector,
    get_fast_screen_config,
    get_high_precision_config,
    get_optimized_config,
)
from pdf_splitter.detection.llm_detector import LLMDetector

__all__ = [
    "BaseDetector",
    "BoundaryResult",
    "BoundaryType",
    "DetectionContext",
    "DetectorType",
    "ProcessedPage",
    "LLMDetector",
    "HeuristicDetector",
    "get_optimized_config",
    "get_fast_screen_config",
    "get_high_precision_config",
]
