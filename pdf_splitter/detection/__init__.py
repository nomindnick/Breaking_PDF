"""Document boundary detection using embeddings."""

from pdf_splitter.detection.base_detector import (
    BaseDetector,
    BoundaryResult,
    BoundaryType,
    DetectionContext,
    DetectorType,
    ProcessedPage,
)
from pdf_splitter.detection.embeddings_detector import EmbeddingsDetector
from pdf_splitter.detection.heuristic_detector import HeuristicDetector
from pdf_splitter.detection.llm_detector import LLMDetector
from pdf_splitter.detection.production_detector import create_production_detector
from pdf_splitter.detection.visual_detector import VisualDetector

__all__ = [
    "BaseDetector",
    "BoundaryResult",
    "BoundaryType",
    "DetectionContext",
    "DetectorType",
    "ProcessedPage",
    "EmbeddingsDetector",
    "VisualDetector",
    "LLMDetector",
    "HeuristicDetector",
    "create_production_detector",
]
