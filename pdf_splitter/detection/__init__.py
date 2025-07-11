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
    get_general_purpose_config,
    get_production_config,
    get_conservative_config,
    EnhancedHeuristicDetector,
    create_enhanced_config,
)
from pdf_splitter.detection.llm_detector import LLMDetector
from pdf_splitter.detection.visual_detector import VisualDetector
from pdf_splitter.detection.embeddings_detector import EmbeddingsDetector
from pdf_splitter.detection.optimized_embeddings_detector import OptimizedEmbeddingsDetector
from pdf_splitter.detection.calibrated_heuristic_detector import (
    CalibratedHeuristicDetector,
    create_calibrated_config,
)
from pdf_splitter.detection.signal_combiner import (
    SignalCombiner, 
    SignalCombinerConfig, 
    CombinationStrategy,
    get_production_cascade_config,
    get_high_accuracy_config,
    get_balanced_config,
    get_fast_screening_config,
)

__all__ = [
    "BaseDetector",
    "BoundaryResult",
    "BoundaryType",
    "DetectionContext",
    "DetectorType",
    "ProcessedPage",
    "LLMDetector",
    "VisualDetector",
    "HeuristicDetector",
    "EnhancedHeuristicDetector",
    "CalibratedHeuristicDetector",
    "EmbeddingsDetector",
    "OptimizedEmbeddingsDetector",
    "get_optimized_config",
    "get_fast_screen_config",
    "get_high_precision_config",
    "get_general_purpose_config",
    "get_production_config",
    "get_conservative_config",
    "create_enhanced_config",
    "create_calibrated_config",
    "SignalCombiner",
    "SignalCombinerConfig",
    "CombinationStrategy",
    "get_production_cascade_config",
    "get_high_accuracy_config",
    "get_balanced_config",
    "get_fast_screening_config",
]
