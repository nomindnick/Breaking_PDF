"""
Cascade configuration that includes embeddings detector.

This configuration integrates the embeddings detector as an intermediate
verification step between heuristics and LLM.
"""

from pdf_splitter.detection.base_detector import DetectorType
from pdf_splitter.detection.signal_combiner import (
    CombinationStrategy,
    SignalCombinerConfig,
)


def get_embeddings_cascade_config() -> SignalCombinerConfig:
    """
    Get cascade configuration with embeddings detector.
    
    This configuration:
    - Uses heuristics for fast initial screening
    - Uses embeddings for semantic verification (medium cost)
    - Uses visual detector for supplementary signals
    - Uses LLM only for the most uncertain cases
    
    Cascade flow:
    1. High confidence heuristics (>= 0.85) -> Accept
    2. Medium confidence heuristics (0.5-0.85) -> Verify with embeddings
    3. Low confidence heuristics (< 0.5) -> Verify with embeddings + visual
    4. If still uncertain -> LLM verification
    
    Returns:
        SignalCombinerConfig with embeddings integration
    """
    return SignalCombinerConfig(
        combination_strategy=CombinationStrategy.CASCADE_ENSEMBLE,
        
        # Heuristic thresholds
        heuristic_confidence_threshold=0.85,
        
        # Embeddings verification range (replaces some LLM calls)
        # Note: We're overloading the visual range for embeddings
        visual_verification_range=(0.5, 0.85),
        
        # Only very uncertain cases go to LLM
        require_llm_verification_below=0.4,
        
        # Weights when combining signals
        detector_weights={
            DetectorType.HEURISTIC: 0.25,      # General patterns
            DetectorType.EMBEDDINGS: 0.35,     # Semantic understanding
            DetectorType.VISUAL: 0.15,         # Supplementary
            DetectorType.LLM: 0.25,            # Most accurate but expensive
        },
        
        # Agreement settings
        min_agreement_threshold=0.7,
        confidence_boost_on_agreement=0.15,
        
        # Performance settings
        enable_parallel_processing=True,
        max_workers=3,
        
        # Document settings
        min_document_pages=1,
        merge_adjacent_threshold=2,
        add_implicit_start_boundary=False,
    )


def get_embeddings_only_cascade_config() -> SignalCombinerConfig:
    """
    Get cascade configuration for testing without LLM.
    
    This configuration:
    - Uses heuristics for pattern detection
    - Uses embeddings as primary verification
    - No LLM usage (for testing/environments without LLM)
    
    Returns:
        SignalCombinerConfig optimized for heuristic + embeddings
    """
    return SignalCombinerConfig(
        combination_strategy=CombinationStrategy.CASCADE_ENSEMBLE,
        
        # Moderate heuristic threshold
        heuristic_confidence_threshold=0.7,
        
        # Wide embeddings verification range
        visual_verification_range=(0.0, 0.7),  # Using visual range for embeddings
        
        # Never use LLM
        require_llm_verification_below=-1.0,
        
        # Heavier embeddings weight since no LLM
        detector_weights={
            DetectorType.HEURISTIC: 0.3,
            DetectorType.EMBEDDINGS: 0.5,
            DetectorType.VISUAL: 0.2,
        },
        
        min_agreement_threshold=0.6,
        confidence_boost_on_agreement=0.2,
        
        enable_parallel_processing=True,
        max_workers=2,
        
        min_document_pages=1,
        merge_adjacent_threshold=2,
        add_implicit_start_boundary=False,
    )