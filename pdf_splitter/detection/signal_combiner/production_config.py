"""
Production-ready configurations for the Signal Combiner.

These configurations prioritize accuracy while maintaining acceptable performance
(target: <15-20 seconds per page, ideal: <5 seconds per page).
"""

from typing import Dict

from pdf_splitter.detection.base_detector import DetectorType
from pdf_splitter.detection.signal_combiner import (
    CombinationStrategy,
    SignalCombinerConfig,
)


def get_production_cascade_config() -> SignalCombinerConfig:
    """
    Get production-ready cascade configuration.
    
    This configuration:
    - Uses general-purpose heuristics (not overfitted)
    - Triggers LLM verification for most boundaries (accuracy > speed)
    - Keeps processing under 15-20 seconds per page
    - Achieves high accuracy across diverse document types
    
    Returns:
        Production-ready SignalCombinerConfig
    """
    return SignalCombinerConfig(
        # Use cascade strategy for intelligent LLM usage
        combination_strategy=CombinationStrategy.CASCADE_ENSEMBLE,
        
        # Conservative thresholds to ensure accuracy
        # Only very high confidence heuristics skip LLM verification
        heuristic_confidence_threshold=0.85,
        
        # Most boundaries get LLM verification for accuracy
        # This ensures we don't miss boundaries due to weak heuristics
        require_llm_verification_below=0.7,
        
        # Visual detector for medium confidence cases
        # Can help reduce LLM calls while maintaining accuracy
        visual_verification_range=(0.7, 0.85),
        
        # Weights for when using weighted voting (not primary strategy)
        detector_weights={
            DetectorType.HEURISTIC: 0.2,  # Low weight - general purpose
            DetectorType.VISUAL: 0.2,      # Low weight - supplementary
            DetectorType.LLM: 0.6,         # High weight - most accurate
        },
        
        # High agreement threshold for consensus strategy
        min_agreement_threshold=0.8,
        
        # Enable parallel processing for performance
        enable_parallel_processing=True,
        max_workers=3,
        
        # Document constraints
        min_document_pages=1,  # Allow single-page documents
        merge_adjacent_threshold=2,  # Merge boundaries within 2 pages
        
        # Confidence boost when detectors agree
        confidence_boost_on_agreement=0.15,
        
        # Don't add implicit boundaries - let detectors decide
        add_implicit_start_boundary=False,
    )


def get_high_accuracy_config() -> SignalCombinerConfig:
    """
    Get configuration optimized for maximum accuracy.
    
    This configuration:
    - Uses LLM for almost all boundaries
    - Accepts longer processing times for better accuracy
    - Suitable when accuracy is critical
    
    Returns:
        High-accuracy SignalCombinerConfig
    """
    return SignalCombinerConfig(
        combination_strategy=CombinationStrategy.CASCADE_ENSEMBLE,
        
        # Very high threshold - almost all boundaries get LLM verification
        heuristic_confidence_threshold=0.95,
        require_llm_verification_below=0.9,
        
        # Narrow visual range - mostly rely on LLM
        visual_verification_range=(0.9, 0.95),
        
        # Heavy LLM weighting
        detector_weights={
            DetectorType.HEURISTIC: 0.1,
            DetectorType.VISUAL: 0.1,
            DetectorType.LLM: 0.8,
        },
        
        min_agreement_threshold=0.9,
        enable_parallel_processing=True,
        max_workers=4,
        min_document_pages=1,
        merge_adjacent_threshold=1,
        confidence_boost_on_agreement=0.2,
    )


def get_balanced_config() -> SignalCombinerConfig:
    """
    Get balanced configuration for general use.
    
    This configuration:
    - Balances accuracy and speed
    - Uses LLM for uncertain cases only
    - Targets 5-10 seconds per page
    
    Returns:
        Balanced SignalCombinerConfig
    """
    return SignalCombinerConfig(
        combination_strategy=CombinationStrategy.CASCADE_ENSEMBLE,
        
        # Moderate thresholds
        heuristic_confidence_threshold=0.75,
        require_llm_verification_below=0.5,
        
        # Wider visual range for more visual verification
        visual_verification_range=(0.5, 0.75),
        
        # Balanced weights
        detector_weights={
            DetectorType.HEURISTIC: 0.3,
            DetectorType.VISUAL: 0.2,
            DetectorType.LLM: 0.5,
        },
        
        min_agreement_threshold=0.7,
        enable_parallel_processing=True,
        max_workers=3,
        min_document_pages=1,
        merge_adjacent_threshold=2,
        confidence_boost_on_agreement=0.1,
    )


def get_fast_screening_config() -> SignalCombinerConfig:
    """
    Get configuration optimized for fast screening.
    
    This configuration:
    - Minimizes LLM calls
    - Accepts lower accuracy for speed
    - Targets <5 seconds per page
    - Suitable for initial screening or similar document batches
    
    Returns:
        Fast screening SignalCombinerConfig
    """
    return SignalCombinerConfig(
        combination_strategy=CombinationStrategy.CASCADE_ENSEMBLE,
        
        # Low thresholds - trust heuristics more
        heuristic_confidence_threshold=0.6,
        require_llm_verification_below=0.3,
        
        # Wide visual range
        visual_verification_range=(0.3, 0.6),
        
        # Higher heuristic weight
        detector_weights={
            DetectorType.HEURISTIC: 0.4,
            DetectorType.VISUAL: 0.3,
            DetectorType.LLM: 0.3,
        },
        
        min_agreement_threshold=0.6,
        enable_parallel_processing=True,
        max_workers=4,
        min_document_pages=1,
        merge_adjacent_threshold=3,
        confidence_boost_on_agreement=0.05,
    )