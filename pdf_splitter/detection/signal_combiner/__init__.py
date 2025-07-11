"""Signal combiner for intelligent document boundary detection.

This module provides a sophisticated signal combination system that integrates
multiple detection methods (heuristic, LLM, visual) to achieve high accuracy
and performance in document boundary detection.
"""

from .signal_combiner import SignalCombiner, SignalCombinerConfig, CombinationStrategy
from .production_config import (
    get_production_cascade_config,
    get_high_accuracy_config,
    get_balanced_config,
    get_fast_screening_config,
)

__all__ = [
    "SignalCombiner",
    "SignalCombinerConfig",
    "CombinationStrategy",
    "get_production_cascade_config",
    "get_high_accuracy_config",
    "get_balanced_config",
    "get_fast_screening_config",
]