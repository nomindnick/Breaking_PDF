#!/bin/bash
# Cleanup script for detection module
# Run with: bash cleanup_detection_module.sh

echo "Detection Module Cleanup Script"
echo "==============================="
echo "This will remove experimental code and consolidate the detection module."
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Phase 1: Remove experimental detector implementations
echo "Removing experimental detectors..."
rm -f pdf_splitter/detection/optimized_embeddings_detector.py
rm -f pdf_splitter/detection/balanced_embeddings_detector.py
rm -f pdf_splitter/detection/calibrated_heuristic_detector.py
rm -rf pdf_splitter/detection/embeddings_detector/fixed_embeddings_detector.py
rm -rf pdf_splitter/detection/context_aware_detector.py
rm -rf pdf_splitter/detection/signal_combiner/
rm -rf pdf_splitter/detection/heuristic_detector/enhanced_heuristic_detector.py
rm -rf pdf_splitter/detection/heuristic_detector/general_purpose_config.py
rm -rf pdf_splitter/detection/heuristic_detector/improved_config.py

# Phase 2: Remove test scripts (keep only essential ones)
echo "Removing experimental test scripts..."
rm -f scripts/test_*.py
rm -f scripts/debug_*.py
rm -f scripts/analyze_*.py
rm -f scripts/create_*.py
rm -f scripts/heuristic_analysis.py
rm -f scripts/tune_ensemble_weights.py
rm -f scripts/production_*.py
rm -f scripts/*.json

# Keep only essential scripts
git checkout scripts/check_ollama_setup.py 2>/dev/null || true

# Phase 3: Remove analysis documentation
echo "Removing analysis documentation..."
rm -f boundary_detection_progress_report.md
rm -f cascade_workaround.md
rm -f detection_accuracy_analysis.md
rm -f detection_fixes_summary.md
rm -f detection_issues_analysis.md
rm -f detection_module_fixes_complete.md
rm -f ensemble_voting_summary.md
rm -f integration_test_analysis.md
rm -f production_ready_detection.md
rm -f boundary_detection_improvements_documentation.md
rm -f boundary_detection_explained.md
rm -f boundary_detection_overfitting_analysis.md
rm -f boundary_detection_solution_achieved.md
rm -f boundary_detection_final_results.md
rm -f next_steps_boundary_detection.md
rm -f practical_visual_llm_integration.md

# Phase 4: Remove test data
echo "Removing test-specific data..."
rm -rf test_files/validation_set/
rm -rf test_files/diverse_tests/

# Phase 5: Clean up detection module structure
echo "Cleaning up detection module structure..."

# Create simplified production_detector.py
cat > pdf_splitter/detection/production_detector.py << 'EOF'
"""Production boundary detector factory."""

from pdf_splitter.detection import EmbeddingsDetector


def create_production_detector():
    """
    Create the production boundary detector.
    
    Returns an EmbeddingsDetector with optimal settings:
    - Model: all-MiniLM-L6-v2
    - Threshold: 0.5
    - Expected F1: ~0.65-0.70
    """
    return EmbeddingsDetector(
        model_name='all-MiniLM-L6-v2',
        similarity_threshold=0.5
    )
EOF

# Update detection __init__.py
cat > pdf_splitter/detection/__init__.py << 'EOF'
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
from pdf_splitter.detection.visual_detector import VisualDetector
from pdf_splitter.detection.llm_detector import LLMDetector
from pdf_splitter.detection.heuristic_detector import HeuristicDetector
from pdf_splitter.detection.production_detector import create_production_detector

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
EOF

echo "Cleanup complete!"
echo ""
echo "Summary of changes:"
echo "- Removed experimental detectors and complex ensemble code"
echo "- Removed analysis scripts and documentation"
echo "- Simplified detection module to core components"
echo "- Updated production_detector.py for simple usage"
echo ""
echo "Next steps:"
echo "1. Review and commit these changes"
echo "2. Update CLAUDE.md with final detection approach"
echo "3. Update development_progress.md with consolidated findings"
echo "4. Run tests to ensure core functionality still works"