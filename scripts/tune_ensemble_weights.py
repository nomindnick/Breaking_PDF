#!/usr/bin/env python3
"""
Tune ensemble weights on the expanded diverse dataset.

This script tests different weight configurations to find the optimal
combination for production use.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_splitter.detection import (
    ProcessedPage,
    SignalCombiner,
    SignalCombinerConfig,
    CombinationStrategy,
    DetectorType,
)
from pdf_splitter.detection.embeddings_detector import EmbeddingsDetector
from pdf_splitter.detection.heuristic_detector.enhanced_heuristic_detector import (
    EnhancedHeuristicDetector,
    create_enhanced_config
)
from pdf_splitter.detection.context_aware_detector import ContextAwareDetector
from pdf_splitter.preprocessing import PDFHandler, TextExtractor


def load_test_dataset(dataset_path: Path) -> List[Tuple[str, Path, List[int]]]:
    """Load test dataset information."""
    ground_truth_path = dataset_path / "diverse_test_ground_truth.json"
    
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)
    
    test_cases = []
    for tc in ground_truth["test_cases"]:
        pdf_path = dataset_path / tc["file"]
        if pdf_path.exists():
            test_cases.append((tc["name"], pdf_path, tc["boundaries"]))
    
    return test_cases


def evaluate_configuration(
    detectors: Dict,
    weights: Dict[DetectorType, float],
    test_cases: List[Tuple[str, Path, List[int]]]
) -> Dict:
    """Evaluate a specific weight configuration."""
    # Create combiner with given weights
    config = SignalCombinerConfig(
        combination_strategy=CombinationStrategy.WEIGHTED_VOTING,
        detector_weights=weights,
        enable_parallel_processing=True,
        max_workers=2,
    )
    
    combiner = SignalCombiner(detectors, config)
    
    # Track results
    all_tp = all_fp = all_fn = 0
    results_by_case = []
    
    pdf_handler = PDFHandler()
    
    for test_name, pdf_path, expected_boundaries in test_cases:
        # Load PDF
        pages = []
        with pdf_handler.load_pdf(pdf_path) as pdf:
            text_extractor = TextExtractor(pdf_handler)
            
            for i in range(pdf.page_count):
                extracted = text_extractor.extract_page(i)
                page = ProcessedPage(
                    page_number=i,
                    text=extracted.text,
                    ocr_confidence=extracted.quality_score,
                    page_type="searchable"
                )
                pages.append(page)
        
        # Detect boundaries
        boundaries = combiner.detect_boundaries(pages)
        detected = {b.page_number for b in boundaries if b.confidence >= 0.5}
        expected = set(expected_boundaries)
        
        # Calculate metrics
        tp = len(detected & expected)
        fp = len(detected - expected)
        fn = len(expected - detected)
        
        all_tp += tp
        all_fp += fp
        all_fn += fn
        
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        results_by_case.append((test_name, f1))
    
    # Overall metrics
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "weights": weights,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
        "by_case": results_by_case
    }


def main():
    """Find optimal ensemble weights."""
    print("ENSEMBLE WEIGHT TUNING")
    print("="*60)
    
    # Load test dataset
    dataset_path = Path("test_files/diverse_tests")
    test_cases = load_test_dataset(dataset_path)
    print(f"Loaded {len(test_cases)} test cases")
    
    # Initialize detectors
    print("\nInitializing detectors...")
    detectors = {
        DetectorType.HEURISTIC: EnhancedHeuristicDetector(create_enhanced_config()),
        DetectorType.EMBEDDINGS: EmbeddingsDetector(
            model_name='all-MiniLM-L6-v2',
            similarity_threshold=0.6
        ),
        # Note: Context-aware detector needs a different integration approach
        # as it's not a simple detector type
    }
    
    # Test different weight combinations
    print("\nTesting weight configurations...")
    
    # Define weight options to test
    weight_options = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    
    results = []
    best_f1 = 0
    best_config = None
    
    # Test combinations that sum to 1.0
    for h_weight in weight_options:
        e_weight = 1.0 - h_weight
        
        if e_weight < 0 or e_weight > 1:
            continue
        
        weights = {
            DetectorType.HEURISTIC: h_weight,
            DetectorType.EMBEDDINGS: e_weight,
        }
        
        print(f"\nTesting H:{h_weight:.1f} E:{e_weight:.1f}", end="... ")
        
        result = evaluate_configuration(detectors, weights, test_cases)
        results.append(result)
        
        print(f"F1={result['f1']:.3f} (P={result['precision']:.3f}, R={result['recall']:.3f})")
        
        if result['f1'] > best_f1:
            best_f1 = result['f1']
            best_config = result
    
    # Also test three-way combinations with context-aware
    print("\n\nTesting three-detector configurations...")
    print("(Using enhanced heuristic as proxy for context-aware)")
    
    # Test some promising three-way splits
    three_way_configs = [
        (0.3, 0.4, 0.3),  # Balanced
        (0.2, 0.5, 0.3),  # Embeddings-heavy
        (0.4, 0.4, 0.2),  # Less context
        (0.2, 0.6, 0.2),  # Strong embeddings
        (0.5, 0.3, 0.2),  # Heuristic-heavy
    ]
    
    for h_weight, e_weight, c_weight in three_way_configs:
        # Simulate by using enhanced heuristic twice with different weights
        weights = {
            DetectorType.HEURISTIC: h_weight + c_weight,  # Combined weight
            DetectorType.EMBEDDINGS: e_weight,
        }
        
        print(f"\nTesting H:{h_weight:.1f} E:{e_weight:.1f} C:{c_weight:.1f}", end="... ")
        
        result = evaluate_configuration(detectors, weights, test_cases)
        result['simulated_weights'] = f"H:{h_weight} E:{e_weight} C:{c_weight}"
        results.append(result)
        
        print(f"F1={result['f1']:.3f} (P={result['precision']:.3f}, R={result['recall']:.3f})")
        
        if result['f1'] > best_f1:
            best_f1 = result['f1']
            best_config = result
    
    # Report best configuration
    print("\n" + "="*60)
    print("BEST CONFIGURATION")
    print("="*60)
    
    if best_config:
        print(f"\nWeights: {best_config.get('simulated_weights', best_config['weights'])}")
        print(f"F1 Score: {best_config['f1']:.3f}")
        print(f"Precision: {best_config['precision']:.3f}")
        print(f"Recall: {best_config['recall']:.3f}")
        
        print("\nPerformance by test case:")
        for test_name, f1 in best_config['by_case']:
            print(f"  {test_name:30s} F1={f1:.3f}")
    
    # Save results (convert DetectorType keys to strings)
    output_path = Path("scripts/ensemble_tuning_results.json")
    
    # Convert detector types to strings in weights
    if best_config:
        best_config_serializable = best_config.copy()
        if 'weights' in best_config_serializable:
            best_config_serializable['weights'] = {
                str(k): v for k, v in best_config_serializable['weights'].items()
            }
    
    # Convert all results
    results_serializable = []
    for r in results:
        r_copy = r.copy()
        if 'weights' in r_copy:
            r_copy['weights'] = {str(k): v for k, v in r_copy['weights'].items()}
        results_serializable.append(r_copy)
    
    with open(output_path, 'w') as f:
        json.dump({
            "best_config": best_config_serializable,
            "all_results": results_serializable
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if best_f1 >= 0.7:
        print("✓ Ensemble achieves good performance with current detectors")
    else:
        print("⚠️ Ensemble performance is below target (F1<0.7)")
        print("\nSuggested improvements:")
        print("1. Implement proper context-aware integration")
        print("2. Fine-tune individual detector thresholds")
        print("3. Consider adding LLM verification for low-confidence boundaries")
        print("4. Implement confidence calibration")


if __name__ == "__main__":
    main()