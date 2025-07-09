#!/usr/bin/env python3
"""
CLI tool for running visual boundary detection experiments.

This script provides an easy interface for testing different visual
detection techniques on PDF files.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from pdf_splitter.detection.visual_detector.experiments.experiment_runner import (
    VisualExperimentRunner,
)
from pdf_splitter.detection.visual_detector.experiments.metrics import (
    calculate_threshold_sweep,
    generate_summary_report,
    plot_threshold_analysis,
    print_detailed_metrics,
)
from pdf_splitter.detection.visual_detector.experiments.visual_techniques import (
    create_technique,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run visual boundary detection experiments"
    )

    # Required arguments
    parser.add_argument("pdf_path", type=Path, help="Path to the test PDF file")
    parser.add_argument(
        "ground_truth", type=Path, help="Path to the ground truth JSON file"
    )

    # Technique selection
    parser.add_argument(
        "--technique",
        choices=["histogram", "ssim", "phash", "all"],
        default="all",
        help="Visual technique to test (default: all)",
    )

    # Technique parameters
    parser.add_argument(
        "--threshold", type=float, help="Similarity threshold for boundary detection"
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=8,
        help="Hash size for perceptual hashing (default: 8)",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=256,
        help="Number of bins for histogram comparison (default: 256)",
    )
    parser.add_argument(
        "--ssim-window", type=int, default=11, help="Window size for SSIM (default: 11)"
    )

    # Analysis options
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="Perform threshold sweep analysis",
    )
    parser.add_argument("--compare", action="store_true", help="Compare all techniques")
    parser.add_argument(
        "--analyze-failures",
        action="store_true",
        help="Analyze false positives and negatives",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Directory to save results (default: experiments/results)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save results to disk"
    )

    return parser.parse_args()


def get_technique_configs(args) -> List[Tuple[str, dict]]:
    """
    Get technique configurations based on arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        List of (technique_name, parameters) tuples
    """
    configs = []

    if args.technique == "all" or args.technique == "histogram":
        params = {"bins": args.histogram_bins}
        if args.threshold is not None:
            params["threshold"] = args.threshold
        else:
            params["threshold"] = 0.8  # Default for histogram
        configs.append(("histogram", params))

    if args.technique == "all" or args.technique == "ssim":
        params = {"window_size": args.ssim_window}
        if args.threshold is not None:
            params["threshold"] = args.threshold
        else:
            params["threshold"] = 0.7  # Default for SSIM
        configs.append(("ssim", params))

    if args.technique == "all" or args.technique == "phash":
        params = {"hash_size": args.hash_size}
        if args.threshold is not None:
            params["threshold"] = args.threshold
        else:
            params["threshold"] = 10  # Default for phash (Hamming distance)
        configs.append(("phash", params))

    return configs


def run_single_experiment(runner, technique_name, params, pdf_path, ground_truth):
    """Run a single experiment and return the result."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {technique_name} with parameters: {params}")
    logger.info(f"{'='*60}")

    # Create technique
    technique = create_technique(technique_name, **params)

    # Run experiment
    result = runner.run_experiment(technique, pdf_path, ground_truth, save_results=True)

    # Print detailed metrics
    print_detailed_metrics(result.to_dict())

    return result


def run_threshold_sweep(runner, technique_name, base_params, pdf_path, ground_truth):
    """Run threshold sweep analysis for a technique."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running threshold sweep for {technique_name}")
    logger.info(f"{'='*60}")

    # First run with default threshold to get similarities
    technique = create_technique(technique_name, **base_params)
    result = runner.run_experiment(
        technique, pdf_path, ground_truth, save_results=False
    )

    # Extract similarities and true boundaries
    similarities = [comp.similarity_score for comp in result.comparisons]
    true_boundaries = []

    # Load ground truth boundaries
    true_boundary_pages = runner.load_ground_truth(ground_truth)

    for comp in result.comparisons:
        true_boundaries.append(comp.page1_num in true_boundary_pages)

    # Calculate metrics across thresholds
    if technique_name == "phash":
        # For hashing, we use Hamming distance thresholds
        thresholds = list(range(0, 30, 2))
    else:
        # For similarity-based techniques
        thresholds = [i / 100 for i in range(30, 100, 5)]

    threshold_results = calculate_threshold_sweep(
        similarities, true_boundaries, thresholds
    )

    # Plot results
    plot_path = runner.results_dir / f"{technique_name}_threshold_analysis.png"
    plot_threshold_analysis(threshold_results, save_path=plot_path)

    # Find optimal threshold
    best_idx = threshold_results["f1_score"].index(max(threshold_results["f1_score"]))
    best_threshold = threshold_results["thresholds"][best_idx]
    best_f1 = threshold_results["f1_score"][best_idx]

    logger.info(
        f"Optimal threshold for {technique_name}: {best_threshold:.3f} (F1: {best_f1:.3f})"
    )

    return best_threshold


def main():
    """Run visual boundary detection experiments."""
    args = parse_arguments()

    # Validate inputs
    if not args.pdf_path.exists():
        logger.error(f"PDF file not found: {args.pdf_path}")
        sys.exit(1)

    if not args.ground_truth.exists():
        logger.error(f"Ground truth file not found: {args.ground_truth}")
        sys.exit(1)

    # Create experiment runner
    runner = VisualExperimentRunner(results_dir=args.output_dir)

    # Get technique configurations
    technique_configs = get_technique_configs(args)

    if args.threshold_sweep:
        # Run threshold sweep for each technique
        for tech_name, params in technique_configs:
            optimal_threshold = run_threshold_sweep(
                runner, tech_name, params, args.pdf_path, args.ground_truth
            )

            # Update params with optimal threshold
            params["threshold"] = optimal_threshold

    if args.compare or len(technique_configs) > 1:
        # Compare multiple techniques
        logger.info("\nComparing techniques...")
        results = runner.compare_techniques(
            technique_configs, args.pdf_path, args.ground_truth
        )

        # Generate summary report
        if not args.no_save:
            report_path = args.output_dir / "comparison_summary.json"
            result_dicts = [r.to_dict() for r in results.values()]
            generate_summary_report(result_dicts, report_path)

    else:
        # Run single experiment
        tech_name, params = technique_configs[0]
        result = run_single_experiment(
            runner, tech_name, params, args.pdf_path, args.ground_truth
        )

        if args.analyze_failures:
            # Analyze failures
            logger.info("\nAnalyzing failures...")
            analysis = runner.analyze_failures(result, args.ground_truth)

            print(f"\nFalse Positives: {analysis['fp_count']}")
            for fp in analysis["false_positives"]:
                print(f"  Page {fp['page']}: similarity={fp['similarity']:.3f}")

            print(f"\nFalse Negatives: {analysis['fn_count']}")
            for fn in analysis["false_negatives"]:
                print(f"  Page {fn['page']}: similarity={fn['similarity']:.3f}")

    logger.info("\nExperiment complete!")


if __name__ == "__main__":
    main()
