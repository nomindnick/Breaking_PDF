#!/usr/bin/env python3
"""
Quick baseline test with increased timeout for LLM experiments.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.run_experiments import (
    ExperimentConfig,
    LLMExperimentRunner,
    load_ground_truth,
    process_pdf_to_pages,
)


def main():
    """Run baseline experiment with one model."""
    # File paths
    pdf_path = Path("test_files/Test_PDF_Set_1.pdf")
    ground_truth_path = Path("test_files/Test_PDF_Set_Ground_Truth.json")

    # Create experiment config with longer timeout
    config = ExperimentConfig(
        name="llama3_baseline_test",
        model="llama3:8b-instruct-q5_K_M",
        strategy="context_overlap",
        context_overlap_percent=0.3,
        window_size=3,
        temperature=0.1,
        max_tokens=500,
        timeout=120,  # Increased from 30 to 120 seconds
    )

    print(f"Starting baseline test at {datetime.now()}")
    print(f"Model: {config.model}")
    print(f"Strategy: {config.strategy}")
    print(f"Timeout: {config.timeout}s")
    print("-" * 60)

    # Load test data
    print("Loading PDF...")
    pages = process_pdf_to_pages(pdf_path)
    print(f"Loaded {len(pages)} pages")

    print("\nLoading ground truth...")
    ground_truth = load_ground_truth(ground_truth_path)
    print(f"Ground truth boundaries: {ground_truth}")

    # Initialize runner
    runner = LLMExperimentRunner()

    # Run experiment
    print(f"\nRunning experiment...")
    try:
        result = runner.run_experiment(config, pages, ground_truth)

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Precision: {result.precision:.3f}")
        print(f"Recall: {result.recall:.3f}")
        print(f"F1 Score: {result.f1_score:.3f}")
        print(f"Total Time: {result.total_time:.1f}s")
        print(f"Time per Page: {result.avg_time_per_page:.3f}s")
        print(f"Time per Boundary: {result.avg_time_per_boundary:.3f}s")
        print(f"\nPredicted boundaries: {result.predicted_boundaries}")
        print(f"True boundaries: {result.true_boundaries}")

        # Save detailed results
        results_file = Path(
            f"baseline_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(
                {
                    "config": config.__dict__,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1_score": result.f1_score,
                    "total_time": result.total_time,
                    "predicted_boundaries": result.predicted_boundaries,
                    "true_boundaries": result.true_boundaries,
                    "errors": result.errors,
                },
                f,
                indent=2,
                default=str,
            )

        print(f"\nDetailed results saved to: {results_file}")

    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
