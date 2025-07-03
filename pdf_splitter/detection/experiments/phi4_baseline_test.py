#!/usr/bin/env python3
"""
Baseline test with Phi4 model using OCR'd PDF for better performance testing.
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
    """Run baseline experiment with Phi4 model."""
    # File paths - using OCR'd PDF
    pdf_path = Path("test_files/Test_PDF_Set_2_ocr.pdf")
    ground_truth_path = Path("test_files/Test_PDF_Set_Ground_Truth.json")

    # Create experiment config for Phi4
    config = ExperimentConfig(
        name="phi4_baseline_test",
        model="phi4-mini:3.8b",
        strategy="context_overlap",
        context_overlap_percent=0.3,
        window_size=3,
        temperature=0.1,
        max_tokens=500,
        timeout=60,  # Shorter timeout for smaller model
    )

    print(f"Starting Phi4 baseline test at {datetime.now()}")
    print(f"Model: {config.model}")
    print(f"PDF: {pdf_path.name} (OCR'd version)")
    print(f"Strategy: {config.strategy}")
    print(f"Timeout: {config.timeout}s")
    print("-" * 60)

    # Load test data
    print("Loading PDF...")
    pages = process_pdf_to_pages(pdf_path)
    print(f"Loaded {len(pages)} pages")

    # Show sample of actual text from first few pages
    print("\nSample text from first 3 pages:")
    for i in range(min(3, len(pages))):
        page = pages[i]
        text_preview = page.text[:200].replace("\n", " ")
        print(f"Page {page.page_number}: {text_preview}...")

    print("\nLoading ground truth...")
    ground_truth = load_ground_truth(ground_truth_path)
    print(f"Ground truth boundaries: {ground_truth}")

    # Initialize runner
    runner = LLMExperimentRunner()

    # Run experiment
    print(f"\nRunning experiment...")
    start_time = datetime.now()

    try:
        result = runner.run_experiment(config, pages, ground_truth)

        end_time = datetime.now()

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Model: {config.model}")
        print(f"Total Time: {result.total_time:.1f}s")
        print(f"Time per Page: {result.avg_time_per_page:.3f}s")
        print(f"Time per Boundary: {result.avg_time_per_boundary:.3f}s")
        print(f"\nAccuracy Metrics:")
        print(f"  Precision: {result.precision:.3f}")
        print(f"  Recall: {result.recall:.3f}")
        print(f"  F1 Score: {result.f1_score:.3f}")
        print(f"\nBoundary Detection:")
        print(f"  Predicted: {result.predicted_boundaries}")
        print(f"  Ground Truth: {result.true_boundaries}")
        print(f"  Total Errors: {len(result.errors)}")

        # Save detailed results
        results_file = Path(
            f"phi4_baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        results_data = {
            "config": {
                "model": config.model,
                "strategy": config.strategy,
                "timeout": config.timeout,
                "temperature": config.temperature,
            },
            "metrics": {
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "total_time": result.total_time,
                "avg_time_per_page": result.avg_time_per_page,
                "avg_time_per_boundary": result.avg_time_per_boundary,
            },
            "boundaries": {
                "predicted": result.predicted_boundaries,
                "ground_truth": result.true_boundaries,
            },
            "errors": result.errors,
            "timestamp": str(datetime.now()),
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"\nDetailed results saved to: {results_file}")

        # Update experiments log
        log_entry = f"""

### Experiment 2: Phi4 Baseline Test
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **Model**: phi4-mini:3.8b
- **PDF**: Test_PDF_Set_2_ocr.pdf (OCR'd version)
- **Strategy**: context_overlap (30% overlap, window_size=3)
- **Results**:
  - F1 Score: {result.f1_score:.3f}
  - Precision: {result.precision:.3f}
  - Recall: {result.recall:.3f}
  - Total Time: {result.total_time:.1f}s
  - Time per Boundary: {result.avg_time_per_boundary:.3f}s
  - Predicted Boundaries: {result.predicted_boundaries}
- **Observations**: [To be filled after analysis]
"""
        print(f"\nLog entry for experiments log:")
        print(log_entry)

    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
