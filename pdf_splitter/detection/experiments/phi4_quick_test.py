#!/usr/bin/env python3
"""
Quick test with Phi4 on first 10 pages for faster results.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.run_experiments import (
    ExperimentConfig,
    LLMExperimentRunner,
    process_pdf_to_pages,
)


def main():
    """Run quick test with first 10 pages."""
    # File paths
    pdf_path = Path("test_files/Test_PDF_Set_2_ocr.pdf")

    # Create experiment config
    config = ExperimentConfig(
        name="phi4_quick_test",
        model="phi4-mini:3.8b",
        strategy="context_overlap",
        context_overlap_percent=0.3,
        window_size=3,
        temperature=0.1,
        max_tokens=500,
        timeout=30,
    )

    print(f"Quick Phi4 test - First 10 pages")
    print(f"Model: {config.model}")
    print("-" * 60)

    # Load test data
    print("Loading PDF...")
    all_pages = process_pdf_to_pages(pdf_path)
    pages = all_pages[:10]  # Just first 10 pages
    print(f"Using first {len(pages)} pages")

    # Show actual text content
    print("\nActual text from pages:")
    for i, page in enumerate(pages[:3]):
        text_preview = page.text[:150].replace("\n", " ")
        if text_preview.strip():
            print(f"\nPage {page.page_number}: {text_preview}...")

    # Expected boundaries in first 10 pages (from ground truth)
    ground_truth = [5, 7, 9]  # Document boundaries in pages 1-10
    print(f"\nExpected boundaries in first 10 pages: {ground_truth}")

    # Initialize runner
    runner = LLMExperimentRunner()

    # Run experiment
    print(f"\nRunning quick test...")
    start_time = datetime.now()

    try:
        result = runner.run_experiment(config, pages, ground_truth)

        print(f"\n{'='*60}")
        print("QUICK TEST RESULTS")
        print(f"{'='*60}")
        print(f"Total Time: {result.total_time:.1f}s")
        print(f"Time per Page: {result.avg_time_per_page:.3f}s")
        print(f"\nAccuracy:")
        print(f"  Precision: {result.precision:.3f}")
        print(f"  Recall: {result.recall:.3f}")
        print(f"  F1 Score: {result.f1_score:.3f}")
        print(f"\nPredicted boundaries: {result.predicted_boundaries}")
        print(f"Expected boundaries: {ground_truth}")

        # Show some model responses
        if result.model_responses:
            print(f"\nSample model responses:")
            for resp in result.model_responses[:3]:
                print(f"  Page {resp['page']}: {resp['response'][:100]}...")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
