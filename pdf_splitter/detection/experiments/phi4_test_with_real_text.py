#!/usr/bin/env python3
"""
Phi4 test ensuring we get real text from the OCR'd PDF.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.base_detector import ProcessedPage
from pdf_splitter.detection.experiments.experiment_runner import (
    ExperimentConfig,
    LLMExperimentRunner,
)
from pdf_splitter.preprocessing.pdf_handler import PDFHandler
from pdf_splitter.preprocessing.text_extractor import TextExtractor


def process_pdf_correctly(pdf_path: Path, num_pages: int = 10):
    """Process PDF and ensure we get actual text."""
    config = PDFConfig()
    handler = PDFHandler(config)

    with handler.load_pdf(pdf_path) as loaded_handler:
        extractor = TextExtractor(loaded_handler)
        pages = []

        for page_num in range(min(num_pages, loaded_handler.page_count)):
            page_type = loaded_handler.get_page_type(page_num)
            page_type_str = page_type.value

            # Always try to extract text for searchable pages
            if page_type_str in ["searchable", "mixed"]:
                extracted = extractor.extract_page_text(page_num + 1)
                text = extracted.text
            else:
                text = f"[Page {page_num + 1} - Image-based content]"

            page = ProcessedPage(
                page_number=page_num + 1,
                text=text,
                page_type=page_type_str,
                metadata={"page_type": page_type_str},
            )
            pages.append(page)

    return pages


def main():
    """Run test with real text extraction."""
    pdf_path = Path("test_files/Test_PDF_Set_2_ocr.pdf")

    print("Phi4 Test with Real Text Extraction")
    print("=" * 60)

    # Process PDF with correct text extraction
    print("Processing PDF with text extraction...")
    pages = process_pdf_correctly(pdf_path, num_pages=10)

    # Show actual extracted text
    print("\nActual extracted text samples:")
    for i in range(min(3, len(pages))):
        page = pages[i]
        text_preview = page.text[:200].strip().replace("\n", " ")
        print(f"\nPage {page.page_number}:")
        print(f"  Type: {page.page_type}")
        print(f"  Text: {text_preview}...")

    # Ground truth for first 10 pages
    ground_truth = [5, 7, 9]
    print(f"\nExpected boundaries: {ground_truth}")

    # Create experiment config
    config = ExperimentConfig(
        name="phi4_real_text_test",
        model="phi4-mini:3.8b",
        strategy="context_overlap",
        context_overlap_percent=0.3,
        window_size=3,
        temperature=0.1,
        max_tokens=500,
        timeout=45,
    )

    # Run experiment
    runner = LLMExperimentRunner()
    print(f"\nRunning experiment with {config.model}...")

    try:
        result = runner.run_experiment(config, pages, ground_truth)

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Performance:")
        print(f"  Total Time: {result.total_time:.1f}s")
        print(f"  Time per Page: {result.avg_time_per_page:.3f}s")
        print(f"  Time per Boundary: {result.avg_time_per_boundary:.3f}s")

        print(f"\nAccuracy:")
        print(f"  Precision: {result.precision:.3f}")
        print(f"  Recall: {result.recall:.3f}")
        print(f"  F1 Score: {result.f1_score:.3f}")

        print(f"\nBoundaries:")
        print(f"  Predicted: {result.predicted_boundaries}")
        print(f"  Expected: {ground_truth}")

        # Show model reasoning
        if result.boundary_results:
            print(f"\nDetected boundaries with reasoning:")
            for boundary in result.boundary_results:
                print(f"  Page {boundary.page_number}: {boundary.reasoning[:100]}...")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"phi4_real_text_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "model": config.model,
                    "f1_score": result.f1_score,
                    "precision": result.precision,
                    "recall": result.recall,
                    "time_per_page": result.avg_time_per_page,
                    "predicted": result.predicted_boundaries,
                    "expected": ground_truth,
                    "sample_text": pages[0].text[:200] if pages else "No text",
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to: {results_file}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
