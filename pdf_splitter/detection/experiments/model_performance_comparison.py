#!/usr/bin/env python3
"""
Model performance comparison script.

Tests the working simple approach with different models to find the optimal
balance between accuracy and speed.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path before imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.experiment_runner import (  # noqa: E402
    OllamaClient,
)
from pdf_splitter.detection.experiments.phi4_test_with_real_text import (  # noqa: E402
    process_pdf_correctly,
)


class ModelComparisonRunner:
    """Runs the simple transition detection approach with multiple models."""

    def __init__(self):
        """Initialize the comparison runner."""
        self.client = OllamaClient()
        self.results = []

        # Models to test (ordered by expected speed)
        self.models_to_test = [
            "tinyllama:latest",  # 1.1B params - fastest
            "llama3.2:1b",  # 1B params
            "phi3:mini",  # 3.8B params
            "gemma2:2b",  # 2B params
            "phi4-mini:3.8b",  # 3.8B params - our baseline
            "gemma:2b",  # 2B params (older version)
            "llama3.2:3b",  # 3B params
            "mistral:7b-instruct-q4_0",  # 7B params quantized
        ]

        # Test parameters
        self.chars_per_part = 300
        self.temperature = 0.0
        self.max_tokens = 50

    def extract_page_parts(self, text: str) -> Tuple[str, str]:
        """Extract top and bottom portions of page text."""
        lines = text.strip().split("\n")

        # Get approximately chars_per_part from top
        top_text = ""
        for line in lines:
            if len(top_text) + len(line) <= self.chars_per_part:
                top_text += line + "\n"
            else:
                break

        # Get approximately chars_per_part from bottom
        bottom_text = ""
        for line in reversed(lines):
            if len(bottom_text) + len(line) <= self.chars_per_part:
                bottom_text = line + "\n" + bottom_text
            else:
                break

        return top_text.strip(), bottom_text.strip()

    def create_prompt(self, page1_bottom: str, page2_top: str) -> str:
        """Create the detection prompt."""
        return (
            "Your task is to determine if two document snippets are part of a "
            "single document or are different documents.\n\n"
            "You will be given the bottom part of Page 1 and the top portion of "
            "Page 2. Your task is to determine whether Page 1 and Page 2 are part "
            "of a single document or if Page 1 is the end of one document and "
            "Page 2 is the start of a new document.\n\n"
            'Please only respond with "Same Document" or "Different Documents"\n\n'
            "Bottom of Page 1:\n"
            f"{page1_bottom}\n\n"
            "Top of Page 2:\n"
            f"{page2_top}"
        )

    def test_model(
        self, model: str, pages: List, expected_boundaries: List[int]
    ) -> Optional[Dict]:
        """Test a single model and return results."""
        print(f"\nTesting model: {model}")
        print("-" * 40)

        # Check if model is available
        try:
            # Test with a simple prompt first
            test_response = self.client.generate(
                model=model,
                prompt="Say 'yes'",
                temperature=0.0,
                max_tokens=10,
                timeout=10,
            )
            if not test_response.get("response"):
                print(f"  ⚠️  Model {model} not available, skipping...")
                return None
        except Exception as e:
            print(f"  ⚠️  Model {model} not available: {e}")
            return None

        # Run detection
        detected_boundaries = []
        total_time = 0.0
        errors = []
        responses = []

        start_time = time.time()

        for i in range(len(pages) - 1):
            try:
                # Extract text parts
                _, page1_bottom = self.extract_page_parts(pages[i].text)
                page2_top, _ = self.extract_page_parts(pages[i + 1].text)

                # Create prompt
                prompt = self.create_prompt(page1_bottom, page2_top)

                # Get model response
                boundary_start = time.time()
                response = self.client.generate(
                    model=model,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=30,
                )
                boundary_time = time.time() - boundary_start

                response_text = response.get("response", "").strip()
                responses.append(
                    {"page": i + 1, "response": response_text, "time": boundary_time}
                )

                # Check if boundary detected
                if "Different Documents" in response_text:
                    detected_boundaries.append(pages[i + 1].page_number)

            except Exception as e:
                errors.append(f"Error at page {i+1}: {str(e)}")

        total_time = time.time() - start_time

        # Calculate metrics
        true_positives = len(set(detected_boundaries) & set(expected_boundaries))
        false_positives = len(set(detected_boundaries) - set(expected_boundaries))
        false_negatives = len(set(expected_boundaries) - set(detected_boundaries))

        precision = (
            true_positives / (true_positives + false_positives)
            if detected_boundaries
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if expected_boundaries
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Calculate timing metrics
        avg_time_per_boundary = total_time / (len(pages) - 1) if len(pages) > 1 else 0

        result = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(f1, 3),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            },
            "performance": {
                "total_time": round(total_time, 2),
                "avg_time_per_boundary": round(avg_time_per_boundary, 2),
                "boundaries_checked": len(pages) - 1,
            },
            "boundaries": {
                "expected": expected_boundaries,
                "detected": detected_boundaries,
            },
            "errors": errors,
            "config": {
                "chars_per_part": self.chars_per_part,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        }

        # Print summary
        print(f"  ✓ Precision: {precision:.3f}")
        print(f"  ✓ Recall: {recall:.3f}")
        print(f"  ✓ F1 Score: {f1:.3f}")
        print(f"  ✓ Avg time per boundary: {avg_time_per_boundary:.2f}s")
        print(f"  ✓ Total time: {total_time:.2f}s")

        if errors:
            print(f"  ⚠️  Errors: {len(errors)}")

        return result

    def run_comparison(self, pdf_path: Path, num_pages: int = 15):
        """Run comparison across all models."""
        print("Model Performance Comparison")
        print("=" * 60)
        print(f"PDF: {pdf_path.name}")
        print(f"Testing first {num_pages} pages")

        # Load pages
        print("\nLoading PDF pages...")
        pages = process_pdf_correctly(pdf_path, num_pages=num_pages)
        print(f"Loaded {len(pages)} pages")

        # Expected boundaries in first 15 pages
        expected_boundaries = [5, 7, 9, 13, 14]
        print(f"Expected boundaries: {expected_boundaries}")

        # Test each model
        for model in self.models_to_test:
            result = self.test_model(model, pages, expected_boundaries)
            if result:
                self.results.append(result)

        # Save results
        self.save_results()

        # Print comparison summary
        self.print_summary()

    def save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(
            f"pdf_splitter/detection/experiments/results/"
            f"model_comparison_{timestamp}.json"
        )
        results_file.parent.mkdir(exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(
                {
                    "experiment": "model_performance_comparison",
                    "timestamp": datetime.now().isoformat(),
                    "results": self.results,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {results_file}")

    def print_summary(self):
        """Print comparison summary table."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(
            f"{'Model':<25} {'F1':<8} {'Recall':<8} {'Precision':<10} {'Avg Time':<10}"
        )
        print("-" * 80)

        # Sort by F1 score, then by speed
        sorted_results = sorted(
            self.results,
            key=lambda x: (
                -x["metrics"]["f1_score"],
                x["performance"]["avg_time_per_boundary"],
            ),
        )

        for result in sorted_results:
            print(
                f"{result['model']:<25} "
                f"{result['metrics']['f1_score']:<8.3f} "
                f"{result['metrics']['recall']:<8.3f} "
                f"{result['metrics']['precision']:<10.3f} "
                f"{result['performance']['avg_time_per_boundary']:<10.2f}s"
            )

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        # Find best models
        if sorted_results:
            # Best accuracy
            best_f1 = max(sorted_results, key=lambda x: x["metrics"]["f1_score"])
            print(
                f"Best Accuracy: {best_f1['model']} "
                f"(F1: {best_f1['metrics']['f1_score']})"
            )

            # Fastest with 100% recall
            perfect_recall = [
                r for r in sorted_results if r["metrics"]["recall"] == 1.0
            ]
            if perfect_recall:
                fastest_perfect = min(
                    perfect_recall,
                    key=lambda x: x["performance"]["avg_time_per_boundary"],
                )
                print(
                    f"Fastest with 100% Recall: {fastest_perfect['model']} "
                    f"({fastest_perfect['performance']['avg_time_per_boundary']:.2f}s "
                    f"per boundary)"
                )

            # Models under 2s target
            fast_models = [
                r
                for r in sorted_results
                if r["performance"]["avg_time_per_boundary"] < 2.0
                and r["metrics"]["recall"] >= 0.8
            ]
            if fast_models:
                print("\nModels meeting <2s target with >80% recall:")
                for model in fast_models:
                    print(
                        f"  - {model['model']}: "
                        f"{model['performance']['avg_time_per_boundary']:.2f}s, "
                        f"Recall: {model['metrics']['recall']:.3f}"
                    )


def main():
    """Run the model comparison."""
    # Set up test file path
    pdf_path = Path("test_files/Test_PDF_Set_2_ocr.pdf")

    if not pdf_path.exists():
        print(f"Error: Test PDF not found at {pdf_path}")
        return

    # Run comparison
    runner = ModelComparisonRunner()
    runner.run_comparison(pdf_path, num_pages=15)

    print("\nNext steps:")
    print("1. Select the best model based on accuracy/speed tradeoff")
    print("2. Test prompt refinements with the selected model")
    print("3. Implement two-pass approach if needed")


if __name__ == "__main__":
    main()
