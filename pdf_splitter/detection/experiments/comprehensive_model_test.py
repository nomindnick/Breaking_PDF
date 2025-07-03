#!/usr/bin/env python3
"""
Comprehensive model testing for boundary detection.

Tests all available models with the best performing prompt.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path before imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.experiment_runner import (  # noqa: E402
    OllamaClient,
)
from pdf_splitter.detection.experiments.phi4_test_with_real_text import (  # noqa: E402
    process_pdf_correctly,
)


class ComprehensiveModelTester:
    """Test all available models for boundary detection."""

    def __init__(self):
        """Initialize the tester."""
        self.client = OllamaClient()
        self.results = []

        # All models to test (ordered by size)
        self.models_to_test = [
            # Sub-1B models
            "qwen3:0.6b",  # 522 MB - smallest
            # 1-2B models
            "deepseek-r1:1.5b",  # 1.1 GB
            "qwen3:1.7b",  # 1.4 GB
            "granite3.3:2b",  # 1.5 GB
            # 2-4B models
            "phi3:mini",  # 2.2 GB
            "phi4-mini:3.8b",  # 2.5 GB
            "phi4-mini-reasoning:3.8b",  # 3.2 GB - reasoning variant
            "gemma3:latest",  # 3.3 GB
            # 4-8B models
            "granite3.3:8b",  # 4.9 GB
            "qwen3:8b",  # 5.2 GB
            "deepseek-r1:8b",  # 5.2 GB
            "gemma3n:e2b",  # 5.6 GB - experimental
            "llama3:8b-instruct-q5_K_M",  # 5.7 GB
        ]

        # Best performing prompt from experiments
        self.prompt_template = (
            "Your task is to determine if two document snippets are part of a "
            "single document or are different documents.\n\n"
            "You will be given the bottom part of Page 1 and the top portion of "
            "Page 2. Your task is to determine whether Page 1 and Page 2 are part "
            "of a single document or if Page 1 is the end of one document and "
            "Page 2 is the start of a new document.\n\n"
            "IMPORTANT: A simple page break within a document is NOT a document "
            "boundary. Look for signs of a completely new document starting, such as:\n"
            "- New letterhead or header\n"
            "- Complete change in formatting or topic\n"
            "- New date that suggests a different document\n"
            "- Signature at the bottom of Page 1 followed by a new document "
            "header on Page 2\n\n"
            'Please only respond with "Same Document" or "Different Documents"\n\n'
            "Bottom of Page 1:\n"
            "{page1_bottom}\n\n"
            "Top of Page 2:\n"
            "{page2_top}"
        )

        self.chars_per_part = 200  # Optimal from experiments

    def extract_page_parts(self, text: str) -> tuple:
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

    def test_model(
        self, model: str, pages: List, expected_boundaries: List[int]
    ) -> Optional[Dict]:
        """Test a single model and return results."""
        print(f"\nTesting model: {model}")
        print("-" * 60)

        # Check if model is available
        try:
            # Quick availability test
            test_response = self.client.generate(
                model=model,
                prompt="Say 'yes'",
                temperature=0.0,
                max_tokens=10,
                timeout=10,
            )
            if not test_response.get("response"):
                print(f"  ⚠️  Model {model} not responding properly, skipping...")
                return None
        except Exception as e:
            print(f"  ⚠️  Model {model} error: {e}")
            return None

        # Run detection
        detected_boundaries = []
        response_times = []
        errors = []

        start_time = time.time()

        for i in range(len(pages) - 1):
            try:
                # Extract text parts
                _, page1_bottom = self.extract_page_parts(pages[i].text)
                page2_top, _ = self.extract_page_parts(pages[i + 1].text)

                # Create prompt
                prompt = self.prompt_template.format(
                    page1_bottom=page1_bottom, page2_top=page2_top
                )

                # Get model response
                boundary_start = time.time()
                response = self.client.generate(
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=50,
                    timeout=30,
                )
                boundary_time = time.time() - boundary_start
                response_times.append(boundary_time)

                response_text = response.get("response", "").strip()

                # Check for boundary (handle various response formats)
                positive_indicators = [
                    "Different Documents",
                    "different documents",
                    "DIFFERENT DOCUMENTS",
                    "Different documents",
                    "different Documents",
                ]

                if any(indicator in response_text for indicator in positive_indicators):
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

        # Calculate timing statistics
        valid_times = [t for t in response_times if t is not None]
        avg_time = sum(valid_times) / len(valid_times) if valid_times else 0
        min_time = min(valid_times) if valid_times else 0
        max_time = max(valid_times) if valid_times else 0

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
                "avg_time_per_boundary": round(avg_time, 2),
                "min_time": round(min_time, 2),
                "max_time": round(max_time, 2),
                "boundaries_checked": len(pages) - 1,
            },
            "boundaries": {
                "expected": expected_boundaries,
                "detected": detected_boundaries,
            },
            "errors": errors,
        }

        # Print summary
        print(f"  ✓ F1 Score: {f1:.3f} (P: {precision:.3f}, R: {recall:.3f})")
        print(
            f"  ✓ Avg time: {avg_time:.2f}s "
            f"(min: {min_time:.2f}s, max: {max_time:.2f}s)"
        )
        print(f"  ✓ Detected: {detected_boundaries}")

        if errors:
            print(f"  ⚠️  Errors: {len(errors)}")

        return result

    def run_comprehensive_test(self, pdf_path: Path, num_pages: int = 15):
        """Run comprehensive model testing."""
        print("Comprehensive Model Performance Test")
        print("=" * 80)
        print(f"PDF: {pdf_path.name}")
        print(f"Testing first {num_pages} pages")
        print(f"Models to test: {len(self.models_to_test)}")

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

        # Print comprehensive summary
        self.print_comprehensive_summary()

    def save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(
            f"pdf_splitter/detection/experiments/results/"
            f"comprehensive_model_test_{timestamp}.json"
        )
        results_file.parent.mkdir(exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(
                {
                    "experiment": "comprehensive_model_test",
                    "timestamp": datetime.now().isoformat(),
                    "prompt": "page_break_guidance_200chars",
                    "results": self.results,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {results_file}")

    def print_comprehensive_summary(self):
        """Print comprehensive comparison summary."""
        print("\n" + "=" * 100)
        print("COMPREHENSIVE SUMMARY")
        print("=" * 100)

        # Sort by F1 score
        sorted_by_f1 = sorted(self.results, key=lambda x: -x["metrics"]["f1_score"])

        print(
            f"\n{'Model':<30} {'F1':<8} {'Recall':<8} {'Prec':<8} "
            f"{'FP':<5} {'Avg(s)':<8} {'Size':<8}"
        )
        print("-" * 100)

        for result in sorted_by_f1:
            # Get model size from our list
            model_size = "?"
            if "0.6b" in result["model"]:
                model_size = "0.5GB"
            elif "1.5b" in result["model"]:
                model_size = "1.1GB"
            elif "1.7b" in result["model"]:
                model_size = "1.4GB"
            elif "2b" in result["model"]:
                model_size = "1.5GB"
            elif "3.8b" in result["model"] and "reasoning" in result["model"]:
                model_size = "3.2GB"
            elif "3.8b" in result["model"]:
                model_size = "2.5GB"
            elif "mini" in result["model"]:
                model_size = "2.2GB"
            elif "gemma3:latest" in result["model"]:
                model_size = "3.3GB"
            elif "8b" in result["model"] and "granite" in result["model"]:
                model_size = "4.9GB"
            elif "8b" in result["model"] and "qwen" in result["model"]:
                model_size = "5.2GB"
            elif "8b" in result["model"] and "deepseek" in result["model"]:
                model_size = "5.2GB"
            elif "e2b" in result["model"]:
                model_size = "5.6GB"
            elif "llama3" in result["model"]:
                model_size = "5.7GB"

            print(
                f"{result['model']:<30} "
                f"{result['metrics']['f1_score']:<8.3f} "
                f"{result['metrics']['recall']:<8.3f} "
                f"{result['metrics']['precision']:<8.3f} "
                f"{result['metrics']['false_positives']:<5} "
                f"{result['performance']['avg_time_per_boundary']:<8.2f} "
                f"{model_size:<8}"
            )

        # Category analysis
        print("\n" + "=" * 100)
        print("ANALYSIS BY CATEGORY")
        print("=" * 100)

        # Best overall
        if sorted_by_f1:
            best_overall = sorted_by_f1[0]
            print("\n🏆 Best Overall (F1 Score):")
            print(
                f"   {best_overall['model']} - "
                f"F1: {best_overall['metrics']['f1_score']:.3f}"
            )

        # Best speed under 2s with good recall
        fast_good = [
            r
            for r in self.results
            if r["performance"]["avg_time_per_boundary"] < 2.0
            and r["metrics"]["recall"] >= 0.8
        ]
        if fast_good:
            best_fast = min(
                fast_good, key=lambda x: x["performance"]["avg_time_per_boundary"]
            )
            print("\n⚡ Fastest with ≥80% Recall (<2s):")
            print(
                f"   {best_fast['model']} - "
                f"{best_fast['performance']['avg_time_per_boundary']:.2f}s, "
                f"Recall: {best_fast['metrics']['recall']:.3f}"
            )

        # Best small model (under 2GB)
        small_models = [
            r
            for r in self.results
            if any(size in r["model"] for size in ["0.6b", "1.5b", "1.7b"])
        ]
        if small_models:
            best_small = max(small_models, key=lambda x: x["metrics"]["f1_score"])
            print("\n📦 Best Small Model (<2GB):")
            print(
                f"   {best_small['model']} - "
                f"F1: {best_small['metrics']['f1_score']:.3f}, "
                f"Speed: {best_small['performance']['avg_time_per_boundary']:.2f}s"
            )

        # Models with 100% recall
        perfect_recall = [r for r in self.results if r["metrics"]["recall"] == 1.0]
        if perfect_recall:
            print("\n💯 Models with 100% Recall:")
            for model in sorted(
                perfect_recall, key=lambda x: x["metrics"]["precision"], reverse=True
            )[:3]:
                print(
                    f"   {model['model']} - "
                    f"Precision: {model['metrics']['precision']:.3f}, "
                    f"FP: {model['metrics']['false_positives']}"
                )


def main():
    """Run comprehensive model testing."""
    pdf_path = Path("test_files/Test_PDF_Set_2_ocr.pdf")

    if not pdf_path.exists():
        print(f"Error: Test PDF not found at {pdf_path}")
        return

    # Run comprehensive test
    tester = ComprehensiveModelTester()
    tester.run_comprehensive_test(pdf_path, num_pages=15)

    print("\nTesting complete!")
    print("\nNext steps:")
    print("1. Select best model based on your priorities (accuracy vs speed)")
    print("2. Consider model size for deployment constraints")
    print("3. Test two-pass verification with top performers")


if __name__ == "__main__":
    main()
