#!/usr/bin/env python3
"""
Two-pass verification system for boundary detection.

Tests combinations of our finalist models to optimize for both
recall and precision.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, cast

# Add parent directory to path before imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.experiment_runner import (  # noqa: E402
    OllamaClient,
)
from pdf_splitter.detection.experiments.phi4_test_with_real_text import (  # noqa: E402
    process_pdf_correctly,
)


class TwoPassVerificationSystem:
    """Implements two-pass boundary detection with different model combinations."""

    def __init__(self):
        """Initialize the two-pass system."""
        self.client = OllamaClient()
        self.results = []

        # Finalist models
        self.finalist_models = [
            "qwen3:8b",
            "phi4-mini:3.8b",
            "gemma3:latest",
            "qwen3:1.7b",
        ]

        # Define test combinations
        self.test_combinations = [
            # Fast first pass → Accurate second pass
            ("phi4-mini:3.8b", "qwen3:8b", "Speed + Accuracy"),
            ("qwen3:1.7b", "qwen3:8b", "Small + Accurate"),
            ("gemma3:latest", "qwen3:8b", "Consistent + Accurate"),
            # Same model with different strategies
            ("qwen3:8b", "qwen3:8b", "Refined Two-Stage"),
            # Reverse combinations (accurate first)
            ("qwen3:8b", "phi4-mini:3.8b", "Accurate + Fast Verify"),
            # Small model combinations
            ("qwen3:1.7b", "phi4-mini:3.8b", "Small + Fast"),
            ("phi4-mini:3.8b", "qwen3:1.7b", "Fast + Small"),
        ]

        # Prompts for different passes
        self.first_pass_prompt = (
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
            'Please only respond with "Same Document" or '
            '"Different Documents"\n\n'
            "Bottom of Page 1:\n"
            "{page1_bottom}\n\n"
            "Top of Page 2:\n"
            "{page2_top}"
        )

        self.second_pass_prompt = (
            "You are reviewing a potential document boundary. Look VERY CAREFULLY "
            "for these specific signs:\n\n"
            "STRONG indicators of a document boundary:\n"
            '✓ Page 1 ends with a signature block or "Sincerely,"\n'
            '✓ Page 2 starts with a new letter header or "Dear..."\n'
            "✓ Page 2 has a completely new date at the top\n"
            "✓ Clear format change (e.g., letter ends, memo begins)\n\n"
            "WEAK indicators (probably NOT a boundary):\n"
            "✗ Just a page number change\n"
            "✗ Continuing paragraph or sentence\n"
            "✗ Same formatting and style\n"
            "✗ Related content topics\n\n"
            "Based on careful analysis, is this a TRUE document boundary?\n\n"
            "Bottom of Page 1:\n"
            "{page1_bottom}\n\n"
            "Top of Page 2:\n"
            "{page2_top}\n\n"
            'Answer with ONLY "Different Documents" if you see STRONG indicators, '
            'otherwise "Same Document".'
        )

        self.chars_per_part = 200  # Optimal from testing
        self.extended_chars = 400  # For second pass

    def extract_page_parts(self, text: str, chars: int = 200) -> Tuple[str, str]:
        """Extract top and bottom portions of page text."""
        lines = text.strip().split("\n")

        # Get approximately chars from top
        top_text = ""
        for line in lines:
            if len(top_text) + len(line) <= chars:
                top_text += line + "\n"
            else:
                break

        # Get approximately chars from bottom
        bottom_text = ""
        for line in reversed(lines):
            if len(bottom_text) + len(line) <= chars:
                bottom_text = line + "\n" + bottom_text
            else:
                break

        return top_text.strip(), bottom_text.strip()

    def run_detection_pass(
        self, model: str, pages: List, prompt_template: str, chars_per_part: int = 200
    ) -> Tuple[List[int], List[float], List[str]]:
        """Run a single detection pass with specified model and prompt."""
        detected_boundaries = []
        response_times = []
        responses = []

        for i in range(len(pages) - 1):
            try:
                # Extract text parts
                _, page1_bottom = self.extract_page_parts(pages[i].text, chars_per_part)
                page2_top, _ = self.extract_page_parts(
                    pages[i + 1].text, chars_per_part
                )

                # Create prompt
                prompt = prompt_template.format(
                    page1_bottom=page1_bottom, page2_top=page2_top
                )

                # Get model response
                start_time = time.time()
                response = self.client.generate(
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=100,
                    timeout=30,
                )
                response_time = time.time() - start_time
                response_times.append(response_time)

                response_text = response.get("response", "").strip()
                responses.append(response_text)

                # Check for boundary
                if (
                    "Different Documents" in response_text
                    or "different documents" in response_text
                ):
                    detected_boundaries.append(pages[i + 1].page_number)

            except Exception as e:
                print(f"    Error at page {i+1}: {str(e)}")
                responses.append(f"ERROR: {str(e)}")

        return detected_boundaries, response_times, responses

    def run_two_pass_detection(
        self, model1: str, model2: str, pages: List, strategy: str = "standard"
    ) -> Dict:
        """Run two-pass detection with specified models."""
        print(f"\n  Strategy: {strategy}")
        start_time = time.time()

        # First pass - high recall
        print(f"  Pass 1 ({model1})...", end="", flush=True)
        pass1_boundaries, pass1_times, pass1_responses = self.run_detection_pass(
            model1, pages, self.first_pass_prompt, self.chars_per_part
        )
        pass1_time = sum(t for t in pass1_times if t is not None)
        print(f" {len(pass1_boundaries)} boundaries in {pass1_time:.1f}s")

        # Second pass - verify boundaries
        if pass1_boundaries and model2:
            print(f"  Pass 2 ({model2})...", end="", flush=True)

            # Only verify the boundaries detected in pass 1
            verified_boundaries = []
            pass2_times = []

            for boundary_page in pass1_boundaries:
                # Find the page index
                page_idx = boundary_page - 1  # Convert to 0-based index
                if 0 < page_idx < len(pages):
                    # Create a mini-list for verification
                    verify_pages = [pages[page_idx - 1], pages[page_idx]]

                    # Use extended context for verification
                    verified, times, _ = self.run_detection_pass(
                        model2,
                        verify_pages,
                        self.second_pass_prompt,
                        self.extended_chars,
                    )

                    if verified:  # If still detected as boundary
                        verified_boundaries.append(boundary_page)
                    pass2_times.extend(times)

            pass2_time = sum(t for t in pass2_times if t is not None)
            print(f" {len(verified_boundaries)} verified in {pass2_time:.1f}s")

            final_boundaries = verified_boundaries
            total_time = time.time() - start_time
        else:
            # Single pass only
            final_boundaries = pass1_boundaries
            total_time = time.time() - start_time
            pass2_time = 0

        return {
            "boundaries": final_boundaries,
            "pass1_boundaries": pass1_boundaries,
            "pass1_time": pass1_time,
            "pass2_time": pass2_time,
            "total_time": total_time,
            "pass1_count": len(pass1_boundaries),
            "final_count": len(final_boundaries),
        }

    def test_combination(
        self,
        model1: str,
        model2: str,
        description: str,
        pages: List,
        expected_boundaries: List[int],
    ) -> Dict:
        """Test a specific model combination."""
        print(f"\nTesting: {model1} → {model2}")
        print(f"Description: {description}")
        print("-" * 60)

        # Run two-pass detection
        detection_result: Dict = self.run_two_pass_detection(model1, model2, pages)

        detected_boundaries = detection_result["boundaries"]

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

        # Calculate reduction in false positives
        pass1_fp = detection_result["pass1_count"] - len(
            set(detection_result["pass1_boundaries"]) & set(expected_boundaries)
        )
        final_fp = false_positives
        fp_reduction = ((pass1_fp - final_fp) / pass1_fp * 100) if pass1_fp > 0 else 0

        result = {
            "combination": f"{model1} → {model2}",
            "description": description,
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
                "pass1_time": round(detection_result["pass1_time"], 2),
                "pass2_time": round(detection_result["pass2_time"], 2),
                "total_time": round(detection_result["total_time"], 2),
                "avg_time_per_boundary": round(
                    detection_result["total_time"] / (len(pages) - 1), 2
                ),
            },
            "boundaries": {
                "expected": expected_boundaries,
                "pass1_detected": detection_result["pass1_boundaries"],
                "final_detected": detected_boundaries,
            },
            "improvement": {
                "pass1_count": detection_result["pass1_count"],
                "final_count": detection_result["final_count"],
                "boundaries_filtered": detection_result["pass1_count"]
                - detection_result["final_count"],
                "fp_reduction_percent": round(fp_reduction, 1),
            },
        }

        # Print summary
        print("\n  Results:")
        print(f"    Pass 1: {detection_result['pass1_count']} boundaries detected")
        print(
            f"    Pass 2: {cast(Dict, detection_result)['final_count']} "
            f"boundaries verified"
        )
        print(
            f"    Filtered: {result['improvement']['boundaries_filtered']} "
            f"false positives removed"
        )
        print(f"    Final metrics: F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}")
        print(f"    Total time: {detection_result['total_time']:.2f}s")  # type: ignore[index]

        return result

    def run_all_tests(self, pdf_path: Path, num_pages: int = 15):
        """Run all two-pass combination tests."""
        print("Two-Pass Verification System Test")
        print("=" * 80)
        print(f"PDF: {pdf_path.name}")
        print(f"Testing first {num_pages} pages")

        # Load pages
        print("\nLoading PDF pages...")
        pages = process_pdf_correctly(pdf_path, num_pages=num_pages)
        expected_boundaries = [5, 7, 9, 13, 14]
        print(f"Expected boundaries: {expected_boundaries}")

        # Test each combination
        for model1, model2, description in self.test_combinations:
            result = self.test_combination(
                model1, model2, description, pages, expected_boundaries
            )
            self.results.append(result)

        # Save results
        self.save_results()

        # Print comprehensive summary
        self.print_summary()

    def save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(
            f"pdf_splitter/detection/experiments/results/"
            f"two_pass_verification_{timestamp}.json"
        )
        results_file.parent.mkdir(exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(
                {
                    "experiment": "two_pass_verification",
                    "timestamp": datetime.now().isoformat(),
                    "results": self.results,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {results_file}")

    def print_summary(self):
        """Print comprehensive summary of two-pass results."""
        print("\n" + "=" * 100)
        print("TWO-PASS VERIFICATION SUMMARY")
        print("=" * 100)

        # Sort by F1 score
        sorted_results = sorted(self.results, key=lambda x: -x["metrics"]["f1_score"])

        print(
            f"\n{'Combination':<35} {'F1':<8} {'Recall':<8} {'Prec':<8} "
            f"{'FP':<5} {'Time':<8} {'Filtered':<10}"
        )
        print("-" * 100)

        for result in sorted_results:
            print(
                f"{result['combination']:<35} "
                f"{result['metrics']['f1_score']:<8.3f} "
                f"{result['metrics']['recall']:<8.3f} "
                f"{result['metrics']['precision']:<8.3f} "
                f"{result['metrics']['false_positives']:<5} "
                f"{result['performance']['total_time']:<8.2f}s "
                f"{result['improvement']['boundaries_filtered']:<10}"
            )

        # Best combinations analysis
        print("\n" + "=" * 100)
        print("KEY FINDINGS")
        print("=" * 100)

        if sorted_results:
            # Best overall
            best = sorted_results[0]
            print("\n🏆 Best Overall:")
            print(f"   {best['combination']} - F1: {best['metrics']['f1_score']:.3f}")
            print(
                f"   Filtered {best['improvement']['boundaries_filtered']} "
                f"false positives "
                f"({best['improvement']['fp_reduction_percent']:.1f}% reduction)"
            )

            # Most improvement
            most_improved = max(
                self.results, key=lambda x: x["improvement"]["boundaries_filtered"]
            )
            if most_improved["improvement"]["boundaries_filtered"] > 0:
                print("\n📈 Most Improvement:")
                print(
                    f"   {most_improved['combination']} - "
                    f"Filtered {most_improved['improvement']['boundaries_filtered']} "
                    f"boundaries"
                )

            # Fastest with good accuracy
            fast_good = [r for r in self.results if r["metrics"]["f1_score"] >= 0.5]
            if fast_good:
                fastest = min(fast_good, key=lambda x: x["performance"]["total_time"])
                print("\n⚡ Fastest with F1≥0.5:")
                print(
                    f"   {fastest['combination']} - "
                    f"{fastest['performance']['total_time']:.2f}s"
                )


def main():
    """Run two-pass verification tests."""
    pdf_path = Path("test_files/Test_PDF_Set_2_ocr.pdf")

    if not pdf_path.exists():
        print(f"Error: Test PDF not found at {pdf_path}")
        return

    # Run tests
    system = TwoPassVerificationSystem()
    system.run_all_tests(pdf_path, num_pages=15)

    print("\n\nNext steps:")
    print("1. Test winning combinations on full 36-page document")
    print("2. Try ensemble voting with multiple models")
    print("3. Implement production LLMDetector with best approach")


if __name__ == "__main__":
    main()
