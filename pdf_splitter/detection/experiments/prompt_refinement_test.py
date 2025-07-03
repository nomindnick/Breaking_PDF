#!/usr/bin/env python3
"""
Prompt refinement experiments to reduce false positives.

Tests different prompt variations to improve precision while maintaining recall.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path before imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.experiment_runner import (  # noqa: E402
    OllamaClient,
)
from pdf_splitter.detection.experiments.phi4_test_with_real_text import (  # noqa: E402
    process_pdf_correctly,
)


class PromptRefinementTester:
    """Test different prompt variations to improve boundary detection."""

    def __init__(self, model="phi4-mini:3.8b"):
        """Initialize with chosen model."""
        self.model = model
        self.client = OllamaClient()
        self.results = []

    def extract_page_parts(self, text: str, chars_per_part: int = 300) -> tuple:
        """Extract top and bottom portions of page text."""
        lines = text.strip().split("\n")

        # Get approximately chars_per_part from top
        top_text = ""
        for line in lines:
            if len(top_text) + len(line) <= chars_per_part:
                top_text += line + "\n"
            else:
                break

        # Get approximately chars_per_part from bottom
        bottom_text = ""
        for line in reversed(lines):
            if len(bottom_text) + len(line) <= chars_per_part:
                bottom_text = line + "\n" + bottom_text
            else:
                break

        return top_text.strip(), bottom_text.strip()

    def test_prompt_variation(
        self,
        name: str,
        prompt_template: str,
        pages: List,
        expected_boundaries: List[int],
        context_size: int = 300,
    ) -> Dict:
        """Test a single prompt variation."""
        print(f"\nTesting prompt: {name}")
        print("-" * 60)

        detected_boundaries = []
        total_time = 0.0
        responses = []

        start_time = time.time()

        for i in range(len(pages) - 1):
            _, page1_bottom = self.extract_page_parts(pages[i].text, context_size)
            page2_top, _ = self.extract_page_parts(pages[i + 1].text, context_size)

            # Format prompt with the extracted text
            prompt = prompt_template.format(
                page1_bottom=page1_bottom, page2_top=page2_top
            )

            try:
                boundary_start = time.time()
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=100,
                    timeout=30,
                )
                boundary_time = time.time() - boundary_start

                response_text = response.get("response", "").strip()
                responses.append(
                    {
                        "pages": f"{i+1}-{i+2}",
                        "response": response_text,
                        "time": boundary_time,
                    }
                )

                # Check various positive indicators
                positive_indicators = [
                    "Different Documents",
                    "different documents",
                    "new document",
                    "separate document",
                    "boundary",
                    "YES",
                    "True",
                ]

                if any(indicator in response_text for indicator in positive_indicators):
                    detected_boundaries.append(pages[i + 1].page_number)

            except Exception as e:
                print(f"  Error at page {i+1}: {str(e)}")

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

        result = {
            "name": name,
            "prompt_template": prompt_template,
            "context_size": context_size,
            "metrics": {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(f1, 3),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            },
            "boundaries": {
                "expected": expected_boundaries,
                "detected": detected_boundaries,
            },
            "performance": {
                "total_time": round(total_time, 2),
                "avg_time_per_boundary": round(total_time / (len(pages) - 1), 2),
            },
        }

        # Print summary
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  Detected: {detected_boundaries}")
        print(f"  Expected: {expected_boundaries}")

        return result

    def run_all_tests(self, pdf_path: Path, num_pages: int = 15):
        """Run all prompt variations."""
        print(f"Prompt Refinement Tests - Model: {self.model}")
        print("=" * 80)

        # Load pages
        print("Loading PDF pages...")
        pages = process_pdf_correctly(pdf_path, num_pages=num_pages)
        expected_boundaries = [5, 7, 9, 13, 14]

        # Define prompt variations
        prompts: List[Dict[str, Any]] = [  # noqa: E501
            # Original working prompt
            {
                "name": "Original (baseline)",
                "template": """Your task is to determine if two document snippets are part of a single document or are different documents.

You will be given the bottom part of Page 1 and the top portion of Page 2. Your task is to determine whether Page 1 and Page 2 are part of a single document or if Page 1 is the end of one document and Page 2 is the start of a new document.

Please only respond with "Same Document" or "Different Documents"

Bottom of Page 1:
{page1_bottom}

Top of Page 2:
{page2_top}""",
                "context_size": 300,
            },
            # Add page break guidance
            {
                "name": "With page break guidance",
                "template": """Your task is to determine if two document snippets are part of a single document or are different documents.

You will be given the bottom part of Page 1 and the top portion of Page 2. Your task is to determine whether Page 1 and Page 2 are part of a single document or if Page 1 is the end of one document and Page 2 is the start of a new document.

IMPORTANT: A simple page break within a document is NOT a document boundary. Look for signs of a completely new document starting, such as:
- New letterhead or header
- Complete change in formatting or topic
- New date that suggests a different document
- Signature at the bottom of Page 1 followed by a new document header on Page 2

Please only respond with "Same Document" or "Different Documents"

Bottom of Page 1:
{page1_bottom}

Top of Page 2:
{page2_top}""",
                "context_size": 300,
            },
            # Focus on document endings
            {
                "name": "Focus on endings",
                "template": """Analyze if Page 1 ends a document and Page 2 starts a new one.

Signs that Page 1 ENDS a document:
- Signature line or signature block
- "Sincerely," or similar closing
- Final paragraph with conclusive language
- Document footer or end mark

Signs that Page 2 STARTS a new document:
- New letterhead or document header
- New date at the top
- "Dear" or similar greeting
- Document title or subject line

Bottom of Page 1:
{page1_bottom}

Top of Page 2:
{page2_top}

Based on these signs, are these pages from the Same Document or Different Documents?""",
                "context_size": 300,
            },
            # Stricter criteria
            {
                "name": "Stricter criteria",
                "template": """Determine if there is a document boundary between these pages.

You should ONLY say "Different Documents" if you see CLEAR evidence such as:
1. Page 1 ends with a signature AND Page 2 has a new header/date
2. Page 1 has a clear ending (like "Sincerely") AND Page 2 starts a new letter/memo
3. Obvious format change indicating separate documents

Otherwise, assume they are the Same Document.

Bottom of Page 1:
{page1_bottom}

Top of Page 2:
{page2_top}

Response (Same Document or Different Documents):""",
                "context_size": 300,
            },
            # Few-shot with examples
            {
                "name": "Few-shot examples",
                "template": """Determine if these pages are from the same document or different documents.

Example 1:
Page 1 bottom: "...will continue on the next page with additional details about..."
Page 2 top: "...the project timeline and resource allocation."
Answer: Same Document (continuous text)

Example 2:
Page 1 bottom: "Sincerely,\n\nJohn Smith\nProject Manager"
Page 2 top: "MEMORANDUM\n\nDate: March 15, 2023\nTo: All Staff"
Answer: Different Documents (letter ends, memo begins)

Now analyze:
Bottom of Page 1:
{page1_bottom}

Top of Page 2:
{page2_top}

Answer:""",
                "context_size": 300,
            },
            # Test different context sizes with original prompt
            {
                "name": "Original with 200 chars",
                "template": """Your task is to determine if two document snippets are part of a single document or are different documents.

You will be given the bottom part of Page 1 and the top portion of Page 2. Your task is to determine whether Page 1 and Page 2 are part of a single document or if Page 1 is the end of one document and Page 2 is the start of a new document.

Please only respond with "Same Document" or "Different Documents"

Bottom of Page 1:
{page1_bottom}

Top of Page 2:
{page2_top}""",
                "context_size": 200,
            },
            {
                "name": "Original with 400 chars",
                "template": """Your task is to determine if two document snippets are part of a single document or are different documents.

You will be given the bottom part of Page 1 and the top portion of Page 2. Your task is to determine whether Page 1 and Page 2 are part of a single document or if Page 1 is the end of one document and Page 2 is the start of a new document.

Please only respond with "Same Document" or "Different Documents"

Bottom of Page 1:
{page1_bottom}

Top of Page 2:
{page2_top}""",
                "context_size": 400,
            },
        ]

        # Test each prompt variation
        for prompt_config in prompts:
            result = self.test_prompt_variation(
                name=prompt_config["name"],
                prompt_template=prompt_config["template"],
                pages=pages,
                expected_boundaries=expected_boundaries,
                context_size=prompt_config["context_size"],
            )
            self.results.append(result)

        # Save results
        self.save_results()

        # Print summary
        self.print_summary()

    def save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(
            f"pdf_splitter/detection/experiments/results/"
            f"prompt_refinement_{timestamp}.json"
        )

        with open(results_file, "w") as f:
            json.dump(
                {
                    "experiment": "prompt_refinement",
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "results": self.results,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {results_file}")

    def print_summary(self):
        """Print comparison summary."""
        print("\n" + "=" * 80)
        print("PROMPT COMPARISON SUMMARY")
        print("=" * 80)
        print(
            f"{'Prompt Variation':<30} {'F1':<8} {'Recall':<8} "
            f"{'Precision':<10} {'FP':<5}"
        )
        print("-" * 80)

        # Sort by F1 score
        sorted_results = sorted(self.results, key=lambda x: -x["metrics"]["f1_score"])

        for result in sorted_results:
            print(
                f"{result['name']:<30} "
                f"{result['metrics']['f1_score']:<8.3f} "
                f"{result['metrics']['recall']:<8.3f} "
                f"{result['metrics']['precision']:<10.3f} "
                f"{result['metrics']['false_positives']:<5}"
            )

        # Find best variations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        # Best overall
        best_f1 = max(self.results, key=lambda x: x["metrics"]["f1_score"])
        print(
            f"Best F1 Score: {best_f1['name']} ({best_f1['metrics']['f1_score']:.3f})"
        )

        # Best precision with good recall
        high_precision = [r for r in self.results if r["metrics"]["recall"] >= 0.8]
        if high_precision:
            best_precision = max(
                high_precision, key=lambda x: x["metrics"]["precision"]
            )
            print(
                f"Best Precision (recall ≥ 0.8): {best_precision['name']} "
                f"(P: {best_precision['metrics']['precision']:.3f}, "
                f"R: {best_precision['metrics']['recall']:.3f})"
            )


def main():
    """Run prompt refinement tests."""
    pdf_path = Path("test_files/Test_PDF_Set_2_ocr.pdf")

    if not pdf_path.exists():
        print(f"Error: Test PDF not found at {pdf_path}")
        return

    # Run tests
    tester = PromptRefinementTester(model="phi4-mini:3.8b")
    tester.run_all_tests(pdf_path, num_pages=15)

    print("\nNext steps:")
    print("1. Select the best prompt based on results")
    print("2. Test with phi3:mini (slower but slightly better accuracy)")
    print("3. Consider two-pass approach for verification")


if __name__ == "__main__":
    main()
