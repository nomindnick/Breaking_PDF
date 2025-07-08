#!/usr/bin/env python3
"""
Synthetic boundary test suite for evaluating LLM boundary detection.

Creates controlled test cases ranging from obvious to ambiguous boundaries.
"""

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.experiment_runner import OllamaClient


@dataclass
class TestCase:
    """A single test case for boundary detection."""

    id: str
    difficulty: int  # 1-10, where 1 is obvious, 10 is very hard
    page1_bottom: str
    page2_top: str
    expected: str  # "Same" or "Different"
    category: str
    explanation: str


class SyntheticBoundaryTester:
    """Tests models with synthetic boundary examples."""

    def __init__(self):
        self.client = OllamaClient()
        self.test_cases = self._create_test_cases()

    def _create_test_cases(self) -> List[TestCase]:
        """Create a comprehensive set of test cases."""
        cases = []

        # OBVIOUS DIFFERENT DOCUMENTS (Difficulty 1-3)
        cases.extend(
            [
                TestCase(
                    id="diff_1",
                    difficulty=1,
                    page1_bottom="Sincerely,\n\nJohn Smith\nManager",
                    page2_top="MEMORANDUM\n\nTo: All Staff\nFrom: HR Department\nDate: March 15, 2024\nRe: Updated Vacation Policy",
                    expected="Different",
                    category="obvious_different",
                    explanation="Clear document end (signature) followed by new document header",
                ),
                TestCase(
                    id="diff_2",
                    difficulty=2,
                    page1_bottom="If you have any questions, please contact us at support@example.com.\n\nBest regards,\nThe Support Team",
                    page2_top="Invoice #12345\n\nBill To:\nAcme Corporation\n123 Main Street",
                    expected="Different",
                    category="obvious_different",
                    explanation="Email closing followed by invoice header",
                ),
                TestCase(
                    id="diff_3",
                    difficulty=3,
                    page1_bottom="Page 4 of 4\n\nEnd of Report",
                    page2_top="Dear Mr. Johnson,\n\nI am writing to inform you about the upcoming changes to our service agreement.",
                    expected="Different",
                    category="obvious_different",
                    explanation="Report end followed by letter beginning",
                ),
            ]
        )

        # OBVIOUS SAME DOCUMENT (Difficulty 1-3)
        cases.extend(
            [
                TestCase(
                    id="same_1",
                    difficulty=1,
                    page1_bottom="The quick brown fox jumps over the lazy",
                    page2_top="dog and continues running through the forest.",
                    expected="Same",
                    category="obvious_same",
                    explanation="Sentence continues across page break",
                ),
                TestCase(
                    id="same_2",
                    difficulty=2,
                    page1_bottom="3. Third point about implementation\n4. Fourth point about",
                    page2_top="testing procedures and quality assurance measures.",
                    expected="Same",
                    category="obvious_same",
                    explanation="Numbered list continues across pages",
                ),
                TestCase(
                    id="same_3",
                    difficulty=3,
                    page1_bottom="In conclusion, our analysis demonstrates that the proposed solution will",
                    page2_top="significantly improve operational efficiency while reducing costs.",
                    expected="Same",
                    category="obvious_same",
                    explanation="Paragraph continues with same topic",
                ),
            ]
        )

        # MEDIUM DIFFICULTY (4-6)
        cases.extend(
            [
                TestCase(
                    id="med_1",
                    difficulty=5,
                    page1_bottom="Thank you for your consideration.\n\nSincerely,",
                    page2_top="Sarah Williams\nProject Manager\n\nAttachment A: Budget Breakdown",
                    expected="Same",
                    category="medium_same",
                    explanation="Signature split across pages - still same document",
                ),
                TestCase(
                    id="med_2",
                    difficulty=6,
                    page1_bottom="For more information, please see the attached documents.",
                    page2_top="Section 2: Technical Specifications\n\nThe following requirements must be met:",
                    expected="Same",
                    category="medium_same",
                    explanation="Could be same document with section break",
                ),
                TestCase(
                    id="med_3",
                    difficulty=5,
                    page1_bottom="This concludes our monthly report.",
                    page2_top="Executive Summary\n\nThis document outlines the key findings from our recent audit.",
                    expected="Different",
                    category="medium_different",
                    explanation="Report end followed by new document start",
                ),
            ]
        )

        # HARD CASES (7-9)
        cases.extend(
            [
                TestCase(
                    id="hard_1",
                    difficulty=8,
                    page1_bottom="Additional notes and observations are included below.",
                    page2_top="Notes:\n• System performance improved by 15%\n• User satisfaction increased",
                    expected="Same",
                    category="hard_same",
                    explanation="Notes section continues - ambiguous without more context",
                ),
                TestCase(
                    id="hard_2",
                    difficulty=9,
                    page1_bottom="2024-03-15",
                    page2_top="2024-03-16\n\nDaily Report",
                    expected="Different",
                    category="hard_different",
                    explanation="Date change might indicate new document",
                ),
                TestCase(
                    id="hard_3",
                    difficulty=7,
                    page1_bottom="See reverse side for terms and conditions.",
                    page2_top="Terms and Conditions\n\n1. Payment is due within 30 days",
                    expected="Same",
                    category="hard_same",
                    explanation="Referenced content on next page",
                ),
            ]
        )

        # EDGE CASES (Difficulty 10)
        cases.extend(
            [
                TestCase(
                    id="edge_1",
                    difficulty=10,
                    page1_bottom="[This page intentionally left blank]",
                    page2_top="Chapter 2: Implementation Details",
                    expected="Same",
                    category="edge_same",
                    explanation="Blank pages within documents are common",
                ),
                TestCase(
                    id="edge_2",
                    difficulty=10,
                    page1_bottom="---",
                    page2_top="---",
                    expected="Same",
                    category="edge_ambiguous",
                    explanation="Separator lines could mean anything",
                ),
            ]
        )

        return cases

    def test_prompt_variants(self) -> Dict[str, str]:
        """Different prompt strategies to test."""
        return {
            "baseline": """Your task is to determine if two document snippets are part of a single document or are different documents.

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
            "balanced": """Determine if these two page snippets are from the same document or different documents.

Consider:
- Documents often continue across pages
- Look for clear endings (signatures, "End of Report", etc.)
- Look for clear beginnings (headers, "Dear...", dates, etc.)
- When uncertain, consider if the content flows naturally

Bottom of Page 1:
{page1_bottom}

Top of Page 2:
{page2_top}

Respond with only "Same Document" or "Different Documents".""",
            "bias_same": """You are reviewing a document that may span multiple pages. Most page transitions are simply page breaks within the same document.

Only mark as different documents if you see very clear evidence like:
- A signature/closing followed by a completely new header
- "End of document" followed by a new document title
- Completely unrelated topics with formal document boundaries

Bottom of Page 1:
{page1_bottom}

Top of Page 2:
{page2_top}

Response: "Same Document" or "Different Documents".""",
            "neutral": """Page 1 ends with:
{page1_bottom}

Page 2 begins with:
{page2_top}

Are these from the same document or different documents? Answer only "Same Document" or "Different Documents".""",
            "think_first": """Analyze these page transitions step by step:

Bottom of Page 1:
{page1_bottom}

Top of Page 2:
{page2_top}

First, identify any document markers (signatures, headers, dates, etc.).
Then determine: Same Document or Different Documents?""",
            "examples_first": """Examples:
- "Sincerely, John" → "MEMO TO: Staff" = Different Documents
- "the lazy" → "dog jumped over" = Same Document

Now analyze:
Bottom of Page 1:
{page1_bottom}

Top of Page 2:
{page2_top}

Answer: Same Document or Different Documents""",
        }

    def test_model_with_cases(self, model: str, prompt_template: str) -> Dict:
        """Test a model with all synthetic cases."""
        results = {
            "model": model,
            "correct": 0,
            "total": len(self.test_cases),
            "by_difficulty": {},
            "by_category": {},
            "errors": [],
            "details": [],
        }

        for case in self.test_cases:
            prompt = prompt_template.format(
                page1_bottom=case.page1_bottom, page2_top=case.page2_top
            )

            try:
                response = self.client.generate(
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=50,
                    timeout=30,
                )

                response_text = response.get("response", "").strip()

                # Parse response
                if "Different Document" in response_text:
                    prediction = "Different"
                elif "Same Document" in response_text:
                    prediction = "Same"
                else:
                    prediction = "Unknown"
                    results["errors"].append(
                        f"{case.id}: Unclear response - {response_text[:50]}"
                    )

                correct = prediction == case.expected
                if correct:
                    results["correct"] += 1

                # Track by difficulty
                diff_key = f"difficulty_{case.difficulty}"
                if diff_key not in results["by_difficulty"]:
                    results["by_difficulty"][diff_key] = {"correct": 0, "total": 0}
                results["by_difficulty"][diff_key]["total"] += 1
                if correct:
                    results["by_difficulty"][diff_key]["correct"] += 1

                # Track by category
                if case.category not in results["by_category"]:
                    results["by_category"][case.category] = {"correct": 0, "total": 0}
                results["by_category"][case.category]["total"] += 1
                if correct:
                    results["by_category"][case.category]["correct"] += 1

                # Store details
                results["details"].append(
                    {
                        "case_id": case.id,
                        "expected": case.expected,
                        "predicted": prediction,
                        "correct": correct,
                        "response": response_text[:100],
                    }
                )

            except Exception as e:
                results["errors"].append(f"{case.id}: {str(e)}")
                results["details"].append(
                    {
                        "case_id": case.id,
                        "expected": case.expected,
                        "predicted": "Error",
                        "correct": False,
                        "response": str(e),
                    }
                )

        results["accuracy"] = results["correct"] / results["total"]
        return results

    def run_comprehensive_test(self, models: List[str], prompt_names: List[str] = None):
        """Run comprehensive testing across models and prompts."""
        prompts = self.test_prompt_variants()
        if prompt_names:
            prompts = {k: v for k, v in prompts.items() if k in prompt_names}

        all_results = []

        for model in models:
            print(f"\nTesting {model}...")
            print("=" * 60)

            for prompt_name, prompt_template in prompts.items():
                print(f"  Testing prompt: {prompt_name}...")
                start_time = time.time()

                results = self.test_model_with_cases(model, prompt_template)
                results["prompt_name"] = prompt_name
                results["time_taken"] = time.time() - start_time

                all_results.append(results)

                # Print summary
                print(
                    f"    Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})"
                )
                print(f"    Time: {results['time_taken']:.1f}s")

                # Print category breakdown
                print("    By category:")
                for cat, stats in results["by_category"].items():
                    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                    print(
                        f"      {cat}: {acc:.1%} ({stats['correct']}/{stats['total']})"
                    )

        return all_results

    def save_results(self, results: List[Dict], output_dir: Path):
        """Save test results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"synthetic_boundary_test_{timestamp}.json"

        # Include test cases for reference
        output_data = {
            "experiment": "synthetic_boundary_test",
            "timestamp": datetime.now().isoformat(),
            "test_cases": [
                {
                    "id": case.id,
                    "difficulty": case.difficulty,
                    "expected": case.expected,
                    "category": case.category,
                    "explanation": case.explanation,
                    "page1_bottom": case.page1_bottom,
                    "page2_top": case.page2_top,
                }
                for case in self.test_cases
            ],
            "results": results,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        return output_file


def main():
    """Run synthetic boundary tests."""
    tester = SyntheticBoundaryTester()

    # Test all models that were previously considered
    models = [
        "phi4-mini:3.8b",
        "gemma3:latest",
        "qwen3:8b",
        "qwen3:1.7b",
        "llama3:8b-instruct-q5_K_M",
        "phi3:mini",
    ]

    # Test key prompt variants
    prompt_names = ["baseline", "balanced", "bias_same", "neutral"]

    print("Running synthetic boundary tests...")
    print(f"Testing {len(models)} models with {len(prompt_names)} prompts")
    print(f"Total test cases: {len(tester.test_cases)}")

    results = tester.run_comprehensive_test(models, prompt_names)

    # Save results
    output_dir = Path("pdf_splitter/detection/experiments/results")
    output_dir.mkdir(exist_ok=True)
    output_file = tester.save_results(results, output_dir)

    # Print best combinations
    print("\n" + "=" * 60)
    print("BEST MODEL-PROMPT COMBINATIONS")
    print("=" * 60)

    # Sort by accuracy
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)

    for i, result in enumerate(sorted_results[:10]):
        print(f"\n{i+1}. {result['model']} + {result['prompt_name']}")
        print(f"   Accuracy: {result['accuracy']:.1%}")
        print(
            f"   Obvious Same: {result['by_category'].get('obvious_same', {}).get('correct', 0)}/3"
        )
        print(
            f"   Obvious Diff: {result['by_category'].get('obvious_different', {}).get('correct', 0)}/3"
        )


if __name__ == "__main__":
    main()
