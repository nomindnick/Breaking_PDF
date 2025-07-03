#!/usr/bin/env python3
"""
Enhanced synthetic boundary test suite with comprehensive prompt engineering experiments.

This module extends the original synthetic tests with:
1. More test cases across difficulty levels
2. Specific prompt templates from user requirements
3. Progressive difficulty testing
4. Detailed performance analysis
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.synthetic_boundary_tests import (
    SyntheticBoundaryTester,
    TestCase,
)


class EnhancedSyntheticTester(SyntheticBoundaryTester):
    """Enhanced synthetic boundary tester with more comprehensive test cases and prompts."""

    def __init__(self):
        """Initialize enhanced synthetic tester with additional test cases."""
        super().__init__()
        # Extend test cases with more examples
        self.test_cases.extend(self._create_additional_test_cases())
        # Sort test cases by difficulty for progressive testing
        self.test_cases.sort(key=lambda x: x.difficulty)

    def _create_additional_test_cases(self) -> List[TestCase]:
        """Create additional test cases for comprehensive testing."""
        cases = []

        # MORE OBVIOUS DIFFERENT (Difficulty 1-2)
        cases.extend(
            [
                TestCase(
                    id="diff_4",
                    difficulty=1,
                    page1_bottom="Thank you for your business.\n\n[Company Logo]\nwww.company.com",
                    page2_top="REQUEST FOR INFORMATION\n\nProject: Highway Bridge Renovation\nRFI Number: 2024-0156\nDate: April 1, 2024",
                    expected="Different",
                    category="obvious_different",
                    explanation="Company footer followed by RFI header",
                ),
                TestCase(
                    id="diff_5",
                    difficulty=2,
                    page1_bottom="Approved by: _____________\nDate: _____________",
                    page2_top="CONFIDENTIAL\n\nMeeting Minutes\nBoard of Directors\nMarch 15, 2024",
                    expected="Different",
                    category="obvious_different",
                    explanation="Approval signature block followed by meeting minutes header",
                ),
            ]
        )

        # MORE OBVIOUS SAME (Difficulty 1-2)
        cases.extend(
            [
                TestCase(
                    id="same_4",
                    difficulty=1,
                    page1_bottom="Furthermore, the analysis indicates that the proposed",
                    page2_top="solution would reduce processing time by approximately 40%.",
                    expected="Same",
                    category="obvious_same",
                    explanation="Clear sentence continuation",
                ),
                TestCase(
                    id="same_5",
                    difficulty=2,
                    page1_bottom="Table 2.1: Quarterly Revenue (continued)",
                    page2_top="Q3 2023    $1,250,000    $980,000    $270,000\nQ4 2023    $1,480,000    $1,100,000  $380,000",
                    expected="Same",
                    category="obvious_same",
                    explanation="Table explicitly marked as continued",
                ),
            ]
        )

        # MORE MEDIUM CASES (Difficulty 4-6)
        cases.extend(
            [
                TestCase(
                    id="med_4",
                    difficulty=4,
                    page1_bottom="Best regards,\n\nMichael Chen",
                    page2_top="cc: Sarah Johnson, David Williams\n\nAttachment: Q3_Financial_Report.pdf",
                    expected="Same",
                    category="medium_same",
                    explanation="Email signature with cc line on next page",
                ),
                TestCase(
                    id="med_5",
                    difficulty=5,
                    page1_bottom="This agreement shall remain in effect until terminated.",
                    page2_top="SCHEDULE A - PRICING\n\nThe following rates apply:",
                    expected="Same",
                    category="medium_same",
                    explanation="Contract followed by its schedule/appendix",
                ),
                TestCase(
                    id="med_6",
                    difficulty=6,
                    page1_bottom="Project Status: Complete\nDelivery Date: March 15, 2024",
                    page2_top="Purchase Order #45678\n\nVendor: ABC Supplies Inc.\nDate: March 16, 2024",
                    expected="Different",
                    category="medium_different",
                    explanation="Project completion followed by new PO with next day's date",
                ),
            ]
        )

        # MORE HARD CASES (Difficulty 7-9)
        cases.extend(
            [
                TestCase(
                    id="hard_4",
                    difficulty=7,
                    page1_bottom="Figure 3: System Architecture Diagram",
                    page2_top="3.2 Component Specifications\n\nEach component in the system architecture",
                    expected="Same",
                    category="hard_same",
                    explanation="Figure reference followed by related section",
                ),
                TestCase(
                    id="hard_5",
                    difficulty=8,
                    page1_bottom="Total: $45,678.90",
                    page2_top="Invoice #12346\n\nDate: March 20, 2024\nCustomer: XYZ Corp",
                    expected="Different",
                    category="hard_different",
                    explanation="Total amount followed by new invoice - could be summary or new doc",
                ),
                TestCase(
                    id="hard_6",
                    difficulty=9,
                    page1_bottom="Conclusion",
                    page2_top="The findings presented in this report demonstrate",
                    expected="Same",
                    category="hard_same",
                    explanation="Section header followed by content - ambiguous without more context",
                ),
            ]
        )

        # MORE EDGE CASES (Difficulty 10)
        cases.extend(
            [
                TestCase(
                    id="edge_3",
                    difficulty=10,
                    page1_bottom="1/2",
                    page2_top="2/2",
                    expected="Same",
                    category="edge_same",
                    explanation="Page numbering indicates same document",
                ),
                TestCase(
                    id="edge_4",
                    difficulty=10,
                    page1_bottom="",
                    page2_top="Reference: DOC-2024-001\n\nSubject: Annual Review",
                    expected="Different",
                    category="edge_different",
                    explanation="Empty page end followed by document header",
                ),
            ]
        )

        return cases

    def get_user_specified_prompts(self) -> Dict[str, Dict[str, any]]:
        """Get the specific prompt templates requested by the user."""
        return {
            # Group A: Output Control + Conservative Bias
            "A1_asymmetric": {
                "template": """Assume pages are consecutive unless you see a clear document break.
Output 'D' ONLY when certain a break exists, else 'S'.

Page 1: {page1_bottom}
Page 2: {page2_top}

Reply with S or D only.""",
                "config": {"max_tokens": 1, "stop": ["S", "D"], "temperature": 0.1},
            },
            "A2_high_confidence": {
                "template": """Only classify as different documents if very confident. When uncertain, choose same document.

Page 1: {page1_bottom}
Page 2: {page2_top}

Answer S (same) or D (different):""",
                "config": {"max_tokens": 1, "temperature": 0.05},
            },
            # Group B: Confidence Scoring
            "B1_json_confidence": {
                "template": """Classify this page transition and provide confidence.

Page 1: {page1_bottom}
Page 2: {page2_top}

Respond JSON: {"label":"S","conf":0.85} or {"label":"D","conf":0.92}""",
                "config": {"temperature": 0.0, "max_tokens": 50},
                "post_process": "json_confidence",
            },
            "B2_confidence_threshold": {
                "template": """Rate document boundary likelihood (0-1) then classify.

Page 1: {page1_bottom}
Page 2: {page2_top}

Confidence: 0.X
Label: S or D""",
                "config": {"temperature": 0.0, "max_tokens": 30},
                "post_process": "parse_confidence",
            },
            # Group C: Structured Decision Making
            "C1_silent_checklist": {
                "template": """Think silently: ① Topic continuity? ② Page sequence? ③ Document markers?
Then output 'S' or 'D'.

Page 1: {page1_bottom}
Page 2: {page2_top}

Classification:""",
                "config": {"temperature": 0.0, "max_tokens": 1},
            },
            "C2_self_check": {
                "template": """Classify this transition twice silently, then give final answer.
If both classifications match, use that; if different, choose 'S'.

Page 1: {page1_bottom}
Page 2: {page2_top}

Final: S or D""",
                "config": {"temperature": 0.0, "max_tokens": 1},
            },
            # Group D: Few-Shot with Strategic Examples
            "D1_conservative_few_shot": {
                "template": """Examples:
Page 1: "...meeting adjourned at 3 PM." Page 2: "Minutes of Board Meeting..." → S
Page 1: "...thank you. Sincerely, John" Page 2: "INVOICE #12345 Date:..." → D
Page 1: "...continued on next page." Page 2: "The following items were..." → S

Your turn:
Page 1: {page1_bottom}
Page 2: {page2_top}

Answer: S or D""",
                "config": {"temperature": 0.0, "max_tokens": 5},
            },
        }

    def post_process_response(
        self, response: str, post_process_type: str, threshold: float = 0.4
    ) -> Tuple[str, Optional[float]]:
        """Post-process model responses based on the specified type."""
        if post_process_type == "json_confidence":
            try:
                # Extract JSON from response
                json_match = re.search(r"\{.*?\}", response)
                if json_match:
                    data = json.loads(json_match.group())
                    label = data.get("label", "").upper()
                    conf = float(data.get("conf", 0))

                    # Apply threshold - only accept "D" if confidence > threshold
                    if label == "D" and conf <= threshold:
                        return "S", conf
                    return label, conf
            except Exception:
                pass
            return "Unknown", None

        elif post_process_type == "parse_confidence":
            try:
                # Parse confidence and label
                conf_match = re.search(r"Confidence:\s*(\d*\.?\d+)", response)
                label_match = re.search(r"Label:\s*([SD])", response)

                if conf_match and label_match:
                    conf = float(conf_match.group(1))
                    label = label_match.group(1).upper()

                    # Apply threshold
                    if label == "D" and conf <= threshold:
                        return "S", conf
                    return label, conf
            except Exception:
                pass
            return "Unknown", None

        # Default: parse simple S/D response
        response = response.strip().upper()
        if response.startswith("S"):
            return "S", None
        elif response.startswith("D"):
            return "D", None
        else:
            return "Unknown", None

    def test_prompts_progressively(
        self,
        model: str,
        difficulty_levels: List[int] = None,
        confidence_threshold: float = 0.4,
    ) -> Dict:
        """Test prompts progressively by difficulty level."""
        if difficulty_levels is None:
            difficulty_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        prompts = self.get_user_specified_prompts()
        results = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "by_prompt": {},
            "by_difficulty": {},
            "best_prompts_by_difficulty": {},
        }

        # Test each prompt
        for prompt_name, prompt_info in prompts.items():
            print(f"\nTesting prompt: {prompt_name}")
            prompt_results = {
                "name": prompt_name,
                "by_difficulty": {},
                "overall": {"correct": 0, "total": 0},
            }

            # Test by difficulty level
            for difficulty in difficulty_levels:
                # Get test cases for this difficulty
                diff_cases = [c for c in self.test_cases if c.difficulty == difficulty]
                if not diff_cases:
                    continue

                diff_results = {"correct": 0, "total": len(diff_cases), "details": []}

                for case in diff_cases:
                    # Format prompt
                    prompt = prompt_info["template"].format(
                        page1_bottom=case.page1_bottom, page2_top=case.page2_top
                    )

                    # Get model response
                    try:
                        config = prompt_info.get("config", {})
                        response = self.client.generate(
                            model=model,
                            prompt=prompt,
                            temperature=config.get("temperature", 0.0),
                            max_tokens=config.get("max_tokens", 50),
                            stop=config.get("stop", None),
                            timeout=30,
                        )

                        response_text = response.get("response", "").strip()

                        # Post-process if needed
                        post_process = prompt_info.get("post_process")
                        if post_process:
                            prediction, confidence = self.post_process_response(
                                response_text, post_process, confidence_threshold
                            )
                        else:
                            prediction, confidence = self.post_process_response(
                                response_text, None
                            )

                        # Map to expected format
                        if prediction == "S":
                            prediction = "Same"
                        elif prediction == "D":
                            prediction = "Different"

                        # Check if correct
                        correct = prediction == case.expected
                        if correct:
                            diff_results["correct"] += 1
                            prompt_results["overall"]["correct"] += 1

                        diff_results["details"].append(
                            {
                                "case_id": case.id,
                                "correct": correct,
                                "expected": case.expected,
                                "predicted": prediction,
                                "confidence": confidence,
                                "response": response_text[:100],
                            }
                        )

                    except Exception as e:
                        diff_results["details"].append(
                            {"case_id": case.id, "correct": False, "error": str(e)}
                        )

                prompt_results["overall"]["total"] += diff_results["total"]
                prompt_results["by_difficulty"][difficulty] = diff_results

                # Track accuracy by difficulty
                accuracy = (
                    diff_results["correct"] / diff_results["total"]
                    if diff_results["total"] > 0
                    else 0
                )
                print(
                    f"  Difficulty {difficulty}: {accuracy:.1%} ({diff_results['correct']}/{diff_results['total']})"
                )

            results["by_prompt"][prompt_name] = prompt_results

        # Find best prompts for each difficulty level
        for difficulty in difficulty_levels:
            best_prompt = None
            best_accuracy = 0

            for prompt_name, prompt_results in results["by_prompt"].items():
                if difficulty in prompt_results["by_difficulty"]:
                    diff_data = prompt_results["by_difficulty"][difficulty]
                    accuracy = (
                        diff_data["correct"] / diff_data["total"]
                        if diff_data["total"] > 0
                        else 0
                    )
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_prompt = prompt_name

            if best_prompt:
                results["best_prompts_by_difficulty"][difficulty] = {
                    "prompt": best_prompt,
                    "accuracy": best_accuracy,
                }

        return results

    def test_easy_medium_hard_progression(
        self, models: List[str], confidence_threshold: float = 0.4
    ) -> Dict:
        """Test models with easy->medium->hard progression."""
        # Define difficulty groups
        easy = [1, 2, 3]
        medium = [4, 5, 6]
        hard = [7, 8, 9, 10]

        all_results = {
            "experiment": "progressive_difficulty_test",
            "timestamp": datetime.now().isoformat(),
            "models": {},
        }

        for model in models:
            print(f"\n{'='*60}")
            print(f"Testing {model}")
            print(f"{'='*60}")

            model_results = {
                "easy": None,
                "medium": None,
                "hard": None,
                "best_prompt": None,
            }

            # Test on easy cases first
            print("\n1. Testing EASY cases (difficulty 1-3)...")
            easy_results = self.test_prompts_progressively(
                model, easy, confidence_threshold
            )

            # Find best performing prompt on easy cases
            best_prompt = None
            best_accuracy = 0
            for prompt_name, prompt_data in easy_results["by_prompt"].items():
                if prompt_data["overall"]["total"] > 0:
                    accuracy = (
                        prompt_data["overall"]["correct"]
                        / prompt_data["overall"]["total"]
                    )
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_prompt = prompt_name

            model_results["easy"] = {
                "best_prompt": best_prompt,
                "best_accuracy": best_accuracy,
                "all_results": easy_results,
            }

            print(f"\nBest prompt for easy cases: {best_prompt} ({best_accuracy:.1%})")

            # Only proceed to medium if accuracy > 80% on easy
            if best_accuracy >= 0.8:
                print("\n2. Testing MEDIUM cases (difficulty 4-6)...")
                medium_results = self.test_prompts_progressively(
                    model, medium, confidence_threshold
                )

                # Check performance of best easy prompt on medium
                if best_prompt in medium_results["by_prompt"]:
                    medium_data = medium_results["by_prompt"][best_prompt]["overall"]
                    medium_accuracy = (
                        medium_data["correct"] / medium_data["total"]
                        if medium_data["total"] > 0
                        else 0
                    )

                    model_results["medium"] = {
                        "accuracy_with_best_easy_prompt": medium_accuracy,
                        "all_results": medium_results,
                    }

                    print(f"Best easy prompt on medium cases: {medium_accuracy:.1%}")

                    # Only proceed to hard if accuracy > 70% on medium
                    if medium_accuracy >= 0.7:
                        print("\n3. Testing HARD cases (difficulty 7-10)...")
                        hard_results = self.test_prompts_progressively(
                            model, hard, confidence_threshold
                        )

                        if best_prompt in hard_results["by_prompt"]:
                            hard_data = hard_results["by_prompt"][best_prompt][
                                "overall"
                            ]
                            hard_accuracy = (
                                hard_data["correct"] / hard_data["total"]
                                if hard_data["total"] > 0
                                else 0
                            )

                            model_results["hard"] = {
                                "accuracy_with_best_easy_prompt": hard_accuracy,
                                "all_results": hard_results,
                            }

                            print(
                                f"Best easy prompt on hard cases: {hard_accuracy:.1%}"
                            )
                    else:
                        print("\nSkipping HARD cases (medium accuracy < 70%)")
                else:
                    print("\nSkipping MEDIUM/HARD cases (easy accuracy < 80%)")
            else:
                print("\nSkipping MEDIUM/HARD cases (easy accuracy < 80%)")

            model_results["best_prompt"] = best_prompt
            all_results["models"][model] = model_results

        return all_results

    def save_progressive_results(self, results: Dict, output_dir: Path) -> Path:
        """Save progressive test results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"progressive_boundary_test_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        return output_file


def main():
    """Run enhanced synthetic boundary tests."""
    tester = EnhancedSyntheticTester()

    # Models to test
    models = ["phi4-mini:3.8b", "gemma3:latest", "qwen3:8b", "qwen3:1.7b"]

    print("Enhanced Synthetic Boundary Testing")
    print("=" * 60)
    print(f"Total test cases: {len(tester.test_cases)}")
    print(f"Models to test: {len(models)}")
    print(f"Prompt variations: {len(tester.get_user_specified_prompts())}")

    # Run progressive testing
    results = tester.test_easy_medium_hard_progression(models, confidence_threshold=0.4)

    # Save results
    output_dir = Path("pdf_splitter/detection/experiments/results")
    output_dir.mkdir(exist_ok=True)
    tester.save_progressive_results(results, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF PROGRESSIVE TESTING")
    print("=" * 60)

    for model, model_results in results["models"].items():
        print(f"\n{model}:")
        print(f"  Best prompt: {model_results['best_prompt']}")

        if model_results["easy"]:
            print(f"  Easy cases: {model_results['easy']['best_accuracy']:.1%}")

        if model_results["medium"]:
            print(
                f"  Medium cases: {model_results['medium']['accuracy_with_best_easy_prompt']:.1%}"
            )

        if model_results["hard"]:
            print(
                f"  Hard cases: {model_results['hard']['accuracy_with_best_easy_prompt']:.1%}"
            )


if __name__ == "__main__":
    main()
