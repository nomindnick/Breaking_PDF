#!/usr/bin/env python3
"""Test prompts with balanced dataset to get more accurate metrics."""

import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.enhanced_synthetic_tests import (
    EnhancedSyntheticTester,
)
from pdf_splitter.detection.experiments.test_optimal_prompts import OptimalPromptTester


class BalancedDatasetTester(OptimalPromptTester):
    """Tester that uses a balanced dataset."""

    def __init__(self):
        super().__init__()
        # Create balanced dataset
        self._create_balanced_dataset()

    def _create_balanced_dataset(self):
        """Create a balanced dataset with equal Same/Different cases."""
        # Get all test cases
        all_cases = self.synthetic_tester.test_cases

        # Separate by expected outcome
        same_cases = [c for c in all_cases if c.expected == "Same"]
        different_cases = [c for c in all_cases if c.expected == "Different"]

        # Balance by sampling Same cases to match Different count
        random.seed(42)  # For reproducibility
        balanced_same = random.sample(same_cases, len(different_cases))

        # Combine and shuffle
        self.balanced_cases = balanced_same + different_cases
        random.shuffle(self.balanced_cases)

        # Replace the test cases in synthetic_tester
        self.synthetic_tester.test_cases = self.balanced_cases

    def test_with_balanced_dataset(self, models, prompts_to_test):
        """Test specific prompts with the balanced dataset."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "dataset": "balanced",
            "total_cases": len(self.balanced_cases),
            "distribution": {
                "Same": sum(1 for c in self.balanced_cases if c.expected == "Same"),
                "Different": sum(
                    1 for c in self.balanced_cases if c.expected == "Different"
                ),
            },
            "models": {},
        }

        for model in models:
            print(f"\n{'='*60}")
            print(f"Testing {model} with balanced dataset")
            print(f"{'='*60}")

            model_results = {}

            for prompt_name in prompts_to_test:
                print(f"\nTesting prompt: {prompt_name}")

                # Test the prompt
                prompt_results = self._test_single_prompt(
                    model,
                    prompt_name,
                    difficulty_levels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                )

                model_results[prompt_name] = prompt_results

                # Print quick summary
                if "overall" in prompt_results:
                    overall = prompt_results["overall"]
                    print(f"  Accuracy: {overall.get('accuracy', 0):.2%}")
                    print(f"  F1 Score: {overall.get('f1_score', 0):.3f}")
                    print(f"  Precision: {overall.get('precision', 0):.3f}")
                    print(f"  Recall: {overall.get('recall', 0):.3f}")

            results["models"][model] = model_results

        return results


def main():
    """Test best prompts with balanced dataset."""
    print("Testing with Balanced Dataset")
    print("=" * 60)

    # Initialize tester
    tester = BalancedDatasetTester()

    print(f"Balanced dataset size: {len(tester.balanced_cases)} cases")
    print(f"Distribution: 50% Same, 50% Different")

    # Test configuration
    models = ["phi4-mini:3.8b", "gemma3:latest"]

    # Test only the best performing prompts
    prompts_to_test = [
        "gemma3_optimal",  # Best overall (F1=0.700 on imbalanced)
        "E1_cod_reasoning",  # Second best (F1=0.500 on imbalanced)
        "D1_conservative_few_shot",  # For comparison
    ]

    # Run tests
    results = tester.test_with_balanced_dataset(models, prompts_to_test)

    # Save results
    output_dir = Path("pdf_splitter/detection/experiments/results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"balanced_dataset_test_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON: IMBALANCED vs BALANCED DATASET")
    print("=" * 60)

    print("\nNote: F1 scores on balanced dataset are more reliable indicators")
    print("of true model performance for boundary detection.")

    # Summary table
    print("\n| Model | Prompt | Balanced F1 | Original F1 | Change |")
    print("|-------|--------|-------------|-------------|---------|")

    for model, model_data in results["models"].items():
        for prompt_name, prompt_data in model_data.items():
            if "overall" in prompt_data:
                balanced_f1 = prompt_data["overall"].get("f1_score", 0)
                # Note: We'd need to load original results to show comparison
                print(f"| {model} | {prompt_name} | {balanced_f1:.3f} | - | - |")


if __name__ == "__main__":
    main()
