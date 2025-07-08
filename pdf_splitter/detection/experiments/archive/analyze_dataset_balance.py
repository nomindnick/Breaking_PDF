#!/usr/bin/env python3
"""Analyze the current dataset balance and create a balanced version."""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.enhanced_synthetic_tests import \
    EnhancedSyntheticTester


def analyze_dataset():
    """Analyze the current test dataset."""
    tester = EnhancedSyntheticTester()

    print("Current Dataset Analysis")
    print("=" * 60)
    print(f"Total test cases: {len(tester.test_cases)}")

    # Count by expected outcome
    same_count = sum(1 for c in tester.test_cases if c.expected == "Same")
    different_count = len(tester.test_cases) - same_count

    print(f"\nDistribution:")
    print(f"  Same: {same_count} ({same_count/len(tester.test_cases)*100:.1f}%)")
    print(
        f"  Different: {different_count} ({different_count/len(tester.test_cases)*100:.1f}%)"
    )

    # Analyze by difficulty
    by_difficulty = defaultdict(lambda: {"Same": 0, "Different": 0})
    for case in tester.test_cases:
        by_difficulty[case.difficulty][case.expected] += 1

    print(f"\nBy Difficulty Level:")
    print(f"{'Level':<10} {'Same':<10} {'Different':<10} {'Total':<10}")
    print("-" * 40)
    for diff in sorted(by_difficulty.keys()):
        stats = by_difficulty[diff]
        total = stats["Same"] + stats["Different"]
        print(f"{diff:<10} {stats['Same']:<10} {stats['Different']:<10} {total:<10}")

    # Analyze by category
    by_category = defaultdict(int)
    for case in tester.test_cases:
        by_category[case.category] += 1

    print(f"\nBy Category:")
    for category, count in sorted(by_category.items()):
        print(f"  {category}: {count}")

    return tester.test_cases, same_count, different_count


def create_balanced_dataset(test_cases, same_count, different_count):
    """Create a balanced dataset with equal Same/Different cases."""

    print("\n\nCreating Balanced Dataset")
    print("=" * 60)

    # Separate cases
    same_cases = [c for c in test_cases if c.expected == "Same"]
    different_cases = [c for c in test_cases if c.expected == "Different"]

    # We have fewer "Different" cases, so we'll balance by:
    # 1. Using all Different cases
    # 2. Randomly sampling Same cases to match

    import random

    random.seed(42)  # For reproducibility

    # Sample Same cases to match Different count
    balanced_same = random.sample(same_cases, different_count)

    # Combine and shuffle
    balanced_dataset = balanced_same + different_cases
    random.shuffle(balanced_dataset)

    print(f"Balanced dataset size: {len(balanced_dataset)}")
    print(f"  Same: {len(balanced_same)} (50%)")
    print(f"  Different: {len(different_cases)} (50%)")

    # Analyze balance by difficulty
    by_difficulty = defaultdict(lambda: {"Same": 0, "Different": 0})
    for case in balanced_dataset:
        by_difficulty[case.difficulty][case.expected] += 1

    print(f"\nBalanced Dataset by Difficulty:")
    print(f"{'Level':<10} {'Same':<10} {'Different':<10} {'Total':<10}")
    print("-" * 40)
    for diff in sorted(by_difficulty.keys()):
        stats = by_difficulty[diff]
        total = stats["Same"] + stats["Different"]
        print(f"{diff:<10} {stats['Same']:<10} {stats['Different']:<10} {total:<10}")

    return balanced_dataset


def main():
    """Analyze dataset and create balanced version."""
    # Analyze current dataset
    test_cases, same_count, different_count = analyze_dataset()

    # Create balanced dataset
    balanced_dataset = create_balanced_dataset(test_cases, same_count, different_count)

    # Recommendations
    print("\n\nRecommendations:")
    print("-" * 60)
    print("1. The original dataset has 61.5% Same cases vs 38.5% Different cases")
    print(
        "2. This imbalance affects F1 scores and biases models toward 'Same' predictions"
    )
    print("3. The balanced dataset has exactly 50/50 distribution")
    print("4. Some difficulty levels may be underrepresented in the balanced set")
    print("5. Consider creating more 'Different' test cases for better coverage")


if __name__ == "__main__":
    main()
