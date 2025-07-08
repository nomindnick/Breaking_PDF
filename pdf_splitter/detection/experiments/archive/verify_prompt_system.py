#!/usr/bin/env python3
"""
Quick verification script to test the prompt engineering system.

This script tests:
1. Synthetic test case generation
2. Prompt template loading
3. Basic model testing with a few prompts
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.enhanced_synthetic_tests import \
    EnhancedSyntheticTester
from pdf_splitter.detection.experiments.experiment_runner import \
    LLMExperimentRunner


def verify_synthetic_tests():
    """Verify synthetic test cases are created properly."""
    print("1. Verifying Synthetic Test Cases")
    print("-" * 40)

    tester = EnhancedSyntheticTester()

    print(f"Total test cases: {len(tester.test_cases)}")

    # Count by difficulty
    difficulty_counts = {}
    for case in tester.test_cases:
        diff = case.difficulty
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

    print("\nCases by difficulty:")
    for diff in sorted(difficulty_counts.keys()):
        print(f"  Difficulty {diff}: {difficulty_counts[diff]} cases")

    # Show a few examples
    print("\nExample test cases:")
    for case in tester.test_cases[:3]:
        print(f"\n  ID: {case.id} (Difficulty: {case.difficulty})")
        print(f"  Expected: {case.expected}")
        print(f"  Category: {case.category}")
        print(f"  Page 1 bottom: {case.page1_bottom[:50]}...")
        print(f"  Page 2 top: {case.page2_top[:50]}...")

    return True


def verify_prompt_templates():
    """Verify prompt templates are loaded correctly."""
    print("\n2. Verifying Prompt Templates")
    print("-" * 40)

    runner = LLMExperimentRunner()

    # Check that all user-specified templates are loaded
    expected_templates = [
        "A1_asymmetric",
        "A2_high_confidence",
        "B1_json_confidence",
        "B2_confidence_threshold",
        "C1_silent_checklist",
        "C2_self_check",
        "D1_conservative_few_shot",
    ]

    print(f"Total templates loaded: {len(runner.prompt_templates)}")
    print("\nUser-specified templates:")

    for template in expected_templates:
        if template in runner.prompt_templates:
            print(f"  ✓ {template}")
            # Show first 100 chars
            content = runner.prompt_templates[template][:100].replace("\n", " ")
            print(f"    Preview: {content}...")
        else:
            print(f"  ✗ {template} - NOT FOUND")

    return True


def verify_prompt_configs():
    """Verify prompt configurations from enhanced tester."""
    print("\n3. Verifying Prompt Configurations")
    print("-" * 40)

    tester = EnhancedSyntheticTester()
    prompts = tester.get_user_specified_prompts()

    print(f"Total prompt configurations: {len(prompts)}")

    for name, config in prompts.items():
        print(f"\n  {name}:")
        print(f"    Has template: {'template' in config}")
        print(f"    Has config: {'config' in config}")
        if "config" in config:
            print(f"    Temperature: {config['config'].get('temperature', 'default')}")
            print(f"    Max tokens: {config['config'].get('max_tokens', 'default')}")
        if "post_process" in config:
            print(f"    Post-processing: {config['post_process']}")

    return True


def test_simple_prompt():
    """Test a simple prompt with Ollama."""
    print("\n4. Testing Simple Prompt with Ollama")
    print("-" * 40)

    try:
        from pdf_splitter.detection.experiments.experiment_runner import \
            OllamaClient

        client = OllamaClient()

        # Test with a very simple prompt
        response = client.generate(
            model="phi4-mini:3.8b",
            prompt="Reply with just the letter S",
            temperature=0.0,
            max_tokens=1,
            timeout=10,
        )

        if "error" in response:
            print(f"  ✗ Ollama error: {response['error']}")
            print("  Make sure Ollama is running: ollama serve")
            return False
        else:
            print(f"  ✓ Ollama response: {response.get('response', 'No response')}")
            return True

    except Exception as e:
        print(f"  ✗ Error testing Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False


def main():
    """Run all verification tests."""
    print("PDF Splitter Prompt Engineering System Verification")
    print("=" * 60)

    # Run verification tests
    tests = [
        ("Synthetic Tests", verify_synthetic_tests),
        ("Prompt Templates", verify_prompt_templates),
        ("Prompt Configs", verify_prompt_configs),
        ("Ollama Connection", test_simple_prompt),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\nError in {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False

    if all_passed:
        print("\nAll verification tests passed! ✓")
        print("\nYou can now run the full systematic tests with:")
        print("  python pdf_splitter/detection/experiments/systematic_prompt_test.py")
        print("\nOr test a specific prompt with:")
        print(
            "  python pdf_splitter/detection/experiments/systematic_prompt_test.py \\"
        )
        print("    --quick-test phi4-mini:3.8b A1_asymmetric")
    else:
        print("\nSome tests failed. Please fix the issues above before proceeding.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
