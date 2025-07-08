#!/usr/bin/env python3
"""Test ONLY the optimal prompts to get complete results."""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.test_optimal_prompts import \
    OptimalPromptTester

# Initialize tester
tester = OptimalPromptTester()

# Test configuration - ONLY optimal prompts
models = ["phi4-mini:3.8b", "gemma3:latest"]
results = {
    "timestamp": datetime.now().isoformat(),
    "test_type": "optimal_prompts_only",
    "models": {},
}

for model in models:
    print(f"\n{'='*60}")
    print(f"Testing {model} - OPTIMAL PROMPT ONLY")
    print(f"{'='*60}")

    model_results = {"optimal": {}}

    # Determine optimal prompt
    model_family = tester.formatter.detect_model_family(model)
    if model_family == "phi":
        optimal_prompt = "phi4_optimal"
    elif model_family == "gemma":
        optimal_prompt = "gemma3_optimal"
    else:
        continue

    print(f"Using prompt: {optimal_prompt}")

    # Test with all difficulty levels
    prompt_results = tester._test_single_prompt(
        model, optimal_prompt, difficulty_levels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    model_results["optimal"][optimal_prompt] = prompt_results

    # Print quick summary
    if "overall" in prompt_results:
        overall = prompt_results["overall"]
        print(f"\nResults for {optimal_prompt}:")
        print(f"  Total cases: {overall.get('total', 0)}")
        print(f"  Correct: {overall.get('correct', 0)}")
        print(f"  Accuracy: {overall.get('accuracy', 0):.2%}")
        print(f"  F1 Score: {overall.get('f1_score', 0):.3f}")
        print(f"  Precision: {overall.get('precision', 0):.3f}")
        print(f"  Recall: {overall.get('recall', 0):.3f}")

    results["models"][model] = model_results

# Save results
output_dir = Path("pdf_splitter/detection/experiments/results")
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"optimal_prompts_only_{timestamp}.json"

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\nResults saved to: {output_file}")

# Print final summary
print("\n" + "=" * 60)
print("FINAL SUMMARY - OPTIMAL PROMPTS")
print("=" * 60)

for model, model_data in results["models"].items():
    for prompt_name, prompt_data in model_data.get("optimal", {}).items():
        if "overall" in prompt_data:
            overall = prompt_data["overall"]
            print(f"\n{model} - {prompt_name}:")
            print(f"  F1 Score: {overall.get('f1_score', 0):.3f}")
            print(f"  Accuracy: {overall.get('accuracy', 0):.2%}")
            print(f"  Precision: {overall.get('precision', 0):.3f}")
            print(f"  Recall: {overall.get('recall', 0):.3f}")
