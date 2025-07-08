#!/usr/bin/env python3
"""Test the effect of model preloading on performance."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.experiment_runner import OllamaClient


def test_preloading_effect():
    """Test cold start vs warm model performance."""
    ollama = OllamaClient()

    # Simple test prompt
    test_prompt = """Determine if these texts are from the same document.
Page 1: "Thank you for your business."
Page 2: "INVOICE #123"
Answer with one word: SAME or DIFFERENT"""

    models = ["gemma3:1b-it-q4_K_M", "qwen3:0.6b", "gemma3:latest"]
    results = {}

    print("Testing model preloading effects")
    print("=" * 60)

    for model in models:
        print(f"\nTesting {model}...")
        timings = {"cold_start": [], "warm": [], "consecutive": []}

        # Test 1: Cold start (first call after model switch)
        print("  Cold start test...")
        for i in range(3):
            # Force a different model first to ensure cold start
            try:
                if model != models[0]:
                    ollama.generate(
                        model=models[0], prompt="test", max_tokens=1, timeout=10
                    )
            except:
                pass

            time.sleep(1)  # Brief pause

            start = time.time()
            try:
                response = ollama.generate(
                    model=model,
                    prompt=test_prompt,
                    temperature=0.0,
                    max_tokens=50,
                    timeout=30,
                )
                elapsed = time.time() - start
                timings["cold_start"].append(elapsed)
                print(f"    Run {i+1}: {elapsed:.2f}s")
            except Exception as e:
                print(f"    Run {i+1}: ERROR - {str(e)}")

        # Test 2: Warm model (repeated calls)
        print("  Warm model test...")
        # First, warm up the model
        ollama.generate(model=model, prompt="warm up", max_tokens=1, timeout=10)
        time.sleep(0.5)

        for i in range(5):
            start = time.time()
            try:
                response = ollama.generate(
                    model=model,
                    prompt=test_prompt,
                    temperature=0.0,
                    max_tokens=50,
                    timeout=30,
                )
                elapsed = time.time() - start
                timings["warm"].append(elapsed)
                print(f"    Run {i+1}: {elapsed:.2f}s")
            except Exception as e:
                print(f"    Run {i+1}: ERROR - {str(e)}")

            time.sleep(0.1)  # Very brief pause between calls

        # Test 3: Rapid consecutive calls
        print("  Consecutive calls test...")
        for i in range(10):
            start = time.time()
            try:
                response = ollama.generate(
                    model=model,
                    prompt=test_prompt
                    + f" (Call {i})",  # Slightly different to avoid caching
                    temperature=0.0,
                    max_tokens=50,
                    timeout=30,
                )
                elapsed = time.time() - start
                timings["consecutive"].append(elapsed)
            except Exception as e:
                print(f"    ERROR: {str(e)}")

        if timings["consecutive"]:
            print(f"    10 calls in {sum(timings['consecutive']):.2f}s total")

        # Calculate statistics
        results[model] = {
            "cold_start_avg": sum(timings["cold_start"]) / len(timings["cold_start"])
            if timings["cold_start"]
            else 0,
            "warm_avg": sum(timings["warm"]) / len(timings["warm"])
            if timings["warm"]
            else 0,
            "consecutive_avg": sum(timings["consecutive"]) / len(timings["consecutive"])
            if timings["consecutive"]
            else 0,
            "speedup_factor": None,
        }

        if results[model]["cold_start_avg"] > 0 and results[model]["warm_avg"] > 0:
            results[model]["speedup_factor"] = (
                results[model]["cold_start_avg"] / results[model]["warm_avg"]
            )

    # Print summary
    print("\n" + "=" * 60)
    print("PRELOADING EFFECT SUMMARY")
    print("=" * 60)
    print(
        f"{'Model':<25} {'Cold Start':<12} {'Warm':<12} {'Consecutive':<12} {'Speedup':<10}"
    )
    print("-" * 60)

    for model, data in results.items():
        print(
            f"{model:<25} {data['cold_start_avg']:<12.2f} {data['warm_avg']:<12.2f} "
            f"{data['consecutive_avg']:<12.2f} {data['speedup_factor'] or 0:<10.1f}x"
        )

    # Save results
    output_file = (
        Path("pdf_splitter/detection/experiments/results") / "preloading_test.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Recommendations
    print("\nRECOMMENDATIONS:")
    if any(r["speedup_factor"] and r["speedup_factor"] > 1.5 for r in results.values()):
        print("- Model preloading shows significant benefits (>1.5x speedup)")
        print("- Consider keeping models loaded in memory for production")
    else:
        print("- Model preloading shows minimal benefits")
        print("- Cold start penalty is relatively low")


if __name__ == "__main__":
    test_preloading_effect()
