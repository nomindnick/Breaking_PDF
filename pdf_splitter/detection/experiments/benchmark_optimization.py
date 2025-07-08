#!/usr/bin/env python3
"""Comprehensive benchmark of model and prompt combinations for optimization."""

import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.enhanced_synthetic_tests import (
    EnhancedSyntheticTester,
)
from pdf_splitter.detection.experiments.experiment_runner import OllamaClient


class OptimizationBenchmark:
    """Benchmark different models and prompts for speed/accuracy optimization."""

    def __init__(self):
        self.synthetic_tester = EnhancedSyntheticTester()
        self.ollama = OllamaClient()
        self.results_dir = Path("pdf_splitter/detection/experiments/results")
        self.results_dir.mkdir(exist_ok=True)

        # Model configurations
        self.models = {
            "gemma3:1b-it-q4_K_M": {"type": "gemma", "size": "815MB"},
            "qwen3:0.6b": {"type": "generic", "size": "522MB"},
            "granite3.3:2b": {"type": "generic", "size": "1.5GB"},
            "deepseek-r1:1.5b": {"type": "generic", "size": "1.1GB"},
            "gemma3:latest": {"type": "gemma", "size": "3.3GB"},  # baseline
        }

        # Prompt configurations
        self.prompts = {
            "optimal": {
                "gemma": "prompts/gemma3_optimal.txt",
                "generic": None,  # Skip for generic models
            },
            "compressed": {
                "gemma": "prompts/gemma3_compressed.txt",
                "generic": "prompts/generic_minimal.txt",  # Use minimal for generic
            },
            "minimal": {
                "gemma": "prompts/gemma3_minimal.txt",
                "generic": "prompts/generic_minimal.txt",
            },
        }

    def load_prompt(self, prompt_file: str) -> str:
        """Load a prompt template."""
        prompt_path = Path(__file__).parent / prompt_file
        with open(prompt_path, "r") as f:
            return f.read()

    def test_model_prompt_combo(
        self,
        model: str,
        prompt_name: str,
        prompt_template: str,
        test_cases: List,
        warm_up: bool = True,
    ) -> Dict:
        """Test a specific model/prompt combination."""
        results = {
            "model": model,
            "prompt": prompt_name,
            "correct": 0,
            "total": 0,
            "timings": [],
            "cold_start": None,
            "errors": 0,
        }

        # Determine stop tokens based on model type
        model_type = self.models[model]["type"]
        stop_tokens = ["<end_of_turn>"] if model_type == "gemma" else ["</answer>"]

        # Warm-up run (not counted in timings)
        if warm_up and test_cases:
            print(f"  Warming up model...")
            warm_up_case = test_cases[0]
            prompt = prompt_template.format(
                page1_bottom=warm_up_case.page1_bottom, page2_top=warm_up_case.page2_top
            )

            try:
                start = time.time()
                self.ollama.generate(
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=100,
                    stop=stop_tokens,
                    timeout=30,
                )
                results["cold_start"] = time.time() - start
            except:
                pass

        # Test all cases
        for i, case in enumerate(test_cases):
            prompt = prompt_template.format(
                page1_bottom=case.page1_bottom, page2_top=case.page2_top
            )

            start_time = time.time()
            try:
                response = self.ollama.generate(
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=100,
                    stop=stop_tokens,
                    timeout=30,
                )
                elapsed = time.time() - start_time
                results["timings"].append(elapsed)

                # Extract answer
                response_text = response.get("response", "")
                prediction = None

                if "<answer>" in response_text:
                    start_idx = response_text.find("<answer>") + len("<answer>")
                    end_idx = response_text.find("</answer>")
                    if end_idx > start_idx:
                        answer = response_text[start_idx:end_idx].strip().upper()
                        if "DIFFERENT" in answer:
                            prediction = "Different"
                        elif "SAME" in answer:
                            prediction = "Same"

                if prediction == case.expected:
                    results["correct"] += 1

                results["total"] += 1

                # Progress indicator
                if (i + 1) % 5 == 0:
                    print(f"    Processed {i+1}/{len(test_cases)} cases...")

            except Exception as e:
                results["errors"] += 1
                print(f"    Error on case {i+1}: {str(e)}")

        return results

    def calculate_metrics(self, results: Dict) -> Dict:
        """Calculate performance metrics from results."""
        if not results["timings"]:
            return results

        results["accuracy"] = (
            results["correct"] / results["total"] if results["total"] > 0 else 0
        )
        results["avg_time"] = statistics.mean(results["timings"])
        results["median_time"] = statistics.median(results["timings"])
        results["std_time"] = (
            statistics.stdev(results["timings"]) if len(results["timings"]) > 1 else 0
        )
        results["min_time"] = min(results["timings"])
        results["max_time"] = max(results["timings"])
        results["p95_time"] = (
            sorted(results["timings"])[int(len(results["timings"]) * 0.95)]
            if results["timings"]
            else 0
        )

        return results

    def run_benchmark(self, num_cases: int = 26):
        """Run comprehensive benchmark."""
        test_cases = self.synthetic_tester.test_cases[:num_cases]
        all_results = []

        print(f"Running optimization benchmark with {num_cases} test cases")
        print("=" * 80)

        for model_name, model_info in self.models.items():
            print(f"\nTesting {model_name} ({model_info['size']})...")

            for prompt_name, prompt_paths in self.prompts.items():
                # Get appropriate prompt for model type
                prompt_file = prompt_paths.get(model_info["type"])
                if not prompt_file:
                    continue

                print(f"  Prompt: {prompt_name}")

                try:
                    prompt_template = self.load_prompt(prompt_file)
                    results = self.test_model_prompt_combo(
                        model_name,
                        prompt_name,
                        prompt_template,
                        test_cases,
                        warm_up=True,
                    )

                    results = self.calculate_metrics(results)
                    all_results.append(results)

                    # Print summary
                    print(f"    Accuracy: {results['accuracy']:.1%}")
                    print(
                        f"    Avg time: {results['avg_time']:.2f}s (±{results['std_time']:.2f}s)"
                    )
                    print(f"    P95 time: {results['p95_time']:.2f}s")
                    if results["cold_start"]:
                        print(f"    Cold start: {results['cold_start']:.2f}s")

                except Exception as e:
                    print(f"    ERROR: {str(e)}")

        return all_results

    def save_results(self, results: List[Dict]):
        """Save benchmark results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"optimization_benchmark_{timestamp}.json"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_test_cases": len(self.synthetic_tester.test_cases),
            "results": results,
            "best_by_speed": min(
                results, key=lambda x: x.get("avg_time", float("inf"))
            ),
            "best_by_accuracy": max(results, key=lambda x: x.get("accuracy", 0)),
            "best_balanced": max(
                results,
                key=lambda x: x.get("accuracy", 0) / max(x.get("avg_time", 1), 0.1),
            ),
        }

        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        return output_file

    def print_summary_table(self, results: List[Dict]):
        """Print a formatted summary table."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Sort by average time
        sorted_results = sorted(results, key=lambda x: x.get("avg_time", float("inf")))

        print(
            f"{'Model':<25} {'Prompt':<12} {'Accuracy':<10} {'Avg Time':<10} {'P95 Time':<10} {'Throughput':<12}"
        )
        print("-" * 80)

        for r in sorted_results:
            if r.get("avg_time", 0) > 0:
                throughput = 1 / r["avg_time"]
            else:
                throughput = 0

            print(
                f"{r['model']:<25} {r['prompt']:<12} {r.get('accuracy', 0):<10.1%} "
                f"{r.get('avg_time', 0):<10.2f} {r.get('p95_time', 0):<10.2f} {throughput:<12.2f}"
            )

        # Find best combinations
        print("\n" + "-" * 80)
        print("RECOMMENDATIONS:")

        # Best for speed
        fastest = min(results, key=lambda x: x.get("avg_time", float("inf")))
        print(
            f"Fastest: {fastest['model']} + {fastest['prompt']} "
            f"({fastest.get('avg_time', 0):.2f}s avg, {fastest.get('accuracy', 0):.1%} accuracy)"
        )

        # Best for accuracy
        most_accurate = max(results, key=lambda x: x.get("accuracy", 0))
        print(
            f"Most Accurate: {most_accurate['model']} + {most_accurate['prompt']} "
            f"({most_accurate.get('accuracy', 0):.1%} accuracy, {most_accurate.get('avg_time', 0):.2f}s avg)"
        )

        # Best balanced (high accuracy, reasonable speed)
        balanced = [
            r
            for r in results
            if r.get("accuracy", 0) >= 0.7 and r.get("avg_time", float("inf")) < 5
        ]
        if balanced:
            best_balanced = min(balanced, key=lambda x: x.get("avg_time", float("inf")))
            print(
                f"Best Balanced: {best_balanced['model']} + {best_balanced['prompt']} "
                f"({best_balanced.get('accuracy', 0):.1%} accuracy, {best_balanced.get('avg_time', 0):.2f}s avg)"
            )


def main():
    """Run the optimization benchmark."""
    benchmark = OptimizationBenchmark()

    # Run benchmark
    results = benchmark.run_benchmark(num_cases=26)  # Use all 26 test cases

    # Save results
    output_file = benchmark.save_results(results)

    # Print summary
    benchmark.print_summary_table(results)

    print(f"\n\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
