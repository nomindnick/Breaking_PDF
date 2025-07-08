#!/usr/bin/env python3
"""
Comprehensive benchmarking of Ollama model performance.

This script tests different models with various prompt sizes to understand:
1. Model loading times
2. Prompt processing speed
3. Token generation speed
4. Overall latency characteristics
"""

import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.timing_analysis import \
    TimingAwareOllamaClient


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    models: List[str]
    prompt_sizes: List[int]  # Approximate character counts
    max_tokens: int = 50
    repetitions: int = 5
    warmup_runs: int = 2


class OllamaPerformanceBenchmark:
    """Benchmark Ollama model performance."""

    def __init__(self):
        self.client = TimingAwareOllamaClient()
        self.results_dir = Path("pdf_splitter/detection/experiments/results/benchmarks")
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Sample prompts of different sizes
        self.prompt_templates = {
            50: "Is this a document boundary? Answer Yes or No.",
            150: "Analyze these two pages and determine if they are from the same document. Page 1 ends with 'Thank you' and Page 2 starts with 'Invoice #123'. Answer only 'Same Document' or 'Different Documents'.",
            300: "You are analyzing pages from a PDF document. The bottom of page 1 contains: 'This concludes our analysis of the financial data for Q3 2024. For questions, contact finance@company.com.' The top of page 2 contains: 'MEMORANDUM TO: All Staff FROM: HR Department DATE: October 15, 2024'. Determine if this is a document boundary. Consider signatures, headers, document endings, and topic changes.",
            500: """You are a document boundary detection system. Your task is to determine whether two consecutive pages from a PDF file belong to the same document or if there is a document boundary between them.

Page 1 ends with:
"Thank you for your consideration of this proposal. We look forward to working with you on this project. Should you have any questions, please don't hesitate to contact us.

Sincerely,
John Smith
Project Manager"

Page 2 begins with:
"Attachment A: Technical Specifications

System Requirements:
- Operating System: Windows 10 or later
- Memory: 8GB RAM minimum"

Analyze these pages and determine: Same Document or Different Documents?""",
            1000: """You are an expert document analyst specializing in identifying document boundaries in concatenated PDF files. Your task is to analyze page transitions and determine whether consecutive pages belong to the same document or represent a boundary between different documents.

Consider the following factors in your analysis:
1. Document structure markers (headers, footers, page numbers)
2. Content continuity (does the text flow naturally?)
3. Formatting changes (font, layout, style)
4. Document type indicators (letter, memo, report, invoice)
5. Explicit endings (signatures, "End of Document", closing statements)
6. Explicit beginnings (letterheads, "Page 1", document titles)

IMPORTANT: Not every page break is a document boundary. Many documents span multiple pages. Only identify a boundary when there is clear evidence of a new, separate document beginning.

Now analyze this specific case:

Page 1 ends with:
"This concludes our quarterly financial review. The results demonstrate strong growth across all divisions, with particularly impressive performance in the Asia-Pacific region. We remain optimistic about future prospects and will continue to invest in strategic initiatives.

For detailed financial statements, please refer to the appendices.

Respectfully submitted,

Sarah Johnson
Chief Financial Officer
October 30, 2024"

Page 2 begins with:
"CONFIDENTIAL MEMORANDUM

TO: All Department Heads
FROM: Human Resources
DATE: November 1, 2024
RE: Updated Holiday Schedule and PTO Policy

Effective immediately, the following changes to our holiday schedule and paid time off policy will be implemented:"

Based on your analysis, are these pages from the Same Document or Different Documents? Provide your answer in the exact format requested.""",
        }

    def generate_prompt(self, target_size: int) -> str:
        """Generate or select a prompt of approximately the target size."""
        # Find the closest template
        sizes = sorted(self.prompt_templates.keys())
        closest_size = min(sizes, key=lambda x: abs(x - target_size))

        prompt = self.prompt_templates[closest_size]

        # Adjust prompt length if needed
        if len(prompt) < target_size:
            # Pad with context
            padding = "\n\nAdditional context: " + "This is padding text. " * (
                (target_size - len(prompt)) // 20
            )
            prompt += padding[: target_size - len(prompt)]
        elif len(prompt) > target_size:
            # Truncate
            prompt = prompt[:target_size]

        return prompt

    def benchmark_model(self, model: str, config: BenchmarkConfig) -> Dict:
        """Benchmark a single model."""
        print(f"\nBenchmarking {model}...")
        print("-" * 40)

        results = {
            "model": model,
            "prompt_sizes": {},
            "model_load_time_ms": 0,
            "errors": [],
        }

        # First, do a cold start to measure model loading
        print("  Measuring model load time...")
        cold_prompt = self.generate_prompt(50)
        cold_response = self.client.generate(model, cold_prompt, max_tokens=10)

        if "error" in cold_response:
            results["errors"].append(f"Failed to load model: {cold_response['error']}")
            return results

        if self.client.timing_stats:
            results["model_load_time_ms"] = self.client.timing_stats[
                -1
            ].load_duration_ms

        # Warmup runs
        print(f"  Running {config.warmup_runs} warmup iterations...")
        for _ in range(config.warmup_runs):
            warmup_prompt = self.generate_prompt(150)
            self.client.generate(model, warmup_prompt, max_tokens=config.max_tokens)
            time.sleep(0.1)

        # Benchmark different prompt sizes
        for prompt_size in config.prompt_sizes:
            print(f"  Testing prompt size ~{prompt_size} chars...")

            prompt = self.generate_prompt(prompt_size)
            actual_size = len(prompt)

            timings = []

            for i in range(config.repetitions):
                response = self.client.generate(
                    model=model,
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=config.max_tokens,
                    timeout=60,
                )

                if "error" in response:
                    results["errors"].append(
                        f"Prompt size {prompt_size}, iteration {i+1}: {response['error']}"
                    )
                    continue

                # Get the latest timing stats
                if self.client.timing_stats:
                    latest_timing = self.client.timing_stats[-1]
                    timings.append(
                        {
                            "total_ms": latest_timing.total_duration_ms,
                            "prompt_eval_ms": latest_timing.prompt_eval_duration_ms,
                            "eval_ms": latest_timing.eval_duration_ms,
                            "tokens_per_sec": latest_timing.tokens_per_second,
                            "prompt_tokens": latest_timing.prompt_eval_count,
                            "response_tokens": latest_timing.eval_count,
                        }
                    )

                # Small delay between requests
                time.sleep(0.2)

            # Calculate statistics
            if timings:
                results["prompt_sizes"][prompt_size] = {
                    "actual_prompt_size": actual_size,
                    "iterations": len(timings),
                    "avg_total_ms": statistics.mean(t["total_ms"] for t in timings),
                    "std_total_ms": statistics.stdev(t["total_ms"] for t in timings)
                    if len(timings) > 1
                    else 0,
                    "min_total_ms": min(t["total_ms"] for t in timings),
                    "max_total_ms": max(t["total_ms"] for t in timings),
                    "avg_prompt_eval_ms": statistics.mean(
                        t["prompt_eval_ms"] for t in timings
                    ),
                    "avg_eval_ms": statistics.mean(t["eval_ms"] for t in timings),
                    "avg_tokens_per_sec": statistics.mean(
                        t["tokens_per_sec"] for t in timings
                    ),
                    "avg_prompt_tokens": statistics.mean(
                        t["prompt_tokens"] for t in timings
                    ),
                    "avg_response_tokens": statistics.mean(
                        t["response_tokens"] for t in timings
                    ),
                    "raw_timings": timings,
                }

        return results

    def run_benchmark(self, config: BenchmarkConfig) -> Dict:
        """Run the complete benchmark."""
        print("=" * 60)
        print("Ollama Performance Benchmark")
        print("=" * 60)

        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "models": config.models,
                "prompt_sizes": config.prompt_sizes,
                "max_tokens": config.max_tokens,
                "repetitions": config.repetitions,
                "warmup_runs": config.warmup_runs,
            },
            "results": {},
        }

        # Clear timing stats
        self.client.timing_stats = []

        # Benchmark each model
        for model in config.models:
            model_results = self.benchmark_model(model, config)
            benchmark_results["results"][model] = model_results

        # Save results
        self.save_results(benchmark_results)

        # Print summary
        self.print_summary(benchmark_results)

        return benchmark_results

    def save_results(self, results: Dict):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Full results
        full_results_file = self.results_dir / f"benchmark_full_{timestamp}.json"
        with open(full_results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nFull results saved to: {full_results_file}")

        # Summary CSV for easy analysis
        summary_file = self.results_dir / f"benchmark_summary_{timestamp}.csv"
        with open(summary_file, "w") as f:
            # Header
            f.write(
                "Model,Prompt Size,Avg Total (ms),Std Dev (ms),Min (ms),Max (ms),"
                "Avg Prompt Eval (ms),Avg Generation (ms),Tokens/sec\n"
            )

            # Data
            for model, model_results in results["results"].items():
                if "errors" in model_results and model_results["errors"]:
                    continue

                for prompt_size, data in model_results["prompt_sizes"].items():
                    f.write(
                        f"{model},{prompt_size},"
                        f"{data['avg_total_ms']:.1f},{data['std_total_ms']:.1f},"
                        f"{data['min_total_ms']:.1f},{data['max_total_ms']:.1f},"
                        f"{data['avg_prompt_eval_ms']:.1f},{data['avg_eval_ms']:.1f},"
                        f"{data['avg_tokens_per_sec']:.1f}\n"
                    )

        print(f"Summary CSV saved to: {summary_file}")

    def print_summary(self, results: Dict):
        """Print a summary of benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        for model, model_results in results["results"].items():
            print(f"\n{model}:")
            print(f"  Model load time: {model_results['model_load_time_ms']:.1f} ms")

            if model_results["errors"]:
                print(f"  Errors: {len(model_results['errors'])}")
                continue

            print("\n  Prompt Size | Avg Total | Std Dev | Tokens/sec")
            print("  " + "-" * 50)

            for prompt_size in sorted(model_results["prompt_sizes"].keys()):
                data = model_results["prompt_sizes"][prompt_size]
                print(
                    f"  {prompt_size:>10} | {data['avg_total_ms']:>9.1f} | "
                    f"{data['std_total_ms']:>7.1f} | {data['avg_tokens_per_sec']:>10.1f}"
                )

        # Model comparison for 150-char prompts (typical use case)
        print("\n" + "=" * 60)
        print("MODEL COMPARISON (150-char prompts)")
        print("=" * 60)
        print("Model               | Avg Latency | Tokens/sec | Load Time")
        print("-" * 60)

        for model, model_results in results["results"].items():
            if 150 in model_results["prompt_sizes"]:
                data = model_results["prompt_sizes"][150]
                print(
                    f"{model:<18} | {data['avg_total_ms']:>11.1f} | "
                    f"{data['avg_tokens_per_sec']:>10.1f} | "
                    f"{model_results['model_load_time_ms']:>9.1f}"
                )


def main():
    """Run the benchmark with default configuration."""
    config = BenchmarkConfig(
        models=["qwen3:0.6b", "granite3.3:2b", "phi3:mini"],
        prompt_sizes=[50, 150, 300, 500, 1000],
        max_tokens=50,
        repetitions=5,
        warmup_runs=2,
    )

    benchmark = OllamaPerformanceBenchmark()
    benchmark.run_benchmark(config)


if __name__ == "__main__":
    main()
