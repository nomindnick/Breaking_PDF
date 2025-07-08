#!/usr/bin/env python3
"""
Analyze timing information from Ollama API calls.

This script extends the OllamaClient to capture and analyze timing data
from the API responses to understand performance characteristics.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests  # type: ignore


@dataclass
class TimingStats:
    """Statistics for API timing."""

    model: str
    prompt_length: int
    response_length: int
    total_duration_ms: float
    load_duration_ms: float
    prompt_eval_duration_ms: float
    eval_duration_ms: float
    prompt_eval_count: int
    eval_count: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens generated per second."""
        if self.eval_duration_ms > 0:
            return (self.eval_count / self.eval_duration_ms) * 1000
        return 0.0

    @property
    def prompt_tokens_per_second(self) -> float:
        """Calculate prompt tokens processed per second."""
        if self.prompt_eval_duration_ms > 0:
            return (self.prompt_eval_count / self.prompt_eval_duration_ms) * 1000
        return 0.0


class TimingAwareOllamaClient:
    """Ollama client that tracks timing information."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize client."""
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.timing_stats: List[TimingStats] = []

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 500,
        timeout: int = 30,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a response and capture timing information."""
        options = {"num_predict": max_tokens}
        if stop:
            options["stop"] = stop

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "options": options,
            "stream": False,
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            # Extract timing information if available
            if all(key in data for key in ["total_duration", "eval_duration"]):
                stats = TimingStats(
                    model=model,
                    prompt_length=len(prompt),
                    response_length=len(data.get("response", "")),
                    total_duration_ms=data.get("total_duration", 0)
                    / 1_000_000,  # Convert ns to ms
                    load_duration_ms=data.get("load_duration", 0) / 1_000_000,
                    prompt_eval_duration_ms=data.get("prompt_eval_duration", 0)
                    / 1_000_000,
                    eval_duration_ms=data.get("eval_duration", 0) / 1_000_000,
                    prompt_eval_count=data.get("prompt_eval_count", 0),
                    eval_count=data.get("eval_count", 0),
                )
                self.timing_stats.append(stats)

            return data
        except Exception as e:
            print(f"Ollama API error: {e}")
            return {"error": str(e)}

    def get_timing_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all timing data."""
        if not self.timing_stats:
            return {"error": "No timing data collected"}

        # Group by model
        model_stats = {}
        for stat in self.timing_stats:
            if stat.model not in model_stats:
                model_stats[stat.model] = []
            model_stats[stat.model].append(stat)

        summary = {}
        for model, stats in model_stats.items():
            total_calls = len(stats)
            avg_total_ms = sum(s.total_duration_ms for s in stats) / total_calls
            avg_eval_ms = sum(s.eval_duration_ms for s in stats) / total_calls
            avg_prompt_eval_ms = (
                sum(s.prompt_eval_duration_ms for s in stats) / total_calls
            )
            avg_tokens_per_sec = sum(s.tokens_per_second for s in stats) / total_calls

            # Exclude first call (model loading) for more accurate timing
            if total_calls > 1:
                stats_no_first = stats[1:]
                avg_total_no_load = sum(
                    s.total_duration_ms for s in stats_no_first
                ) / len(stats_no_first)
            else:
                avg_total_no_load = avg_total_ms

            summary[model] = {
                "total_calls": total_calls,
                "avg_total_duration_ms": avg_total_ms,
                "avg_total_duration_no_load_ms": avg_total_no_load,
                "avg_eval_duration_ms": avg_eval_ms,
                "avg_prompt_eval_duration_ms": avg_prompt_eval_ms,
                "avg_tokens_per_second": avg_tokens_per_sec,
                "first_call_load_time_ms": stats[0].load_duration_ms if stats else 0,
            }

        return summary

    def save_timing_data(self, filepath: Path):
        """Save timing data to JSON file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "stats": [
                {
                    "model": s.model,
                    "prompt_length": s.prompt_length,
                    "response_length": s.response_length,
                    "total_duration_ms": s.total_duration_ms,
                    "load_duration_ms": s.load_duration_ms,
                    "prompt_eval_duration_ms": s.prompt_eval_duration_ms,
                    "eval_duration_ms": s.eval_duration_ms,
                    "prompt_eval_count": s.prompt_eval_count,
                    "eval_count": s.eval_count,
                    "tokens_per_second": s.tokens_per_second,
                    "prompt_tokens_per_second": s.prompt_tokens_per_second,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in self.timing_stats
            ],
            "summary": self.get_timing_summary(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Timing data saved to {filepath}")


def run_timing_analysis(models: List[str], prompts: List[str], output_dir: Path):
    """Run timing analysis on different models and prompts."""
    client = TimingAwareOllamaClient()

    print("Running timing analysis...")
    print("=" * 60)

    for model in models:
        print(f"\nTesting model: {model}")

        # First call includes model loading
        print("  First call (includes model loading)...")
        response = client.generate(model, prompts[0], max_tokens=50)
        if "error" in response:
            print(f"  Error: {response['error']}")
            continue

        # Subsequent calls without loading
        for i, prompt in enumerate(prompts[1:], 1):
            print(f"  Call {i+1}/{len(prompts)}...")
            response = client.generate(model, prompt, max_tokens=50)
            time.sleep(0.1)  # Small delay between requests

    # Print summary
    print("\nTiming Summary:")
    print("=" * 60)
    summary = client.get_timing_summary()

    for model, stats in summary.items():
        print(f"\n{model}:")
        print(f"  Total calls: {stats['total_calls']}")
        print(f"  Avg total duration: {stats['avg_total_duration_ms']:.1f} ms")
        print(
            f"  Avg duration (no load): {stats['avg_total_duration_no_load_ms']:.1f} ms"
        )
        print(f"  Avg eval duration: {stats['avg_eval_duration_ms']:.1f} ms")
        print(f"  Avg tokens/second: {stats['avg_tokens_per_second']:.1f}")
        print(f"  First call load time: {stats['first_call_load_time_ms']:.1f} ms")

    # Save results
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"timing_analysis_{timestamp}.json"
    client.save_timing_data(output_file)

    return client


if __name__ == "__main__":
    # Test prompts of varying lengths
    test_prompts = [
        "Is this a document boundary? Answer with Yes or No.",
        "Analyze these two pages and determine if they are from the same document or different documents. Page 1 ends with 'Thank you' and Page 2 starts with 'Invoice #123'.",
        "You are analyzing pages from a PDF document. The bottom of page 1 contains: 'This concludes our analysis of the financial data for Q3 2024.' The top of page 2 contains: 'MEMORANDUM TO: All Staff FROM: HR Department'. Determine if this is a document boundary.",
        "Consider the following context: We have two consecutive pages from a PDF file. Your task is to determine whether these pages belong to the same document or if there is a document boundary between them. Look for clear indicators such as signatures, headers, document endings, or topic changes. Page 1 bottom: 'For more information, please contact us.' Page 2 top: 'Chapter 3: Implementation Details'. What is your determination?",
    ]

    # Models to test
    test_models = ["qwen3:0.6b", "granite3.3:2b", "phi3:mini"]

    # Output directory
    output_dir = Path("pdf_splitter/detection/experiments/results/timing")

    # Run analysis
    run_timing_analysis(test_models, test_prompts, output_dir)
