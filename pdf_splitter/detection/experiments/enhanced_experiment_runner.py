#!/usr/bin/env python3
"""
Enhanced experiment runner with detailed timing tracking.

This module extends the existing experiment runner to capture and analyze
timing data from Ollama API responses.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pdf_splitter.detection.experiments.experiment_runner import (
    ExperimentConfig,
    ExperimentResult,
    LLMExperimentRunner,
    OllamaClient,
    ProcessedPage,
)


@dataclass
class DetailedTiming:
    """Detailed timing information for an API call."""

    page_number: int
    prompt_length: int
    response_length: int
    total_duration_ms: float
    load_duration_ms: float
    prompt_eval_duration_ms: float
    eval_duration_ms: float
    prompt_eval_count: int
    eval_count: int
    tokens_per_second: float

    @classmethod
    def from_ollama_response(
        cls, response: Dict[str, Any], page_number: int, prompt: str
    ) -> Optional["DetailedTiming"]:
        """Create DetailedTiming from Ollama API response."""
        if "total_duration" not in response:
            return None

        eval_count = response.get("eval_count", 0)
        eval_duration_ms = response.get("eval_duration", 0) / 1_000_000

        return cls(
            page_number=page_number,
            prompt_length=len(prompt),
            response_length=len(response.get("response", "")),
            total_duration_ms=response.get("total_duration", 0) / 1_000_000,
            load_duration_ms=response.get("load_duration", 0) / 1_000_000,
            prompt_eval_duration_ms=response.get("prompt_eval_duration", 0) / 1_000_000,
            eval_duration_ms=eval_duration_ms,
            prompt_eval_count=response.get("prompt_eval_count", 0),
            eval_count=eval_count,
            tokens_per_second=(eval_count / eval_duration_ms * 1000)
            if eval_duration_ms > 0
            else 0,
        )


@dataclass
class EnhancedExperimentResult(ExperimentResult):
    """Experiment result with detailed timing information."""

    timing_details: List[DetailedTiming] = field(default_factory=list)

    # Additional timing metrics
    avg_api_latency_ms: float = 0.0  # Average time for API calls (excluding first)
    first_call_load_time_ms: float = 0.0
    total_api_time_ms: float = 0.0
    avg_tokens_per_second: float = 0.0

    def calculate_timing_metrics(self):
        """Calculate timing metrics from detailed timing data."""
        if not self.timing_details:
            return

        # First call load time
        self.first_call_load_time_ms = self.timing_details[0].load_duration_ms

        # Total API time
        self.total_api_time_ms = sum(t.total_duration_ms for t in self.timing_details)

        # Average latency (excluding first call which includes model loading)
        if len(self.timing_details) > 1:
            subsequent_calls = self.timing_details[1:]
            self.avg_api_latency_ms = sum(
                t.total_duration_ms for t in subsequent_calls
            ) / len(subsequent_calls)
        else:
            self.avg_api_latency_ms = self.timing_details[0].total_duration_ms

        # Average tokens per second
        valid_timings = [t for t in self.timing_details if t.tokens_per_second > 0]
        if valid_timings:
            self.avg_tokens_per_second = sum(
                t.tokens_per_second for t in valid_timings
            ) / len(valid_timings)


class TimingAwareOllamaClient(OllamaClient):
    """Ollama client that captures timing information."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__(base_url)
        self.timing_data: List[DetailedTiming] = []

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 500,
        timeout: int = 30,
        stop: Optional[List[str]] = None,
        page_number: int = 0,  # Added for tracking
    ) -> Dict[str, Any]:
        """Generate a response and capture timing information."""
        response = super().generate(
            model, prompt, temperature, max_tokens, timeout, stop
        )

        # Extract timing if available
        timing = DetailedTiming.from_ollama_response(response, page_number, prompt)
        if timing:
            self.timing_data.append(timing)

        return response


class EnhancedLLMExperimentRunner(LLMExperimentRunner):
    """Experiment runner with enhanced timing capabilities."""

    def __init__(self, results_dir: Optional[Path] = None):
        super().__init__(results_dir)
        # Replace the ollama client with timing-aware version
        self.ollama = TimingAwareOllamaClient()

    def run_experiment(
        self,
        config: ExperimentConfig,
        pages: List[ProcessedPage],
        ground_truth: List[int],
    ) -> EnhancedExperimentResult:
        """Run experiment with timing tracking."""
        # Clear previous timing data
        self.ollama.timing_data = []

        # Run the base experiment
        base_result = super().run_experiment(config, pages, ground_truth)

        # Create enhanced result with timing data
        enhanced_result = EnhancedExperimentResult(
            config=base_result.config,
            true_boundaries=base_result.true_boundaries,
            predicted_boundaries=base_result.predicted_boundaries,
            precision=base_result.precision,
            recall=base_result.recall,
            f1_score=base_result.f1_score,
            total_time=base_result.total_time,
            avg_time_per_boundary=base_result.avg_time_per_boundary,
            avg_time_per_page=base_result.avg_time_per_page,
            boundary_results=base_result.boundary_results,
            errors=base_result.errors,
            total_pages=base_result.total_pages,
            model_responses=base_result.model_responses,
            timing_details=self.ollama.timing_data.copy(),
        )

        # Calculate timing metrics
        enhanced_result.calculate_timing_metrics()

        return enhanced_result

    def _run_synthetic_pairs_strategy(
        self,
        config: ExperimentConfig,
        pages: List[ProcessedPage],
        result: ExperimentResult,
    ) -> List[BoundaryResult]:
        """Override to add page number tracking for timing."""
        predictions = []

        for i in range(len(pages) - 1):
            page1 = pages[i]
            page2 = pages[i + 1]

            # Extract page text
            page1_text = page1.text.strip()
            page2_text = page2.text.strip()

            page1_bottom = page1_text[-300:] if len(page1_text) > 300 else page1_text
            page2_top = page2_text[:300] if len(page2_text) > 300 else page2_text

            # Format prompt
            template = self.prompt_templates.get(
                config.prompt_template, self.prompt_templates["default"]
            )

            if "{page1_bottom}" in template and "{page2_top}" in template:
                prompt = template.format(
                    page1_bottom=page1_bottom,
                    page2_top=page2_top,
                    page1=page1_bottom,
                    page2=page2_top,
                )
            else:
                prompt = (
                    f"Page {page1.page_number} ends with:\n{page1_bottom}\n\n"
                    f"Page {page2.page_number} begins with:\n{page2_top}\n\n"
                    "Are these pages from the Same Document or Different Documents?"
                )

            # Call LLM with page number for timing tracking
            response = self.ollama.generate(
                model=config.model,
                prompt=prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                stop=config.stop_tokens if hasattr(config, "stop_tokens") else None,
                page_number=page2.page_number,  # Track which page this is for
            )

            # Process response
            boundary = self._process_pair_response(
                response, page2.page_number, result, config
            )
            if boundary:
                predictions.append(boundary)

        return predictions

    def save_enhanced_result(
        self, result: EnhancedExperimentResult, filename_suffix: str = ""
    ):
        """Save enhanced result with timing information."""
        timestamp = result.config.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{result.config.name}_{timestamp}{filename_suffix}.json"
        filepath = self.results_dir / filename

        # Prepare data for serialization
        result_data = {
            "config": {
                "name": result.config.name,
                "model": result.config.model,
                "strategy": result.config.strategy,
                "timestamp": result.config.timestamp.isoformat(),
            },
            "metrics": {
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "total_pages": result.total_pages,
                "true_boundaries": result.true_boundaries,
                "predicted_boundaries": result.predicted_boundaries,
            },
            "timing": {
                "total_time": result.total_time,
                "avg_time_per_page": result.avg_time_per_page,
                "avg_time_per_boundary": result.avg_time_per_boundary,
                "total_api_time_ms": result.total_api_time_ms,
                "avg_api_latency_ms": result.avg_api_latency_ms,
                "first_call_load_time_ms": result.first_call_load_time_ms,
                "avg_tokens_per_second": result.avg_tokens_per_second,
            },
            "timing_details": [
                {
                    "page": t.page_number,
                    "prompt_length": t.prompt_length,
                    "response_length": t.response_length,
                    "total_ms": t.total_duration_ms,
                    "eval_ms": t.eval_duration_ms,
                    "tokens_per_sec": t.tokens_per_second,
                }
                for t in result.timing_details
            ],
            "errors": result.errors,
        }

        with open(filepath, "w") as f:
            json.dump(result_data, f, indent=2)

        print(f"Enhanced results saved to {filepath}")

        # Also create a timing summary
        self._save_timing_summary(result, timestamp)

    def _save_timing_summary(self, result: EnhancedExperimentResult, timestamp: str):
        """Save a summary of timing metrics."""
        summary_file = self.results_dir / f"timing_summary_{timestamp}.txt"

        with open(summary_file, "w") as f:
            f.write(f"Timing Summary for {result.config.name}\n")
            f.write(f"Model: {result.config.model}\n")
            f.write("=" * 60 + "\n\n")

            f.write("Overall Performance:\n")
            f.write(f"  Total experiment time: {result.total_time:.1f} seconds\n")
            f.write(f"  Total API time: {result.total_api_time_ms:.1f} ms\n")
            f.write(f"  Average API latency: {result.avg_api_latency_ms:.1f} ms/call\n")
            f.write(
                f"  First call load time: {result.first_call_load_time_ms:.1f} ms\n"
            )
            f.write(f"  Average tokens/second: {result.avg_tokens_per_second:.1f}\n\n")

            f.write("Per-page Performance:\n")
            f.write(
                f"  Average time per page: {result.avg_time_per_page:.2f} seconds\n"
            )
            f.write(
                f"  Average time per boundary: {result.avg_time_per_boundary:.2f} seconds\n\n"
            )

            if result.timing_details:
                f.write("Detailed API Calls:\n")
                for i, timing in enumerate(
                    result.timing_details[:10]
                ):  # First 10 calls
                    f.write(f"  Call {i+1} (Page {timing.page_number}):\n")
                    f.write(f"    Total: {timing.total_duration_ms:.1f} ms\n")
                    f.write(f"    Eval: {timing.eval_duration_ms:.1f} ms\n")
                    f.write(f"    Tokens/sec: {timing.tokens_per_second:.1f}\n")

        print(f"Timing summary saved to {summary_file}")


# Example usage
if __name__ == "__main__":
    from pdf_splitter.detection.experiments.synthetic_boundary_tests import (
        SyntheticBoundaryTester,
    )

    # Create synthetic test data
    tester = SyntheticBoundaryTester()

    # Convert test cases to ProcessedPage format
    pages = []
    for i, case in enumerate(tester.test_cases[:5]):  # Just first 5 cases
        # Page 1
        pages.append(
            ProcessedPage(
                page_number=i * 2 + 1,
                text=case.page1_bottom,
                ocr_performed=False,
                confidence_score=1.0,
            )
        )
        # Page 2
        pages.append(
            ProcessedPage(
                page_number=i * 2 + 2,
                text=case.page2_top,
                ocr_performed=False,
                confidence_score=1.0,
            )
        )

    # Ground truth boundaries (simplified)
    ground_truth = [2, 4, 6, 8, 10]

    # Run enhanced experiment
    runner = EnhancedLLMExperimentRunner()

    config = ExperimentConfig(
        name="timing_test",
        model="qwen3:0.6b",
        strategy="synthetic_pairs",
        prompt_template="baseline",
    )

    print("Running enhanced experiment with timing...")
    result = runner.run_experiment(config, pages, ground_truth)

    # Save results
    runner.save_enhanced_result(result, "_with_timing")

    print(f"\nExperiment complete!")
    print(f"F1 Score: {result.f1_score:.3f}")
    print(f"Average API latency: {result.avg_api_latency_ms:.1f} ms")
    print(f"Average tokens/second: {result.avg_tokens_per_second:.1f}")
