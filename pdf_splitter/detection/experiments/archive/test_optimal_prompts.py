#!/usr/bin/env python3
"""
Test script for optimal prompts based on prompt engineering research.

This script specifically tests the model-specific optimal prompts (phi4_optimal, gemma3_optimal)
and Chain-of-Draft prompts against the baseline prompts to measure improvements.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.enhanced_synthetic_tests import \
    EnhancedSyntheticTester
from pdf_splitter.detection.experiments.experiment_runner import (
    ExperimentConfig, LLMExperimentRunner, OllamaClient)
from pdf_splitter.detection.experiments.model_formatting import ModelFormatter


class OptimalPromptTester:
    """Test harness for evaluating optimal prompts from research."""

    def __init__(self):
        self.synthetic_tester = EnhancedSyntheticTester()
        self.experiment_runner = LLMExperimentRunner()
        self.ollama = OllamaClient()
        self.formatter = ModelFormatter()
        self.results_dir = Path("pdf_splitter/detection/experiments/results")
        self.results_dir.mkdir(exist_ok=True)

    def test_optimal_prompts(
        self,
        models: List[str],
        baseline_prompts: List[str] = None,
        difficulty_levels: List[int] = None,
    ) -> Dict:
        """
        Test optimal prompts against baseline prompts.

        Args:
            models: List of model names to test
            baseline_prompts: List of baseline prompt names to compare against
            difficulty_levels: List of difficulty levels to test (default: all)

        Returns:
            Dictionary with comprehensive results
        """
        if baseline_prompts is None:
            baseline_prompts = ["A1_asymmetric", "D1_conservative_few_shot"]

        if difficulty_levels is None:
            difficulty_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        results = {"timestamp": datetime.now().isoformat(), "models": {}, "summary": {}}

        for model in models:
            print(f"\n{'='*60}")
            print(f"Testing model: {model}")
            print(f"{'='*60}")

            model_results = {
                "baseline": {},
                "optimal": {},
                "cod": {},
                "improvements": {},
            }

            # Determine which optimal prompt to use
            model_family = self.formatter.detect_model_family(model)
            if model_family == "phi":
                optimal_prompt = "phi4_optimal"
            elif model_family == "gemma":
                optimal_prompt = "gemma3_optimal"
            else:
                optimal_prompt = None

            # Test baseline prompts
            print("\nTesting baseline prompts...")
            for prompt_name in baseline_prompts:
                print(f"  - {prompt_name}")
                prompt_results = self._test_single_prompt(
                    model, prompt_name, difficulty_levels
                )
                model_results["baseline"][prompt_name] = prompt_results

            # Test optimal prompt if available
            if optimal_prompt:
                print(f"\nTesting optimal prompt: {optimal_prompt}")
                optimal_results = self._test_single_prompt(
                    model, optimal_prompt, difficulty_levels
                )
                model_results["optimal"][optimal_prompt] = optimal_results

            # Test CoD prompts
            print("\nTesting Chain-of-Draft prompts...")
            for cod_prompt in ["E1_cod_reasoning", "E2_cod_minimal"]:
                print(f"  - {cod_prompt}")
                cod_results = self._test_single_prompt(
                    model, cod_prompt, difficulty_levels
                )
                model_results["cod"][cod_prompt] = cod_results

            # Calculate improvements
            model_results["improvements"] = self._calculate_improvements(model_results)

            results["models"][model] = model_results

        # Generate summary
        results["summary"] = self._generate_summary(results)

        # Save results
        self._save_results(results)

        return results

    def _test_single_prompt(
        self, model: str, prompt_name: str, difficulty_levels: List[int]
    ) -> Dict:
        """Test a single prompt across difficulty levels."""
        start_time = time.time()

        # Get prompt info
        prompts = self.synthetic_tester.get_user_specified_prompts()
        prompt_info = prompts.get(prompt_name)

        if not prompt_info or not prompt_info.get("template"):
            return {"error": f"Prompt {prompt_name} not found or empty"}

        # Run tests
        results = {
            "prompt_name": prompt_name,
            "by_difficulty": {},
            "overall": {
                "correct": 0,
                "total": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
            "timing": {"total_time": 0.0, "avg_per_case": 0.0},
        }

        for difficulty in difficulty_levels:
            diff_cases = [
                c
                for c in self.synthetic_tester.test_cases
                if c.difficulty == difficulty
            ]

            if not diff_cases:
                continue

            diff_results = {
                "correct": 0,
                "total": len(diff_cases),
                "accuracy": 0.0,
                "details": [],
            }

            for case in diff_cases:
                # Format prompt
                prompt = prompt_info["template"].format(
                    page1_bottom=case.page1_bottom, page2_top=case.page2_top
                )

                # Get response
                try:
                    config = prompt_info.get("config", {})
                    response = self.ollama.generate(
                        model=model,
                        prompt=prompt,
                        temperature=config.get("temperature", 0.0),
                        max_tokens=config.get("max_tokens", 50),
                        stop=config.get("stop"),
                        timeout=30,
                    )

                    response_text = response.get("response", "").strip()

                    # Post-process response
                    post_process = prompt_info.get("post_process")
                    if post_process:
                        (
                            prediction,
                            confidence,
                        ) = self.synthetic_tester.post_process_response(
                            response_text, post_process
                        )
                    else:
                        prediction, _ = self.synthetic_tester.post_process_response(
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
                        results["overall"]["correct"] += 1

                    diff_results["details"].append(
                        {
                            "case_id": case.id,
                            "correct": correct,
                            "expected": case.expected,
                            "predicted": prediction,
                        }
                    )

                except Exception as e:
                    diff_results["details"].append(
                        {"case_id": case.id, "error": str(e)}
                    )

            diff_results["accuracy"] = (
                diff_results["correct"] / diff_results["total"]
                if diff_results["total"] > 0
                else 0.0
            )

            results["overall"]["total"] += diff_results["total"]
            results["by_difficulty"][difficulty] = diff_results

        # Calculate overall metrics
        if results["overall"]["total"] > 0:
            results["overall"]["accuracy"] = (
                results["overall"]["correct"] / results["overall"]["total"]
            )

            # Calculate precision/recall for boundary detection
            true_positives = sum(
                1
                for d in results["by_difficulty"].values()
                for detail in d["details"]
                if detail.get("expected") == "Different"
                and detail.get("predicted") == "Different"
            )
            false_positives = sum(
                1
                for d in results["by_difficulty"].values()
                for detail in d["details"]
                if detail.get("expected") == "Same"
                and detail.get("predicted") == "Different"
            )
            false_negatives = sum(
                1
                for d in results["by_difficulty"].values()
                for detail in d["details"]
                if detail.get("expected") == "Different"
                and detail.get("predicted") == "Same"
            )

            if true_positives + false_positives > 0:
                results["overall"]["precision"] = true_positives / (
                    true_positives + false_positives
                )
            if true_positives + false_negatives > 0:
                results["overall"]["recall"] = true_positives / (
                    true_positives + false_negatives
                )

            if results["overall"]["precision"] + results["overall"]["recall"] > 0:
                results["overall"]["f1_score"] = (
                    2
                    * results["overall"]["precision"]
                    * results["overall"]["recall"]
                    / (results["overall"]["precision"] + results["overall"]["recall"])
                )

        # Timing
        results["timing"]["total_time"] = time.time() - start_time
        if results["overall"]["total"] > 0:
            results["timing"]["avg_per_case"] = (
                results["timing"]["total_time"] / results["overall"]["total"]
            )

        return results

    def _calculate_improvements(self, model_results: Dict) -> Dict:
        """Calculate improvements of optimal/CoD prompts over baselines."""
        improvements = {}

        # Find best baseline
        best_baseline = None
        best_baseline_f1 = 0.0

        for name, results in model_results["baseline"].items():
            if not isinstance(results, dict) or "error" in results:
                continue
            f1 = results.get("overall", {}).get("f1_score", 0.0)
            if f1 > best_baseline_f1:
                best_baseline_f1 = f1
                best_baseline = name

        if not best_baseline:
            return improvements

        baseline_results = model_results["baseline"][best_baseline]
        improvements["best_baseline"] = best_baseline
        improvements["baseline_f1"] = best_baseline_f1

        # Compare optimal prompt
        for name, results in model_results["optimal"].items():
            if isinstance(results, dict) and "error" not in results:
                optimal_f1 = results.get("overall", {}).get("f1_score", 0.0)
                improvements["optimal_f1"] = optimal_f1
                improvements["optimal_improvement"] = (
                    (optimal_f1 - best_baseline_f1) / best_baseline_f1 * 100
                    if best_baseline_f1 > 0
                    else 0.0
                )

                # Latency improvement
                baseline_time = baseline_results.get("timing", {}).get(
                    "avg_per_case", 0.0
                )
                optimal_time = results.get("timing", {}).get("avg_per_case", 0.0)
                if baseline_time > 0 and optimal_time > 0:
                    improvements["latency_improvement"] = (
                        (baseline_time - optimal_time) / baseline_time * 100
                    )

        # Compare CoD prompts
        best_cod_f1 = 0.0
        best_cod = None
        for name, results in model_results["cod"].items():
            if isinstance(results, dict) and "error" not in results:
                cod_f1 = results.get("overall", {}).get("f1_score", 0.0)
                if cod_f1 > best_cod_f1:
                    best_cod_f1 = cod_f1
                    best_cod = name

        if best_cod:
            improvements["best_cod"] = best_cod
            improvements["cod_f1"] = best_cod_f1
            improvements["cod_improvement"] = (
                (best_cod_f1 - best_baseline_f1) / best_baseline_f1 * 100
                if best_baseline_f1 > 0
                else 0.0
            )

        return improvements

    def _generate_summary(self, results: Dict) -> Dict:
        """Generate a summary of results across all models."""
        summary = {
            "best_overall": {"model": None, "prompt": None, "f1_score": 0.0},
            "average_improvements": {
                "optimal_over_baseline": 0.0,
                "cod_over_baseline": 0.0,
            },
            "model_recommendations": {},
        }

        # Find best overall
        for model, model_results in results["models"].items():
            for category in ["baseline", "optimal", "cod"]:
                for prompt_name, prompt_results in model_results.get(
                    category, {}
                ).items():
                    if (
                        isinstance(prompt_results, dict)
                        and "error" not in prompt_results
                    ):
                        f1 = prompt_results.get("overall", {}).get("f1_score", 0.0)
                        if f1 > summary["best_overall"]["f1_score"]:
                            summary["best_overall"] = {
                                "model": model,
                                "prompt": prompt_name,
                                "f1_score": f1,
                            }

        # Calculate average improvements
        optimal_improvements = []
        cod_improvements = []

        for model, model_results in results["models"].items():
            improvements = model_results.get("improvements", {})
            if "optimal_improvement" in improvements:
                optimal_improvements.append(improvements["optimal_improvement"])
            if "cod_improvement" in improvements:
                cod_improvements.append(improvements["cod_improvement"])

            # Model-specific recommendation
            best_f1 = 0.0
            best_prompt = None

            for category in ["baseline", "optimal", "cod"]:
                for name, res in model_results.get(category, {}).items():
                    if isinstance(res, dict) and "error" not in res:
                        f1 = res.get("overall", {}).get("f1_score", 0.0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_prompt = name

            summary["model_recommendations"][model] = {
                "best_prompt": best_prompt,
                "f1_score": best_f1,
            }

        if optimal_improvements:
            summary["average_improvements"]["optimal_over_baseline"] = sum(
                optimal_improvements
            ) / len(optimal_improvements)
        if cod_improvements:
            summary["average_improvements"]["cod_over_baseline"] = sum(
                cod_improvements
            ) / len(cod_improvements)

        return summary

    def _save_results(self, results: Dict) -> Path:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"optimal_prompt_test_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        return output_file

    def print_summary(self, results: Dict):
        """Print a human-readable summary of results."""
        print("\n" + "=" * 60)
        print("OPTIMAL PROMPT TEST SUMMARY")
        print("=" * 60)

        summary = results.get("summary", {})

        # Best overall
        best = summary.get("best_overall", {})
        print(f"\nBest Overall Performance:")
        print(f"  Model: {best.get('model', 'N/A')}")
        print(f"  Prompt: {best.get('prompt', 'N/A')}")
        print(f"  F1 Score: {best.get('f1_score', 0.0):.3f}")

        # Average improvements
        avg_imp = summary.get("average_improvements", {})
        print(f"\nAverage Improvements:")
        print(f"  Optimal prompts: {avg_imp.get('optimal_over_baseline', 0.0):.1f}%")
        print(f"  CoD prompts: {avg_imp.get('cod_over_baseline', 0.0):.1f}%")

        # Model-specific recommendations
        print(f"\nRecommendations by Model:")
        for model, rec in summary.get("model_recommendations", {}).items():
            print(
                f"  {model}: {rec.get('best_prompt')} (F1: {rec.get('f1_score', 0.0):.3f})"
            )

        # Detailed improvements by model
        print(f"\nDetailed Improvements:")
        for model, model_results in results.get("models", {}).items():
            improvements = model_results.get("improvements", {})
            if improvements:
                print(f"\n  {model}:")
                print(
                    f"    Baseline: {improvements.get('best_baseline')} (F1: {improvements.get('baseline_f1', 0.0):.3f})"
                )
                if "optimal_f1" in improvements:
                    print(
                        f"    Optimal: F1: {improvements.get('optimal_f1', 0.0):.3f} ({improvements.get('optimal_improvement', 0.0):+.1f}%)"
                    )
                if "cod_f1" in improvements:
                    print(
                        f"    Best CoD: {improvements.get('best_cod')} F1: {improvements.get('cod_f1', 0.0):.3f} ({improvements.get('cod_improvement', 0.0):+.1f}%)"
                    )
                if "latency_improvement" in improvements:
                    print(
                        f"    Latency: {improvements.get('latency_improvement', 0.0):+.1f}% faster"
                    )


def main():
    """Main entry point for testing optimal prompts."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test optimal prompts from prompt engineering research"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["phi4-mini:3.8b", "gemma3:latest"],
        help="Models to test (default: phi4-mini:3.8b gemma3:latest)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["A1_asymmetric", "D1_conservative_few_shot"],
        help="Baseline prompts to compare against",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        type=int,
        help="Difficulty levels to test (default: all)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test with only easy/medium difficulties",
    )

    args = parser.parse_args()

    # Adjust difficulties for quick test
    if args.quick_test:
        difficulties = [1, 2, 3, 4, 5]
    else:
        difficulties = args.difficulties

    # Run tests
    tester = OptimalPromptTester()
    results = tester.test_optimal_prompts(
        models=args.models,
        baseline_prompts=args.baselines,
        difficulty_levels=difficulties,
    )

    # Print summary
    tester.print_summary(results)


if __name__ == "__main__":
    main()
