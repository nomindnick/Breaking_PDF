#!/usr/bin/env python3
"""
Systematic prompt testing using synthetic pairs and real PDFs.

This script:
1. Tests all prompt variations on synthetic test cases (easy->medium->hard)
2. Identifies best performing prompts per difficulty level
3. Tests winning combinations on real PDF documents
"""

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.experiments.enhanced_synthetic_tests import \
    EnhancedSyntheticTester
from pdf_splitter.detection.experiments.experiment_runner import (
    ExperimentConfig, LLMExperimentRunner, ProcessedPage)
from pdf_splitter.preprocessing import PDFHandler, TextExtractor


@dataclass
class PromptTestResult:
    """Result from testing a prompt configuration."""

    prompt_name: str
    model: str
    difficulty_level: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positives: int
    false_negatives: int
    avg_time: float
    errors: List[str]


class SystematicPromptTester:
    """Systematic testing of prompts with progressive difficulty."""

    def __init__(self):
        self.synthetic_tester = EnhancedSyntheticTester()
        self.experiment_runner = LLMExperimentRunner()
        self.results_dir = Path("pdf_splitter/detection/experiments/results")
        self.results_dir.mkdir(exist_ok=True)

    def test_prompts_on_synthetic(
        self, models: List[str], confidence_threshold: float = 0.4
    ) -> Dict:
        """Test all prompts on synthetic data with progressive difficulty."""
        print("Phase 1: Testing on Synthetic Data")
        print("=" * 60)

        # Run enhanced synthetic tests
        results = self.synthetic_tester.test_easy_medium_hard_progression(
            models, confidence_threshold
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"systematic_synthetic_test_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nSynthetic test results saved to: {output_file}")
        return results

    def identify_best_combinations(self, synthetic_results: Dict) -> List[Dict]:
        """Identify best model/prompt combinations from synthetic tests."""
        best_combinations = []

        # Find best performers for each difficulty level
        for model, model_results in synthetic_results["models"].items():
            if not model_results["easy"]:
                continue

            best_prompt = model_results["best_prompt"]
            easy_accuracy = model_results["easy"]["best_accuracy"]

            combo = {
                "model": model,
                "prompt": best_prompt,
                "easy_accuracy": easy_accuracy,
                "medium_accuracy": None,
                "hard_accuracy": None,
            }

            if model_results.get("medium"):
                combo["medium_accuracy"] = model_results["medium"].get(
                    "accuracy_with_best_easy_prompt", 0
                )

            if model_results.get("hard"):
                combo["hard_accuracy"] = model_results["hard"].get(
                    "accuracy_with_best_easy_prompt", 0
                )

            # Calculate overall score (weighted average)
            scores = []
            weights = []
            if combo["easy_accuracy"] is not None:
                scores.append(combo["easy_accuracy"])
                weights.append(3)  # Easy cases more important
            if combo["medium_accuracy"] is not None:
                scores.append(combo["medium_accuracy"])
                weights.append(2)
            if combo["hard_accuracy"] is not None:
                scores.append(combo["hard_accuracy"])
                weights.append(1)

            if scores:
                combo["overall_score"] = sum(
                    s * w for s, w in zip(scores, weights)
                ) / sum(weights)
                best_combinations.append(combo)

        # Sort by overall score
        best_combinations.sort(key=lambda x: x["overall_score"], reverse=True)
        return best_combinations

    def test_on_real_pdf(
        self, model: str, prompt_name: str, pdf_path: Optional[Path] = None
    ) -> PromptTestResult:
        """Test a model/prompt combination on real PDF data."""
        if pdf_path is None:
            pdf_path = Path("Test_PDF_Set_1.pdf")

        print(f"\nTesting {model} with {prompt_name} on {pdf_path.name}...")

        # Load PDF and extract text
        config = PDFConfig()
        pdf_handler = PDFHandler(config)
        text_extractor = TextExtractor(config)

        # Load PDF
        pdf_handler.load_pdf(pdf_path)

        # Extract text from pages
        pages = []
        for i in range(pdf_handler.page_count):
            page_data = pdf_handler.get_page_data(i)
            text = text_extractor.extract_page_text(page_data)

            pages.append(
                ProcessedPage(page_number=i + 1, text=text, metadata={}, layout_info={})
            )

        # Load ground truth
        ground_truth_file = Path("Test_PDF_Set_Ground_Truth.json")
        with open(ground_truth_file) as f:
            ground_truth_data = json.load(f)

        # Extract boundary page numbers
        true_boundaries = []
        for doc in ground_truth_data["documents"]:
            if doc["start_page"] > 1:  # Not the first document
                true_boundaries.append(doc["start_page"])

        # Create experiment config
        exp_config = ExperimentConfig(
            name=f"real_pdf_test_{model}_{prompt_name}",
            model=model,
            strategy="synthetic_pairs",
            prompt_template=prompt_name,
            temperature=0.0,
            max_tokens=10,
        )

        # Add stop tokens for certain prompts
        if prompt_name in ["A1_asymmetric"]:
            exp_config.stop_tokens = ["S", "D"]

        # Run experiment
        start_time = time.time()
        result = self.experiment_runner.run_experiment(
            exp_config, pages, true_boundaries
        )
        elapsed_time = time.time() - start_time

        # Calculate additional metrics
        predicted_set = set(result.predicted_boundaries)
        true_set = set(true_boundaries)

        false_positives = len(predicted_set - true_set)
        false_negatives = len(true_set - predicted_set)

        return PromptTestResult(
            prompt_name=prompt_name,
            model=model,
            difficulty_level="real_pdf",
            accuracy=(result.precision + result.recall) / 2
            if result.precision + result.recall > 0
            else 0,
            precision=result.precision,
            recall=result.recall,
            f1_score=result.f1_score,
            false_positives=false_positives,
            false_negatives=false_negatives,
            avg_time=elapsed_time / len(pages) if pages else 0,
            errors=result.errors,
        )

    def run_full_systematic_test(
        self, models: List[str], test_real_pdf: bool = True
    ) -> Dict:
        """Run complete systematic testing pipeline."""
        all_results = {
            "experiment": "systematic_prompt_test",
            "timestamp": datetime.now().isoformat(),
            "phases": {},
        }

        # Phase 1: Synthetic tests
        print("\n" + "=" * 60)
        print("PHASE 1: SYNTHETIC DATA TESTING")
        print("=" * 60)

        synthetic_results = self.test_prompts_on_synthetic(models)
        all_results["phases"]["synthetic"] = synthetic_results

        # Identify best combinations
        best_combos = self.identify_best_combinations(synthetic_results)
        all_results["best_combinations"] = best_combos

        print("\n" + "=" * 60)
        print("BEST MODEL/PROMPT COMBINATIONS")
        print("=" * 60)

        for i, combo in enumerate(best_combos[:5]):  # Top 5
            print(f"\n{i+1}. {combo['model']} + {combo['prompt']}")
            print(f"   Overall Score: {combo['overall_score']:.1%}")
            print(f"   Easy: {combo['easy_accuracy']:.1%}")
            if combo["medium_accuracy"]:
                print(f"   Medium: {combo['medium_accuracy']:.1%}")
            if combo["hard_accuracy"]:
                print(f"   Hard: {combo['hard_accuracy']:.1%}")

        # Phase 2: Real PDF testing
        if test_real_pdf and best_combos:
            print("\n" + "=" * 60)
            print("PHASE 2: REAL PDF TESTING")
            print("=" * 60)

            real_pdf_results = []

            # Test top 3 combinations on real PDFs
            for combo in best_combos[:3]:
                try:
                    result = self.test_on_real_pdf(combo["model"], combo["prompt"])
                    real_pdf_results.append(result)

                    print(f"\nResults for {combo['model']} + {combo['prompt']}:")
                    print(f"  F1 Score: {result.f1_score:.3f}")
                    print(f"  Precision: {result.precision:.3f}")
                    print(f"  Recall: {result.recall:.3f}")
                    print(f"  False Positives: {result.false_positives}")
                    print(f"  False Negatives: {result.false_negatives}")
                    print(f"  Avg Time/Page: {result.avg_time:.2f}s")

                except Exception as e:
                    print(f"\nError testing {combo['model']} + {combo['prompt']}: {e}")

            all_results["phases"]["real_pdf"] = [
                {
                    "prompt_name": r.prompt_name,
                    "model": r.model,
                    "f1_score": r.f1_score,
                    "precision": r.precision,
                    "recall": r.recall,
                    "false_positives": r.false_positives,
                    "false_negatives": r.false_negatives,
                    "avg_time": r.avg_time,
                    "errors": r.errors,
                }
                for r in real_pdf_results
            ]

        # Save complete results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"systematic_test_complete_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n\nComplete results saved to: {output_file}")
        return all_results

    def quick_test_single_prompt(self, model: str, prompt_name: str):
        """Quick test of a single model/prompt combination."""
        print(f"\nQuick test: {model} with {prompt_name}")
        print("=" * 60)

        # Test on a few synthetic cases
        prompts = self.synthetic_tester.get_user_specified_prompts()
        if prompt_name not in prompts:
            print(f"Error: Prompt {prompt_name} not found")
            return

        prompt_info = prompts[prompt_name]

        # Test on easy cases first
        easy_cases = [c for c in self.synthetic_tester.test_cases if c.difficulty <= 3][
            :5
        ]
        correct = 0

        for case in easy_cases:
            prompt = prompt_info["template"].format(
                page1_bottom=case.page1_bottom,
                page2_top=case.page2_top,
                page1=case.page1_bottom,
                page2=case.page2_top,
            )

            config = prompt_info.get("config", {})
            response = self.synthetic_tester.client.generate(
                model=model,
                prompt=prompt,
                temperature=config.get("temperature", 0.0),
                max_tokens=config.get("max_tokens", 50),
                stop=config.get("stop", None),
                timeout=30,
            )

            response_text = response.get("response", "").strip()
            print(f"\nCase: {case.id} (Expected: {case.expected})")
            print(f"Response: {response_text}")

            # Simple classification
            if case.expected == "Same" and "S" in response_text.upper():
                correct += 1
            elif case.expected == "Different" and "D" in response_text.upper():
                correct += 1

        print(
            f"\nAccuracy on easy cases: {correct}/{len(easy_cases)} = {correct/len(easy_cases):.1%}"
        )


def main():
    """Run systematic prompt testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Systematic prompt testing")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["phi4-mini:3.8b", "gemma3:latest"],
        help="Models to test",
    )
    parser.add_argument(
        "--quick-test",
        nargs=2,
        metavar=("MODEL", "PROMPT"),
        help="Quick test a single model/prompt combination",
    )
    parser.add_argument(
        "--no-real-pdf", action="store_true", help="Skip real PDF testing"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.4,
        help="Confidence threshold for B1/B2 prompts",
    )

    args = parser.parse_args()

    tester = SystematicPromptTester()

    if args.quick_test:
        model, prompt = args.quick_test
        tester.quick_test_single_prompt(model, prompt)
    else:
        results = tester.run_full_systematic_test(
            args.models, test_real_pdf=not args.no_real_pdf
        )

        print("\n" + "=" * 60)
        print("TESTING COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()
