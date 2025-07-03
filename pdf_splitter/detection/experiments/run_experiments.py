"""
Script to run LLM boundary detection experiments.

This script provides a command-line interface for running experiments
with different models and strategies.
"""

import argparse
import json
from pathlib import Path
from typing import List

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.base_detector import ProcessedPage
from pdf_splitter.detection.experiments.experiment_runner import (
    ExperimentConfig,
    LLMExperimentRunner,
)
from pdf_splitter.preprocessing.pdf_handler import PDFHandler
from pdf_splitter.preprocessing.text_extractor import TextExtractor


def load_ground_truth(ground_truth_path: Path) -> List[int]:
    """Load ground truth boundaries from JSON file."""
    with open(ground_truth_path) as f:
        data = json.load(f)

    boundaries = []
    for doc in data["documents"]:
        # Extract the first page number from the range
        page_range = doc["pages"]
        if "-" in page_range:
            start_page = int(page_range.split("-")[0])
        else:
            start_page = int(page_range)
        boundaries.append(start_page)

    return boundaries[1:]  # Skip the first document (page 1 is always a start)


def process_pdf_to_pages(pdf_path: Path) -> List[ProcessedPage]:
    """Process a PDF file and convert to ProcessedPage objects."""
    config = PDFConfig()
    handler = PDFHandler(config)
    extractor = TextExtractor(config)

    # Load PDF
    with handler.load_pdf(pdf_path) as loaded_handler:
        pages = []
        for page_num in range(1, loaded_handler.page_count + 1):
            # Get page info
            page_info = loaded_handler.get_page_info(page_num)

            # Extract text
            if page_info["type"] in ["SEARCHABLE", "MIXED"]:
                page_text = extractor.extract_page_text(page_num)
                text = page_text.text if hasattr(page_text, "text") else str(page_text)
            else:
                # For scanned pages, we'd use OCR here
                text = f"[Page {page_num} - Scanned content]"

            # Create ProcessedPage
            page = ProcessedPage(
                page_number=page_num,
                text=text,
                page_type=page_info["type"],
                metadata=page_info,
            )
            pages.append(page)

        return pages


def create_experiment_configs(
    models: List[str], strategies: List[str]
) -> List[ExperimentConfig]:
    """Create experiment configurations for all combinations."""
    configs = []

    for model in models:
        for strategy in strategies:
            # Base config
            base_config = {
                "name": f"{model.replace(':', '_')}_{strategy}",
                "model": model,
                "strategy": strategy,
            }

            # Strategy-specific parameters
            if strategy == "context_overlap":
                for overlap in [0.2, 0.3, 0.4]:
                    config = ExperimentConfig(
                        name=base_config["name"],
                        model=base_config["model"],
                        strategy=base_config["strategy"],
                        context_overlap_percent=overlap,
                        window_size=3,
                    )
                    config.name += f"_{int(overlap*100)}pct"
                    configs.append(config)

            elif strategy == "chain_of_thought":
                config = ExperimentConfig(
                    name=base_config["name"],
                    model=base_config["model"],
                    strategy=base_config["strategy"],
                    temperature=0.3,  # Slightly higher for reasoning
                    max_tokens=1000,  # More tokens for CoT
                )
                configs.append(config)

            else:
                configs.append(
                    ExperimentConfig(
                        name=base_config["name"],
                        model=base_config["model"],
                        strategy=base_config["strategy"],
                    )
                )

    return configs


def main():
    """Run experiments."""
    parser = argparse.ArgumentParser(
        description="Run LLM boundary detection experiments"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("test_files/Test_PDF_Set_1.pdf"),
        help="Path to test PDF file",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("test_files/Test_PDF_Set_Ground_Truth.json"),
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3:8b-instruct-q5_K_M"],
        help="Ollama models to test",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["context_overlap"],
        help="Strategies to test: context_overlap, type_first, chain_of_thought",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("pdf_splitter/detection/experiments/results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare existing results, don't run new experiments",
    )

    args = parser.parse_args()

    # Initialize runner
    runner = LLMExperimentRunner(results_dir=args.results_dir)

    if args.compare_only:
        # Compare existing results
        experiment_names = [
            f"{model.replace(':', '_')}_{strategy}"
            for model in args.models
            for strategy in args.strategies
        ]
        comparison = runner.compare_experiments(experiment_names)

        print("\nExperiment Comparison:")
        print("-" * 80)
        for name, metrics in comparison.items():
            print(f"\n{name}:")
            print(f"  Runs: {metrics.get('runs', 0)}")
            print(f"  Avg F1 Score: {metrics.get('avg_f1', 0):.3f}")
            print(f"  Avg Precision: {metrics.get('avg_precision', 0):.3f}")
            print(f"  Avg Recall: {metrics.get('avg_recall', 0):.3f}")
            print(f"  Avg Time/Page: {metrics.get('avg_time', 0):.3f}s")
            print(f"  Total Errors: {metrics.get('total_errors', 0)}")

        return

    # Check if Ollama is available
    available_models = runner.ollama.list_models()
    if not available_models:
        print("Error: Could not connect to Ollama. Make sure it's running.")
        return

    print(f"Available Ollama models: {available_models}")

    # Verify requested models are available
    for model in args.models:
        if model not in available_models:
            print(f"Warning: Model '{model}' not found in Ollama")

    # Load test data
    print(f"\nLoading PDF: {args.pdf}")
    pages = process_pdf_to_pages(args.pdf)
    print(f"Loaded {len(pages)} pages")

    print(f"\nLoading ground truth: {args.ground_truth}")
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Ground truth boundaries: {ground_truth}")

    # Create experiment configurations
    configs = create_experiment_configs(args.models, args.strategies)
    print(f"\nCreated {len(configs)} experiment configurations")

    # Run experiments
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running experiment: {config.name}")
        try:
            result = runner.run_experiment(config, pages, ground_truth)
            results.append(result)

            print(f"  Precision: {result.precision:.3f}")
            print(f"  Recall: {result.recall:.3f}")
            print(f"  F1 Score: {result.f1_score:.3f}")
            print(
                f"  Time: {result.total_time:.1f}s "
                f"({result.avg_time_per_page:.3f}s/page)"
            )
            print(f"  Predicted boundaries: {result.predicted_boundaries}")

        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    best_f1 = max(results, key=lambda r: r.f1_score) if results else None
    if best_f1:
        print(f"\nBest F1 Score: {best_f1.config.name}")
        print(f"  F1: {best_f1.f1_score:.3f}")
        print(f"  Precision: {best_f1.precision:.3f}")
        print(f"  Recall: {best_f1.recall:.3f}")

    fastest = min(results, key=lambda r: r.avg_time_per_page) if results else None
    if fastest:
        print(f"\nFastest: {fastest.config.name}")
        print(f"  Time/page: {fastest.avg_time_per_page:.3f}s")
        print(f"  F1: {fastest.f1_score:.3f}")


if __name__ == "__main__":
    main()
