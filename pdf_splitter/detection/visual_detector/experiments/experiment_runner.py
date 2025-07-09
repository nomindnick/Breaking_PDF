"""
Experiment runner for visual boundary detection techniques.

This module provides the framework for running experiments on different
visual detection techniques and evaluating their performance.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pdf_splitter.preprocessing.pdf_handler import PDFHandler
from pdf_splitter.preprocessing.text_extractor import TextExtractor

from .visual_techniques import BaseVisualTechnique, VisualComparison, create_technique

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetrics:
    """Metrics for evaluating boundary detection performance."""

    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    total_processing_time: float = 0.0
    pages_processed: int = 0

    @property
    def precision(self) -> float:
        """Calculate precision: TP / (TP + FP)."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Calculate recall: TP / (TP + FN)."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """Calculate F1 score: harmonic mean of precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        """Calculate accuracy: (TP + TN) / total."""
        total = (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total

    @property
    def avg_time_per_page(self) -> float:
        """Average processing time per page."""
        if self.pages_processed == 0:
            return 0.0
        return self.total_processing_time / self.pages_processed


@dataclass
class ExperimentResult:
    """Result of a visual detection experiment."""

    technique_name: str
    parameters: Dict[str, Any]
    metrics: ExperimentMetrics
    comparisons: List[VisualComparison]

    timestamp: str = ""
    pdf_path: str = ""
    ground_truth_path: str = ""

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "technique_name": self.technique_name,
            "parameters": self.parameters,
            "metrics": asdict(self.metrics),
            "comparisons": [asdict(c) for c in self.comparisons],
            "timestamp": self.timestamp,
            "pdf_path": self.pdf_path,
            "ground_truth_path": self.ground_truth_path,
            "summary": {
                "f1_score": self.metrics.f1_score,
                "precision": self.metrics.precision,
                "recall": self.metrics.recall,
                "accuracy": self.metrics.accuracy,
                "avg_time_per_page": self.metrics.avg_time_per_page,
            },
        }


class VisualExperimentRunner:
    """
    Runner for visual boundary detection experiments.

    Handles loading PDFs, applying techniques, and evaluating results
    against ground truth.
    """

    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize the experiment runner.

        Args:
            results_dir: Directory to save experiment results
        """
        self.results_dir = results_dir or Path("experiments/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize PDF handler for rendering pages
        self.pdf_handler = PDFHandler()
        self.text_extractor = TextExtractor(self.pdf_handler)

    def load_ground_truth(self, ground_truth_path: Path) -> List[int]:
        """
        Load ground truth boundaries from JSON file.

        Args:
            ground_truth_path: Path to ground truth JSON

        Returns:
            List of page numbers where boundaries occur
        """
        with open(ground_truth_path, "r") as f:
            data = json.load(f)

        boundaries = []

        # Extract boundaries from document ranges
        if "documents" in data:
            for doc in data["documents"]:
                pages = doc["pages"]
                if "-" in pages:
                    start, end = pages.split("-")
                    end_page = int(end)
                    # Boundary after the last page of each document
                    boundaries.append(end_page)

        # Also check for explicit boundaries
        if "visual_boundaries" in data:
            for boundary in data["visual_boundaries"]:
                boundaries.append(boundary["after_page"])

        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))

        # Remove the last boundary if it's the last page
        # (no boundary after the last page of the PDF)
        return boundaries[:-1] if boundaries else []

    def render_pdf_pages(self, pdf_path: Path, dpi: int = 150) -> List[np.ndarray]:
        """
        Render all pages of a PDF as images.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for rendering

        Returns:
            List of page images as numpy arrays
        """
        images = []

        # Load the PDF using context manager
        with self.pdf_handler.load_pdf(pdf_path):
            # Render each page
            for page_num in range(self.pdf_handler.page_count):
                # render_page expects 0-based indexing
                image = self.pdf_handler.render_page(page_num, dpi=dpi)

                # Image is already a numpy array from render_page
                images.append(image)

        return images

    def run_experiment(
        self,
        technique: BaseVisualTechnique,
        pdf_path: Path,
        ground_truth_path: Path,
        save_results: bool = True,
    ) -> ExperimentResult:
        """
        Run an experiment with a visual technique.

        Args:
            technique: Visual technique to test
            pdf_path: Path to test PDF
            ground_truth_path: Path to ground truth JSON
            save_results: Whether to save results to disk

        Returns:
            Experiment results
        """
        logger.info(f"Starting experiment with {technique.name}")

        # Load ground truth
        true_boundaries = self.load_ground_truth(ground_truth_path)
        logger.info(f"Loaded {len(true_boundaries)} true boundaries")

        # Render PDF pages
        logger.info(f"Rendering pages from {pdf_path}")
        start_time = time.time()
        page_images = self.render_pdf_pages(pdf_path)
        render_time = time.time() - start_time
        logger.info(f"Rendered {len(page_images)} pages in {render_time:.2f}s")

        # Run comparisons between adjacent pages
        comparisons = []
        metrics = ExperimentMetrics()

        for i in range(len(page_images) - 1):
            page1_num = i + 1
            page2_num = i + 2

            # Compare adjacent pages
            comparison = technique.detect_boundary(
                page_images[i], page_images[i + 1], page1_num, page2_num
            )
            comparisons.append(comparison)

            # Update metrics
            metrics.total_processing_time += comparison.processing_time
            metrics.pages_processed += 1

            # Check against ground truth
            is_true_boundary = page1_num in true_boundaries
            is_predicted_boundary = comparison.is_boundary

            if is_true_boundary and is_predicted_boundary:
                metrics.true_positives += 1
            elif not is_true_boundary and is_predicted_boundary:
                metrics.false_positives += 1
            elif not is_true_boundary and not is_predicted_boundary:
                metrics.true_negatives += 1
            else:  # is_true_boundary and not is_predicted_boundary
                metrics.false_negatives += 1

        # Create result
        result = ExperimentResult(
            technique_name=technique.name,
            parameters={
                "threshold": technique.threshold,
                **getattr(technique, "__dict__", {}),
            },
            metrics=metrics,
            comparisons=comparisons,
            pdf_path=str(pdf_path),
            ground_truth_path=str(ground_truth_path),
        )

        # Log summary
        logger.info(f"Experiment complete for {technique.name}:")
        logger.info(f"  F1 Score: {metrics.f1_score:.3f}")
        logger.info(f"  Precision: {metrics.precision:.3f}")
        logger.info(f"  Recall: {metrics.recall:.3f}")
        logger.info(f"  Avg time/page: {metrics.avg_time_per_page:.3f}s")

        # Save results if requested
        if save_results:
            self.save_result(result)

        return result

    def save_result(self, result: ExperimentResult) -> Path:
        """
        Save experiment result to disk.

        Args:
            result: Experiment result to save

        Returns:
            Path to saved result file
        """
        # Create filename with timestamp and technique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{result.technique_name.lower()}_result.json"
        filepath = self.results_dir / filename

        # Save as JSON
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved results to {filepath}")
        return filepath

    def compare_techniques(
        self,
        techniques: List[Tuple[str, Dict[str, Any]]],
        pdf_path: Path,
        ground_truth_path: Path,
    ) -> Dict[str, ExperimentResult]:
        """
        Compare multiple techniques on the same PDF.

        Args:
            techniques: List of (technique_name, parameters) tuples
            pdf_path: Path to test PDF
            ground_truth_path: Path to ground truth

        Returns:
            Dictionary mapping technique names to results
        """
        results = {}

        for tech_name, params in techniques:
            logger.info(f"\nTesting {tech_name} with params: {params}")

            # Create technique
            technique = create_technique(tech_name, **params)

            # Run experiment
            result = self.run_experiment(technique, pdf_path, ground_truth_path)

            results[f"{tech_name}_{params}"] = result

        # Print comparison summary
        print("\n=== Technique Comparison ===")
        print(
            f"{'Technique':<30} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Time/Page':<10}"
        )
        print("-" * 70)

        for name, result in results.items():
            metrics = result.metrics
            print(
                f"{name:<30} {metrics.f1_score:<8.3f} {metrics.precision:<10.3f} "
                f"{metrics.recall:<8.3f} {metrics.avg_time_per_page:<10.3f}"
            )

        return results

    def analyze_failures(
        self, result: ExperimentResult, ground_truth_path: Path
    ) -> Dict[str, Any]:
        """
        Analyze false positives and false negatives.

        Args:
            result: Experiment result to analyze
            ground_truth_path: Path to ground truth for context

        Returns:
            Analysis of failures
        """
        true_boundaries = self.load_ground_truth(Path(ground_truth_path))

        false_positives = []
        false_negatives = []

        for comparison in result.comparisons:
            page_num = comparison.page1_num
            is_true_boundary = page_num in true_boundaries
            is_predicted = comparison.is_boundary

            if is_predicted and not is_true_boundary:
                false_positives.append(
                    {
                        "page": page_num,
                        "similarity": comparison.similarity_score,
                        "metadata": comparison.metadata,
                    }
                )
            elif is_true_boundary and not is_predicted:
                false_negatives.append(
                    {
                        "page": page_num,
                        "similarity": comparison.similarity_score,
                        "metadata": comparison.metadata,
                    }
                )

        return {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "fp_count": len(false_positives),
            "fn_count": len(false_negatives),
        }
