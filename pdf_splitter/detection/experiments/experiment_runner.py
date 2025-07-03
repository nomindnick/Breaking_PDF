"""
LLM Experiment Runner for Document Boundary Detection.

This module provides a framework for testing different LLM models and strategies
to find the optimal approach for detecting document boundaries.
"""

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests  # type: ignore

from pdf_splitter.core.logging import setup_logging
from pdf_splitter.detection.base_detector import (
    BoundaryResult,
    BoundaryType,
    DetectorType,
    ProcessedPage,
)

setup_logging(__name__)
logger = setup_logging(__name__)  # type: ignore


@dataclass
class ExperimentConfig:
    """Configuration for an LLM experiment."""

    name: str
    model: str  # Ollama model name
    # e.g., "context_overlap", "type_first", "multi_signal", "chain_of_thought"
    strategy: str

    # Strategy-specific parameters
    context_overlap_percent: float = 0.3
    window_size: int = 3  # Number of pages to consider
    temperature: float = 0.1  # Low temperature for consistency
    max_tokens: int = 500

    # Prompt template name
    prompt_template: str = "default"

    # Other parameters
    batch_size: int = 10
    timeout: int = 30  # seconds per request

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    config: ExperimentConfig

    # Ground truth comparison
    true_boundaries: List[int]
    predicted_boundaries: List[int]

    # Metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Performance
    total_time: float = 0.0
    avg_time_per_boundary: float = 0.0
    avg_time_per_page: float = 0.0

    # Detailed results
    boundary_results: List[BoundaryResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Additional metadata
    total_pages: int = 0
    model_responses: List[Dict[str, Any]] = field(default_factory=list)


class OllamaClient:
    """Simple client for interacting with Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama client."""
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 500,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Generate a response from Ollama."""
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "options": {"num_predict": max_tokens},
            "stream": False,
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return {"error": str(e)}

    def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []


class LLMExperimentRunner:
    """Runner for LLM boundary detection experiments."""

    def __init__(self, results_dir: Optional[Path] = None):
        """Initialize the experiment runner."""
        self.ollama = OllamaClient()
        self.results_dir = results_dir or Path(
            "pdf_splitter/detection/experiments/results"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates from the prompts directory."""
        templates = {}
        prompts_dir = Path("pdf_splitter/detection/experiments/prompts")

        # Default template
        default_prompt = (
            "You are analyzing pages from a PDF document to identify document "
            "boundaries.\n\n"
            "Current page {current_page}:\n"
            "{current_text}\n\n"
            "{context_info}\n\n"
            "Question: Is there a document boundary (start of a new document) "
            "at or after page {current_page}?\n"
            'Answer with JSON: {{"boundary": true/false, "confidence": 0.0-1.0, '
            '"reasoning": "brief explanation"}}'
        )
        templates["default"] = default_prompt

        # Load custom templates from files
        if prompts_dir.exists():
            for template_file in prompts_dir.glob("*.txt"):
                template_name = template_file.stem
                templates[template_name] = template_file.read_text()

        return templates

    def run_experiment(
        self,
        config: ExperimentConfig,
        pages: List[ProcessedPage],
        ground_truth: List[int],
    ) -> ExperimentResult:
        """Run a single experiment with the given configuration."""
        logger.info(f"Starting experiment: {config.name}")
        start_time = time.time()

        result = ExperimentResult(
            config=config,
            true_boundaries=ground_truth,
            predicted_boundaries=[],
            total_pages=len(pages),
        )

        # Strategy dispatcher
        if config.strategy == "context_overlap":
            predictions = self._run_context_overlap_strategy(config, pages, result)
        elif config.strategy == "type_first":
            predictions = self._run_type_first_strategy(config, pages, result)
        elif config.strategy == "multi_signal":
            predictions = self._run_multi_signal_strategy(config, pages, result)
        elif config.strategy == "chain_of_thought":
            predictions = self._run_chain_of_thought_strategy(config, pages, result)
        else:
            result.errors.append(f"Unknown strategy: {config.strategy}")
            return result

        # Extract predicted boundaries
        result.predicted_boundaries = [p.page_number for p in predictions]
        result.boundary_results = predictions

        # Calculate metrics
        result.precision, result.recall, result.f1_score = self._calculate_metrics(
            ground_truth, result.predicted_boundaries
        )

        # Calculate timing
        result.total_time = time.time() - start_time
        if result.predicted_boundaries:
            result.avg_time_per_boundary = result.total_time / len(
                result.predicted_boundaries
            )
        result.avg_time_per_page = result.total_time / len(pages)

        # Save results
        self._save_result(result)

        logger.info(
            f"Experiment complete: F1={result.f1_score:.3f}, "
            f"Time={result.total_time:.1f}s"
        )
        return result

    def _run_context_overlap_strategy(
        self,
        config: ExperimentConfig,
        pages: List[ProcessedPage],
        result: ExperimentResult,
    ) -> List[BoundaryResult]:
        """Run the context overlap strategy."""
        predictions = []
        window_size = config.window_size
        overlap_size = int(window_size * config.context_overlap_percent)

        i = 0
        while i < len(pages):
            # Get window of pages
            window_end = min(i + window_size, len(pages))
            window = pages[i:window_end]

            # Check each page in the window for boundaries
            for j, page in enumerate(window):
                if j == 0 and i > 0:
                    # Skip first page if we've already processed it
                    continue

                # Prepare context
                context_start = max(0, i + j - 1)
                context_end = min(len(pages), i + j + 2)
                context_pages = pages[context_start:context_end]

                # Generate prompt
                prompt = self._generate_prompt(config, page, context_pages, i + j)

                # Call LLM
                response = self.ollama.generate(
                    model=config.model,
                    prompt=prompt,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout,
                )

                # Process response
                boundary = self._process_llm_response(
                    response, page.page_number, result
                )
                if boundary and boundary.confidence >= 0.7:
                    predictions.append(boundary)

            # Move window with overlap
            i += window_size - overlap_size

        return predictions

    def _run_type_first_strategy(
        self,
        config: ExperimentConfig,
        pages: List[ProcessedPage],
        result: ExperimentResult,
    ) -> List[BoundaryResult]:
        """Run type-first strategy (classify type, then detect boundaries)."""
        predictions = []

        # First pass: classify document types
        doc_types = []
        for page in pages:
            prompt = f"""Classify the document type for this page:

{page.text[:500]}

Options: email, invoice, letter, form, technical_drawing, memo, other
Answer with just the type:"""

            response = self.ollama.generate(
                model=config.model, prompt=prompt, temperature=0.1, max_tokens=50
            )

            doc_type = response.get("response", "other").strip().lower()
            doc_types.append(doc_type)

        # Second pass: detect boundaries based on type changes
        for i in range(1, len(pages)):
            if doc_types[i] != doc_types[i - 1]:
                # Type change detected, check if it's a boundary
                prompt = self._generate_prompt(
                    config, pages[i], pages[max(0, i - 2) : min(len(pages), i + 2)], i
                )
                prompt += (
                    f"\nNote: Document type changed from '{doc_types[i-1]}' "
                    f"to '{doc_types[i]}'"
                )

                response = self.ollama.generate(
                    model=config.model,
                    prompt=prompt,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )

                boundary = self._process_llm_response(
                    response, pages[i].page_number, result
                )
                if boundary:
                    predictions.append(boundary)

        return predictions

    def _run_multi_signal_strategy(
        self,
        config: ExperimentConfig,
        pages: List[ProcessedPage],
        result: ExperimentResult,
    ) -> List[BoundaryResult]:
        """Run multi-signal strategy (combine visual, text, metadata signals)."""
        # This would integrate with visual and heuristic detectors
        # For now, we'll use a simplified version
        return self._run_context_overlap_strategy(config, pages, result)

    def _run_chain_of_thought_strategy(
        self,
        config: ExperimentConfig,
        pages: List[ProcessedPage],
        result: ExperimentResult,
    ) -> List[BoundaryResult]:
        """Run the chain-of-thought strategy."""
        predictions = []

        for i in range(1, len(pages)):
            # Get context (unused in this strategy, but kept for consistency)
            # context_pages = pages[max(0, i - 1) : min(len(pages), i + 2)]

            # Generate CoT prompt
            prompt = (
                "Analyze these pages to determine if there's a document boundary.\n\n"
                f"Previous page {pages[i-1].page_number}:\n"
                f"{pages[i-1].text[:300]}\n\n"
                f"Current page {pages[i].page_number}:\n"
                f"{pages[i].text[:300]}\n\n"
                "Think step by step:\n"
                "1. What type of document is the previous page?\n"
                "2. What type of document is the current page?\n"
                "3. Are there any clear ending markers on the previous page?\n"
                "4. Are there any clear starting markers on the current page?\n"
                "5. Is this likely a document boundary?\n\n"
                "Provide your analysis and then answer with JSON: "
                '{{"boundary": true/false, "confidence": 0.0-1.0, '
                '"reasoning": "your analysis"}}'
            )

            response = self.ollama.generate(
                model=config.model,
                prompt=prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens * 2,  # More tokens for reasoning
            )

            boundary = self._process_llm_response(
                response, pages[i].page_number, result
            )
            if boundary and boundary.confidence >= 0.7:
                predictions.append(boundary)

        return predictions

    def _generate_prompt(
        self,
        config: ExperimentConfig,
        current_page: ProcessedPage,
        context_pages: List[ProcessedPage],
        page_idx: int,
    ) -> str:
        """Generate a prompt based on the template and context."""
        template = self.prompt_templates.get(
            config.prompt_template, self.prompt_templates["default"]
        )

        # Build context info
        context_parts = []
        for page in context_pages:
            if page.page_number != current_page.page_number:
                preview = page.text[:200] + "..." if len(page.text) > 200 else page.text
                context_parts.append(f"Page {page.page_number}: {preview}")

        context_info = (
            "Context pages:\n" + "\n".join(context_parts) if context_parts else ""
        )

        # Format the prompt
        prompt = template.format(
            current_page=current_page.page_number,
            current_text=current_page.text[:500],
            context_info=context_info,
        )

        return prompt

    def _process_llm_response(
        self, response: Dict[str, Any], page_number: int, result: ExperimentResult
    ) -> Optional[BoundaryResult]:
        """Process LLM response and extract boundary information."""
        if "error" in response:
            result.errors.append(f"Page {page_number}: {response['error']}")
            return None

        try:
            response_text = response.get("response", "")
            result.model_responses.append(
                {"page": page_number, "response": response_text}
            )

            # Try to parse JSON from response
            import re

            json_match = re.search(r"\{[^}]+\}", response_text)
            if json_match:
                data = json.loads(json_match.group())

                if data.get("boundary", False):
                    return BoundaryResult(
                        page_number=page_number,
                        boundary_type=BoundaryType.DOCUMENT_START,
                        confidence=float(data.get("confidence", 0.5)),
                        detector_type=DetectorType.LLM,
                        reasoning=data.get("reasoning", ""),
                    )

        except Exception as e:
            result.errors.append(f"Page {page_number}: Failed to parse response - {e}")

        return None

    def _calculate_metrics(
        self,
        true_boundaries: List[int],
        predicted_boundaries: List[int],
        tolerance: int = 1,
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score with tolerance."""
        true_set = set(true_boundaries)
        predicted_set = set(predicted_boundaries)

        # Count true positives with tolerance
        true_positives = 0
        for pred in predicted_set:
            for true in true_set:
                if abs(pred - true) <= tolerance:
                    true_positives += 1
                    break

        # Calculate false positives and negatives (currently unused but may be useful)
        # false_positives = len(predicted_set) - true_positives
        # false_negatives = len(true_set) - true_positives

        precision = true_positives / len(predicted_set) if predicted_set else 0.0
        recall = true_positives / len(true_set) if true_set else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return precision, recall, f1

    def _save_result(self, result: ExperimentResult) -> None:
        """Save experiment result to disk."""
        timestamp = result.config.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{result.config.name}_{timestamp}.json"
        filepath = self.results_dir / filename

        # Convert to serializable format
        result_dict = asdict(result)
        result_dict["config"]["timestamp"] = result.config.timestamp.isoformat()
        for br in result_dict["boundary_results"]:
            br["boundary_type"] = (
                br["boundary_type"].value
                if hasattr(br["boundary_type"], "value")
                else br["boundary_type"]
            )
            br["detector_type"] = (
                br["detector_type"].value
                if hasattr(br["detector_type"], "value")
                else br["detector_type"]
            )
            br["timestamp"] = (
                br["timestamp"].isoformat()
                if hasattr(br["timestamp"], "isoformat")
                else str(br["timestamp"])
            )

        with open(filepath, "w") as f:
            json.dump(result_dict, f, indent=2)

        logger.info(f"Saved result to {filepath}")

    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """Compare results from multiple experiments."""
        results = []

        # Load results
        for name in experiment_names:
            pattern = f"{name}_*.json"
            for result_file in self.results_dir.glob(pattern):
                with open(result_file) as f:
                    results.append(json.load(f))

        if not results:
            return {"error": "No results found"}

        # Aggregate metrics
        comparison = defaultdict(list)
        for result in results:
            name = result["config"]["name"]
            comparison[name].append(
                {
                    "f1_score": result["f1_score"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "avg_time_per_page": result["avg_time_per_page"],
                    "errors": len(result["errors"]),
                }
            )

        # Calculate averages
        summary = {}
        for name, metrics_list in comparison.items():
            summary[name] = {
                "runs": len(metrics_list),
                "avg_f1": sum(m["f1_score"] for m in metrics_list) / len(metrics_list),
                "avg_precision": sum(m["precision"] for m in metrics_list)
                / len(metrics_list),
                "avg_recall": sum(m["recall"] for m in metrics_list)
                / len(metrics_list),
                "avg_time": sum(m["avg_time_per_page"] for m in metrics_list)
                / len(metrics_list),
                "total_errors": sum(m["errors"] for m in metrics_list),
            }

        return summary
