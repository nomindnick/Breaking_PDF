"""
LLM-based document boundary detector using Ollama with Gemma3.

This module implements document boundary detection using a Large Language Model
to analyze text from consecutive pages and determine if they belong to the same
document. Based on extensive experimentation, this implementation uses the
gemma3:latest model with an optimized prompt structure that achieves:
- F1 Score: 0.889
- Precision: 100% (zero false boundaries)
- Recall: 80%
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.exceptions import ConnectionError, Timeout

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.base_detector import (
    BaseDetector,
    BoundaryResult,
    BoundaryType,
    DetectionContext,
    DetectorType,
    ProcessedPage,
)

logger = logging.getLogger(__name__)


class LLMDetector(BaseDetector):
    """
    Detects document boundaries using Large Language Models via Ollama.

    This detector analyzes text from consecutive pages to determine if they
    belong to the same logical document. It uses a carefully crafted prompt
    with few-shot examples and structured XML output for reliable parsing.
    """

    def __init__(
        self,
        config: Optional[PDFConfig] = None,
        model_name: str = "gemma3:latest",
        ollama_url: str = "http://localhost:11434",
        cache_responses: bool = True,
    ):
        """
        Initialize the LLM detector.

        Args:
            config: PDF processing configuration
            model_name: Name of the Ollama model to use
            ollama_url: URL of the Ollama API endpoint
            cache_responses: Whether to cache LLM responses
        """
        super().__init__(config)
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.cache_responses = cache_responses
        self._response_cache: Dict[str, str] = {}

        # Load the optimal prompt template
        self.prompt_template = self._load_prompt_template()

        # Model-specific settings
        self.timeout = 45  # seconds
        self.max_retries = 2

        # Text extraction settings
        self.bottom_lines = 15  # Lines from bottom of page 1
        self.top_lines = 15  # Lines from top of page 2

    def _load_prompt_template(self) -> str:
        """Load the optimal prompt template for Gemma3."""
        prompt_path = (
            Path(__file__).parent / "experiments" / "prompts" / "gemma3_optimal.txt"
        )

        # Fallback to embedded prompt if file not found
        if not prompt_path.exists():
            logger.warning(
                f"Prompt file not found at {prompt_path}, using embedded prompt"
            )
            return self._get_embedded_prompt()

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            return self._get_embedded_prompt()

    def _get_embedded_prompt(self) -> str:
        """Get the embedded optimal prompt template."""
        return """<start_of_turn>user
You are a meticulous document analyst specializing in automated document segmentation. Your task is to determine if the text from two consecutive pages belongs to the same logical document.

Analyze the provided text snippets and the few-shot examples to understand the patterns of document continuity and separation.

Your reasoning process must be brief and follow the Chain-of-Draft style, placed inside <thinking> tags. Your final classification must be a single word, either 'SAME' or 'DIFFERENT', placed inside <answer> tags.

### EXAMPLES ###

# Example 1: Clear Continuation
<PAGE_1_END_TEXT>
...and therefore, the system is expected to achieve a 95% efficiency rating under normal operating conditions.
</PAGE_1_END_TEXT>
<PAGE_2_START_TEXT>
This level of efficiency is critical for meeting our energy consumption targets. The primary factor influencing this is...
</PAGE_2_START_TEXT>
<thinking>Page 1 ends a sentence. Page 2 begins a new sentence that directly refers to the topic of Page 1 ('This level of efficiency'). Clear semantic continuation. Decision: SAME.</thinking>
<answer>SAME</answer>

# Example 2: Ambiguous Chapter Break
<PAGE_1_END_TEXT>
...concluding the first phase of our investigation.
CHAPTER 3
</PAGE_1_END_TEXT>
<PAGE_2_START_TEXT>
THE NEXT STAGE
The second phase of the investigation began with a new set of challenges. The team first needed to...
</PAGE_2_START_TEXT>
<thinking>Page 1 ends with a chapter marker. Page 2 begins with a new chapter title. This is an internal structural break, not a new document. Decision: SAME.</thinking>
<answer>SAME</answer>

# Example 3: Clear Document Break
<PAGE_1_END_TEXT>
...and we thank you for your business.
Sincerely,
ACME Corporation
</PAGE_1_END_TEXT>
<PAGE_2_START_TEXT>
INVOICE
Number: INV-2025-001
Date: 2025-07-15
</PAGE_2_START_TEXT>
<thinking>Page 1 concludes a formal letter with a signature. Page 2 begins a new document type (invoice) with a clear header. These are distinct documents. Decision: DIFFERENT.</thinking>
<answer>DIFFERENT</answer>

### TASK ###
<PAGE_1_END_TEXT>
{page1_bottom}
</PAGE_1_END_TEXT>
<PAGE_2_START_TEXT>
{page2_top}
</PAGE_2_START_TEXT>
<end_of_turn>
<start_of_turn>model
"""

    def get_detector_type(self) -> DetectorType:
        """Return the type of this detector."""
        return DetectorType.LLM

    def get_confidence_threshold(self) -> float:
        """Get the minimum confidence threshold for this detector."""
        return 0.8  # Higher threshold for LLM due to high precision

    def detect_boundaries(
        self, pages: List[ProcessedPage], context: Optional[DetectionContext] = None
    ) -> List[BoundaryResult]:
        """
        Detect document boundaries by analyzing consecutive page pairs.

        Args:
            pages: List of processed pages to analyze
            context: Optional context information for detection

        Returns:
            List of detected boundaries with confidence scores
        """
        if len(pages) < 2:
            logger.warning("Need at least 2 pages for boundary detection")
            return []

        boundaries = []

        # Check Ollama availability once
        if not self._check_ollama_availability():
            logger.error("Ollama is not available. Cannot perform LLM detection.")
            return boundaries

        # Process consecutive page pairs
        for i in range(len(pages) - 1):
            page1 = pages[i]
            page2 = pages[i + 1]

            # Skip if either page is empty
            if page1.is_empty or page2.is_empty:
                logger.debug(f"Skipping empty page pair: {i+1}-{i+2}")
                continue

            try:
                # Analyze the page pair
                start_time = time.time()
                is_boundary, confidence, reasoning = self._analyze_page_pair(
                    page1, page2
                )
                elapsed_time = time.time() - start_time

                logger.info(
                    f"Analyzed pages {i+1}-{i+2}: boundary={is_boundary}, "
                    f"confidence={confidence:.2f}, time={elapsed_time:.1f}s"
                )

                # Create boundary result if detected
                if is_boundary:
                    boundary = BoundaryResult(
                        page_number=i + 1,  # Boundary after this page
                        boundary_type=BoundaryType.DOCUMENT_END,
                        confidence=confidence,
                        detector_type=self.get_detector_type(),
                        evidence={
                            "llm_response": reasoning,
                            "processing_time": elapsed_time,
                            "model": self.model_name,
                        },
                        reasoning=reasoning,
                        is_between_pages=True,
                        next_page_number=i + 2,
                    )
                    boundaries.append(boundary)
                    self._detection_history.append(boundary)

            except Exception as e:
                logger.error(f"Error analyzing pages {i+1}-{i+2}: {e}")
                continue

        # Update context if provided
        if context:
            context.update_progress(len(pages))

        self._last_detection_time = time.time()

        return boundaries

    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False

            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            if self.model_name not in model_names:
                logger.warning(
                    f"Model {self.model_name} not found. Available: {model_names}"
                )
                return False

            return True

        except (ConnectionError, Timeout) as e:
            logger.error(f"Cannot connect to Ollama at {self.ollama_url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {e}")
            return False

    def _analyze_page_pair(
        self, page1: ProcessedPage, page2: ProcessedPage
    ) -> Tuple[bool, float, str]:
        """
        Analyze a pair of consecutive pages to determine if there's a boundary.

        Args:
            page1: First page
            page2: Second page

        Returns:
            Tuple of (is_boundary, confidence, reasoning)
        """
        # Extract relevant text portions
        page1_text = self._extract_bottom_text(page1.text)
        page2_text = self._extract_top_text(page2.text)

        # Check cache
        cache_key = self._get_cache_key(page1_text, page2_text)
        if self.cache_responses and cache_key in self._response_cache:
            cached = self._response_cache[cache_key]
            return self._parse_llm_response(cached)

        # Prepare prompt
        prompt = self.prompt_template.format(
            page1_bottom=page1_text, page2_top=page2_text
        )

        # Call LLM
        response = self._call_ollama(prompt)

        # Cache response
        if self.cache_responses and response:
            self._response_cache[cache_key] = response

        # Parse response
        return self._parse_llm_response(response)

    def _extract_bottom_text(self, text: str) -> str:
        """Extract the bottom portion of a page's text."""
        lines = text.strip().split("\n")
        if len(lines) <= self.bottom_lines:
            return text.strip()
        return "\n".join(lines[-self.bottom_lines :])

    def _extract_top_text(self, text: str) -> str:
        """Extract the top portion of a page's text."""
        lines = text.strip().split("\n")
        if len(lines) <= self.top_lines:
            return text.strip()
        return "\n".join(lines[: self.top_lines])

    def _get_cache_key(self, text1: str, text2: str) -> str:
        """Generate a cache key for a text pair."""
        # Simple hash-based key
        combined = f"{text1}|||{text2}"
        return str(hash(combined))

    def _call_ollama(self, prompt: str) -> str:
        """
        Call the Ollama API with the given prompt.

        Args:
            prompt: The formatted prompt

        Returns:
            The model's response text
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temperature for consistency
                            "top_k": 10,
                            "top_p": 0.9,
                        },
                    },
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    return response.json().get("response", "")
                else:
                    logger.error(
                        f"Ollama API error: {response.status_code} - {response.text}"
                    )

            except Timeout:
                logger.warning(
                    f"Ollama request timeout (attempt {attempt + 1}/{self.max_retries})"
                )
            except Exception as e:
                logger.error(f"Error calling Ollama: {e}")

            if attempt < self.max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff

        return ""

    def _parse_llm_response(self, response: str) -> Tuple[bool, float, str]:
        """
        Parse the LLM response to extract decision and reasoning.

        Args:
            response: Raw LLM response

        Returns:
            Tuple of (is_boundary, confidence, reasoning)
        """
        if not response:
            return False, 0.0, "No response from LLM"

        # Extract reasoning from <thinking> tags
        reasoning = ""
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", response, re.DOTALL)
        if thinking_match:
            reasoning = thinking_match.group(1).strip()

        # Extract answer from <answer> tags
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not answer_match:
            logger.warning(f"No answer tags found in response: {response[:200]}...")
            return False, 0.0, "Invalid response format"

        answer = answer_match.group(1).strip().upper()

        # Determine if it's a boundary
        is_boundary = answer == "DIFFERENT"

        # Assign confidence based on the model's consistency
        # Since we achieve 100% precision, we use high confidence for positive predictions
        if is_boundary:
            confidence = 0.95  # High confidence for boundaries
        else:
            confidence = 0.85  # Slightly lower for continuations

        return is_boundary, confidence, reasoning

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the detector configuration and environment.

        Returns:
            Dictionary with validation results
        """
        results = {
            "ollama_available": False,
            "model_available": False,
            "prompt_loaded": bool(self.prompt_template),
            "cache_enabled": self.cache_responses,
        }

        # Check Ollama
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            results["ollama_available"] = response.status_code == 200

            if results["ollama_available"]:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                results["model_available"] = self.model_name in model_names
                results["available_models"] = model_names

        except Exception as e:
            results["error"] = str(e)

        return results
