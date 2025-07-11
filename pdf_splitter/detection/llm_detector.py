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
from pdf_splitter.detection.base_detector import (BaseDetector, BoundaryResult,
                                                  BoundaryType,
                                                  DetectionContext,
                                                  DetectorType, ProcessedPage)
from pdf_splitter.detection.llm_cache import LLMResponseCache
from pdf_splitter.detection.llm_config import LLMDetectorConfig, get_config

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
        llm_config: Optional[LLMDetectorConfig] = None,
        **kwargs,
    ):
        """
        Initialize the LLM detector.

        Args:
            config: PDF processing configuration
            llm_config: LLM detector specific configuration
            **kwargs: Override configuration parameters
        """
        super().__init__(config)

        # Get LLM configuration
        self.llm_config = llm_config or get_config(**kwargs)

        # Set instance attributes from config
        self.model_name = self.llm_config.model_name
        self.ollama_url = self.llm_config.ollama_url
        self.cache_responses = self.llm_config.cache_enabled
        self.timeout = self.llm_config.timeout
        self.max_retries = self.llm_config.max_retries
        self.bottom_lines = self.llm_config.bottom_lines
        self.top_lines = self.llm_config.top_lines
        self._prompt_version = self.llm_config.prompt_version

        # Initialize persistent cache
        if self.llm_config.cache_enabled:
            self._cache = LLMResponseCache(
                cache_path=self.llm_config.cache_path,
                max_age_days=self.llm_config.cache_max_age_days,
                max_size_mb=self.llm_config.cache_max_size_mb,
            )
        else:
            self._cache = None

        # Load the optimal prompt template
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load the optimal prompt template for Gemma3."""
        # Check if custom template path is specified
        if (
            self.llm_config.prompt_template_path
            and self.llm_config.prompt_template_path.exists()
        ):
            try:
                with open(
                    self.llm_config.prompt_template_path, "r", encoding="utf-8"
                ) as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error loading custom prompt template: {e}")

        # Default path
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

        # Check if we should only process specific target pages
        target_pages = None
        if context and context.document_metadata:
            target_pages = context.document_metadata.get("target_pages", None)
            if target_pages:
                logger.info(f"Processing only target pages: {target_pages}")

        # Process consecutive page pairs
        for i in range(len(pages) - 1):
            page1 = pages[i]
            page2 = pages[i + 1]

            # Skip if not in target pages (when specified)
            if target_pages is not None:
                # Check if either page in the pair is a target
                if i not in target_pages and (i + 1) not in target_pages:
                    logger.debug(f"Skipping pages {i+1}-{i+2}: not in target pages")
                    continue

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
                    # Extract text previews for evidence
                    page1_text = self._extract_bottom_text(page1.text)
                    page2_text = self._extract_top_text(page2.text)

                    boundary = BoundaryResult(
                        page_number=i + 1,  # Boundary after this page
                        boundary_type=BoundaryType.DOCUMENT_END,
                        confidence=confidence,
                        detector_type=self.get_detector_type(),
                        evidence={
                            "llm_response": reasoning,
                            "processing_time": elapsed_time,
                            "model": self.model_name,
                            "page1_preview": page1_text[:100] + "..."
                            if len(page1_text) > 100
                            else page1_text,
                            "page2_preview": page2_text[:100] + "..."
                            if len(page2_text) > 100
                            else page2_text,
                        },
                        reasoning=reasoning,
                        is_between_pages=True,
                        next_page_number=i + 2,
                    )
                    boundaries.append(boundary)
                    self._detection_history.append(boundary)

            except Exception as e:
                logger.error(
                    f"Error analyzing pages {i+1}-{i+2}: {type(e).__name__}: {e}"
                )
                # Create error boundary result for tracking
                error_boundary = BoundaryResult(
                    page_number=i + 1,
                    boundary_type=BoundaryType.UNCERTAIN,
                    confidence=0.0,
                    detector_type=self.get_detector_type(),
                    evidence={
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    reasoning=f"Error during detection: {e}",
                    is_between_pages=True,
                    next_page_number=i + 2,
                )
                self._detection_history.append(error_boundary)
                continue

        # Update context if provided
        if context:
            context.update_progress(len(pages))

        self._last_detection_time = time.time()

        # Log summary of processing
        if target_pages is not None:
            logger.info(f"LLM detector processed {len(boundaries)} boundaries from target pages")
        else:
            logger.info(f"LLM detector found {len(boundaries)} boundaries from all {len(pages)} pages")

        return boundaries

    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.error(f"Ollama API returned status {response.status_code}")
                return False

            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            if self.model_name not in model_names:
                logger.warning(
                    f"Model {self.model_name} not found. Available: {model_names}"
                )
                # Suggest similar models if available
                similar = [m for m in model_names if "gemma" in m.lower()]
                if similar:
                    logger.info(f"Similar models available: {similar}")
                return False

            return True

        except (ConnectionError, Timeout) as e:
            logger.error(f"Cannot connect to Ollama at {self.ollama_url}: {e}")
            logger.info("Ensure Ollama is running with: ollama serve")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid response from Ollama API: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking Ollama: {type(e).__name__}: {e}")
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

        # Check persistent cache first
        if self._cache:
            cached_result = self._cache.get(
                page1_text, page2_text, self.model_name, self._prompt_version
            )
            if cached_result:
                return cached_result

        # Prepare prompt
        prompt = self.prompt_template.format(
            page1_bottom=page1_text, page2_top=page2_text
        )

        # Call LLM
        response = self._call_ollama(prompt)

        # Parse response
        result = self._parse_llm_response(response)

        # Cache response in persistent storage
        if self._cache and response:
            self._cache.put(
                page1_text,
                page2_text,
                self.model_name,
                response,
                result[0],
                result[1],
                result[2],
                self._prompt_version,
            )

        return result

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

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._cache:
            return self._cache.get_stats()
        return {"cache_enabled": False}

    def clear_cache(self):
        """Clear the cache - useful for testing."""
        if self._cache:
            self._cache.clear()
            logger.info("LLM detector cache cleared")

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
                            "temperature": self.llm_config.temperature,
                            "top_k": self.llm_config.top_k,
                            "top_p": self.llm_config.top_p,
                        },
                    },
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    try:
                        data = response.json()
                        if "error" in data:
                            logger.error(f"Ollama returned error: {data['error']}")
                            return ""
                        return data.get("response", "")
                    except json.JSONDecodeError:
                        logger.error("Failed to parse Ollama response as JSON")
                        return ""
                elif response.status_code == 404:
                    logger.error(
                        f"Model {self.model_name} not found. Pull it with: ollama pull {self.model_name}"
                    )
                    return ""
                else:
                    logger.error(
                        f"Ollama API error: {response.status_code} - {response.text[:200]}"
                    )

            except Timeout:
                logger.warning(
                    f"Ollama request timeout after {self.timeout}s (attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt == 0:
                    logger.info(
                        "Consider increasing timeout for slow models or reducing prompt size"
                    )
            except ConnectionError as e:
                logger.error(f"Cannot connect to Ollama: {e}")
                logger.info("Ensure Ollama is running with: ollama serve")
                return ""  # No point retrying connection errors
            except Exception as e:
                logger.error(
                    f"Unexpected error calling Ollama: {type(e).__name__}: {e}"
                )

            if attempt < self.max_retries - 1:
                wait_time = 2**attempt
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

        logger.error(f"Failed to get response after {self.max_retries} attempts")
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
            logger.warning("Empty response from LLM, defaulting to no boundary")
            return False, 0.0, "No response from LLM"

        # Extract reasoning from <thinking> tags
        reasoning = ""
        thinking_match = re.search(
            r"<thinking>(.*?)</thinking>", response, re.DOTALL | re.IGNORECASE
        )
        if thinking_match:
            reasoning = thinking_match.group(1).strip()
        else:
            logger.debug("No thinking tags found in response")

        # Extract answer from <answer> tags
        answer_match = re.search(
            r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE
        )
        if not answer_match:
            # Fallback: look for SAME/DIFFERENT anywhere in response
            if "DIFFERENT" in response.upper():
                logger.warning("No answer tags but found DIFFERENT in response")
                return True, 0.7, reasoning or "Inferred from response"
            elif "SAME" in response.upper():
                logger.warning("No answer tags but found SAME in response")
                return False, 0.7, reasoning or "Inferred from response"

            logger.warning(f"No valid answer found in response: {response[:200]}...")
            return False, 0.0, "Invalid response format"

        answer = answer_match.group(1).strip().upper()

        # Validate answer
        if answer not in ["SAME", "DIFFERENT"]:
            logger.warning(f"Unexpected answer: '{answer}', treating as SAME")
            return False, 0.5, reasoning or "Uncertain response"

        # Determine if it's a boundary
        is_boundary = answer == "DIFFERENT"

        # Assign confidence based on reasoning quality and configuration
        if reasoning and len(reasoning) > 20:
            # Good reasoning provided
            confidence = (
                self.llm_config.boundary_confidence_high
                if is_boundary
                else self.llm_config.continuation_confidence_high
            )
        elif reasoning:
            # Brief reasoning
            confidence = (
                self.llm_config.boundary_confidence_medium
                if is_boundary
                else self.llm_config.continuation_confidence_medium
            )
        else:
            # No reasoning
            confidence = (
                self.llm_config.boundary_confidence_medium
                if is_boundary
                else self.llm_config.continuation_confidence_low
            )

        return is_boundary, confidence, reasoning or "No reasoning provided"

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
            "cache_stats": self.get_cache_stats() if self._cache else None,
            "config": self.llm_config.to_dict(),
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
