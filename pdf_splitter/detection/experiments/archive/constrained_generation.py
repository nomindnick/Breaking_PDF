#!/usr/bin/env python3
"""
Implement constrained generation for boundary detection using Outlines.

This module provides constrained generation to ensure models only output
valid tokens (S/D or SAME/DIFFERENT) for boundary detection.
"""

import re
import sys
from pathlib import Path
from typing import Literal, Optional, Union

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiment_runner import OllamaClient
from pydantic import BaseModel


class BoundaryDecision(BaseModel):
    """Structured output for boundary detection."""

    decision: Literal["SAME", "DIFFERENT"]
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class SingleTokenDecision(BaseModel):
    """Single token output."""

    decision: Literal["S", "D"]


def create_constrained_prompt(
    page1_bottom: str, page2_top: str, output_format: str = "structured"
) -> str:
    """Create a prompt optimized for constrained generation."""

    if output_format == "single_token":
        return f"""Determine if these pages are from the same document.
Page 1 ends: {page1_bottom}
Page 2 starts: {page2_top}
Output only 'S' for same document or 'D' for different documents.
Decision:"""

    elif output_format == "structured":
        return f"""You are a document boundary detector. Analyze these page transitions and output a JSON object.

Page 1 ends with: {page1_bottom}
Page 2 starts with: {page2_top}

Output a JSON object with:
- decision: "SAME" or "DIFFERENT"
- confidence: 0.0 to 1.0
- reasoning: Brief explanation

Example: {{"decision": "SAME", "confidence": 0.9, "reasoning": "Sentence continues across pages"}}

JSON:"""

    else:  # xml format
        return f"""<system>You are a document boundary detector.</system>
<user>
Page 1 ends: {page1_bottom}
Page 2 starts: {page2_top}
</user>
<assistant>
I need to determine if these pages are from the same document.
<decision>"""


class ConstrainedBoundaryDetector:
    """Boundary detector with constrained generation."""

    def __init__(self, model: str = "phi4-mini:3.8b"):
        self.model = model
        self.client = OllamaClient()

    def detect_with_regex(
        self, page1_bottom: str, page2_top: str, pattern: str = r"^(S|D)$"
    ) -> str:
        """Use regex-constrained generation for single token output."""
        prompt = create_constrained_prompt(page1_bottom, page2_top, "single_token")

        # With Ollama, we can't directly use Outlines' regex constraints,
        # but we can use stop tokens and post-processing
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=1,
            stop=[" ", "\n", "\t", ".", ",", ":", ";", "!"],
        )

        text = response.get("response", "").strip().upper()

        # Validate against pattern
        if re.match(pattern, text):
            return text
        else:
            # Fallback: extract first S or D
            match = re.search(r"[SD]", text.upper())
            return match.group() if match else "S"

    def detect_with_json(self, page1_bottom: str, page2_top: str) -> BoundaryDecision:
        """Use JSON-constrained generation for structured output."""
        prompt = create_constrained_prompt(page1_bottom, page2_top, "structured")

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=100,
            stop=["\n\n", "```", "</json>"],
        )

        text = response.get("response", "").strip()

        # Try to parse JSON
        try:
            import json

            # Extract JSON if embedded in text
            json_match = re.search(r"\{[^}]+\}", text)
            if json_match:
                data = json.loads(json_match.group())
                return BoundaryDecision(
                    decision=data.get("decision", "SAME").upper(),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", ""),
                )
        except:
            pass

        # Fallback: extract decision from text
        if "DIFFERENT" in text.upper():
            return BoundaryDecision(decision="DIFFERENT", confidence=0.5)
        else:
            return BoundaryDecision(decision="SAME", confidence=0.5)

    def detect_with_xml(self, page1_bottom: str, page2_top: str) -> str:
        """Use XML-constrained generation."""
        prompt = create_constrained_prompt(page1_bottom, page2_top, "xml")

        # We want the model to complete the <decision> tag
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=20,
            stop=["</decision>", "\n", "<"],
        )

        text = response.get("response", "").strip().upper()

        # Extract SAME or DIFFERENT
        if "DIFFERENT" in text:
            return "DIFFERENT"
        elif "SAME" in text:
            return "SAME"
        else:
            # Try single letter
            match = re.search(r"[SD]", text)
            if match:
                return "SAME" if match.group() == "S" else "DIFFERENT"
            return "SAME"  # Default

    def benchmark_approaches(self, test_cases):
        """Benchmark different constrained generation approaches."""
        results = {
            "regex": {"correct": 0, "total": 0, "errors": 0},
            "json": {"correct": 0, "total": 0, "errors": 0},
            "xml": {"correct": 0, "total": 0, "errors": 0},
        }

        for case in test_cases:
            # Test regex approach
            try:
                prediction = self.detect_with_regex(case.page1_bottom, case.page2_top)
                if prediction == "S":
                    prediction = "Same"
                elif prediction == "D":
                    prediction = "Different"

                results["regex"]["total"] += 1
                if prediction == case.expected:
                    results["regex"]["correct"] += 1
            except Exception as e:
                results["regex"]["errors"] += 1
                print(f"Regex error: {e}")

            # Test JSON approach
            try:
                decision = self.detect_with_json(case.page1_bottom, case.page2_top)
                prediction = "Same" if decision.decision == "SAME" else "Different"

                results["json"]["total"] += 1
                if prediction == case.expected:
                    results["json"]["correct"] += 1
            except Exception as e:
                results["json"]["errors"] += 1
                print(f"JSON error: {e}")

            # Test XML approach
            try:
                prediction = self.detect_with_xml(case.page1_bottom, case.page2_top)
                if prediction == "SAME":
                    prediction = "Same"
                else:
                    prediction = "Different"

                results["xml"]["total"] += 1
                if prediction == case.expected:
                    results["xml"]["correct"] += 1
            except Exception as e:
                results["xml"]["errors"] += 1
                print(f"XML error: {e}")

        return results


def main():
    """Test constrained generation approaches."""
    from enhanced_synthetic_tests import EnhancedSyntheticTester

    print("Testing Constrained Generation for Boundary Detection")
    print("=" * 60)

    # Get test cases
    tester = EnhancedSyntheticTester()
    test_cases = tester.test_cases[:5]  # Just test with 5 cases

    # Test with different models
    models = ["phi4-mini:3.8b", "gemma3:latest"]

    for model in models:
        print(f"\nTesting {model}")
        print("-" * 40)

        detector = ConstrainedBoundaryDetector(model)
        results = detector.benchmark_approaches(test_cases)

        # Print results
        for approach, stats in results.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                print(f"\n{approach.upper()} approach:")
                print(
                    f"  Accuracy: {accuracy:.2%} ({stats['correct']}/{stats['total']})"
                )
                print(f"  Errors: {stats['errors']}")


if __name__ == "__main__":
    main()
