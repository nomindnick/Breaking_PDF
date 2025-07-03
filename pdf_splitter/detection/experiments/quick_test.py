#!/usr/bin/env python3
"""
Quick test to verify LLM detection is working with a small subset of pages.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.base_detector import (
    BoundaryResult,
    BoundaryType,
    DetectorType,
    ProcessedPage,
)
from pdf_splitter.detection.experiments.experiment_runner import OllamaClient


def main():
    """Run a quick test with just a few pages."""
    print("Quick LLM Detection Test")
    print("=" * 60)

    # Create test pages with actual content
    test_pages = [
        ProcessedPage(
            page_number=1,
            text="From: John Smith\nTo: Jane Doe\nSubject: Project Update\nDate: March 1, 2024\n\nHello Jane,\n\nI wanted to update you on the project status...",
            page_type="SEARCHABLE",
        ),
        ProcessedPage(
            page_number=2,
            text="...continued from previous page. The timeline looks good and we should be able to deliver on schedule.\n\nBest regards,\nJohn",
            page_type="SEARCHABLE",
        ),
        ProcessedPage(
            page_number=3,
            text="INVOICE\n\nInvoice #: 2024-001\nDate: March 2, 2024\n\nBill To:\nAcme Corporation\n123 Main St\n\nDescription: Consulting Services\nAmount: $5,000.00",
            page_type="SEARCHABLE",
        ),
    ]

    # Initialize Ollama client
    client = OllamaClient()

    # Test each page
    for i, page in enumerate(test_pages):
        print(f"\nChecking page {page.page_number}...")

        # Create a simple prompt
        prompt = f"""You are analyzing a PDF to find document boundaries.

Current page {page.page_number}:
{page.text}

Is this the start of a new document (not a continuation)?
Answer with JSON: {{"boundary": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

        try:
            # Call Ollama
            response = client.generate(
                model="llama3:8b-instruct-q5_K_M",
                prompt=prompt,
                temperature=0.1,
                max_tokens=200,
                timeout=30,
            )

            print(f"Response: {response.get('response', 'No response')}")

        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()
