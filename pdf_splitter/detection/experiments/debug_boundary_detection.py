#!/usr/bin/env python3
"""
Debug why boundary detection is failing by testing specific pages.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.experiment_runner import OllamaClient
from pdf_splitter.detection.experiments.phi4_test_with_real_text import (
    process_pdf_correctly,
)


def test_specific_boundary(pages, page_num, model="phi4-mini:3.8b"):
    """Test boundary detection for a specific page."""
    client = OllamaClient()

    # Get context pages
    context_pages = []
    for i in range(max(0, page_num - 2), min(len(pages), page_num + 1)):
        if i != page_num - 1:  # Don't include current page in context
            context_pages.append(
                f"Page {pages[i].page_number}: {pages[i].text[:150]}..."
            )

    context_info = "\n".join(context_pages) if context_pages else "No context available"

    # Create prompt
    prompt = f"""You are analyzing pages from a PDF document to identify document boundaries.

Current page {page_num}:
{pages[page_num - 1].text[:300]}

Context pages:
{context_info}

Question: Is page {page_num} the start of a new document (not a continuation of the previous document)?
Answer with JSON: {{"boundary": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    print(f"\n{'='*60}")
    print(f"Testing Page {page_num}")
    print(f"{'='*60}")
    print(f"Expected: {'BOUNDARY' if page_num in [5, 7, 9] else 'NO BOUNDARY'}")

    try:
        response = client.generate(
            model=model, prompt=prompt, temperature=0.1, max_tokens=200, timeout=30
        )

        print(f"\nModel Response:")
        print(response.get("response", "No response"))

        # Try to parse JSON
        response_text = response.get("response", "")
        import re

        json_match = re.search(r"\{[^}]+\}", response_text)
        if json_match:
            result = json.loads(json_match.group())
            print(f"\nParsed Result:")
            print(f"  Boundary: {result.get('boundary', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 'N/A')}")
            print(f"  Reasoning: {result.get('reasoning', 'N/A')}")

            return result.get("boundary", False)
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Debug boundary detection."""
    pdf_path = Path("test_files/Test_PDF_Set_2_ocr.pdf")

    print("Debugging Boundary Detection")
    print("=" * 80)

    # Load first 10 pages
    pages = process_pdf_correctly(pdf_path, num_pages=10)

    # Test key boundaries
    test_pages = [4, 5, 6, 7, 8, 9]
    results = {}

    for page_num in test_pages:
        detected = test_specific_boundary(pages, page_num)
        expected = page_num in [5, 7, 9]
        results[page_num] = {
            "detected": detected,
            "expected": expected,
            "correct": detected == expected,
        }

    # Summary
    print(f"\n{'='*60}")
    print("DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Page':<10} {'Expected':<15} {'Detected':<15} {'Correct':<10}")
    print("-" * 60)

    correct = 0
    for page, result in results.items():
        print(
            f"{page:<10} {'BOUNDARY' if result['expected'] else 'NO BOUNDARY':<15} "
            f"{'BOUNDARY' if result['detected'] else 'NO BOUNDARY':<15} "
            f"{'✓' if result['correct'] else '✗':<10}"
        )
        if result["correct"]:
            correct += 1

    print(f"\nAccuracy: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")


if __name__ == "__main__":
    main()
