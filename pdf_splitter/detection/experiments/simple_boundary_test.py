#!/usr/bin/env python3
"""
Test simplified boundary detection approach focusing on page transitions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.experiment_runner import OllamaClient
from pdf_splitter.detection.experiments.phi4_test_with_real_text import (
    process_pdf_correctly,
)


def extract_page_parts(text, chars_per_part=300):
    """Extract top and bottom portions of page text."""
    lines = text.strip().split("\n")

    # Get approximately chars_per_part from top
    top_text = ""
    for line in lines:
        if len(top_text) + len(line) <= chars_per_part:
            top_text += line + "\n"
        else:
            break

    # Get approximately chars_per_part from bottom
    bottom_text = ""
    for line in reversed(lines):
        if len(bottom_text) + len(line) <= chars_per_part:
            bottom_text = line + "\n" + bottom_text
        else:
            break

    return top_text.strip(), bottom_text.strip()


def test_boundary_simple(page1_text, page2_text, model="phi4-mini:3.8b"):
    """Test boundary using simple approach."""
    client = OllamaClient()

    # Extract relevant parts
    _, page1_bottom = extract_page_parts(page1_text)
    page2_top, _ = extract_page_parts(page2_text)

    # Simple, focused prompt
    prompt = f"""Your task is to determine if two document snippets are part of a single document or are different documents.

You will be given the bottom part of Page 1 and the top portion of Page 2. Your task is to determine whether Page 1 and Page 2 are part of a single document or if Page 1 is the end of one document and Page 2 is the start of a new document.

Please only respond with "Same Document" or "Different Documents"

Bottom of Page 1:
{page1_bottom}

Top of Page 2:
{page2_top}"""

    try:
        response = client.generate(
            model=model,
            prompt=prompt,
            temperature=0.0,  # Even lower for consistency
            max_tokens=50,  # We only need a short response
            timeout=30,
        )

        response_text = response.get("response", "").strip()
        return "Different Documents" in response_text

    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Test simplified boundary detection."""
    pdf_path = Path("test_files/Test_PDF_Set_2_ocr.pdf")

    print("Testing Simplified Boundary Detection")
    print("=" * 80)

    # Load pages
    pages = process_pdf_correctly(pdf_path, num_pages=10)

    # Test transitions
    transitions = [
        (3, 4, False),  # Same document
        (4, 5, True),  # Boundary - email ends, new email starts
        (5, 6, False),  # Same email thread
        (6, 7, True),  # Boundary - email ends, submittal starts
        (7, 8, False),  # Same submittal
        (8, 9, True),  # Boundary - submittal ends, schedule starts
    ]

    results = []

    for page1_num, page2_num, expected_boundary in transitions:
        print(f"\n{'='*60}")
        print(f"Testing transition: Page {page1_num} → Page {page2_num}")
        print(f"Expected: {'BOUNDARY' if expected_boundary else 'CONTINUATION'}")

        page1_text = pages[page1_num - 1].text
        page2_text = pages[page2_num - 1].text

        # Show what we're analyzing
        _, bottom = extract_page_parts(page1_text, 150)
        top, _ = extract_page_parts(page2_text, 150)

        print(f"\nBottom of Page {page1_num}:")
        print(f"  {bottom[:100]}...")
        print(f"\nTop of Page {page2_num}:")
        print(f"  {top[:100]}...")

        # Test detection
        detected_boundary = test_boundary_simple(page1_text, page2_text)

        print(f"\nDetected: {'BOUNDARY' if detected_boundary else 'CONTINUATION'}")
        correct = detected_boundary == expected_boundary
        print(f"Result: {'✓ CORRECT' if correct else '✗ INCORRECT'}")

        results.append(
            {
                "transition": f"{page1_num}→{page2_num}",
                "expected": expected_boundary,
                "detected": detected_boundary,
                "correct": correct,
            }
        )

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    correct_count = sum(1 for r in results if r["correct"])
    total = len(results)

    print(f"\n{'Transition':<15} {'Expected':<15} {'Detected':<15} {'Result':<10}")
    print("-" * 60)

    for r in results:
        print(
            f"{r['transition']:<15} "
            f"{'BOUNDARY' if r['expected'] else 'CONTINUATION':<15} "
            f"{'BOUNDARY' if r['detected'] else 'CONTINUATION':<15} "
            f"{'✓' if r['correct'] else '✗':<10}"
        )

    print(f"\nAccuracy: {correct_count}/{total} = {correct_count/total*100:.1f}%")

    # Compare to original approach
    print(f"\nOriginal approach: 0% recall (detected 0/3 boundaries)")
    print(f"Simplified approach: {correct_count}/{total} transitions correct")


if __name__ == "__main__":
    main()
