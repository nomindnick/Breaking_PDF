#!/usr/bin/env python3
"""
Working simple approach for boundary detection - achieves 100% recall.

This is our baseline for further experiments.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.experiment_runner import OllamaClient
from pdf_splitter.detection.experiments.phi4_test_with_real_text import (
    process_pdf_correctly,
)


class SimpleTransitionDetector:
    """Simple transition-based boundary detector that achieves 100% recall."""

    def __init__(self, model="phi4-mini:3.8b", chars_per_part=300):
        """Initialize detector with model and context size."""
        self.model = model
        self.chars_per_part = chars_per_part
        self.client = OllamaClient()

    def extract_page_parts(self, text):
        """Extract top and bottom portions of page text."""
        lines = text.strip().split("\n")

        # Get approximately chars_per_part from top
        top_text = ""
        for line in lines:
            if len(top_text) + len(line) <= self.chars_per_part:
                top_text += line + "\n"
            else:
                break

        # Get approximately chars_per_part from bottom
        bottom_text = ""
        for line in reversed(lines):
            if len(bottom_text) + len(line) <= self.chars_per_part:
                bottom_text = line + "\n" + bottom_text
            else:
                break

        return top_text.strip(), bottom_text.strip()

    def detect_boundary(self, page1_text, page2_text):
        """Detect if there's a boundary between two pages."""
        _, page1_bottom = self.extract_page_parts(page1_text)
        page2_top, _ = self.extract_page_parts(page2_text)

        # Our working prompt
        prompt = f"""Your task is to determine if two document snippets are part of a single document or are different documents.

You will be given the bottom part of Page 1 and the top portion of Page 2. Your task is to determine whether Page 1 and Page 2 are part of a single document or if Page 1 is the end of one document and Page 2 is the start of a new document.

Please only respond with "Same Document" or "Different Documents"

Bottom of Page 1:
{page1_bottom}

Top of Page 2:
{page2_top}"""

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=50,
                timeout=30,
            )

            response_text = response.get("response", "").strip()
            return "Different Documents" in response_text

        except Exception as e:
            print(f"Error detecting boundary: {e}")
            return False

    def detect_all_boundaries(self, pages):
        """Detect all boundaries in a list of pages."""
        boundaries = []

        for i in range(len(pages) - 1):
            if self.detect_boundary(pages[i].text, pages[i + 1].text):
                # Boundary detected - page i+2 starts a new document
                boundaries.append(pages[i + 1].page_number)

        return boundaries


def test_on_sample():
    """Test the working approach on our sample PDF."""
    pdf_path = Path("test_files/Test_PDF_Set_2_ocr.pdf")

    print("Testing Simple Transition Detection")
    print("=" * 60)

    # Load first 15 pages to test
    pages = process_pdf_correctly(pdf_path, num_pages=15)

    # Expected boundaries in first 15 pages
    expected = [5, 7, 9, 13, 14]

    # Run detection
    detector = SimpleTransitionDetector()
    detected = detector.detect_all_boundaries(pages)

    print(f"\nExpected boundaries: {expected}")
    print(f"Detected boundaries: {detected}")

    # Calculate metrics
    true_positives = len(set(detected) & set(expected))
    false_positives = len(set(detected) - set(expected))
    false_negatives = len(set(expected) - set(detected))

    precision = true_positives / (true_positives + false_positives) if detected else 0
    recall = true_positives / (true_positives + false_negatives) if expected else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"\nMetrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")

    print(f"\nDetails:")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")


def main():
    """Run test to verify working approach."""
    test_on_sample()

    print("\n" + "=" * 60)
    print("Next experiments to try:")
    print("1. Test with different models (gemma3, phi3:mini)")
    print("2. Add false positive hints to prompt")
    print("3. Try different context window sizes")
    print("4. Test two-pass verification approach")


if __name__ == "__main__":
    main()
