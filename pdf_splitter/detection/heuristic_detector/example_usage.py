#!/usr/bin/env python3
"""
Example usage of the heuristic detector with optimized configurations.

This script demonstrates how to use the heuristic detector for document
boundary detection with different configuration options.
"""

from pathlib import Path

from pdf_splitter.detection import (
    HeuristicDetector,
    ProcessedPage,
    get_fast_screen_config,
    get_high_precision_config,
    get_optimized_config,
)
from pdf_splitter.preprocessing import PDFHandler, TextExtractor


def main():
    """Demonstrate usage of heuristic detector with different configurations."""
    # Example PDF path
    pdf_path = Path("test_files/Test_PDF_Set_2_ocr.pdf")

    # Initialize PDF handler
    pdf_handler = PDFHandler()

    with pdf_handler.load_pdf(pdf_path):
        # Create text extractor
        text_extractor = TextExtractor(pdf_handler)

        # Process pages
        pages = []
        for page_num in range(1, min(10, pdf_handler.page_count + 1)):
            extracted = text_extractor.extract_page(page_num - 1)
            page = ProcessedPage(
                page_number=page_num,
                text=extracted.text,
                ocr_confidence=0.95,
                metadata={"word_count": extracted.word_count},
            )
            pages.append(page)

        print(f"Loaded {len(pages)} pages from {pdf_path.name}\n")

        # Example 1: Optimized configuration (balanced)
        print("1. Using Optimized Configuration (Balanced)")
        print("-" * 50)
        detector = HeuristicDetector(get_optimized_config())
        results = detector.detect_boundaries(pages[: len(pages) - 1])

        for i, result in enumerate(results):
            if result.boundary_type.value != "continuation":
                print(f"Boundary after page {i+1}: {result.boundary_type.value}")
                print(f"  Confidence: {result.confidence:.2f}")
                print(
                    f"  Active patterns: {result.evidence.get('active_patterns', [])}"
                )

        # Example 2: Fast screening configuration (high recall)
        print("\n2. Using Fast Screen Configuration (High Recall)")
        print("-" * 50)
        detector = HeuristicDetector(get_fast_screen_config())
        results = detector.detect_boundaries(pages[: len(pages) - 1])

        boundaries_found = sum(
            1 for r in results if r.boundary_type.value != "continuation"
        )
        print(f"Found {boundaries_found} potential boundaries")
        print("This configuration is designed for use with cascade architecture")

        # Example 3: High precision configuration
        print("\n3. Using High Precision Configuration")
        print("-" * 50)
        detector = HeuristicDetector(get_high_precision_config())
        results = detector.detect_boundaries(pages[: len(pages) - 1])

        for i, result in enumerate(results):
            if result.boundary_type.value != "continuation":
                print(f"High-confidence boundary after page {i+1}")
                print(f"  Confidence: {result.confidence:.2f}")
                print(f"  Pattern: {result.evidence.get('active_patterns', [])}")

        # Example 4: Custom configuration
        print("\n4. Using Custom Configuration")
        print("-" * 50)
        from pdf_splitter.detection.heuristic_detector import HeuristicConfig

        config = HeuristicConfig()
        # Only enable email detection for email-heavy documents
        for pattern_name in config.patterns:
            config.patterns[pattern_name].enabled = False

        config.patterns["email_header"].enabled = True
        config.patterns["email_header"].weight = 1.0

        detector = HeuristicDetector(config)
        results = detector.detect_boundaries(pages[: len(pages) - 1])

        email_boundaries = sum(
            1 for r in results if r.boundary_type.value != "continuation"
        )
        print(f"Found {email_boundaries} email boundaries using custom config")


if __name__ == "__main__":
    main()
