#!/usr/bin/env python3
"""
Example usage of the VisualDetector for document boundary detection.

This script demonstrates how to use the visual detector both standalone
and as a supplementary signal to other detection methods.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from pdf_splitter.core.config import PDFConfig  # noqa: E402
from pdf_splitter.detection.base_detector import ProcessedPage  # noqa: E402
from pdf_splitter.detection.visual_detector import VisualDetector  # noqa: E402
from pdf_splitter.preprocessing.pdf_handler import PDFHandler  # noqa: E402
from pdf_splitter.preprocessing.text_extractor import TextExtractor  # noqa: E402


def demonstrate_visual_detection(pdf_path: Path) -> None:
    """
    Demonstrate visual boundary detection on a PDF file.

    Args:
        pdf_path: Path to the PDF file to analyze
    """
    print("\n=== Visual Boundary Detection Demo ===")
    print(f"PDF: {pdf_path.name}")
    print("-" * 50)

    # Initialize components
    config = PDFConfig()
    pdf_handler = PDFHandler(config)
    text_extractor = TextExtractor(pdf_handler)

    # Create visual detector with sensitive settings
    visual_detector = VisualDetector(
        config=config,
        pdf_handler=pdf_handler,
        voting_threshold=1,  # Require only 1 algorithm to vote for boundary
    )

    # Load and process PDF
    with pdf_handler.load_pdf(pdf_path):
        num_pages = pdf_handler.page_count
        print(f"Pages: {num_pages}")

        # Extract text and create processed pages
        processed_pages = []
        for page_num in range(num_pages):
            extracted_page = text_extractor.extract_page(page_num)

            processed_page = ProcessedPage(
                page_number=page_num + 1,
                text=extracted_page.text,
                page_type="SEARCHABLE",  # Visual detector doesn't need OCR info
                metadata={
                    "word_count": extracted_page.word_count,
                    "quality_score": extracted_page.quality_score,
                    "has_headers": extracted_page.has_headers,
                    "has_footers": extracted_page.has_footers,
                },
            )
            processed_pages.append(processed_page)

        # Detect boundaries (while PDF is still loaded)
        print("\nDetecting visual boundaries...")
        boundaries = visual_detector.detect_boundaries(processed_pages)

        # Display results
        if not boundaries:
            print("No visual boundaries detected.")
        else:
            print(f"\nFound {len(boundaries)} visual boundaries:")
            for boundary in boundaries:
                print(f"\n  Boundary after page {boundary.page_number}:")
                print(f"    Confidence: {boundary.confidence:.3f}")
                print(f"    Votes: {boundary.evidence.get('votes', 'N/A')}/3")
                print("    Hash distances:")
                print(
                    f"      - pHash: {boundary.evidence.get('phash_distance', 'N/A')}"
                )
                print(
                    f"      - aHash: {boundary.evidence.get('ahash_distance', 'N/A')}"
                )
                print(
                    f"      - dHash: {boundary.evidence.get('dhash_distance', 'N/A')}"
                )

        # Show statistics
        stats = visual_detector.get_detection_stats()
        print("\nDetection Statistics:")
        print(f"  Total detections: {stats['detections']}")
        if stats["detections"] > 0:
            print(f"  Average confidence: {stats['avg_confidence']:.3f}")
            print(f"  Min confidence: {stats['min_confidence']:.3f}")
            print(f"  Max confidence: {stats['max_confidence']:.3f}")

    # Clean up
    visual_detector.clear_cache()


def demonstrate_combined_detection(pdf_path: Path) -> None:
    """
    Demonstrate how visual detection can supplement other detectors.

    This shows the intended production usage where visual signals
    provide additional confidence to semantic boundary detection.

    Args:
        pdf_path: Path to the PDF file to analyze
    """
    print("\n\n=== Combined Detection Demo ===")
    print("(Simulating visual + semantic detection)")
    print("-" * 50)

    # Initialize visual detector
    config = PDFConfig()
    pdf_handler = PDFHandler(config)
    _ = VisualDetector(
        config=config,
        pdf_handler=pdf_handler,
        voting_threshold=2,  # More conservative for supplementary use
    )

    # In production, you would also have semantic detectors
    # For this demo, we'll just show how to combine signals

    with pdf_handler.load_pdf(pdf_path):
        print(f"Processing {pdf_handler.page_count} pages...")

    # Simulate boundary detection
    # In production: semantic_boundaries = semantic_detector.detect_boundaries(pages)
    # visual_boundaries = visual_detector.detect_boundaries(pages)

    print("\nIn production, you would:")
    print("1. Run semantic detection (LLM-based) as primary")
    print("2. Run visual detection as supplementary")
    print("3. Combine results with weighted confidence:")
    print("   - High confidence if both agree")
    print("   - Medium confidence if only semantic detects")
    print("   - Low confidence if only visual detects")


def main():
    """Run the demonstration."""
    # Check if PDF path provided
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <path_to_pdf>")
        print("\nExample PDFs to try:")
        print("  - test_files/Test_PDF_Set_1.pdf")
        print("  - test_files/visual_test_pdf.pdf")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    # Run demonstrations
    demonstrate_visual_detection(pdf_path)
    demonstrate_combined_detection(pdf_path)


if __name__ == "__main__":
    main()
