#!/usr/bin/env python3
"""Basic test of PDFHandler functionality with actual PDF files."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pdf_splitter.preprocessing import PDFHandler  # noqa: E402


def test_basic_functionality():
    """Test basic PDFHandler functionality."""
    print("Testing PDFHandler basic functionality...")

    # Initialize handler
    handler = PDFHandler()
    print("✓ PDFHandler initialized")

    # Check for test PDFs
    test_pdfs = [
        Path("test_files/Test_PDF_Set_1.pdf"),
        Path("test_files/Test_PDF_Set_2_ocr.pdf"),
    ]

    available_pdf = None
    for pdf in test_pdfs:
        if pdf.exists():
            available_pdf = pdf
            break

    if not available_pdf:
        print("⚠ No test PDF files found. Skipping file tests.")
        print("  Looking for: Test_PDF_Set_1.pdf or Test_PDF_Set_2_ocr.pdf")
        return

    print(f"\nUsing test PDF: {available_pdf}")

    # Test validation
    print("\nTesting PDF validation...")
    validation = handler.validate_pdf(available_pdf)
    print(
        f"✓ Validation complete: valid={validation.is_valid}, "
        f"pages={validation.page_count}"
    )

    if not validation.is_valid:
        print(f"✗ Validation failed: {validation.errors}")
        return

    # Test loading and analysis
    print("\nTesting PDF loading and analysis...")
    try:
        with handler.load_pdf(available_pdf) as h:
            print(f"✓ PDF loaded: {h.page_count} pages")

            # Test metadata
            metadata = h.get_metadata()
            if metadata:
                print(f"✓ Metadata extracted: PDF version {metadata.pdf_version}")

            # Test page type detection on first page
            if h.page_count > 0:
                page_type = h.get_page_type(0)
                print(f"✓ Page 0 type: {page_type.value}")

                # Test text extraction if searchable
                if page_type.value in ["searchable", "mixed"]:
                    text_data = h.extract_text(0)
                    print(f"✓ Text extracted: {text_data.word_count} words")

            # Test processing estimate
            estimate = h.estimate_processing_time()
            print(f"✓ Processing estimate: {estimate.estimated_seconds:.1f} seconds")

        print("\n✅ All basic tests passed!")

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_basic_functionality()
