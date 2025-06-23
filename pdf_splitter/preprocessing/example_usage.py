"""
Example usage of the PDFHandler class.

This script demonstrates how to use the PDFHandler for various PDF processing tasks.
"""

import sys
from pathlib import Path

# from pprint import pprint  # Not used in this example

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from pdf_splitter.core.config import PDFConfig  # noqa: E402
from pdf_splitter.preprocessing import PageType, PDFHandler  # noqa: E402


def main():
    """Demonstrate PDFHandler usage."""
    # Create custom configuration
    config = PDFConfig(
        default_dpi=150, max_dpi=300, page_cache_size=10, stream_batch_size=5
    )

    # Initialize handler
    handler = PDFHandler(config=config)

    # Example PDF path (update this to your test PDF)
    pdf_path = Path("../../Test_PDF_Set_1.pdf")

    if not pdf_path.exists():
        print(f"PDF file not found: {pdf_path}")
        print("Please update the pdf_path variable to point to a valid PDF file.")
        return

    print(f"Processing PDF: {pdf_path}")
    print("-" * 50)

    # Step 1: Validate the PDF
    print("1. Validating PDF...")
    validation = handler.validate_pdf(pdf_path)
    print(f"   Valid: {validation.is_valid}")
    print(f"   Pages: {validation.page_count}")
    print(f"   Size: {validation.file_size_mb:.2f} MB")
    print(f"   Warnings: {validation.warnings}")
    print(f"   Errors: {validation.errors}")
    print()

    if not validation.is_valid:
        print("PDF validation failed. Exiting.")
        return

    # Step 2: Load and analyze the PDF
    with handler.load_pdf(pdf_path) as h:
        print("2. PDF loaded successfully")
        print(f"   Total pages: {h.page_count}")
        print()

        # Get metadata
        print("3. Document metadata:")
        metadata = h.get_metadata()
        if metadata:
            print(f"   Title: {metadata.title}")
            print(f"   Author: {metadata.author}")
            print(f"   Producer: {metadata.producer}")
            print(f"   PDF Version: {metadata.pdf_version}")
        print()

        # Analyze all pages
        print("4. Analyzing all pages...")
        page_infos = h.analyze_all_pages()

        # Summary by page type
        page_type_summary = {}
        for info in page_infos:
            page_type_summary[info.page_type] = (
                page_type_summary.get(info.page_type, 0) + 1
            )

        print("   Page type summary:")
        for page_type, count in page_type_summary.items():
            print(f"   - {page_type.value}: {count} pages")
        print()

        # Estimate processing time
        print("5. Processing time estimate:")
        estimate = h.estimate_processing_time()
        print(f"   Total pages: {estimate.total_pages}")
        print(f"   Estimated time: {estimate.estimated_seconds:.1f} seconds")
        print(f"   Estimated memory: {estimate.estimated_memory_mb:.1f} MB")
        print(f"   Pages requiring OCR: {estimate.requires_ocr_pages}")
        print()

        # Demo: Process first few pages
        print("6. Processing first 3 pages...")
        for i in range(min(3, h.page_count)):
            print(f"\n   Page {i}:")

            # Get page type
            page_type = h.get_page_type(i)
            print(f"   - Type: {page_type.value}")

            # Extract text if searchable
            if page_type in [PageType.SEARCHABLE, PageType.MIXED]:
                text_data = h.extract_text(i)
                print(f"   - Text preview: {text_data.text[:100]}...")
                print(f"   - Word count: {text_data.word_count}")
                print(f"   - Confidence: {text_data.confidence}")
                print(f"   - Has tables: {text_data.has_tables}")
                print(f"   - Has images: {text_data.has_images}")

            # Render page (just to demonstrate - not displaying)
            if page_type != PageType.EMPTY:
                img_array = h.render_page(i)
                print(f"   - Rendered shape: {img_array.shape}")

        print()

        # Demo: Stream processing
        print("7. Streaming pages in batches...")
        batch_count = 0
        for batch in h.stream_pages(batch_size=5, end_page=10):
            batch_count += 1
            print(f"   Batch {batch_count}: pages {batch.start_idx}-{batch.end_idx-1}")
            print(f"   - Batch size: {batch.batch_size}")

            # Show info for first page in batch
            if batch.pages:
                first_page = batch.pages[0]
                if "error" not in first_page:
                    print(f"   - First page type: {first_page['page_type'].value}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
