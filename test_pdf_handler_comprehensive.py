#!/usr/bin/env python3
"""Comprehensive test of PDFHandler functionality with actual PDF files."""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pdf_splitter.preprocessing import PageType, PDFHandler  # noqa: E402


def test_comprehensive():
    """Test comprehensive PDFHandler functionality."""
    print("=== Comprehensive PDFHandler Test ===\n")

    # Initialize handler
    handler = PDFHandler()

    # Test both PDFs
    test_pdfs = [
        ("Non-OCR PDF", Path("test_files/Test_PDF_Set_1.pdf")),
        ("OCR'd PDF", Path("test_files/Test_PDF_Set_2_ocr.pdf")),
    ]

    for pdf_name, pdf_path in test_pdfs:
        if not pdf_path.exists():
            print(f"⚠ {pdf_name} not found at {pdf_path}")
            continue

        print(f"\n{'='*50}")
        print(f"Testing {pdf_name}: {pdf_path}")
        print("=" * 50)

        # Validation
        print("\n1. Validation:")
        validation = handler.validate_pdf(pdf_path)
        print(f"   Valid: {validation.is_valid}")
        print(f"   Pages: {validation.page_count}")
        print(f"   Size: {validation.file_size_mb:.2f} MB")
        print(f"   Encrypted: {validation.is_encrypted}")
        print(f"   Damaged: {validation.is_damaged}")
        print(f"   PDF Version: {validation.pdf_version}")

        if not validation.is_valid:
            print(f"   ❌ Validation failed: {validation.errors}")
            continue

        # Load and analyze
        with handler.load_pdf(pdf_path) as h:
            # Metadata
            print("\n2. Metadata:")
            metadata = h.get_metadata()
            if metadata:
                print(f"   Title: {metadata.title or 'N/A'}")
                print(f"   Author: {metadata.author or 'N/A'}")
                print(f"   Creator: {metadata.creator or 'N/A'}")
                print(f"   Producer: {metadata.producer or 'N/A'}")
                print(f"   Creation Date: {metadata.creation_date or 'N/A'}")

            # Analyze all pages (with timing)
            print("\n3. Page Analysis:")
            start_time = time.time()
            page_infos = h.analyze_all_pages(max_workers=4)
            analysis_time = time.time() - start_time
            print(f"   Analyzed {len(page_infos)} pages in {analysis_time:.2f} seconds")

            # Page type summary
            if metadata and metadata.page_info_summary:
                print("   Page types:")
                for page_type, count in metadata.page_info_summary.items():
                    print(f"   - {page_type}: {count} pages")

            # Processing estimate
            print("\n4. Processing Estimate:")
            estimate = h.estimate_processing_time()
            print(f"   Total pages: {estimate.total_pages}")
            print(f"   Searchable: {estimate.searchable_pages}")
            print(f"   Requires OCR: {estimate.requires_ocr_pages}")
            print(f"   Mixed: {estimate.mixed_pages}")
            print(f"   Empty: {estimate.empty_pages}")
            print(f"   Estimated time: {estimate.estimated_seconds:.1f} seconds")
            print(f"   Estimated memory: {estimate.estimated_memory_mb:.1f} MB")

            # Test first 3 pages in detail
            print("\n5. Page Details (first 3 pages):")
            for i in range(min(3, h.page_count)):
                info = page_infos[i] if i < len(page_infos) else None
                if info:
                    print(f"\n   Page {i}:")
                    print(f"   - Type: {info.page_type}")
                    print(f"   - Dimensions: {info.width:.0f}x{info.height:.0f}")
                    print(f"   - Rotation: {info.rotation}°")
                    print(f"   - Text coverage: {info.text_percentage:.1f}%")
                    print(f"   - Images: {info.image_count}")
                    print(f"   - Has annotations: {info.has_annotations}")

                    # Extract text if searchable
                    if info.page_type in [PageType.SEARCHABLE, PageType.MIXED]:
                        text_data = h.extract_text(i)
                        print(f"   - Words: {text_data.word_count}")
                        print(f"   - Characters: {text_data.char_count}")
                        print(f"   - Confidence: {text_data.confidence}")
                        print(f"   - Has tables: {text_data.has_tables}")
                        if text_data.text:
                            preview = text_data.text[:100].replace("\n", " ")
                            print(f"   - Text preview: {preview}...")

                    # Test rendering
                    try:
                        start_render = time.time()
                        img = h.render_page(i)
                        render_time = time.time() - start_render
                        print(f"   - Rendered in {render_time:.3f}s: {img.shape}")
                    except Exception as e:
                        print(f"   - Render error: {e}")

            # Test streaming
            print("\n6. Streaming Test:")
            batch_count = 0
            total_pages = 0
            for batch in h.stream_pages(batch_size=5, end_page=min(10, h.page_count)):
                batch_count += 1
                total_pages += batch.batch_size
                print(
                    f"   Batch {batch_count}: {batch.batch_size} pages "
                    f"(idx {batch.start_idx}-{batch.end_idx-1})"
                )
            print(f"   Total: {total_pages} pages in {batch_count} batches")

    print("\n✅ Comprehensive test complete!")


if __name__ == "__main__":
    test_comprehensive()
