#!/usr/bin/env python3
"""
Example usage of the TextExtractor module.

This script demonstrates how to use the TextExtractor to extract text
from searchable PDFs for various purposes including ground truth generation
for OCR testing.
"""

from pathlib import Path

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.preprocessing.pdf_handler import PDFHandler
from pdf_splitter.preprocessing.text_extractor import TextExtractor


def main():
    """Demonstrate text extraction capabilities."""
    # Setup paths
    test_pdf = (
        Path(__file__).parent.parent.parent / "test_files" / "Test_PDF_Set_2_ocr.pdf"
    )

    if not test_pdf.exists():
        print(f"Test PDF not found: {test_pdf}")
        return

    # Initialize components
    config = PDFConfig()
    handler = PDFHandler(config)

    print(f"Loading PDF: {test_pdf.name}")
    print("=" * 50)

    with handler.load_pdf(test_pdf) as pdf_handler:
        extractor = TextExtractor(pdf_handler)

        # Example 1: Extract text from a single page
        print("\n1. Extract text from page 1:")
        print("-" * 50)
        page1 = extractor.extract_page(0)  # 0-indexed
        print(f"Quality Score: {page1.quality_score:.3f}")
        print(f"Word Count: {page1.word_count}")
        print(f"Font: {page1.dominant_font} (avg size: {page1.avg_font_size:.1f})")
        print(f"Has Headers: {page1.has_headers}, Has Footers: {page1.has_footers}")
        print("\nFirst 200 characters of text:")
        print(page1.text[:200] + "...")

        # Example 2: Extract all pages with statistics
        print("\n\n2. Extract all pages:")
        print("-" * 50)
        all_pages = extractor.extract_all_pages()
        print(f"Total pages extracted: {len(all_pages)}")

        # Calculate statistics
        total_words = sum(p.word_count for p in all_pages)
        avg_quality = sum(p.quality_score for p in all_pages) / len(all_pages)
        pages_with_tables = sum(1 for p in all_pages if p.tables)

        print(f"Total words: {total_words:,}")
        print(f"Average quality score: {avg_quality:.3f}")
        print(f"Pages with potential tables: {pages_with_tables}")

        # Example 3: Extract specific document segments
        print("\n\n3. Extract document segments (pages 1-4, 5-6, 7-8):")
        print("-" * 50)
        segments = extractor.extract_document_segments([(1, 4), (5, 6), (7, 8)])

        for i, segment in enumerate(segments, 1):
            pages = segment["page_range"]
            print(f"\nDocument {i} (pages {pages[0]}-{pages[1]}):")
            print(f"  Words: {segment['total_word_count']}")
            print(f"  Avg Quality: {segment['avg_quality_score']:.3f}")
            print(f"  Preview: {segment['full_text'][:100]}...")

        # Example 4: Analyze text blocks on a page
        print("\n\n4. Analyze text blocks on page 10:")
        print("-" * 50)
        page10 = extractor.extract_page(9)  # 0-indexed
        print(f"Total text blocks: {len(page10.blocks)}")

        if page10.blocks:
            # Show first few blocks
            for i, block in enumerate(page10.blocks[:3], 1):
                print(f"\nBlock {i}:")
                print(f"  Text: {block.text[:50]}...")
                print(f"  Font: {block.font_name} (size: {block.font_size:.1f})")
                print(f"  Bold: {block.is_bold}, Italic: {block.is_italic}")
                print(f"  Position: ({block.bbox[0]:.1f}, {block.bbox[1]:.1f})")

        # Example 5: Get PageText for compatibility
        print("\n\n5. PageText compatibility (for use with other modules):")
        print("-" * 50)
        page_text = extractor.extract_page_text(0)
        print(f"Extraction method: {page_text.extraction_method}")
        print(f"Confidence: {page_text.confidence:.3f}")
        print(f"Has tables: {page_text.has_tables}")
        print(f"BBox count: {page_text.bbox_count}")


if __name__ == "__main__":
    main()
