#!/usr/bin/env python3
"""
Example usage of the OCR processor module.

This script demonstrates how to:
- Initialize the OCR processor
- Process individual pages
- Handle different page types
- Use caching for performance
- Process pages in parallel
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_splitter.core.config import PDFConfig  # noqa: E402
from pdf_splitter.preprocessing.ocr_processor import (  # noqa: E402
    OCRConfig,
    OCREngine,
    OCRProcessor,
)
from pdf_splitter.preprocessing.pdf_handler import PageType, PDFHandler  # noqa: E402


def main():
    """Demonstrate OCR processor usage."""
    # Configuration
    pdf_path = Path("test_files/Test_PDF_Set_1.pdf")

    if not pdf_path.exists():
        print(f"Test PDF not found: {pdf_path}")
        print("Please ensure test PDFs are available.")
        return

    # Initialize components
    pdf_config = PDFConfig(
        default_dpi=150,
        enable_cache_metrics=True,
    )

    ocr_config = OCRConfig(
        primary_engine=OCREngine.PADDLEOCR,
        fallback_engines=[OCREngine.EASYOCR],
        preprocessing_enabled=True,
        cache_enabled=True,
        max_workers=4,
    )

    pdf_handler = PDFHandler(config=pdf_config)
    ocr_processor = OCRProcessor(
        config=ocr_config,
        pdf_config=pdf_config,
        cache_manager=pdf_handler.cache_manager,
    )

    print(f"Processing PDF: {pdf_path}")
    print("-" * 60)

    with pdf_handler.load_pdf(pdf_path):
        print(f"Total pages: {pdf_handler.page_count}")

        # Analyze page types
        print("\nAnalyzing page types...")
        page_infos = pdf_handler.analyze_all_pages(max_workers=4)

        # Count page types
        page_type_counts = {}
        for info in page_infos:
            page_type_counts[info.page_type] = (
                page_type_counts.get(info.page_type, 0) + 1
            )

        print("\nPage type distribution:")
        for page_type, count in page_type_counts.items():
            print(f"  {page_type}: {count} pages")

        # Process first few IMAGE_BASED pages
        print("\nProcessing IMAGE_BASED pages with OCR...")
        processed_count = 0
        max_to_process = 3

        for page_num in range(min(10, pdf_handler.page_count)):
            page_type = pdf_handler.get_page_type(page_num)

            if page_type == PageType.IMAGE_BASED and processed_count < max_to_process:
                print(f"\nPage {page_num + 1}:")

                # Render page
                image = pdf_handler.render_page(page_num)

                # Process with OCR
                result = ocr_processor.process_image(
                    image,
                    page_num,
                    page_type,
                    pdf_path=str(pdf_path),
                )

                # Display results
                print(f"  Processing time: {result.processing_time:.2f}s")
                print(f"  Engine used: {result.engine_used.value}")
                print(f"  Confidence: {result.avg_confidence:.2f}")
                print(f"  Quality score: {result.quality_score:.2f}")
                print(f"  Word count: {result.word_count}")
                print(f"  Text preview: {result.full_text[:100]}...")

                if result.warnings:
                    print(f"  Warnings: {', '.join(result.warnings)}")

                processed_count += 1

        # Demonstrate batch processing
        print("\n" + "-" * 60)
        print("Demonstrating batch processing...")

        # Collect a batch of IMAGE_BASED pages
        batch_images = []
        for page_num in range(min(20, pdf_handler.page_count)):
            page_type = pdf_handler.get_page_type(page_num)
            if page_type == PageType.IMAGE_BASED and len(batch_images) < 5:
                image = pdf_handler.render_page(page_num)
                batch_images.append((image, page_num, page_type))

        if batch_images:
            print(f"\nProcessing batch of {len(batch_images)} pages...")

            import time

            start_time = time.time()

            batch_results = ocr_processor.process_batch(
                batch_images,
                max_workers=2,
                pdf_path=str(pdf_path),
            )

            batch_time = time.time() - start_time

            print(f"Batch processing completed in {batch_time:.2f}s")
            print(f"Average time per page: {batch_time / len(batch_images):.2f}s")

            # Show batch results summary
            avg_confidence = sum(r.avg_confidence for r in batch_results) / len(
                batch_results
            )
            print(f"Average confidence: {avg_confidence:.2f}")

        # Display performance statistics
        print("\n" + "-" * 60)
        print("Performance Statistics:")
        stats = ocr_processor.get_performance_stats()

        print(f"  Total pages processed: {stats['total_pages_processed']}")
        print(f"  Total processing time: {stats['total_processing_time']:.2f}s")
        print(f"  Average time per page: {stats['avg_time_per_page']:.2f}s")
        print(f"  Engine usage: {stats['engine_usage']}")

        # Display cache statistics
        cache_stats = pdf_handler.get_cache_stats()
        print("\nCache Statistics:")
        print(f"  Total memory usage: {cache_stats['total_memory_mb']:.1f}MB")

        if "analysis_cache" in cache_stats:
            analysis = cache_stats["analysis_cache"]
            print(f"  OCR cache hit rate: {analysis.get('hit_rate', 0):.1%}")

    # Cleanup
    ocr_processor.cleanup()
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
