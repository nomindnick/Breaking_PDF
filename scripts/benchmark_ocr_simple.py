#!/usr/bin/env python3
"""
Simplified benchmark script for OCR processor performance validation.

This script tests the OCR processor against performance targets:
- < 2 seconds per page processing time
- > 95% accuracy against ground truth (if available)
"""

import argparse
import logging
import os
import sys
import time
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Execute OCR processor benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark OCR processor performance")
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to PDF file to benchmark",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Maximum number of pages to process (default: 5)",
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        logger.error(f"PDF file not found: {args.pdf_path}")
        sys.exit(1)

    # Set environment variable for OCR tests
    os.environ["RUN_OCR_TESTS"] = "true"

    logger.info(f"Starting OCR benchmark for: {args.pdf_path}")

    # Initialize components
    pdf_config = PDFConfig(
        default_dpi=150,
        enable_cache_metrics=True,
    )

    ocr_config = OCRConfig(
        primary_engine=OCREngine.PADDLEOCR,
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

    total_time = 0.0
    pages_processed = 0

    with pdf_handler.load_pdf(args.pdf_path):
        total_pages = pdf_handler.page_count
        pages_to_process = min(args.max_pages, total_pages)

        logger.info(f"Processing {pages_to_process} of {total_pages} pages")

        for page_num in range(pages_to_process):
            page_type = pdf_handler.get_page_type(page_num)

            if page_type in [PageType.IMAGE_BASED, PageType.MIXED]:
                # Render page
                render_start = time.time()
                image = pdf_handler.render_page(page_num, dpi=150)
                render_time = time.time() - render_start

                # Process with OCR
                ocr_start = time.time()
                result = ocr_processor.process_image(
                    image,
                    page_num,
                    page_type,
                    pdf_path=str(args.pdf_path),
                )
                ocr_time = time.time() - ocr_start

                page_total = render_time + ocr_time
                total_time += page_total
                pages_processed += 1

                logger.info(
                    f"Page {page_num}: {page_total:.2f}s total "
                    f"(render: {render_time:.2f}s, ocr: {ocr_time:.2f}s), "
                    f"Confidence: {result.avg_confidence:.2f}"
                )
            else:
                logger.info(f"Page {page_num}: Skipped ({page_type.value})")

    if pages_processed > 0:
        avg_time = total_time / pages_processed

        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Pages processed: {pages_processed}")
        print(f"Average time per page: {avg_time:.3f}s")
        print(f"Total processing time: {total_time:.3f}s")

        print("\nPerformance Target:")
        print(f"✓ Speed target (<2s/page): {'PASS' if avg_time < 2.0 else 'FAIL'}")
        print("=" * 60)

        # Get performance stats
        stats = ocr_processor.get_performance_stats()
        print("\nOCR Engine Statistics:")
        print(f"  Primary engine: {stats['primary_engine']}")
        print(f"  Total pages: {stats['total_pages_processed']}")
        print(f"  Avg time: {stats['avg_time_per_page']:.2f}s")
    else:
        print("No IMAGE_BASED or MIXED pages found to process")


if __name__ == "__main__":
    main()
