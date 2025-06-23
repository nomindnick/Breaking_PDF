"""
Demo script showing the performance benefits of advanced caching.

This script demonstrates how the Level 3 caching implementation
dramatically improves performance for repeated page access patterns
common in document boundary detection.
"""

import time
from pathlib import Path

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.preprocessing import PDFHandler, TextExtractor


def main():
    """Run the caching demo."""
    # Configure with caching enabled
    config = PDFConfig(
        enable_cache_metrics=True,
        render_cache_memory_mb=100,
        text_cache_memory_mb=50,
        cache_warmup_pages=10,
    )

    # Create handler with advanced caching
    handler = PDFHandler(config)

    # Find test PDF
    test_pdf = Path("test_files/Test_PDF_Set_2_ocr.pdf")
    if not test_pdf.exists():
        print("Test PDF not found. Please ensure test files are available.")
        return

    print("=== PDF Caching Demo ===\n")

    with handler.load_pdf(test_pdf) as pdf:
        print(f"Loaded PDF with {pdf.page_count} pages")

        # Demonstrate cache warmup
        print("\n1. Cache Warmup Demo")
        print("   Warming up first 5 pages...")
        warmup_start = time.time()
        handler.warmup_cache(range(5))
        warmup_time = time.time() - warmup_start
        print(f"   Warmup completed in {warmup_time:.2f}s")

        # Demonstrate render caching
        print("\n2. Render Cache Demo")
        print("   First render (cache miss)...")
        start = time.time()
        img1 = handler.render_page(0)  # noqa: F841
        first_time = time.time() - start
        print(f"   Rendered in {first_time*1000:.1f}ms")

        print("   Second render (cache hit)...")
        start = time.time()
        img2 = handler.render_page(0)  # noqa: F841
        cached_time = time.time() - start
        print(f"   Rendered in {cached_time*1000:.1f}ms")
        print(f"   Speed improvement: {first_time/cached_time:.1f}x faster!")

        # Demonstrate text extraction caching
        print("\n3. Text Extraction Cache Demo")
        extractor = TextExtractor(handler)

        print("   First extraction (cache miss)...")
        start = time.time()
        text1 = extractor.extract_page(1)  # noqa: F841
        first_extract = time.time() - start
        print(f"   Extracted in {first_extract*1000:.1f}ms")

        print("   Second extraction (cache hit)...")
        start = time.time()
        text2 = extractor.extract_page(1)  # noqa: F841
        cached_extract = time.time() - start
        print(f"   Extracted in {cached_extract*1000:.1f}ms")
        print(f"   Speed improvement: {first_extract/cached_extract:.1f}x faster!")

        # Simulate boundary detection pattern
        print("\n4. Boundary Detection Pattern Demo")
        print("   Simulating multi-detector access pattern...")

        # LLM detector checks pages 10-12
        print("   - LLM detector analyzing pages 10-12...")
        start = time.time()
        for i in range(10, 13):
            extractor.extract_page(i)
        llm_time = time.time() - start

        # Visual detector checks same pages (should hit cache)
        print("   - Visual detector analyzing pages 10-12...")
        start = time.time()
        for i in range(10, 13):
            handler.render_page(i)
        visual_time = time.time() - start

        # Heuristic detector checks text again (should hit cache)
        print("   - Heuristic detector analyzing pages 10-12...")
        start = time.time()
        for i in range(10, 13):
            extractor.extract_page(i)
        heuristic_time = time.time() - start

        print(
            f"\n   Total time: {(llm_time + visual_time + heuristic_time)*1000:.1f}ms"
        )
        print(f"   Without cache (estimated): {(llm_time * 3)*1000:.1f}ms")
        print(
            "   Time saved: "
            f"{((llm_time * 3) - (llm_time + visual_time + heuristic_time))*1000:.1f}ms"
        )

        # Show cache statistics
        print("\n5. Cache Performance Summary")
        stats = handler.get_cache_stats()

        print("   Render Cache:")
        print(f"   - Hit Rate: {stats['render_cache']['hit_rate']}")
        print(f"   - Items Cached: {stats['render_cache']['size']}")
        print(f"   - Memory Used: {stats['render_cache']['memory_mb']:.1f}MB")
        print(f"   - Time Saved: {stats['render_cache']['time_saved_seconds']:.2f}s")

        print("\n   Text Cache:")
        print(f"   - Hit Rate: {stats['text_cache']['hit_rate']}")
        print(f"   - Items Cached: {stats['text_cache']['size']}")
        print(f"   - Memory Used: {stats['text_cache']['memory_mb']:.1f}MB")

        print(f"\n   Total Memory: {stats['total_memory_mb']:.1f}MB")

        # Log detailed performance metrics
        print("\n6. Detailed Performance Logs")
        handler.log_cache_performance()


if __name__ == "__main__":
    main()
