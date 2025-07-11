"""Performance benchmarking tests for the full PDF processing pipeline.

These tests verify that performance targets are met:
- OCR: 1-2 seconds per page (when needed)
- Boundary Detection: < 0.1 seconds per page
- Overall Processing: < 5 seconds per page

Tests include memory usage tracking, scalability tests, cache effectiveness,
and parallel processing performance.
"""

import asyncio
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

import psutil
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection import create_production_detector
from pdf_splitter.preprocessing import PDFHandler, TextExtractor
from pdf_splitter.preprocessing.ocr_processor import OCRConfig, OCRProcessor
from pdf_splitter.splitting import PDFSplitter
from pdf_splitter.test_utils import create_mixed_test_pdf, create_test_pdf


class TestPerformanceBenchmarks:
    """Performance benchmarking tests for the PDF processing pipeline."""

    @pytest.fixture
    def performance_pdfs(self, temp_dir: Path) -> Dict[str, Path]:
        """Create test PDFs of various sizes for performance testing."""
        pdfs = {}

        # Small PDF (5 pages)
        pdfs["small"] = temp_dir / "small_test.pdf"
        create_test_pdf(
            num_pages=5,
            include_text=True,
            include_images=False,
            output_path=pdfs["small"],
        )

        # Medium PDF (20 pages)
        pdfs["medium"] = temp_dir / "medium_test.pdf"
        create_test_pdf(
            num_pages=20,
            include_text=True,
            include_images=True,
            output_path=pdfs["medium"],
        )

        # Large PDF (50 pages)
        pdfs["large"] = temp_dir / "large_test.pdf"
        create_test_pdf(
            num_pages=50,
            include_text=True,
            include_images=True,
            output_path=pdfs["large"],
        )

        # Mixed content PDF (30 pages with OCR requirements)
        pdfs["mixed"] = temp_dir / "mixed_test.pdf"
        documents = [
            {"type": "email", "pages": 3, "page_type": "searchable"},
            {"type": "invoice", "pages": 2, "page_type": "scanned", "quality": "high"},
            {"type": "letter", "pages": 5, "page_type": "searchable"},
            {"type": "rfi", "pages": 3, "page_type": "scanned", "quality": "medium"},
            {"type": "report", "pages": 7, "page_type": "mixed"},
            {"type": "contract", "pages": 10, "page_type": "searchable"},
        ]
        create_mixed_test_pdf(documents, pdfs["mixed"])

        return pdfs

    @pytest.fixture
    def pdf_config(self) -> PDFConfig:
        """Optimized PDF configuration for performance testing."""
        return PDFConfig(
            default_dpi=300,  # Production DPI
            max_file_size_mb=200,
            enable_cache_metrics=True,
            render_cache_memory_mb=100,
            text_cache_memory_mb=50,
        )

    @pytest.fixture
    def ocr_config(self) -> OCRConfig:
        """Optimized OCR configuration for performance testing."""
        return OCRConfig(
            confidence_threshold=0.6,
            preprocessing_enabled=True,
            max_workers=4,  # Production worker count
            batch_size=5,
        )

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    @pytest.mark.integration
    def test_ocr_performance(
        self, performance_pdfs: Dict[str, Path], ocr_config: OCRConfig
    ):
        """Test OCR performance meets target of 1-2 seconds per page."""
        ocr_processor = OCRProcessor(ocr_config)

        # Test with scanned pages from mixed PDF
        pdf_path = performance_pdfs["mixed"]

        with PDFHandler(PDFConfig()).load_pdf(pdf_path) as handler:
            scanned_pages = []

            # Find scanned pages
            for i in range(handler.page_count):
                page_type = handler.analyze_page_type(i)
                if page_type.name in ["IMAGE_BASED", "MIXED"]:
                    scanned_pages.append(i)

            # Benchmark OCR on scanned pages
            ocr_times = []
            for page_num in scanned_pages[:5]:  # Test first 5 scanned pages
                start_time = time.time()
                _ = ocr_processor.process_page(handler.current_doc[page_num])
                elapsed = time.time() - start_time
                ocr_times.append(elapsed)

                # Verify OCR target: 1-2 seconds per page
                assert elapsed < 2.0, f"OCR took too long: {elapsed:.2f}s > 2.0s"

            avg_ocr_time = sum(ocr_times) / len(ocr_times) if ocr_times else 0
            print(f"\nOCR Performance: {avg_ocr_time:.3f}s average per page")

            # Target achieved: < 2 seconds per page
            assert avg_ocr_time < 2.0

    @pytest.mark.integration
    def test_boundary_detection_performance(
        self, performance_pdfs: Dict[str, Path], pdf_config: PDFConfig
    ):
        """Test boundary detection meets target of < 0.1 seconds per page."""
        detector = create_production_detector()

        for pdf_name, pdf_path in performance_pdfs.items():
            with PDFHandler(pdf_config).load_pdf(pdf_path) as handler:
                text_extractor = TextExtractor(handler)

                # Extract all pages first (mimics real usage)
                pages = []
                for i in range(handler.page_count):
                    pages.append(text_extractor.extract_page(i))

                # Benchmark detection
                start_time = time.time()
                _ = detector.detect_boundaries(pages)
                elapsed = time.time() - start_time

                per_page_time = elapsed / handler.page_count
                print(f"\n{pdf_name} PDF ({handler.page_count} pages):")
                print(f"  Total detection time: {elapsed:.3f}s")
                print(f"  Per-page time: {per_page_time:.3f}s")

                # Verify target: < 0.1 seconds per page
                assert per_page_time < 0.1, (
                    f"Detection too slow for {pdf_name}: "
                    f"{per_page_time:.3f}s > 0.1s per page"
                )

    @pytest.mark.integration
    def test_full_pipeline_performance(
        self, performance_pdfs: Dict[str, Path], pdf_config: PDFConfig, temp_dir: Path
    ):
        """Test complete pipeline meets target of < 5 seconds per page."""
        for pdf_name, pdf_path in performance_pdfs.items():
            print(f"\nProcessing {pdf_name} PDF...")

            start_time = time.time()
            start_memory = self.get_memory_usage()

            # Full pipeline
            with PDFHandler(pdf_config).load_pdf(pdf_path) as handler:
                # 1. Text extraction
                text_extractor = TextExtractor(handler)
                pages = []
                for i in range(handler.page_count):
                    pages.append(text_extractor.extract_page(i))

                # 2. Boundary detection
                detector = create_production_detector()
                boundaries = detector.detect_boundaries(pages)

                # 3. PDF splitting
                splitter = PDFSplitter(handler)
                output_dir = temp_dir / f"{pdf_name}_output"
                output_dir.mkdir(exist_ok=True)

                split_files = splitter.split_pdf(boundaries, output_dir)

            elapsed = time.time() - start_time
            memory_used = self.get_memory_usage() - start_memory
            per_page_time = elapsed / handler.page_count

            print(f"  Total time: {elapsed:.2f}s")
            print(f"  Per-page time: {per_page_time:.2f}s")
            print(f"  Memory used: {memory_used:.1f} MB")
            print(f"  Documents created: {len(split_files)}")

            # Verify target: < 5 seconds per page
            assert per_page_time < 5.0, (
                f"Pipeline too slow for {pdf_name}: "
                f"{per_page_time:.2f}s > 5.0s per page"
            )

    @pytest.mark.integration
    def test_memory_usage_scaling(
        self, performance_pdfs: Dict[str, Path], pdf_config: PDFConfig
    ):
        """Test memory usage remains reasonable as PDF size increases."""
        memory_usage = {}

        for pdf_name in ["small", "medium", "large"]:
            gc.collect()  # Clean slate
            start_memory = self.get_memory_usage()

            with PDFHandler(pdf_config).load_pdf(performance_pdfs[pdf_name]) as handler:
                text_extractor = TextExtractor(handler)

                # Process all pages
                for i in range(handler.page_count):
                    _ = text_extractor.extract_page(i)

                # Check peak memory
                peak_memory = self.get_memory_usage()
                memory_usage[pdf_name] = {
                    "pages": handler.page_count,
                    "memory_mb": peak_memory - start_memory,
                }

        # Verify memory scales reasonably (not exponentially)
        small_ratio = (
            memory_usage["small"]["memory_mb"] / memory_usage["small"]["pages"]
        )
        large_ratio = (
            memory_usage["large"]["memory_mb"] / memory_usage["large"]["pages"]
        )

        print("\nMemory Usage Scaling:")
        for name, data in memory_usage.items():
            print(f"  {name}: {data['memory_mb']:.1f} MB for {data['pages']} pages")
            print(f"    ({data['memory_mb']/data['pages']:.2f} MB/page)")

        # Memory per page shouldn't increase dramatically
        assert (
            large_ratio < small_ratio * 2
        ), "Memory usage scaling poorly with PDF size"

    @pytest.mark.integration
    def test_cache_effectiveness(
        self, performance_pdfs: Dict[str, Path], pdf_config: PDFConfig
    ):
        """Test that caching provides significant performance improvements."""
        pdf_path = performance_pdfs["medium"]

        # First pass - cold cache
        with PDFHandler(pdf_config).load_pdf(pdf_path) as handler:
            text_extractor = TextExtractor(handler)

            cold_times = []
            for i in range(5):  # First 5 pages
                start = time.time()
                _ = text_extractor.extract_page(i)
                cold_times.append(time.time() - start)

        # Second pass - warm cache
        with PDFHandler(pdf_config).load_pdf(pdf_path) as handler:
            text_extractor = TextExtractor(handler)

            warm_times = []
            for i in range(5):  # Same 5 pages
                start = time.time()
                _ = text_extractor.extract_page(i)
                warm_times.append(time.time() - start)

        cold_avg = sum(cold_times) / len(cold_times)
        warm_avg = sum(warm_times) / len(warm_times)
        speedup = cold_avg / warm_avg

        print("\nCache Effectiveness:")
        print(f"  Cold cache: {cold_avg:.3f}s average")
        print(f"  Warm cache: {warm_avg:.3f}s average")
        print(f"  Speedup: {speedup:.1f}x")

        # Cache should provide at least 5x speedup
        assert speedup > 5.0, f"Cache not effective enough: {speedup:.1f}x < 5x"

    @pytest.mark.integration
    def test_parallel_processing_performance(
        self, performance_pdfs: Dict[str, Path], pdf_config: PDFConfig
    ):
        """Test parallel processing performance with multiple workers."""
        pdf_path = performance_pdfs["large"]

        def process_pages_sequential(handler, page_range):
            """Process pages sequentially."""
            text_extractor = TextExtractor(handler)
            results = []
            for i in page_range:
                results.append(text_extractor.extract_page(i))
            return results

        def process_page_parallel(args):
            """Process single page (for parallel execution)."""
            handler, page_num = args
            text_extractor = TextExtractor(handler)
            return text_extractor.extract_page(page_num)

        with PDFHandler(pdf_config).load_pdf(pdf_path) as handler:
            test_pages = list(range(20))  # First 20 pages

            # Sequential processing
            start = time.time()
            _ = process_pages_sequential(handler, test_pages)
            seq_time = time.time() - start

            # Parallel processing with threads
            start = time.time()
            with ThreadPoolExecutor(max_workers=4) as executor:
                args = [(handler, i) for i in test_pages]
                _ = list(executor.map(process_page_parallel, args))
            par_time = time.time() - start

            speedup = seq_time / par_time

            print("\nParallel Processing Performance:")
            print(f"  Sequential: {seq_time:.2f}s")
            print(f"  Parallel (4 workers): {par_time:.2f}s")
            print(f"  Speedup: {speedup:.1f}x")

            # Parallel should provide meaningful speedup
            assert (
                speedup > 1.5
            ), f"Parallel processing not effective: {speedup:.1f}x < 1.5x"

    @pytest.mark.integration
    @pytest.mark.benchmark(group="pipeline")
    def test_benchmark_full_pipeline(
        self,
        benchmark: BenchmarkFixture,
        performance_pdfs: Dict[str, Path],
        pdf_config: PDFConfig,
        temp_dir: Path,
    ):
        """Benchmark the complete pipeline using pytest-benchmark."""
        pdf_path = performance_pdfs["small"]  # Use small PDF for quick benchmarks

        def run_pipeline():
            with PDFHandler(pdf_config).load_pdf(pdf_path) as handler:
                # Extract text
                text_extractor = TextExtractor(handler)
                pages = [
                    text_extractor.extract_page(i) for i in range(handler.page_count)
                ]

                # Detect boundaries
                detector = create_production_detector()
                boundaries = detector.detect_boundaries(pages)

                # Split PDF
                splitter = PDFSplitter(handler)
                output_dir = temp_dir / "benchmark_output"
                output_dir.mkdir(exist_ok=True)
                splitter.split_pdf(boundaries, output_dir)

        # Run benchmark
        benchmark(run_pipeline)

        # Verify performance
        pages_processed = 5  # small PDF has 5 pages
        time_per_page = benchmark.stats["mean"] / pages_processed

        print("\nBenchmark Results:")
        print(f"  Mean time: {benchmark.stats['mean']:.3f}s")
        print(f"  Std dev: {benchmark.stats['stddev']:.3f}s")
        print(f"  Per page: {time_per_page:.3f}s")

        assert (
            time_per_page < 5.0
        ), f"Pipeline too slow: {time_per_page:.3f}s > 5.0s per page"

    @pytest.mark.integration
    def test_detection_accuracy_vs_performance(
        self, performance_pdfs: Dict[str, Path], pdf_config: PDFConfig
    ):
        """Test trade-off between detection accuracy and performance."""
        from pdf_splitter.detection import (
            EmbeddingsDetector,
            HeuristicDetector,
            VisualDetector,
        )

        detectors = {
            "production": create_production_detector(),
            "embeddings": EmbeddingsDetector(),
            "heuristic": HeuristicDetector(),
            "visual": VisualDetector(),
        }

        pdf_path = performance_pdfs["medium"]

        with PDFHandler(pdf_config).load_pdf(pdf_path) as handler:
            text_extractor = TextExtractor(handler)
            pages = [text_extractor.extract_page(i) for i in range(handler.page_count)]

            results = {}
            for name, detector in detectors.items():
                start = time.time()
                boundaries = detector.detect_boundaries(pages)
                elapsed = time.time() - start

                results[name] = {
                    "time": elapsed,
                    "per_page": elapsed / len(pages),
                    "boundaries": len([b for b in boundaries if b.is_boundary]),
                }

            print("\nDetector Performance Comparison:")
            for name, data in results.items():
                print(f"  {name}:")
                print(f"    Total time: {data['time']:.3f}s")
                print(f"    Per page: {data['per_page']:.3f}s")
                print(f"    Boundaries found: {data['boundaries']}")

            # Production detector should be fast
            assert results["production"]["per_page"] < 0.1

    @pytest.mark.integration
    @pytest.mark.slow
    def test_stress_test_large_pdf(self, temp_dir: Path, pdf_config: PDFConfig):
        """Stress test with very large PDF (100+ pages)."""
        # Create large PDF
        large_pdf = temp_dir / "stress_test.pdf"
        create_test_pdf(
            num_pages=100, include_text=True, include_images=True, output_path=large_pdf
        )

        start_time = time.time()
        start_memory = self.get_memory_usage()

        with PDFHandler(pdf_config).load_pdf(large_pdf) as handler:
            text_extractor = TextExtractor(handler)

            # Process in batches to test memory management
            batch_size = 10
            for batch_start in range(0, handler.page_count, batch_size):
                batch_end = min(batch_start + batch_size, handler.page_count)

                pages = []
                for i in range(batch_start, batch_end):
                    pages.append(text_extractor.extract_page(i))

                # Detect boundaries for batch
                detector = create_production_detector()
                _ = detector.detect_boundaries(pages)

                # Check memory usage doesn't grow unbounded
                current_memory = self.get_memory_usage()
                memory_growth = current_memory - start_memory
                assert (
                    memory_growth < 500
                ), f"Memory usage too high: {memory_growth:.1f} MB"

        total_time = time.time() - start_time
        per_page_time = total_time / 100

        print("\nStress Test Results (100 pages):")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Per page: {per_page_time:.2f}s")
        print(f"  Peak memory growth: {memory_growth:.1f} MB")

        # Even for stress test, should maintain < 5s per page
        assert per_page_time < 5.0

    @pytest.mark.integration
    def test_concurrent_pdf_processing(
        self, performance_pdfs: Dict[str, Path], pdf_config: PDFConfig, temp_dir: Path
    ):
        """Test processing multiple PDFs concurrently."""

        async def process_pdf_async(pdf_path: Path, output_dir: Path) -> float:
            """Process PDF asynchronously."""
            start = time.time()

            with PDFHandler(pdf_config).load_pdf(pdf_path) as handler:
                text_extractor = TextExtractor(handler)
                pages = [
                    text_extractor.extract_page(i) for i in range(handler.page_count)
                ]

                detector = create_production_detector()
                boundaries = detector.detect_boundaries(pages)

                splitter = PDFSplitter(handler)
                await asyncio.to_thread(splitter.split_pdf, boundaries, output_dir)

            return time.time() - start

        async def run_concurrent_test():
            """Run concurrent processing test."""
            tasks = []
            for name, pdf_path in list(performance_pdfs.items())[:3]:  # First 3 PDFs
                output_dir = temp_dir / f"concurrent_{name}"
                output_dir.mkdir(exist_ok=True)
                tasks.append(process_pdf_async(pdf_path, output_dir))

            times = await asyncio.gather(*tasks)
            return times

        # Sequential processing for comparison
        seq_start = time.time()
        seq_times = []
        for name, pdf_path in list(performance_pdfs.items())[:3]:
            output_dir = temp_dir / f"sequential_{name}"
            output_dir.mkdir(exist_ok=True)

            start = time.time()
            with PDFHandler(pdf_config).load_pdf(pdf_path) as handler:
                text_extractor = TextExtractor(handler)
                pages = [
                    text_extractor.extract_page(i) for i in range(handler.page_count)
                ]
                detector = create_production_detector()
                boundaries = detector.detect_boundaries(pages)
                splitter = PDFSplitter(handler)
                splitter.split_pdf(boundaries, output_dir)
            seq_times.append(time.time() - start)
        seq_total = time.time() - seq_start

        # Concurrent processing
        conc_start = time.time()
        _ = asyncio.run(run_concurrent_test())
        conc_total = time.time() - conc_start

        print("\nConcurrent Processing Results:")
        print(f"  Sequential total: {seq_total:.2f}s")
        print(f"  Concurrent total: {conc_total:.2f}s")
        print(f"  Speedup: {seq_total/conc_total:.1f}x")

        # Concurrent should be faster
        assert conc_total < seq_total
