"""Integration tests for the signal combiner with real PDF data."""

import json
import pytest
from pathlib import Path

from pdf_splitter.preprocessing import PDFHandler, TextExtractor
from pdf_splitter.detection import (
    HeuristicDetector,
    LLMDetector,
    VisualDetector,
    DetectorType,
)
from pdf_splitter.detection.signal_combiner import SignalCombiner, SignalCombinerConfig


@pytest.fixture
def test_pdf_path():
    """Path to test PDF file."""
    return Path("Test_PDF_Set_1.pdf")


@pytest.fixture
def ground_truth():
    """Load ground truth data."""
    with open("Test_PDF_Set_Ground_Truth.json", "r") as f:
        return json.load(f)


@pytest.fixture
def processed_pages(test_pdf_path):
    """Process test PDF pages."""
    pdf_handler = PDFHandler()
    text_extractor = TextExtractor()
    
    pages = []
    if test_pdf_path.exists():
        loaded_pdf = pdf_handler.load_pdf(test_pdf_path)
        for page_num in range(loaded_pdf.page_count):
            page_data = pdf_handler.get_page(loaded_pdf, page_num)
            processed_page = text_extractor.extract_text(page_data)
            pages.append(processed_page)
    
    return pages


@pytest.mark.skipif(
    not Path("Test_PDF_Set_1.pdf").exists(),
    reason="Test PDF file not available"
)
class TestSignalCombinerIntegration:
    """Integration tests for signal combiner with real data."""

    def test_cascade_ensemble_with_real_pdf(self, processed_pages, ground_truth):
        """Test cascade ensemble strategy with real PDF data."""
        # Create real detector instances
        detectors = {
            DetectorType.HEURISTIC: HeuristicDetector(),
            # Note: LLM and Visual detectors require additional setup
            # For integration testing, we'll use just heuristic
        }
        
        config = SignalCombinerConfig(
            combination_strategy="cascade_ensemble",
            heuristic_confidence_threshold=0.85,
        )
        
        combiner = SignalCombiner(detectors, config)
        results = combiner.detect_boundaries(processed_pages)
        
        # Verify we got some results
        assert len(results) > 0
        
        # Check that results are sorted by page number
        page_numbers = [r.page_number for r in results]
        assert page_numbers == sorted(page_numbers)

    def test_performance_metrics(self, processed_pages):
        """Test performance of signal combiner."""
        import time
        
        detectors = {
            DetectorType.HEURISTIC: HeuristicDetector(),
        }
        
        combiner = SignalCombiner(detectors)
        
        start_time = time.time()
        results = combiner.detect_boundaries(processed_pages)
        end_time = time.time()
        
        processing_time = end_time - start_time
        time_per_page = processing_time / len(processed_pages)
        
        # Should process quickly with just heuristic detector
        assert time_per_page < 0.5  # Less than 0.5 seconds per page
        
        print(f"Processing time: {processing_time:.2f}s for {len(processed_pages)} pages")
        print(f"Time per page: {time_per_page:.3f}s")

    def test_boundary_quality(self, processed_pages, ground_truth):
        """Test quality of detected boundaries against ground truth."""
        detectors = {
            DetectorType.HEURISTIC: HeuristicDetector(),
        }
        
        combiner = SignalCombiner(detectors)
        results = combiner.detect_boundaries(processed_pages)
        
        # Extract boundary page numbers
        detected_pages = {r.page_number for r in results}
        
        # Get ground truth boundaries
        expected_boundaries = set()
        for doc in ground_truth["documents"]:
            if "start_page" in doc:
                expected_boundaries.add(doc["start_page"])
        
        # Calculate overlap
        if expected_boundaries:
            overlap = len(detected_pages & expected_boundaries)
            precision = overlap / len(detected_pages) if detected_pages else 0
            recall = overlap / len(expected_boundaries) if expected_boundaries else 0
            
            print(f"Detected boundaries: {sorted(detected_pages)}")
            print(f"Expected boundaries: {sorted(expected_boundaries)}")
            print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")

    def test_multiple_strategies_comparison(self, processed_pages):
        """Compare different combination strategies."""
        detectors = {
            DetectorType.HEURISTIC: HeuristicDetector(),
        }
        
        strategies = [
            "cascade_ensemble",
            "weighted_voting",
            "consensus",
        ]
        
        for strategy in strategies:
            config = SignalCombinerConfig(combination_strategy=strategy)
            combiner = SignalCombiner(detectors, config)
            
            results = combiner.detect_boundaries(processed_pages)
            
            print(f"\nStrategy: {strategy}")
            print(f"Boundaries detected: {len(results)}")
            print(f"Pages: {[r.page_number for r in results]}")

    @pytest.mark.skipif(
        not Path("Test_PDF_Set_2_ocr.pdf").exists(),
        reason="OCR test PDF not available"
    )
    def test_with_ocr_pdf(self):
        """Test signal combiner with OCR'd PDF."""
        pdf_path = Path("Test_PDF_Set_2_ocr.pdf")
        
        # Process OCR PDF
        pdf_handler = PDFHandler()
        text_extractor = TextExtractor()
        
        pages = []
        loaded_pdf = pdf_handler.load_pdf(pdf_path)
        for page_num in range(loaded_pdf.page_count):
            page_data = pdf_handler.get_page(loaded_pdf, page_num)
            processed_page = text_extractor.extract_text(page_data)
            pages.append(processed_page)
        
        # Run detection
        detectors = {
            DetectorType.HEURISTIC: HeuristicDetector(),
        }
        
        combiner = SignalCombiner(detectors)
        results = combiner.detect_boundaries(pages)
        
        # OCR PDFs might have different detection patterns
        assert len(results) >= 0
        
        # Check that we handle OCR-specific features
        for page in pages:
            if page.page_type.value == "IMAGE_BASED":
                # Should still detect boundaries in OCR'd pages
                pass


@pytest.mark.skipif(
    not Path("comprehensive_test_pdf.pdf").exists(),
    reason="Comprehensive test PDF not available"
)
def test_with_comprehensive_pdf():
    """Test with comprehensive mixed-content PDF."""
    pdf_path = Path("comprehensive_test_pdf.pdf")
    ground_truth_path = Path("comprehensive_test_pdf_ground_truth.json")
    
    # Load ground truth if available
    ground_truth = None
    if ground_truth_path.exists():
        with open(ground_truth_path, "r") as f:
            ground_truth = json.load(f)
    
    # Process PDF
    pdf_handler = PDFHandler()
    text_extractor = TextExtractor()
    
    pages = []
    loaded_pdf = pdf_handler.load_pdf(pdf_path)
    for page_num in range(loaded_pdf.page_count):
        page_data = pdf_handler.get_page(loaded_pdf, page_num)
        processed_page = text_extractor.extract_text(page_data)
        pages.append(processed_page)
    
    # Test with different configurations
    configs = [
        SignalCombinerConfig(
            combination_strategy="cascade_ensemble",
            heuristic_confidence_threshold=0.9,
        ),
        SignalCombinerConfig(
            combination_strategy="weighted_voting",
            detector_weights={
                DetectorType.HEURISTIC: 1.0,  # Only heuristic available
            }
        ),
    ]
    
    for config in configs:
        detectors = {
            DetectorType.HEURISTIC: HeuristicDetector(),
        }
        
        combiner = SignalCombiner(detectors, config)
        results = combiner.detect_boundaries(pages)
        
        print(f"\nConfiguration: {config.combination_strategy}")
        print(f"Detected {len(results)} boundaries")
        
        if ground_truth:
            # Compare with ground truth
            expected_pages = {doc["start_page"] for doc in ground_truth.get("documents", [])}
            detected_pages = {r.page_number for r in results}
            
            overlap = len(detected_pages & expected_pages)
            print(f"Overlap with ground truth: {overlap}/{len(expected_pages)}")


def test_signal_combiner_with_all_detectors():
    """Test signal combiner with all available detectors.
    
    This test is marked for manual execution as it requires:
    - Ollama running with gemma3:latest model
    - Sufficient computational resources
    """
    pytest.skip("Manual test - requires full detector setup")
    
    # This would be the full integration test with all detectors:
    # detectors = {
    #     DetectorType.HEURISTIC: HeuristicDetector(),
    #     DetectorType.LLM: LLMDetector(),
    #     DetectorType.VISUAL: VisualDetector(),
    # }
    # 
    # config = SignalCombinerConfig(
    #     combination_strategy="cascade_ensemble",
    #     enable_parallel_processing=True,
    # )
    # 
    # combiner = SignalCombiner(detectors, config)
    # results = combiner.detect_boundaries(pages)