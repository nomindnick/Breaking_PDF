"""Example usage of the SignalCombiner for document boundary detection.

This script demonstrates how to use the SignalCombiner with multiple detectors
to achieve high-accuracy document boundary detection.
"""

from pathlib import Path
from pdf_splitter.preprocessing import PDFHandler, TextExtractor
from pdf_splitter.detection import (
    HeuristicDetector,
    LLMDetector,
    VisualDetector,
    SignalCombiner,
    SignalCombinerConfig,
    DetectorType,
)


def main():
    """Demonstrate signal combiner usage."""
    # Load and process PDF
    pdf_path = Path("Test_PDF_Set_1.pdf")
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        return
    
    print(f"Processing {pdf_path}...")
    
    # Initialize preprocessing components
    pdf_handler = PDFHandler()
    text_extractor = TextExtractor()
    
    # Load PDF and extract text
    loaded_pdf = pdf_handler.load_pdf(pdf_path)
    pages = []
    
    for page_num in range(loaded_pdf.page_count):
        page_data = pdf_handler.get_page(loaded_pdf, page_num)
        processed_page = text_extractor.extract_text(page_data)
        pages.append(processed_page)
    
    print(f"Loaded {len(pages)} pages")
    
    # Example 1: Basic usage with just heuristic detector
    print("\n=== Example 1: Basic Heuristic Detection ===")
    detectors = {
        DetectorType.HEURISTIC: HeuristicDetector(),
    }
    
    combiner = SignalCombiner(detectors)
    results = combiner.detect_boundaries(pages)
    
    print(f"Found {len(results)} boundaries:")
    for result in results:
        print(f"  Page {result.page_number}: {result.boundary_type.value} "
              f"(confidence: {result.confidence:.2f})")
    
    # Example 2: Cascade-ensemble with multiple detectors
    print("\n=== Example 2: Cascade-Ensemble Strategy ===")
    
    # Note: This requires LLM and Visual detectors to be properly configured
    # For demonstration, we'll show the configuration
    
    config = SignalCombinerConfig(
        combination_strategy="cascade_ensemble",
        heuristic_confidence_threshold=0.9,  # High confidence = no verification needed
        require_llm_verification_below=0.7,  # Low confidence = needs LLM
        visual_verification_range=(0.7, 0.9),  # Medium confidence = visual check
        enable_parallel_processing=True,
        max_workers=4,
    )
    
    print(f"Configuration:")
    print(f"  Strategy: {config.combination_strategy}")
    print(f"  Heuristic threshold: {config.heuristic_confidence_threshold}")
    print(f"  LLM verification below: {config.require_llm_verification_below}")
    print(f"  Visual verification range: {config.visual_verification_range}")
    
    # Example 3: Weighted voting configuration
    print("\n=== Example 3: Weighted Voting Configuration ===")
    
    voting_config = SignalCombinerConfig(
        combination_strategy="weighted_voting",
        detector_weights={
            DetectorType.HEURISTIC: 0.3,
            DetectorType.LLM: 0.5,
            DetectorType.VISUAL: 0.2,
        },
        confidence_boost_on_agreement=0.1,  # Boost confidence when detectors agree
    )
    
    print("Detector weights:")
    for detector_type, weight in voting_config.detector_weights.items():
        print(f"  {detector_type.value}: {weight}")
    
    # Example 4: Consensus strategy
    print("\n=== Example 4: Consensus Strategy ===")
    
    consensus_config = SignalCombinerConfig(
        combination_strategy="consensus",
        min_agreement_threshold=0.66,  # At least 2/3 detectors must agree
        add_implicit_start_boundary=True,  # Add boundary at page 0 if missing
        min_document_pages=2,  # Minimum pages per document
    )
    
    print(f"Consensus threshold: {consensus_config.min_agreement_threshold}")
    print(f"Min document pages: {consensus_config.min_document_pages}")
    
    # Example 5: Full integration (requires all detectors)
    print("\n=== Example 5: Full Integration Setup ===")
    print("To use all detectors together:")
    print("""
    # Initialize all detectors
    detectors = {
        DetectorType.HEURISTIC: HeuristicDetector(),
        DetectorType.LLM: LLMDetector(),  # Requires Ollama with gemma3:latest
        DetectorType.VISUAL: VisualDetector(),
    }
    
    # Use cascade-ensemble for optimal performance
    config = SignalCombinerConfig(
        combination_strategy="cascade_ensemble",
        heuristic_confidence_threshold=0.85,
        enable_parallel_processing=True,
    )
    
    # Create combiner and detect boundaries
    combiner = SignalCombiner(detectors, config)
    results = combiner.detect_boundaries(pages)
    """)
    
    # Performance tips
    print("\n=== Performance Tips ===")
    print("1. Use cascade-ensemble to minimize expensive LLM calls")
    print("2. Enable parallel processing for faster results")
    print("3. Adjust confidence thresholds based on your accuracy needs")
    print("4. The LLM detector uses persistent caching for repeated runs")
    print("5. Consider using only heuristic detector for fast screening")


if __name__ == "__main__":
    main()