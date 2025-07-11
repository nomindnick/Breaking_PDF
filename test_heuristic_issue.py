#!/usr/bin/env python3
"""Test script to investigate heuristic detector confidence issues."""

import json
from pathlib import Path

from pdf_splitter.preprocessing.pdf_handler import PDFHandler
from pdf_splitter.preprocessing.text_extractor import TextExtractor
from pdf_splitter.detection.base_detector import ProcessedPage
from pdf_splitter.detection.heuristic_detector import HeuristicDetector

# Load test PDF and process pages
test_file = Path("Test_PDF_Set_1.pdf")
output_dir = Path("analysis_output")
output_dir.mkdir(exist_ok=True)

print(f"Loading {test_file}...")
pdf_handler = PDFHandler()
pdf_handler.load_pdf(test_file)

text_extractor = TextExtractor(pdf_handler)

# Process all pages
pages = []
for i in range(pdf_handler.num_pages):
    page = pdf_handler.get_page(i)
    text = text_extractor.extract_text(page)
    
    processed_page = ProcessedPage(
        page_number=i + 1,
        text=text,
        ocr_confidence=0.95  # Assuming non-OCR PDF
    )
    pages.append(processed_page)

print(f"Processed {len(pages)} pages")

# Initialize detector
detector = HeuristicDetector()

# Detect boundaries
print("\nDetecting boundaries...")
results = detector.detect_all_boundaries(pages)

# Print detailed results
print(f"\nFound {len(results)} potential boundaries:\n")

for i, result in enumerate(results):
    print(f"Between pages {result.page_number - 1} and {result.page_number}:")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Boundary Type: {result.boundary_type}")
    print(f"  Active Patterns: {result.evidence.get('active_patterns', [])}")
    
    # Show individual signal scores
    signals = result.evidence.get('signals', {})
    if signals:
        print("  Signal Scores:")
        for signal_name, score in signals.items():
            if score > 0:
                print(f"    {signal_name}: {score:.3f}")
    print()

# Load ground truth
ground_truth_file = Path("Test_PDF_Set_Ground_Truth.json")
if ground_truth_file.exists():
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    expected_boundaries = ground_truth.get('boundaries', [])
    print(f"\nExpected boundaries: {expected_boundaries}")
    
    # Compare with detection results
    detected_boundaries = [r.page_number for r in results if r.confidence > detector.config.min_confidence_threshold]
    print(f"Detected boundaries (conf > {detector.config.min_confidence_threshold}): {detected_boundaries}")

# Close PDF
pdf_handler.close()