#!/usr/bin/env python3
"""
Optimize embeddings threshold combined with post-processing filters.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_splitter.detection import ProcessedPage
from pdf_splitter.detection.embeddings_detector.fixed_embeddings_detector import FixedEmbeddingsDetector
from pdf_splitter.preprocessing import PDFHandler, TextExtractor


def calculate_metrics(detected, expected):
    """Calculate precision, recall, and F1 score."""
    tp = len(detected & expected)
    fp = len(detected - expected)
    fn = len(expected - detected)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def apply_optimized_filters(boundaries, pages):
    """Apply optimized post-processing filters."""
    filtered = []
    
    # Sort boundaries by page number
    boundaries.sort(key=lambda b: b.page_number)
    
    for i, boundary in enumerate(boundaries):
        page_num = boundary.page_number
        
        # Skip early pages with low confidence (position-based)
        if page_num < 3 and boundary.confidence < 0.7:
            continue
        
        # Skip late pages with low confidence
        if page_num > len(pages) - 4 and boundary.confidence < 0.7:
            continue
        
        # Content-based filtering
        if page_num + 1 < len(pages):
            next_page_text = pages[page_num + 1].text.strip()
            
            # Skip if next page starts with lowercase (continuation)
            if next_page_text and next_page_text[0].islower() and boundary.confidence < 0.8:
                continue
        
        # Minimum document length of 2 (but be more lenient for high confidence)
        if filtered:
            last_boundary = filtered[-1]
            if page_num - last_boundary.page_number < 2 and boundary.confidence < 0.8:
                continue
        
        filtered.append(boundary)
    
    return filtered


def optimize_threshold():
    """Find optimal threshold with post-processing."""
    
    print("THRESHOLD OPTIMIZATION WITH POST-PROCESSING")
    print("="*60)
    
    # Load test PDF
    pdf_path = Path("test_files/Test_PDF_Set_2_ocr.pdf")
    ground_truth_path = Path("test_files/Test_PDF_Set_Ground_Truth.json")
    
    # Load ground truth
    import json
    with open(ground_truth_path) as f:
        data = json.load(f)
    
    expected_boundaries = set()
    for doc in data['documents']:
        if '-' in doc['pages']:
            start, _ = doc['pages'].split('-')
            start_page = int(start) - 1
        else:
            start_page = int(doc['pages']) - 1
        
        if start_page > 0:
            expected_boundaries.add(start_page - 1)
    
    print(f"Expected boundaries: {len(expected_boundaries)}")
    
    # Load PDF pages
    pdf_handler = PDFHandler()
    pages = []
    
    with pdf_handler.load_pdf(pdf_path) as pdf:
        text_extractor = TextExtractor(pdf_handler)
        
        for i in range(pdf.page_count):
            extracted = text_extractor.extract_page(i)
            page = ProcessedPage(
                page_number=i,
                text=extracted.text,
                ocr_confidence=extracted.quality_score,
                page_type="searchable" if extracted.text.strip() else "empty"
            )
            pages.append(page)
    
    # Test different thresholds
    thresholds = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
    
    print("\nTesting thresholds with post-processing:\n")
    print(f"{'Threshold':<10} {'Raw Boundaries':<15} {'Filtered':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 75)
    
    best_f1 = 0
    best_config = None
    
    for threshold in thresholds:
        # Get boundaries with this threshold
        detector = FixedEmbeddingsDetector(
            model_name='all-MiniLM-L6-v2',
            similarity_threshold=threshold
        )
        
        boundaries = detector.detect_boundaries(pages)
        raw_count = len(boundaries)
        
        # Apply post-processing
        filtered = apply_optimized_filters(boundaries, pages)
        filtered_count = len(filtered)
        
        # Calculate metrics
        detected = {b.page_number for b in filtered}
        p, r, f1 = calculate_metrics(detected, expected_boundaries)
        
        print(f"{threshold:<10.2f} {raw_count:<15} {filtered_count:<10} {p:<10.3f} {r:<10.3f} {f1:<10.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_config = {
                'threshold': threshold,
                'raw_count': raw_count,
                'filtered_count': filtered_count,
                'precision': p,
                'recall': r,
                'detected': sorted(detected),
                'missing': sorted(expected_boundaries - detected),
                'false_positives': sorted(detected - expected_boundaries)
            }
    
    # Show best configuration details
    if best_config:
        print("\n" + "="*60)
        print("BEST CONFIGURATION")
        print("="*60)
        print(f"Threshold: {best_config['threshold']}")
        print(f"F1 Score: {best_f1:.3f}")
        print(f"Precision: {best_config['precision']:.3f}")
        print(f"Recall: {best_config['recall']:.3f}")
        print(f"Boundaries: {best_config['filtered_count']} (filtered from {best_config['raw_count']})")
        
        print(f"\nDistance to target F1≥0.75: {0.75 - best_f1:.3f}")
        
        if best_f1 >= 0.75:
            print("\n🎉 TARGET ACHIEVED!")
        else:
            print(f"\nDetected: {best_config['detected']}")
            print(f"Missing: {best_config['missing']}")
            print(f"False positives: {best_config['false_positives']}")
            
            # Analyze what we're missing
            print("\n" + "-"*60)
            print("ANALYSIS OF REMAINING GAPS")
            print("-"*60)
            
            if best_config['missing']:
                print(f"\nMissing {len(best_config['missing'])} boundaries - need to improve recall")
                print("Consider: lower threshold, better embeddings model, or LLM for these specific cases")
            
            if best_config['false_positives']:
                print(f"\nHave {len(best_config['false_positives'])} false positives - need to improve precision")
                print("Consider: stricter filters, confidence calibration, or pattern analysis")


if __name__ == "__main__":
    optimize_threshold()