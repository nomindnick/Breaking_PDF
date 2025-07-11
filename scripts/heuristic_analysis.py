#!/usr/bin/env python3
"""
Detailed analysis of heuristic detector performance to identify specific issues.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_splitter.detection import (
    HeuristicDetector,
    ProcessedPage,
    get_production_config,
)
from pdf_splitter.preprocessing import PDFHandler, TextExtractor


class HeuristicAnalyzer:
    """Analyze heuristic detector performance in detail."""
    
    def __init__(self):
        self.pdf_handler = PDFHandler()
        self.heuristic_detector = HeuristicDetector(get_production_config())
        
    def analyze_pdf(self, pdf_path: Path, ground_truth_path: Path):
        """Analyze heuristic performance on a PDF."""
        # Load ground truth
        with open(ground_truth_path) as f:
            data = json.load(f)
            
        # Convert ground truth to boundary indices
        true_boundaries = set()
        for doc in data['documents']:
            if '-' in doc['pages']:
                start, end = doc['pages'].split('-')
                start_page = int(start) - 1
            else:
                start_page = int(doc['pages']) - 1
                
            if start_page > 0:
                true_boundaries.add(start_page - 1)
        
        print(f"Ground truth boundaries: {sorted(true_boundaries)}")
        
        # Extract pages
        pages = []
        with self.pdf_handler.load_pdf(pdf_path) as pdf:
            text_extractor = TextExtractor(self.pdf_handler)
            
            for i in range(pdf.page_count):
                extracted = text_extractor.extract_page(i)
                page = ProcessedPage(
                    page_number=i,
                    text=extracted.text,
                    ocr_confidence=extracted.quality_score,
                    page_type="searchable" if extracted.text.strip() else "empty"
                )
                pages.append(page)
        
        # Detect boundaries
        boundaries = self.heuristic_detector.detect_boundaries(pages)
        detected_indices = {b.page_number for b in boundaries}
        
        print(f"\nDetected boundaries: {sorted(detected_indices)}")
        print(f"Total detected: {len(detected_indices)} (ground truth: {len(true_boundaries)})")
        
        # Analyze each boundary
        print("\n" + "="*80)
        print("DETAILED BOUNDARY ANALYSIS")
        print("="*80)
        
        # Group boundaries by confidence level
        by_confidence = defaultdict(list)
        for b in boundaries:
            if b.confidence >= 0.9:
                level = "HIGH"
            elif b.confidence >= 0.7:
                level = "MEDIUM"
            else:
                level = "LOW"
            by_confidence[level].append(b)
        
        # Analyze true positives, false positives, and false negatives
        true_positives = detected_indices & true_boundaries
        false_positives = detected_indices - true_boundaries
        false_negatives = true_boundaries - detected_indices
        
        print(f"\nTrue Positives ({len(true_positives)}): {sorted(true_positives)}")
        print(f"False Positives ({len(false_positives)}): {sorted(false_positives)}")
        print(f"False Negatives ({len(false_negatives)}): {sorted(false_negatives)}")
        
        # Analyze patterns for each detected boundary
        print("\n" + "-"*80)
        print("PATTERN ANALYSIS FOR DETECTED BOUNDARIES")
        print("-"*80)
        
        for b in sorted(boundaries, key=lambda x: x.page_number):
            is_correct = b.page_number in true_boundaries
            status = "✓ CORRECT" if is_correct else "✗ FALSE POSITIVE"
            
            print(f"\nBoundary at page {b.page_number} - Confidence: {b.confidence:.3f} [{status}]")
            
            # Show which patterns triggered
            if hasattr(b, 'evidence') and 'patterns' in b.evidence:
                patterns = b.evidence['patterns']
                print("  Triggered patterns:")
                for pattern, score in patterns.items():
                    if score > 0:
                        print(f"    - {pattern}: {score:.3f}")
            
            # Show reasoning
            if b.reasoning:
                print(f"  Reasoning: {b.reasoning[:200]}...")
                
            # For false positives, show the actual text
            if not is_correct and b.page_number < len(pages) - 1:
                print(f"  End page text preview: {pages[b.page_number].text[:100].strip()}...")
                print(f"  Next page text preview: {pages[b.page_number + 1].text[:100].strip()}...")
        
        # Analyze missed boundaries
        print("\n" + "-"*80)
        print("ANALYSIS OF MISSED BOUNDARIES (FALSE NEGATIVES)")
        print("-"*80)
        
        for page_idx in sorted(false_negatives):
            print(f"\nMissed boundary at page {page_idx}")
            if page_idx < len(pages) - 1:
                # Check what patterns should have triggered
                end_page = pages[page_idx]
                start_page = pages[page_idx + 1]
                
                print(f"  End page preview: {end_page.text[:150].strip()}...")
                print(f"  Start page preview: {start_page.text[:150].strip()}...")
                
                # Manually check for patterns
                self._check_patterns_manually(end_page, start_page)
        
        # Confidence distribution
        print("\n" + "-"*80)
        print("CONFIDENCE DISTRIBUTION")
        print("-"*80)
        
        for level, group in by_confidence.items():
            correct = sum(1 for b in group if b.page_number in true_boundaries)
            total = len(group)
            accuracy = correct / total if total > 0 else 0
            
            print(f"\n{level} confidence ({len(group)} boundaries):")
            print(f"  Accuracy: {accuracy:.1%} ({correct}/{total} correct)")
            print(f"  Page numbers: {sorted([b.page_number for b in group])}")
        
        # Overall metrics
        precision = len(true_positives) / len(detected_indices) if detected_indices else 0
        recall = len(true_positives) / len(true_boundaries) if true_boundaries else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n" + "="*80)
        print("SUMMARY METRICS")
        print("="*80)
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
        return {
            'true_positives': sorted(true_positives),
            'false_positives': sorted(false_positives),
            'false_negatives': sorted(false_negatives),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _check_patterns_manually(self, end_page: ProcessedPage, start_page: ProcessedPage):
        """Manually check what patterns should have triggered."""
        config = self.heuristic_detector.config
        
        print("  Pattern analysis:")
        
        # Check email headers
        if any(pattern in start_page.text[:200] for pattern in ['From:', 'To:', 'Subject:', 'Date:']):
            print("    - Email header pattern SHOULD have triggered")
            
        # Check document keywords
        keywords = config.patterns["document_keywords"].params["keywords"]
        found_keywords = [kw for kw in keywords if kw.upper() in start_page.text[:500].upper()]
        if found_keywords:
            print(f"    - Document keywords found: {found_keywords}")
            
        # Check terminal phrases
        phrases = config.patterns["terminal_phrases"].params["phrases"]
        found_phrases = [p for p in phrases if p.lower() in end_page.text[-500:].lower()]
        if found_phrases:
            print(f"    - Terminal phrases found: {found_phrases}")
            
        # Check whitespace
        end_ws = end_page.text.count(' ') / max(len(end_page.text), 1)
        start_ws = start_page.text.count(' ') / max(len(start_page.text), 1)
        print(f"    - Whitespace ratios: end={end_ws:.2f}, start={start_ws:.2f}")


def main():
    """Run the analysis."""
    analyzer = HeuristicAnalyzer()
    
    # Analyze Test_PDF_Set_2_ocr.pdf
    pdf_path = Path("test_files/Test_PDF_Set_2_ocr.pdf")
    ground_truth_path = Path("test_files/Test_PDF_Set_Ground_Truth.json")
    
    print("Analyzing heuristic detector performance...")
    print(f"PDF: {pdf_path}")
    print(f"Ground truth: {ground_truth_path}")
    
    results = analyzer.analyze_pdf(pdf_path, ground_truth_path)
    
    # Save results
    output_path = Path("scripts/heuristic_analysis_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()