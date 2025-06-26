#!/usr/bin/env python
"""
Comprehensive OCR accuracy testing for the PDF splitter preprocessing module.

This test suite evaluates OCR accuracy across different page types, document types,
and quality levels using the comprehensive test PDF.
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import pytest
from fuzzywuzzy import fuzz

from pdf_splitter.preprocessing import PDFHandler, TextExtractor
from pdf_splitter.preprocessing.ocr_processor import OCRProcessor


class TestOCRAccuracyComprehensive:
    """Comprehensive test suite for OCR accuracy evaluation."""

    @pytest.fixture(scope="class")
    def test_pdf_path(self):
        """Path to comprehensive test PDF."""
        return (
            Path(__file__).parent.parent.parent.parent
            / "test_files"
            / "comprehensive_test_pdf.pdf"
        )

    @pytest.fixture(scope="class")
    def ground_truth_path(self):
        """Path to ground truth JSON."""
        return (
            Path(__file__).parent.parent.parent.parent
            / "test_files"
            / "comprehensive_test_pdf_ground_truth.json"
        )

    @pytest.fixture(scope="class")
    def ground_truth(self, ground_truth_path):
        """Load ground truth data."""
        with open(ground_truth_path, "r") as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def ground_truth_text(self):
        """Load ground truth text data."""
        text_path = (
            Path(__file__).parent.parent.parent.parent
            / "test_files"
            / "comprehensive_test_pdf_text_ground_truth.json"
        )
        with open(text_path, "r") as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def pdf_handler(self, test_pdf_path):
        """Create PDF handler with the test PDF loaded."""
        print(f"Loading test PDF from: {test_pdf_path}")
        print(f"File exists: {test_pdf_path.exists()}")
        print(f"File size: {test_pdf_path.stat().st_size / 1024 / 1024:.2f} MB")

        handler = PDFHandler()
        # load_pdf is a context manager, but for testing we'll load it manually
        handler._pdf_path = test_pdf_path
        import fitz

        handler._document = fitz.open(str(test_pdf_path))
        handler._extract_metadata()

        print(f"Is loaded: {handler.is_loaded}")
        print(f"Page count: {handler.page_count}")

        yield handler

        # Cleanup
        handler.close()

    @pytest.fixture(scope="class")
    def text_extractor(self, pdf_handler):
        """Create text extractor."""
        return TextExtractor(pdf_handler)

    def calculate_accuracy_metrics(
        self, expected: str, actual: str
    ) -> Dict[str, float]:
        """
        Calculate various accuracy metrics between expected and actual text.

        Args:
            expected: Expected text (ground truth)
            actual: Actual OCR output

        Returns:
            Dictionary of accuracy metrics
        """
        # Normalize texts
        expected_norm = " ".join(expected.lower().split())
        actual_norm = " ".join(actual.lower().split())

        # Word-level accuracy
        expected_words = expected_norm.split()
        actual_words = actual_norm.split()

        # Calculate word overlap
        common_words = set(expected_words) & set(actual_words)
        word_precision = len(common_words) / len(actual_words) if actual_words else 0
        word_recall = len(common_words) / len(expected_words) if expected_words else 0
        word_f1 = (
            2 * (word_precision * word_recall) / (word_precision + word_recall)
            if (word_precision + word_recall) > 0
            else 0
        )

        # Character-level accuracy using fuzzy matching
        char_accuracy = fuzz.ratio(expected_norm, actual_norm) / 100.0

        # Partial ratio for substring matching
        partial_accuracy = fuzz.partial_ratio(expected_norm, actual_norm) / 100.0

        # Token set ratio for order-independent matching
        token_accuracy = fuzz.token_set_ratio(expected_norm, actual_norm) / 100.0

        return {
            "word_precision": word_precision,
            "word_recall": word_recall,
            "word_f1": word_f1,
            "char_accuracy": char_accuracy,
            "partial_accuracy": partial_accuracy,
            "token_accuracy": token_accuracy,
            "word_count_expected": len(expected_words),
            "word_count_actual": len(actual_words),
            "char_count_expected": len(expected),
            "char_count_actual": len(actual),
        }

    def test_overall_accuracy(
        self, pdf_handler, text_extractor, ground_truth, ground_truth_text
    ):
        """Test overall OCR accuracy across all pages."""
        results = []
        total_pages = pdf_handler.page_count
        ocr_processor = OCRProcessor()

        print(f"\nTesting OCR accuracy on {total_pages} pages...")

        for page_num in range(total_pages):
            start_time = time.time()

            # Get page type
            page_type = pdf_handler.get_page_type(page_num)

            # Extract text based on page type
            if page_type.value == "image_based":
                # Render page and perform OCR
                image = pdf_handler.render_page(page_num)
                ocr_result = ocr_processor.process_image(image, page_num, page_type)
                extracted_text = ocr_result.full_text
                confidence = ocr_result.avg_confidence
            else:
                # Extract text directly for searchable pages
                page_text = pdf_handler.extract_text(page_num)
                extracted_text = page_text.text
                confidence = 1.0  # Perfect confidence for searchable text

            extraction_time = time.time() - start_time

            # Get expected text from ground truth
            page_key = str(page_num + 1)
            expected_text = ground_truth_text["page_texts"][page_key]["expected_text"]
            is_scanned = ground_truth_text["page_texts"][page_key]["is_scanned"]

            # Find document info for this page
            doc_info = None
            for doc in ground_truth["documents"]:
                if "-" in doc["pages"]:
                    start, end = map(int, doc["pages"].split("-"))
                    if start <= page_num + 1 <= end:
                        doc_info = doc
                        break
                elif int(doc["pages"]) == page_num + 1:
                    doc_info = doc
                    break

            # Calculate metrics
            metrics = self.calculate_accuracy_metrics(expected_text, extracted_text)
            metrics.update(
                {
                    "page_num": page_num + 1,
                    "page_type": page_type.value,
                    "is_scanned": is_scanned,
                    "extraction_time": extraction_time,
                    "confidence": confidence,
                    "doc_type": doc_info["type"] if doc_info else "Unknown",
                    "doc_page_type": doc_info.get("page_type", "Unknown"),
                    "doc_quality": doc_info.get("quality", "N/A"),
                }
            )

            # Debug low accuracy pages
            if metrics["char_accuracy"] < 0.85:
                print(
                    f"\nPage {page_num + 1} low accuracy: "
                    f"{metrics['char_accuracy']:.2%}"
                )
                print(
                    f"Expected ({len(expected_text)} chars): {expected_text[:100]}..."
                )
                print(
                    f"Actual ({len(extracted_text)} chars): {extracted_text[:100]}..."
                )

            results.append(metrics)

        # Generate report
        self._generate_accuracy_report(results)

        # Assert minimum accuracy thresholds
        # Different thresholds for searchable vs scanned pages
        searchable_results = [r for r in results if not r["is_scanned"]]
        scanned_results = [r for r in results if r["is_scanned"]]

        if searchable_results:
            avg_searchable_accuracy = sum(
                r["char_accuracy"] for r in searchable_results
            ) / len(searchable_results)
            # Lower threshold for searchable pages due to formatting differences
            assert avg_searchable_accuracy >= 0.75, (
                f"Searchable pages accuracy {avg_searchable_accuracy:.2%} "
                "below 75% threshold"
            )

        if scanned_results:
            avg_scanned_accuracy = sum(
                r["char_accuracy"] for r in scanned_results
            ) / len(scanned_results)
            assert (
                avg_scanned_accuracy >= 0.70
            ), f"Scanned pages accuracy {avg_scanned_accuracy:.2%} below 70% threshold"

    def test_accuracy_by_page_type(self, pdf_handler, text_extractor):
        """Test OCR accuracy grouped by page type."""
        results_by_type = {
            "searchable": [],
            "image_based": [],
            "mixed": [],
            "empty": [],
        }

        ocr_processor = OCRProcessor()

        for page_num in range(pdf_handler.page_count):
            page_type = pdf_handler.get_page_type(page_num)
            page_type_str = page_type.value.lower()

            if page_type.value == "image_based":
                # Render page and perform OCR
                image = pdf_handler.render_page(page_num)
                ocr_result = ocr_processor.process_image(image, page_num, page_type)
                confidence = ocr_result.avg_confidence
                word_count = ocr_result.word_count
                processing_time = ocr_result.processing_time
            else:
                # Extract text directly
                page_text = pdf_handler.extract_text(page_num)
                confidence = 1.0
                word_count = len(page_text.text.split())
                processing_time = 0

            # For testing, we'll use confidence as a proxy for accuracy
            results_by_type[page_type_str].append(
                {
                    "page_num": page_num + 1,
                    "confidence": confidence,
                    "word_count": word_count,
                    "processing_time": processing_time,
                }
            )

        # Analyze results by type
        print("\nAccuracy by Page Type:")
        print("-" * 60)

        for page_type, results in results_by_type.items():
            if results:
                avg_confidence = sum(r["confidence"] for r in results) / len(results)
                avg_words = sum(r["word_count"] for r in results) / len(results)
                total_pages = len(results)

                print(
                    f"{page_type.capitalize():15s}: {total_pages:3d} pages, "
                    f"Avg confidence: {avg_confidence:.2%}, "
                    f"Avg words: {avg_words:.0f}"
                )

                # Assert minimum confidence for each type
                if page_type == "searchable":
                    assert (
                        avg_confidence >= 0.95
                    ), f"Searchable pages confidence too low: {avg_confidence:.2%}"
                elif page_type == "image_based":
                    assert (
                        avg_confidence >= 0.80
                    ), f"Image-based pages confidence too low: {avg_confidence:.2%}"

    def test_accuracy_by_quality(self, pdf_handler, text_extractor, ground_truth):
        """Test OCR accuracy grouped by scan quality."""
        results_by_quality = {"high": [], "medium": [], "low": [], "N/A": []}

        ocr_processor = OCRProcessor()

        for page_num in range(pdf_handler.page_count):
            # Find document info
            doc_info = None
            for doc in ground_truth["documents"]:
                if "-" in doc["pages"]:
                    start, end = map(int, doc["pages"].split("-"))
                    if start <= page_num + 1 <= end:
                        doc_info = doc
                        break
                elif int(doc["pages"]) == page_num + 1:
                    doc_info = doc
                    break

            quality = doc_info.get("quality", "N/A") if doc_info else "N/A"

            page_type = pdf_handler.get_page_type(page_num)

            if page_type.value == "image_based":
                # Render page and perform OCR
                image = pdf_handler.render_page(page_num)
                ocr_result = ocr_processor.process_image(image, page_num, page_type)
                confidence = ocr_result.avg_confidence
                word_count = ocr_result.word_count
            else:
                # Extract text directly
                page_text = pdf_handler.extract_text(page_num)
                confidence = 1.0
                word_count = len(page_text.text.split())

            results_by_quality[quality].append(
                {
                    "page_num": page_num + 1,
                    "confidence": confidence,
                    "word_count": word_count,
                    "page_type": page_type.value,
                }
            )

        # Analyze results by quality
        print("\nAccuracy by Scan Quality:")
        print("-" * 60)

        for quality, results in results_by_quality.items():
            if results:
                avg_confidence = sum(r["confidence"] for r in results) / len(results)
                total_pages = len(results)

                print(
                    f"{quality:10s}: {total_pages:3d} pages, "
                    f"Avg confidence: {avg_confidence:.2%}"
                )

                # Assert minimum confidence by quality
                if quality == "high":
                    assert (
                        avg_confidence >= 0.90
                    ), f"High quality scan confidence too low: {avg_confidence:.2%}"
                elif quality == "medium":
                    assert (
                        avg_confidence >= 0.85
                    ), f"Medium quality scan confidence too low: {avg_confidence:.2%}"
                elif quality == "low":
                    assert (
                        avg_confidence >= 0.75
                    ), f"Low quality scan confidence too low: {avg_confidence:.2%}"

    def test_processing_performance(self, pdf_handler, text_extractor):
        """Test OCR processing performance."""
        processing_times = []
        ocr_processor = OCRProcessor()

        for page_num in range(pdf_handler.page_count):
            page_type = pdf_handler.get_page_type(page_num)

            start_time = time.time()

            if page_type.value == "image_based":
                # Render page and perform OCR
                image = pdf_handler.render_page(page_num)
                ocr_result = ocr_processor.process_image(image, page_num, page_type)
                word_count = ocr_result.word_count
                ocr_performed = True
            else:
                # Extract text directly
                page_text = pdf_handler.extract_text(page_num)
                word_count = len(page_text.text.split())
                ocr_performed = False

            processing_time = time.time() - start_time

            processing_times.append(
                {
                    "page_num": page_num + 1,
                    "processing_time": processing_time,
                    "page_type": page_type.value,
                    "word_count": word_count,
                    "ocr_performed": ocr_performed,
                }
            )

        # Calculate statistics
        avg_time = sum(p["processing_time"] for p in processing_times) / len(
            processing_times
        )
        max_time = max(p["processing_time"] for p in processing_times)

        ocr_pages = [p for p in processing_times if p["ocr_performed"]]
        avg_ocr_time = (
            sum(p["processing_time"] for p in ocr_pages) / len(ocr_pages)
            if ocr_pages
            else 0
        )

        print("\nProcessing Performance:")
        print("-" * 60)
        print(f"Average processing time: {avg_time:.3f}s per page")
        print(f"Maximum processing time: {max_time:.3f}s")
        print(
            f"Average OCR time: {avg_ocr_time:.3f}s per page "
            f"(for {len(ocr_pages)} OCR pages)"
        )

        # Assert performance requirements
        assert (
            avg_time <= 2.0
        ), f"Average processing time {avg_time:.3f}s exceeds 2s limit"
        assert (
            max_time <= 5.0
        ), f"Maximum processing time {max_time:.3f}s exceeds 5s limit"

    def _generate_accuracy_report(self, results: List[Dict]):
        """Generate detailed accuracy report."""
        print("\nDetailed OCR Accuracy Report:")
        print("=" * 80)

        # Overall statistics
        total_pages = len(results)
        avg_char_accuracy = sum(r["char_accuracy"] for r in results) / total_pages
        avg_token_accuracy = sum(r["token_accuracy"] for r in results) / total_pages
        avg_confidence = sum(r["confidence"] for r in results) / total_pages
        avg_time = sum(r["extraction_time"] for r in results) / total_pages

        print("\nOverall Statistics:")
        print(f"  Total Pages: {total_pages}")
        print(f"  Average Character Accuracy: {avg_char_accuracy:.2%}")
        print(f"  Average Token Accuracy: {avg_token_accuracy:.2%}")
        print(f"  Average Confidence: {avg_confidence:.2%}")
        print(f"  Average Processing Time: {avg_time:.3f}s")

        # By page type
        print("\nBy Page Type:")
        for page_type in ["searchable", "image_based", "mixed", "empty"]:
            pages = [r for r in results if r["page_type"] == page_type]
            if pages:
                avg_acc = sum(p["char_accuracy"] for p in pages) / len(pages)
                print(
                    f"  {page_type:15s}: {len(pages):3d} pages, {avg_acc:.2%} accuracy"
                )

        # By document type
        print("\nBy Document Type:")
        doc_types = set(r["doc_type"] for r in results)
        for doc_type in sorted(doc_types):
            pages = [r for r in results if r["doc_type"] == doc_type]
            avg_acc = sum(p["char_accuracy"] for p in pages) / len(pages)
            print(f"  {doc_type:15s}: {len(pages):3d} pages, {avg_acc:.2%} accuracy")

        # Problem pages (accuracy < 85%)
        problem_pages = [r for r in results if r["char_accuracy"] < 0.85]
        if problem_pages:
            print("\nPages with Low Accuracy (< 85%):")
            for page in problem_pages:
                print(
                    f"  Page {page['page_num']:3d}: {page['char_accuracy']:.2%} "
                    f"({page['doc_type']}, {page['doc_page_type']}, "
                    f"quality: {page['doc_quality']})"
                )

        # Separate results by scanned vs searchable
        searchable_pages = [r for r in results if not r.get("is_scanned", False)]
        scanned_pages = [r for r in results if r.get("is_scanned", False)]

        if searchable_pages:
            avg_searchable = sum(p["char_accuracy"] for p in searchable_pages) / len(
                searchable_pages
            )
            print(
                f"\nSearchable pages: {len(searchable_pages):3d} pages, "
                f"{avg_searchable:.2%} avg accuracy"
            )

        if scanned_pages:
            avg_scanned = sum(p["char_accuracy"] for p in scanned_pages) / len(
                scanned_pages
            )
            print(
                f"Scanned pages: {len(scanned_pages):3d} pages, "
                f"{avg_scanned:.2%} avg accuracy"
            )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
