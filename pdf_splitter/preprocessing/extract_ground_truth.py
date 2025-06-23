#!/usr/bin/env python3
"""
Extract ground truth text from OCR'd PDF for testing OCR accuracy.

This script extracts text from Test_PDF_Set_2_ocr.pdf and saves it in a
structured format that can be used to evaluate OCR accuracy when processing
Test_PDF_Set_1.pdf (the non-OCR version).
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.preprocessing.pdf_handler import PDFHandler
from pdf_splitter.preprocessing.text_extractor import TextExtractor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_ground_truth_text():
    """Extract text from OCR'd PDF and save as ground truth."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    test_files_dir = project_root / "test_files"
    ocr_pdf = test_files_dir / "Test_PDF_Set_2_ocr.pdf"
    ground_truth_json = test_files_dir / "Test_PDF_Set_Ground_Truth.json"
    output_file = test_files_dir / "Test_PDF_Set_2_extracted_text.json"

    if not ocr_pdf.exists():
        logger.error(f"OCR PDF not found: {ocr_pdf}")
        return

    # Load existing ground truth for document boundaries
    if not ground_truth_json.exists():
        logger.error(f"Ground truth JSON not found: {ground_truth_json}")
        return

    with open(ground_truth_json, "r") as f:
        ground_truth = json.load(f)

    # Initialize PDF handler and text extractor
    config = PDFConfig()
    handler = PDFHandler(config)

    logger.info(f"Loading PDF: {ocr_pdf}")

    extracted_data = {
        "source_file": ocr_pdf.name,
        "extraction_date": datetime.now().isoformat(),
        "total_pages": 0,
        "pages": [],
        "documents": [],
    }

    with handler.load_pdf(ocr_pdf) as pdf_handler:
        extractor = TextExtractor(pdf_handler)

        extracted_data["total_pages"] = pdf_handler.page_count

        # Extract all pages
        logger.info("Extracting text from all pages...")
        all_pages = extractor.extract_all_pages()

        # Save page-level data
        for page in all_pages:
            page_data = {
                "page_num": page.page_num + 1,  # Convert to 1-indexed
                "text": page.text,
                "word_count": page.word_count,
                "char_count": page.char_count,
                "quality_score": page.quality_score,
                "avg_font_size": page.avg_font_size,
                "dominant_font": page.dominant_font,
                "has_headers": page.has_headers,
                "has_footers": page.has_footers,
                "reading_order_confidence": page.reading_order_confidence,
                "block_count": len(page.blocks),
                "table_count": len(page.tables),
            }
            extracted_data["pages"].append(page_data)

        # Extract document-level data based on ground truth boundaries
        logger.info("Extracting document segments based on ground truth...")

        for doc_info in ground_truth["documents"]:
            pages_str = doc_info["pages"]
            doc_type = doc_info["type"]
            summary = doc_info["summary"]

            # Parse page range
            if "-" in pages_str:
                start_page, end_page = map(int, pages_str.split("-"))
            else:
                start_page = end_page = int(pages_str)

            # Extract document segment
            segments = extractor.extract_document_segments([(start_page, end_page)])

            if segments:
                segment = segments[0]
                doc_data = {
                    "pages": pages_str,
                    "type": doc_type,
                    "summary": summary,
                    "extracted_text": segment["full_text"],
                    "word_count": segment["total_word_count"],
                    "char_count": segment["total_char_count"],
                    "avg_quality_score": segment["avg_quality_score"],
                    "page_count": segment["page_count"],
                }
                extracted_data["documents"].append(doc_data)

                logger.info(
                    f"Extracted document: {doc_type} (pages {pages_str}), "
                    f"{segment['total_word_count']} words"
                )

    # Save extracted data
    logger.info(f"Saving extracted text to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(extracted_data, f, indent=2)

    # Print summary statistics
    total_words = sum(p["word_count"] for p in extracted_data["pages"])
    total_chars = sum(p["char_count"] for p in extracted_data["pages"])
    avg_quality = sum(p["quality_score"] for p in extracted_data["pages"]) / len(
        extracted_data["pages"]
    )

    print("\n=== Extraction Summary ===")
    print(f"Total pages processed: {len(extracted_data['pages'])}")
    print(f"Total words extracted: {total_words:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average quality score: {avg_quality:.3f}")
    print(f"Documents extracted: {len(extracted_data['documents'])}")

    # Sample text from first document
    if extracted_data["documents"]:
        first_doc = extracted_data["documents"][0]
        sample_text = (
            first_doc["extracted_text"][:500] + "..."
            if len(first_doc["extracted_text"]) > 500
            else first_doc["extracted_text"]
        )
        print(f"\nFirst document sample ({first_doc['type']}):")
        print("-" * 50)
        print(sample_text)
        print("-" * 50)

    print(f"\nGround truth text saved to: {output_file}")

    # Also save a simplified version for OCR comparison
    simple_output = test_files_dir / "Test_PDF_Set_2_text_only.json"
    simple_data = {
        "pages": {str(p["page_num"]): p["text"] for p in extracted_data["pages"]}
    }

    with open(simple_output, "w") as f:
        json.dump(simple_data, f, indent=2)

    print(f"Simplified text-only version saved to: {simple_output}")


if __name__ == "__main__":
    extract_ground_truth_text()
