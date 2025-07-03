#!/usr/bin/env python3
"""
Test text extraction to debug why we're not getting actual text.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.preprocessing.pdf_handler import PDFHandler
from pdf_splitter.preprocessing.text_extractor import TextExtractor


def main():
    """Test text extraction directly."""
    pdf_path = Path("test_files/Test_PDF_Set_2_ocr.pdf")

    config = PDFConfig()
    handler = PDFHandler(config)

    print("Testing text extraction...")
    print("-" * 60)

    with handler.load_pdf(pdf_path) as loaded_handler:
        extractor = TextExtractor(loaded_handler)

        # Test first 3 pages
        for page_num in range(1, 4):
            print(f"\nExtracting page {page_num}...")
            extracted_page = extractor.extract_page_text(page_num)

            print(f"Type: {type(extracted_page)}")
            print(f"Has text attr: {hasattr(extracted_page, 'text')}")

            if hasattr(extracted_page, "text"):
                text = extracted_page.text
                print(f"Text length: {len(text)}")
                print(f"First 200 chars: {text[:200]}...")
            else:
                print(f"Object: {extracted_page}")


if __name__ == "__main__":
    main()
