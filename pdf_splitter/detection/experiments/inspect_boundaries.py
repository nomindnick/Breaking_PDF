#!/usr/bin/env python3
"""
Inspect text content around expected boundaries to understand why detection is failing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.preprocessing.pdf_handler import PDFHandler
from pdf_splitter.preprocessing.text_extractor import TextExtractor


def main():
    """Inspect text around boundaries."""
    pdf_path = Path("test_files/Test_PDF_Set_2_ocr.pdf")
    expected_boundaries = [5, 7, 9, 13, 14, 18, 20, 23, 26, 32, 34, 35, 36]

    config = PDFConfig()
    handler = PDFHandler(config)

    print("Inspecting Text Around Expected Boundaries")
    print("=" * 80)

    with handler.load_pdf(pdf_path) as loaded_handler:
        extractor = TextExtractor(loaded_handler)

        # Look at pages around first few boundaries
        for boundary in expected_boundaries[:5]:
            print(f"\n{'='*60}")
            print(f"BOUNDARY AT PAGE {boundary}")
            print(f"{'='*60}")

            # Show page before, at, and after boundary
            for offset in [-1, 0, 1]:
                page_num = boundary + offset
                if 1 <= page_num <= loaded_handler.page_count:
                    page_type = loaded_handler.get_page_type(page_num - 1)

                    if page_type.value in ["searchable", "mixed"]:
                        extracted = extractor.extract_page_text(page_num)
                        text = extracted.text
                    else:
                        text = "[Image-based page]"

                    # Clean up text for display
                    text_preview = text[:300].strip().replace("\n", " ")

                    print(
                        f"\nPage {page_num} ({'BEFORE' if offset < 0 else 'AT BOUNDARY' if offset == 0 else 'AFTER'}):"
                    )
                    print(f"Text: {text_preview}...")

                    # Look for document type indicators
                    doc_indicators = [
                        "From:",
                        "Subject:",
                        "INVOICE",
                        "RFI",
                        "SUBMITTAL",
                        "APPLICATION",
                        "Date:",
                    ]
                    found_indicators = [
                        ind for ind in doc_indicators if ind in text.upper()
                    ]
                    if found_indicators:
                        print(f"Document indicators found: {found_indicators}")


if __name__ == "__main__":
    main()
