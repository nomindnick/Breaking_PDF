#!/usr/bin/env python3
"""
Quick demonstration of the PDF splitting module.

This example shows the simplest way to split a PDF using the production detector.
"""

import asyncio
from pathlib import Path

from pdf_splitter.detection import ProductionDetector
from pdf_splitter.preprocessing import PDFHandler
from pdf_splitter.splitting import PDFSplitter


async def quick_split(pdf_path: Path, output_dir: Path):
    """Quickly split a PDF with minimal configuration."""
    # Initialize components
    pdf_handler = PDFHandler()
    detector = ProductionDetector()  # Uses embeddings detector
    splitter = PDFSplitter()

    try:
        # Load and process PDF
        print(f"📄 Loading {pdf_path.name}...")
        loaded_pdf = await pdf_handler.load_pdf(pdf_path)
        pages = await pdf_handler.process_all_pages(loaded_pdf)
        print(f"✅ Processed {len(pages)} pages")

        # Detect boundaries
        print("\n🔍 Detecting document boundaries...")
        boundaries = await detector.detect_boundaries(pages)
        significant = [b for b in boundaries if b.confidence > 0.5]
        print(f"✅ Found {len(significant)} documents")

        # Generate and execute split
        print("\n✂️  Splitting PDF...")
        proposal = splitter.generate_proposal(boundaries, pages, pdf_path)
        result = splitter.split_pdf(proposal, output_dir)

        # Show results
        print(f"\n✅ Split complete! Created {len(result.output_files)} files:")
        for file in result.output_files:
            size_kb = file.stat().st_size / 1024
            print(f"   📄 {file.name} ({size_kb:.1f} KB)")

        print(f"\n⏱️  Total time: {result.duration_seconds:.2f} seconds")

    finally:
        await pdf_handler.cleanup()


if __name__ == "__main__":
    # Example usage
    pdf_file = Path("test_data/Test_PDF_Set_1.pdf")
    output_folder = Path("output/quick_demo")

    if not pdf_file.exists():
        print(f"❌ Error: {pdf_file} not found!")
        print("Please ensure test data is available.")
    else:
        output_folder.mkdir(parents=True, exist_ok=True)
        asyncio.run(quick_split(pdf_file, output_folder))
