#!/usr/bin/env python3
"""
Complete example demonstrating the full PDF splitting workflow.

This example shows how to:
1. Load and preprocess a PDF
2. Detect document boundaries
3. Generate a split proposal
4. Create a session for user review
5. Apply user modifications
6. Execute the split operation
"""

import asyncio
from pathlib import Path

from pdf_splitter.detection import ProductionDetector
from pdf_splitter.preprocessing import PDFHandler
from pdf_splitter.splitting import PDFSplitter, SplitSessionManager, UserModification


async def split_pdf_example(pdf_path: Path, output_dir: Path):
    """Demonstrate complete PDF splitting workflow."""
    print(f"Processing PDF: {pdf_path}")
    print("-" * 50)

    # Step 1: Load and preprocess the PDF
    print("1. Loading PDF...")
    pdf_handler = PDFHandler()
    loaded_pdf = await pdf_handler.load_pdf(pdf_path)

    if loaded_pdf.requires_ocr:
        print("   PDF requires OCR - this may take a moment...")

    # Extract text from all pages
    pages = await pdf_handler.process_all_pages(loaded_pdf)
    print(f"   Loaded {len(pages)} pages")

    # Step 2: Detect document boundaries
    print("\n2. Detecting document boundaries...")
    detector = ProductionDetector()
    boundaries = await detector.detect_boundaries(pages)

    print(
        f"   Found {len([b for b in boundaries if b.confidence > 0.5])} document boundaries"
    )

    # Step 3: Generate split proposal
    print("\n3. Generating split proposal...")
    splitter = PDFSplitter()
    proposal = splitter.generate_proposal(
        boundaries=boundaries, pages=pages, pdf_path=pdf_path
    )

    print(f"   Proposed {len(proposal.segments)} documents:")
    for i, segment in enumerate(proposal.segments):
        print(
            f"   - Document {i+1}: {segment.document_type} "
            f"(pages {segment.start_page+1}-{segment.end_page+1})"
        )
        print(f"     Suggested name: {segment.suggested_filename}")

    # Step 4: Create session for user review
    print("\n4. Creating session for user review...")
    session_manager = SplitSessionManager()
    session = session_manager.create_session(proposal)
    print(f"   Session ID: {session.session_id}")

    # Step 5: Simulate user modifications (optional)
    print("\n5. Applying user modifications...")

    # Example: Rename the first document
    if proposal.segments:
        modification = UserModification(
            modification_type="rename",
            segment_id=proposal.segments[0].segment_id,
            details={"new_filename": "custom_document_name.pdf"},
        )

        session = session_manager.update_session(
            session.session_id, modifications=[modification]
        )
        print("   Renamed first document to: custom_document_name.pdf")

    # Step 6: Confirm and execute split
    print("\n6. Executing split operation...")

    # Confirm the session
    session = session_manager.update_session(
        session.session_id, status="confirmed", output_directory=output_dir
    )

    # Execute the split
    custom_names = (
        {proposal.segments[0].segment_id: "custom_document_name.pdf"}
        if proposal.segments
        else {}
    )

    result = splitter.split_pdf(
        proposal=proposal, output_dir=output_dir, use_custom_names=custom_names
    )

    # Mark session as completed
    session_manager.update_session(session.session_id, status="completed")

    print("\n✅ Split complete!")
    print(
        f"   Created {len(result.output_files)} files in {result.duration_seconds:.2f} seconds"
    )
    print(f"   Output directory: {output_dir}")

    for output_file in result.output_files:
        print(f"   - {output_file.name}")

    # Cleanup
    await pdf_handler.cleanup()


async def main():
    """Run the example."""
    # Configure paths
    pdf_path = Path("test_data/Test_PDF_Set_1.pdf")
    output_dir = Path("output/split_results")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if test PDF exists
    if not pdf_path.exists():
        print(f"Error: Test PDF not found at {pdf_path}")
        print("Please ensure you have the test data files in place.")
        return

    # Run the example
    try:
        await split_pdf_example(pdf_path, output_dir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
