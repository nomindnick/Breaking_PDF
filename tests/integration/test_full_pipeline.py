"""Integration tests for the complete PDF splitting pipeline.

These tests verify that all modules work together correctly:
1. Preprocessing (PDF loading and text extraction)
2. Detection (boundary identification)
3. Splitting (document separation and output)
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import List

import pytest

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection import ProductionDetector
from pdf_splitter.detection.base_detector import BoundaryType
from pdf_splitter.preprocessing import PDFHandler
from pdf_splitter.splitting import PDFSplitter, SplitSessionManager, UserModification


class TestFullPipeline:
    """Test the complete PDF splitting pipeline."""

    @pytest.fixture
    def test_pdfs(self) -> List[Path]:
        """Get paths to test PDF files."""
        test_files = [
            Path("test_data/Test_PDF_Set_1.pdf"),
            Path("test_data/Test_PDF_Set_2_ocr.pdf"),
        ]

        # Verify test files exist
        missing_files = [f for f in test_files if not f.exists()]
        if missing_files:
            pytest.skip(f"Test PDFs not found: {missing_files}")

        return test_files

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PDFConfig(
            default_dpi=150, parallel_workers=2, enable_cache=True, debug=True
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("test_data/Test_PDF_Set_1.pdf").exists(),
        reason="Test PDF not available",
    )
    async def test_complete_pipeline_non_ocr(self, test_pdfs, temp_output_dir, config):
        """Test complete pipeline with non-OCR PDF."""
        pdf_path = test_pdfs[0]  # Non-OCR PDF

        # Step 1: Initialize components
        pdf_handler = PDFHandler(config=config)
        detector = ProductionDetector(config=config)
        splitter = PDFSplitter(config=config)
        session_manager = SplitSessionManager(
            config=config, db_path=temp_output_dir / "sessions.db"
        )

        try:
            # Step 2: Load and preprocess PDF
            print(f"\n1. Loading PDF: {pdf_path}")
            loaded_pdf = await pdf_handler.load_pdf(pdf_path)
            assert loaded_pdf is not None
            assert loaded_pdf.num_pages == 32
            assert not loaded_pdf.requires_ocr

            # Extract text from all pages
            pages = await pdf_handler.process_all_pages(loaded_pdf)
            assert len(pages) == 32

            # Verify text extraction
            for page in pages[:5]:  # Check first 5 pages
                assert page.text.strip(), f"Page {page.page_number} has no text"
                assert page.page_type in ["SEARCHABLE", "MIXED"]

            # Step 3: Detect boundaries
            print("\n2. Detecting document boundaries...")
            boundaries = await detector.detect_boundaries(pages)
            assert len(boundaries) > 0

            # Filter significant boundaries
            significant_boundaries = [
                b
                for b in boundaries
                if b.boundary_type == BoundaryType.DOCUMENT_START and b.confidence > 0.5
            ]
            print(f"   Found {len(significant_boundaries)} document boundaries")
            assert len(significant_boundaries) >= 2  # Expect at least 2 documents

            # Step 4: Generate split proposal
            print("\n3. Generating split proposal...")
            proposal = splitter.generate_proposal(
                boundaries=boundaries, pages=pages, pdf_path=pdf_path
            )

            assert proposal is not None
            assert len(proposal.segments) >= 2
            assert proposal.total_pages == 32

            # Verify segments
            for i, segment in enumerate(proposal.segments):
                print(
                    f"   Segment {i+1}: {segment.document_type} "
                    f"(pages {segment.start_page+1}-{segment.end_page+1})"
                )
                assert segment.start_page >= 0
                assert segment.end_page < 32
                assert segment.document_type in [
                    "Invoice",
                    "Email",
                    "Letter",
                    "Report",
                    "Contract",
                    "Plans",
                    "Document",
                ]
                assert segment.suggested_filename.endswith(".pdf")
                assert segment.confidence > 0

            # Step 5: Create and manage session
            print("\n4. Creating session...")
            session = session_manager.create_session(proposal)
            assert session is not None
            assert session.status == "pending"

            # Simulate user modification
            if proposal.segments:
                modification = UserModification(
                    modification_type="rename",
                    segment_id=proposal.segments[0].segment_id,
                    details={"new_filename": "integration_test_renamed.pdf"},
                )

                session = session_manager.update_session(
                    session.session_id, modifications=[modification]
                )
                assert session.status == "modified"

            # Step 6: Execute split
            print("\n5. Executing split...")
            output_dir = temp_output_dir / "split_output"
            output_dir.mkdir(exist_ok=True)

            # Confirm session
            session = session_manager.update_session(
                session.session_id, status="confirmed", output_directory=output_dir
            )

            # Execute split with custom names
            custom_names = {}
            if proposal.segments:
                custom_names[
                    proposal.segments[0].segment_id
                ] = "integration_test_renamed.pdf"

            result = splitter.split_pdf(
                proposal=proposal, output_dir=output_dir, use_custom_names=custom_names
            )

            # Verify results
            assert result is not None
            assert len(result.output_files) == len(proposal.segments)
            assert result.duration_seconds > 0

            # Check output files
            for output_file in result.output_files:
                assert output_file.exists()
                assert output_file.stat().st_size > 0
                print(
                    f"   Created: {output_file.name} ({output_file.stat().st_size} bytes)"
                )

            # Verify custom naming worked
            if custom_names:
                renamed_file = output_dir / "integration_test_renamed.pdf"
                assert renamed_file.exists()

            # Mark session complete
            session_manager.update_session(session.session_id, status="completed")

            print("\n✅ Pipeline test completed successfully!")

        finally:
            # Cleanup
            await pdf_handler.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("test_data/Test_PDF_Set_2_ocr.pdf").exists(),
        reason="OCR test PDF not available",
    )
    async def test_complete_pipeline_with_ocr(self, test_pdfs, temp_output_dir, config):
        """Test complete pipeline with OCR PDF."""
        pdf_path = test_pdfs[1]  # OCR PDF

        # Use faster OCR settings for testing
        config.ocr_lang = "eng"
        config.ocr_confidence_threshold = 0.3

        pdf_handler = PDFHandler(config=config)
        detector = ProductionDetector(config=config)
        splitter = PDFSplitter(config=config)

        try:
            # Load PDF
            print(f"\n1. Loading OCR PDF: {pdf_path}")
            loaded_pdf = await pdf_handler.load_pdf(pdf_path)
            assert loaded_pdf is not None

            # Process only first few pages for speed
            print("2. Processing first 5 pages with OCR...")
            pages = await pdf_handler.process_pages(
                loaded_pdf, page_numbers=list(range(5))
            )
            assert len(pages) == 5

            # Verify OCR worked
            for page in pages:
                if page.page_type == "IMAGE_BASED":
                    assert page.text.strip(), f"OCR failed for page {page.page_number}"
                    assert page.ocr_confidence is not None

            # Detect boundaries on partial document
            print("3. Detecting boundaries...")
            boundaries = await detector.detect_boundaries(pages)
            assert len(boundaries) > 0

            # Generate proposal
            print("4. Generating proposal...")
            proposal = splitter.generate_proposal(
                boundaries=boundaries, pages=pages, pdf_path=pdf_path
            )

            assert proposal is not None
            assert len(proposal.segments) >= 1

            print("\n✅ OCR Pipeline test completed!")
            print(f"   Processed {len(pages)} pages")
            print(f"   Found {len(proposal.segments)} documents")

        finally:
            await pdf_handler.cleanup()

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, temp_output_dir, config):
        """Test pipeline error handling."""
        pdf_handler = PDFHandler(config=config)
        splitter = PDFSplitter(config=config)

        # Test with non-existent PDF
        fake_pdf = Path("/nonexistent/fake.pdf")

        with pytest.raises(FileNotFoundError):
            await pdf_handler.load_pdf(fake_pdf)

        # Test with empty boundaries
        from pdf_splitter.splitting.exceptions import PDFSplitError

        with pytest.raises(PDFSplitError):
            splitter.generate_proposal(boundaries=[], pages=[], pdf_path=fake_pdf)

        await pdf_handler.cleanup()

    @pytest.mark.asyncio
    async def test_preview_generation(self, test_pdfs, temp_output_dir, config):
        """Test preview generation functionality."""
        if not test_pdfs:
            pytest.skip("No test PDFs available")

        pdf_path = test_pdfs[0]
        splitter = PDFSplitter(config=config)

        # Create a test segment
        from pdf_splitter.splitting.models import DocumentSegment

        segment = DocumentSegment(
            start_page=0,
            end_page=5,
            document_type="Test",
            suggested_filename="test.pdf",
            confidence=0.9,
        )

        # Generate preview
        preview_bytes = splitter.generate_preview(
            pdf_path=pdf_path, segment=segment, max_pages=3
        )

        assert preview_bytes is not None
        assert len(preview_bytes) > 0
        assert preview_bytes.startswith(b"%PDF")  # PDF magic number

        print(f"✅ Generated preview: {len(preview_bytes)} bytes")

    @pytest.mark.asyncio
    async def test_session_persistence(self, test_pdfs, temp_output_dir, config):
        """Test session persistence across manager instances."""
        if not test_pdfs:
            pytest.skip("No test PDFs available")

        db_path = temp_output_dir / "test_sessions.db"

        # Create first manager and session
        manager1 = SplitSessionManager(config=config, db_path=db_path)

        from pdf_splitter.splitting.models import DocumentSegment, SplitProposal

        proposal = SplitProposal(
            pdf_path=test_pdfs[0],
            total_pages=10,
            segments=[
                DocumentSegment(
                    start_page=0,
                    end_page=5,
                    document_type="Test",
                    suggested_filename="test.pdf",
                    confidence=0.9,
                )
            ],
            detection_results=[],
        )

        session = manager1.create_session(proposal)
        session_id = session.session_id

        # Create second manager and retrieve session
        manager2 = SplitSessionManager(config=config, db_path=db_path)
        retrieved = manager2.get_session(session_id)

        assert retrieved.session_id == session_id
        assert len(retrieved.proposal.segments) == 1
        assert retrieved.status == "pending"

        print("✅ Session persistence verified")

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_pipeline_performance(
        self, test_pdfs, temp_output_dir, config, benchmark
    ):
        """Benchmark pipeline performance."""
        if not test_pdfs:
            pytest.skip("No test PDFs available")

        pdf_path = test_pdfs[0]

        async def run_pipeline():
            """Run complete pipeline for benchmarking."""
            pdf_handler = PDFHandler(config=config)
            detector = ProductionDetector(config=config)
            splitter = PDFSplitter(config=config)

            try:
                # Load and process
                loaded_pdf = await pdf_handler.load_pdf(pdf_path)
                pages = await pdf_handler.process_pages(
                    loaded_pdf, page_numbers=list(range(10))  # First 10 pages
                )

                # Detect boundaries
                boundaries = await detector.detect_boundaries(pages)

                # Generate proposal
                proposal = splitter.generate_proposal(
                    boundaries=boundaries, pages=pages, pdf_path=pdf_path
                )

                return len(proposal.segments)

            finally:
                await pdf_handler.cleanup()

        # Run benchmark
        result = benchmark(lambda: asyncio.run(run_pipeline()))

        print("\n📊 Performance Results:")
        print(f"   Mean time: {benchmark.stats['mean']:.3f}s")
        print(f"   Min time: {benchmark.stats['min']:.3f}s")
        print(f"   Max time: {benchmark.stats['max']:.3f}s")
        print(f"   Documents found: {result}")

        # Verify performance targets
        assert benchmark.stats["mean"] < 5.0  # Should complete in < 5 seconds
