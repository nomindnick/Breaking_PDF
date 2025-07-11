"""Service for handling PDF splitting operations.

This module provides business logic for managing split proposals,
executing splits, and generating previews.
"""

import asyncio
import base64
import io
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import pikepdf
from PIL import Image

from pdf_splitter.api.exceptions import SessionStateError, SplitError, ValidationError
from pdf_splitter.core.config import PDFConfig
from pdf_splitter.core.logging import get_logger
from pdf_splitter.splitting.models import DocumentSegment
from pdf_splitter.splitting.pdf_splitter import PDFSplitter

logger = get_logger(__name__)


class SplittingService:
    """Service for managing PDF splitting operations."""

    def __init__(self, config: PDFConfig, output_base_dir: Path):
        """Initialize splitting service.

        Args:
            config: PDF processing configuration
            output_base_dir: Base directory for output files
        """
        self.config = config
        self.output_base_dir = output_base_dir
        self.pdf_splitter = PDFSplitter(config)
        self._split_tasks: Dict[str, dict] = {}  # Track running splits

    async def generate_preview(
        self,
        session,
        segment_id: str,
        max_pages: int = 3,
        resolution: int = 150,
        format: str = "png",
    ) -> dict:
        """Generate preview for a document segment.

        Args:
            session: Session object
            segment_id: Segment identifier
            max_pages: Maximum pages to include in preview
            resolution: Preview resolution in DPI
            format: Output format (png, jpeg, webp)

        Returns:
            Preview data with base64 encoded images

        Raises:
            ValidationError: If segment not found
            SplitError: If preview generation fails
        """
        try:
            # Find segment
            segment = session.proposal.get_segment(segment_id)
            if not segment:
                raise ValidationError(
                    f"Segment {segment_id} not found", field="segment_id"
                )

            # Generate preview using PyMuPDF
            import fitz  # PyMuPDF

            pdf_path = session.proposal.pdf_path
            images = []

            # Open PDF
            pdf_doc = fitz.open(pdf_path)

            # Calculate pages to preview
            preview_pages = min(max_pages, segment.page_count)
            page_indices = list(
                range(segment.start_page, segment.start_page + preview_pages)
            )

            for page_idx in page_indices:
                if page_idx >= len(pdf_doc):
                    break

                # Render page
                page = pdf_doc[page_idx]
                mat = fitz.Matrix(resolution / 72.0, resolution / 72.0)
                pix = page.get_pixmap(matrix=mat)

                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                # Convert to requested format
                output_buffer = io.BytesIO()
                if format == "jpeg":
                    img = img.convert("RGB")  # JPEG doesn't support transparency
                    img.save(output_buffer, format="JPEG", quality=85)
                elif format == "webp":
                    img.save(output_buffer, format="WEBP", quality=85)
                else:  # png
                    img.save(output_buffer, format="PNG")

                # Encode to base64
                output_buffer.seek(0)
                base64_image = base64.b64encode(output_buffer.read()).decode("utf-8")
                images.append(f"data:image/{format};base64,{base64_image}")

            pdf_doc.close()

            return {
                "segment_id": segment_id,
                "preview_type": format,
                "pages_included": len(images),
                "images": images,
                "metadata": {
                    "total_pages": segment.page_count,
                    "document_type": segment.document_type,
                    "suggested_filename": segment.suggested_filename,
                },
            }

        except Exception as e:
            logger.error(f"Failed to generate preview: {str(e)}")
            raise SplitError(f"Failed to generate preview: {str(e)}")

    async def execute_split(
        self,
        session,
        output_format: str = "pdf",
        compress: bool = False,
        create_zip: bool = True,
        preserve_metadata: bool = True,
        generate_manifest: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Execute PDF split operation.

        Args:
            session: Session object
            output_format: Output format
            compress: Whether to compress PDFs
            create_zip: Whether to create ZIP archive
            preserve_metadata: Whether to preserve PDF metadata
            generate_manifest: Whether to generate manifest
            progress_callback: Progress callback function

        Returns:
            Split ID

        Raises:
            SessionStateError: If session not in correct state
            SplitError: If split execution fails
        """
        split_id = str(uuid4())

        try:
            # Check session state
            if session.status not in ["pending", "modified", "confirmed"]:
                raise SessionStateError(
                    session.session_id,
                    session.status,
                    "pending, modified, or confirmed",
                )

            # Initialize split task
            self._split_tasks[split_id] = {
                "session_id": session.session_id,
                "status": "initializing",
                "progress": 0.0,
                "start_time": time.time(),
                "current_segment": 0,
                "total_segments": len(session.proposal.segments),
                "files_created": 0,
            }

            # Start split in background
            asyncio.create_task(
                self._run_split(
                    split_id,
                    session,
                    output_format,
                    compress,
                    create_zip,
                    preserve_metadata,
                    generate_manifest,
                    progress_callback,
                )
            )

            logger.info(
                f"Started split {split_id} for session {session.session_id} "
                f"with {len(session.proposal.segments)} segments"
            )

            return split_id

        except Exception as e:
            logger.error(f"Failed to start split: {str(e)}")
            if split_id in self._split_tasks:
                del self._split_tasks[split_id]
            raise SplitError(f"Failed to start split: {str(e)}", session.session_id)

    async def _run_split(
        self,
        split_id: str,
        session,
        output_format: str,
        compress: bool,
        create_zip: bool,
        preserve_metadata: bool,
        generate_manifest: bool,
        progress_callback: Optional[callable],
    ):
        """Run split operation asynchronously.

        Args:
            split_id: Split identifier
            session: Session object
            output_format: Output format
            compress: Compress PDFs
            create_zip: Create ZIP archive
            preserve_metadata: Preserve metadata
            generate_manifest: Generate manifest
            progress_callback: Progress callback
        """
        task_info = self._split_tasks[split_id]

        try:
            # Create output directory
            output_dir = self.output_base_dir / "splits" / session.session_id / split_id
            output_dir.mkdir(parents=True, exist_ok=True)

            task_info["status"] = "splitting"
            output_files = []

            # Split each segment
            for i, segment in enumerate(session.proposal.segments):
                # Update progress
                task_info["current_segment"] = i + 1
                progress = (
                    i / len(session.proposal.segments)
                ) * 0.8  # 80% for splitting
                task_info["progress"] = progress

                if progress_callback:
                    await progress_callback(
                        "splitting",
                        progress,
                        f"Processing segment {i + 1}/{len(session.proposal.segments)}",
                    )

                # Generate output filename
                safe_filename = self._sanitize_filename(segment.suggested_filename)
                if not safe_filename.endswith(".pdf"):
                    safe_filename += ".pdf"
                output_path = output_dir / safe_filename

                # Split PDF
                await asyncio.to_thread(
                    self._split_segment,
                    session.proposal.pdf_path,
                    segment,
                    output_path,
                    preserve_metadata,
                    compress,
                )

                output_files.append(output_path)
                task_info["files_created"] += 1

            # Generate manifest if requested
            if generate_manifest:
                task_info["status"] = "generating_manifest"
                task_info["progress"] = 0.85

                if progress_callback:
                    await progress_callback(
                        "generating_manifest", 0.85, "Generating manifest..."
                    )

                manifest_path = output_dir / "manifest.json"
                await asyncio.to_thread(
                    self._generate_manifest,
                    session,
                    output_files,
                    manifest_path,
                )

            # Create ZIP if requested
            zip_path = None
            if create_zip:
                task_info["status"] = "packaging"
                task_info["progress"] = 0.9

                if progress_callback:
                    await progress_callback("packaging", 0.9, "Creating ZIP archive...")

                zip_path = output_dir / f"{session.proposal.pdf_path.stem}_split.zip"
                await asyncio.to_thread(
                    self._create_zip,
                    output_files,
                    manifest_path if generate_manifest else None,
                    zip_path,
                )

            # Complete
            task_info["status"] = "completed"
            task_info["progress"] = 1.0
            task_info["results"] = {
                "output_files": [str(f) for f in output_files],
                "zip_file": str(zip_path) if zip_path else None,
                "manifest_file": str(manifest_path) if generate_manifest else None,
                "output_dir": str(output_dir),
            }
            task_info["end_time"] = time.time()
            task_info["duration"] = task_info["end_time"] - task_info["start_time"]

            if progress_callback:
                await progress_callback(
                    "completed",
                    1.0,
                    f"Split completed. Created {len(output_files)} files.",
                )

            # Update session
            session.complete()

            logger.info(
                f"Split {split_id} completed successfully. "
                f"Created {len(output_files)} files in {task_info['duration']:.2f}s"
            )

        except Exception as e:
            logger.error(f"Split {split_id} failed: {str(e)}")
            task_info["status"] = "failed"
            task_info["error"] = str(e)
            task_info["end_time"] = time.time()

            if progress_callback:
                await progress_callback(
                    "failed", task_info["progress"], f"Split failed: {str(e)}"
                )

    def _split_segment(
        self,
        input_path: Path,
        segment: DocumentSegment,
        output_path: Path,
        preserve_metadata: bool,
        compress: bool,
    ):
        """Split a single segment from PDF.

        Args:
            input_path: Input PDF path
            segment: Document segment
            output_path: Output path
            preserve_metadata: Preserve metadata
            compress: Compress output
        """
        with pikepdf.open(input_path) as pdf:
            # Create new PDF with selected pages
            new_pdf = pikepdf.new()

            # Copy pages
            for page_num in range(segment.start_page, segment.end_page + 1):
                if page_num < len(pdf.pages):
                    new_pdf.pages.append(pdf.pages[page_num])

            # Update metadata if requested
            if preserve_metadata:
                new_pdf.docinfo = pdf.docinfo.copy()

            # Add custom metadata
            new_pdf.docinfo["/Title"] = segment.suggested_filename
            new_pdf.docinfo[
                "/Subject"
            ] = f"{segment.document_type} - {segment.summary or ''}"
            new_pdf.docinfo["/Creator"] = "PDF Splitter"

            # Save with optional compression
            if compress:
                new_pdf.save(
                    output_path,
                    compress_streams=True,
                    object_stream_mode=pikepdf.ObjectStreamMode.generate,
                )
            else:
                new_pdf.save(output_path)

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Remove potentially dangerous characters
        safe_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. "
        )
        sanitized = "".join(c if c in safe_chars else "_" for c in filename)

        # Remove multiple spaces/underscores
        import re

        sanitized = re.sub(r"[_\s]+", "_", sanitized)

        # Ensure not empty
        if not sanitized or sanitized.strip() in [".", ".."]:
            sanitized = "document"

        return sanitized.strip()

    def _generate_manifest(
        self, session, output_files: List[Path], manifest_path: Path
    ):
        """Generate JSON manifest for split operation.

        Args:
            session: Session object
            output_files: List of output files
            manifest_path: Path for manifest file
        """
        import json

        manifest = {
            "session_id": session.session_id,
            "source_file": session.proposal.pdf_path.name,
            "total_pages": session.proposal.total_pages,
            "split_date": datetime.now().isoformat(),
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "filename": output_files[i].name,
                    "start_page": segment.start_page + 1,  # 1-indexed for users
                    "end_page": segment.end_page + 1,
                    "page_count": segment.page_count,
                    "document_type": segment.document_type,
                    "confidence": segment.confidence,
                    "summary": segment.summary,
                    "metadata": segment.metadata,
                }
                for i, segment in enumerate(session.proposal.segments)
            ],
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _create_zip(
        self, files: List[Path], manifest_path: Optional[Path], zip_path: Path
    ):
        """Create ZIP archive of output files.

        Args:
            files: List of files to include
            manifest_path: Optional manifest file
            zip_path: Output ZIP path
        """
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in files:
                zf.write(file_path, file_path.name)

            if manifest_path and manifest_path.exists():
                zf.write(manifest_path, manifest_path.name)

    def get_split_status(self, split_id: str) -> dict:
        """Get status of split operation.

        Args:
            split_id: Split identifier

        Returns:
            Split status information

        Raises:
            ValidationError: If split not found
        """
        task_info = self._split_tasks.get(split_id)
        if not task_info:
            raise ValidationError(f"Split {split_id} not found", field="split_id")

        return {
            "split_id": split_id,
            "session_id": task_info["session_id"],
            "status": task_info["status"],
            "progress": task_info["progress"],
            "current_segment": task_info.get("current_segment", 0),
            "total_segments": task_info.get("total_segments", 0),
            "files_created": task_info.get("files_created", 0),
            "elapsed_time": time.time() - task_info["start_time"],
            "error": task_info.get("error"),
        }

    def get_split_results(self, split_id: str) -> dict:
        """Get results of completed split.

        Args:
            split_id: Split identifier

        Returns:
            Split results

        Raises:
            ValidationError: If split not found or not completed
        """
        task_info = self._split_tasks.get(split_id)
        if not task_info:
            raise ValidationError(f"Split {split_id} not found", field="split_id")

        if task_info["status"] != "completed":
            raise ValidationError(
                f"Split {split_id} is not completed (status: {task_info['status']})",
                field="split_id",
            )

        return task_info.get("results", {})
