"""Service for handling PDF uploads and validation.

This module provides business logic for file uploads, validation,
and initial processing.
"""

import asyncio
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

from fastapi import UploadFile

from pdf_splitter.api.exceptions import (
    FileSizeError,
    FileTypeError,
    UploadError,
    ValidationError,
)
from pdf_splitter.core.config import PDFConfig
from pdf_splitter.core.logging import get_logger
from pdf_splitter.preprocessing.pdf_handler import PDFHandler

logger = get_logger(__name__)


class UploadService:
    """Service for handling PDF uploads."""

    ALLOWED_CONTENT_TYPES = {
        "application/pdf",
        "application/x-pdf",
        "application/acrobat",
        "applications/vnd.pdf",
        "text/pdf",
        "text/x-pdf",
    }

    def __init__(self, config: PDFConfig, upload_dir: Path):
        """Initialize upload service.

        Args:
            config: PDF processing configuration
            upload_dir: Directory for storing uploads
        """
        self.config = config
        self.upload_dir = upload_dir
        self.pdf_handler = PDFHandler(config=config)

    async def validate_upload(
        self, file: UploadFile, max_size_mb: Optional[float] = None
    ) -> dict:
        """Validate uploaded file before saving.

        Args:
            file: FastAPI UploadFile object
            max_size_mb: Optional max size override

        Returns:
            Validation results dictionary

        Raises:
            FileTypeError: If file type is not allowed
            FileSizeError: If file exceeds size limit
        """
        # Check content type
        if file.content_type not in self.ALLOWED_CONTENT_TYPES:
            raise FileTypeError(
                file_type=file.content_type or "unknown",
                allowed_types=list(self.ALLOWED_CONTENT_TYPES),
            )

        # Check file extension
        if not file.filename.lower().endswith(".pdf"):
            raise FileTypeError(
                file_type=Path(file.filename).suffix or "none",
                allowed_types=[".pdf"],
            )

        # Check file size (read first chunk to get size estimate)
        max_size = max_size_mb or self.config.max_file_size_mb
        max_bytes = int(max_size * 1024 * 1024)

        # Read file to check size
        file_data = await file.read()
        file_size = len(file_data)

        if file_size > max_bytes:
            raise FileSizeError(file_size=file_size, max_size=max_bytes)

        # Reset file position
        await file.seek(0)

        return {
            "valid": True,
            "file_size": file_size,
            "content_type": file.content_type,
            "filename": file.filename,
        }

    async def process_upload(
        self, file: UploadFile, validate_only: bool = False
    ) -> Tuple[str, dict]:
        """Process uploaded PDF file.

        Args:
            file: FastAPI UploadFile object
            validate_only: If True, only validate without storing

        Returns:
            Tuple of (upload_id, upload_info)

        Raises:
            UploadError: If upload processing fails
        """
        try:
            # Validate file
            validation_result = await self.validate_upload(file)

            if validate_only:
                # Return validation results without saving
                return str(uuid4()), {
                    **validation_result,
                    "upload_id": None,
                    "status": "validated",
                    "total_pages": None,
                }

            # Generate upload ID and create directory
            upload_id = str(uuid4())
            upload_subdir = self.upload_dir / upload_id
            upload_subdir.mkdir(parents=True, exist_ok=True)

            # Save file
            file_path = upload_subdir / file.filename
            file_data = await file.read()

            await asyncio.to_thread(self._save_file, file_path, file_data)

            # Process PDF to get metadata
            try:
                pdf_info = await asyncio.to_thread(self._process_pdf, file_path)
            except Exception as e:
                # Clean up on processing error
                file_path.unlink()
                upload_subdir.rmdir()
                raise UploadError(f"Failed to process PDF: {str(e)}")

            upload_info = {
                "upload_id": upload_id,
                "file_name": file.filename,
                "file_size": validation_result["file_size"],
                "file_path": str(file_path),
                "total_pages": pdf_info["total_pages"],
                "status": "uploaded",
                "page_types": pdf_info.get("page_types", {}),
                "has_text": pdf_info.get("has_text", False),
                "needs_ocr": pdf_info.get("needs_ocr", False),
                "processing_time": pdf_info.get("processing_time", 0),
            }

            logger.info(
                f"Successfully processed upload {upload_id}: "
                f"{file.filename} ({pdf_info['total_pages']} pages)"
            )

            return upload_id, upload_info

        except (FileTypeError, FileSizeError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Upload processing failed: {str(e)}")
            raise UploadError(f"Upload processing failed: {str(e)}")

    def _save_file(self, file_path: Path, data: bytes):
        """Save file data to disk.

        Args:
            file_path: Path to save file
            data: File data
        """
        with open(file_path, "wb") as f:
            f.write(data)

    def _process_pdf(self, file_path: Path) -> dict:
        """Process PDF to extract metadata.

        Args:
            file_path: Path to PDF file

        Returns:
            PDF metadata dictionary
        """
        import time

        start_time = time.time()

        # Load PDF using context manager
        with self.pdf_handler.load_pdf(file_path):
            # Get basic info
            total_pages = self.pdf_handler.page_count

            # Analyze pages to determine types
            searchable = 0
            image_based = 0
            empty = 0
            mixed = 0

            # Use get_page_type method to analyze each page
            for i in range(total_pages):
                try:
                    page_type = self.pdf_handler.get_page_type(i)
                    if page_type.value == "searchable":
                        searchable += 1
                    elif page_type.value == "image_based":
                        image_based += 1
                    elif page_type.value == "empty":
                        empty += 1
                    elif page_type.value == "mixed":
                        mixed += 1
                except Exception as e:
                    logger.warning(f"Failed to analyze page {i}: {e}")

            # Determine page types
            page_types = {
                "searchable": searchable,
                "image_based": image_based,
                "mixed": mixed,
                "empty": empty,
            }

            processing_time = time.time() - start_time

            return {
                "total_pages": total_pages,
                "page_types": page_types,
                "has_text": searchable > 0,
                "needs_ocr": image_based > 0,
                "processing_time": processing_time,
            }

    async def get_upload_info(self, upload_id: str, file_path: Path) -> dict:
        """Get information about an uploaded file.

        Args:
            upload_id: Upload identifier
            file_path: Path to uploaded file

        Returns:
            Upload information dictionary
        """
        if not file_path.exists():
            raise UploadError(f"Upload {upload_id} not found")

        try:
            # Get file stats
            file_stats = file_path.stat()

            # Process PDF if not already processed
            pdf_info = await asyncio.to_thread(self._process_pdf, file_path)

            return {
                "upload_id": upload_id,
                "file_name": file_path.name,
                "file_size": file_stats.st_size,
                "file_path": str(file_path),
                "total_pages": pdf_info["total_pages"],
                "status": "ready",
                "created_at": file_stats.st_mtime,
                "page_types": pdf_info.get("page_types", {}),
                "has_text": pdf_info.get("has_text", False),
                "needs_ocr": pdf_info.get("needs_ocr", False),
            }

        except Exception as e:
            logger.error(f"Failed to get upload info: {str(e)}")
            raise UploadError(f"Failed to get upload info: {str(e)}")
