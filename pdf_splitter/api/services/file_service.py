"""
File Service Module

Handles file upload, storage, validation, and management.
"""
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple

import aiofiles
from fastapi import UploadFile

from pdf_splitter.api.config import config
from pdf_splitter.api.utils.exceptions import (
    FileNotFoundError,
    FileSizeError,
    FileTypeError,
    FileUploadError,
)
from pdf_splitter.preprocessing.pdf_handler import PDFHandler


class FileService:
    """Service for handling file operations."""

    def __init__(self):
        self.upload_dir = config.upload_dir
        self.chunk_size = config.chunk_size
        self.max_file_size = config.max_upload_size
        self.allowed_extensions = config.allowed_extensions

        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "temp").mkdir(exist_ok=True)
        (self.upload_dir / "metadata").mkdir(exist_ok=True)

    async def save_upload(
        self, file: UploadFile, chunk_number: int = 0, total_chunks: int = 1
    ) -> Dict[str, any]:
        """
        Save uploaded file with validation.

        Args:
            file: Uploaded file object
            chunk_number: Current chunk number for chunked uploads
            total_chunks: Total number of chunks

        Returns:
            Dictionary with upload information

        Raises:
            FileTypeError: If file type is not allowed
            FileSizeError: If file exceeds size limit
            FileUploadError: If upload fails
        """
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            raise FileTypeError(file_ext, list(self.allowed_extensions))

        # Generate unique file ID
        file_id = self._generate_file_id(file.filename)

        # Handle chunked upload
        if total_chunks > 1:
            return await self._handle_chunked_upload(
                file, file_id, chunk_number, total_chunks
            )

        # Validate file size for single upload
        file_size = await self._get_file_size(file)
        if file_size > self.max_file_size:
            raise FileSizeError(file_size, self.max_file_size)

        # Save file
        file_path = self.upload_dir / f"{file_id}.pdf"
        try:
            async with aiofiles.open(file_path, "wb") as f:
                while chunk := await file.read(self.chunk_size):
                    await f.write(chunk)
        except Exception as e:
            # Clean up on failure
            file_path.unlink(missing_ok=True)
            raise FileUploadError(f"Failed to save file: {str(e)}")

        # Validate PDF and get metadata
        try:
            pdf_info = await self._validate_pdf(file_path)
        except Exception as e:
            # Clean up invalid PDF
            file_path.unlink(missing_ok=True)
            raise FileUploadError(f"Invalid PDF file: {str(e)}")

        # Save metadata
        metadata = {
            "file_id": file_id,
            "original_filename": file.filename,
            "file_size": file_size,
            "upload_time": datetime.utcnow().isoformat(),
            "expires_at": (
                datetime.utcnow() + timedelta(seconds=config.session_timeout)
            ).isoformat(),
            "total_pages": pdf_info["page_count"],
            "pdf_version": pdf_info.get("version", "Unknown"),
            "is_encrypted": pdf_info.get("is_encrypted", False),
            "has_text": pdf_info.get("has_text", False),
        }

        await self._save_metadata(file_id, metadata)

        return {
            "file_id": file_id,
            "filename": file.filename,
            "size": file_size,
            "total_pages": pdf_info["page_count"],
            "status": "uploaded",
        }

    async def _handle_chunked_upload(
        self, file: UploadFile, file_id: str, chunk_number: int, total_chunks: int
    ) -> Dict[str, any]:
        """Handle chunked file upload."""
        temp_dir = self.upload_dir / "temp" / file_id
        temp_dir.mkdir(exist_ok=True)

        # Save chunk
        chunk_path = temp_dir / f"chunk_{chunk_number}"
        async with aiofiles.open(chunk_path, "wb") as f:
            while data := await file.read(self.chunk_size):
                await f.write(data)

        # Check if all chunks are uploaded
        uploaded_chunks = list(temp_dir.glob("chunk_*"))
        if len(uploaded_chunks) == total_chunks:
            # Combine chunks
            final_path = self.upload_dir / f"{file_id}.pdf"
            async with aiofiles.open(final_path, "wb") as outfile:
                for i in range(total_chunks):
                    chunk_file = temp_dir / f"chunk_{i}"
                    async with aiofiles.open(chunk_file, "rb") as infile:
                        while data := await infile.read(self.chunk_size):
                            await outfile.write(data)

            # Clean up chunks
            shutil.rmtree(temp_dir)

            # Validate complete file
            file_size = final_path.stat().st_size
            if file_size > self.max_file_size:
                final_path.unlink()
                raise FileSizeError(file_size, self.max_file_size)

            # Validate PDF
            try:
                pdf_info = await self._validate_pdf(final_path)
            except Exception as e:
                final_path.unlink()
                raise FileUploadError(f"Invalid PDF after combining chunks: {str(e)}")

            # Save metadata
            metadata = {
                "file_id": file_id,
                "original_filename": file.filename,
                "file_size": file_size,
                "upload_time": datetime.utcnow().isoformat(),
                "expires_at": (
                    datetime.utcnow() + timedelta(seconds=config.session_timeout)
                ).isoformat(),
                "total_pages": pdf_info["page_count"],
                "pdf_version": pdf_info.get("version", "Unknown"),
                "is_encrypted": pdf_info.get("is_encrypted", False),
                "has_text": pdf_info.get("has_text", False),
            }

            await self._save_metadata(file_id, metadata)

            return {
                "file_id": file_id,
                "filename": file.filename,
                "size": file_size,
                "total_pages": pdf_info["page_count"],
                "status": "uploaded",
                "chunks_processed": total_chunks,
            }
        else:
            return {
                "file_id": file_id,
                "chunk_number": chunk_number,
                "chunks_received": len(uploaded_chunks),
                "total_chunks": total_chunks,
                "status": "uploading",
            }

    async def get_file_path(self, file_id: str) -> Path:
        """Get the path to an uploaded file."""
        file_path = self.upload_dir / f"{file_id}.pdf"
        if not file_path.exists():
            raise FileNotFoundError(file_id)
        return file_path

    async def get_file_metadata(self, file_id: str) -> Dict[str, any]:
        """Get metadata for an uploaded file."""
        metadata_path = self.upload_dir / "metadata" / f"{file_id}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(file_id)

        async with aiofiles.open(metadata_path, "r") as f:
            content = await f.read()
            return json.loads(content)

    async def delete_file(self, file_id: str) -> bool:
        """Delete an uploaded file and its metadata."""
        file_path = self.upload_dir / f"{file_id}.pdf"
        metadata_path = self.upload_dir / "metadata" / f"{file_id}.json"

        deleted = False
        if file_path.exists():
            file_path.unlink()
            deleted = True

        if metadata_path.exists():
            metadata_path.unlink()
            deleted = True

        return deleted

    async def cleanup_expired_files(self) -> int:
        """Clean up expired files. Returns number of files deleted."""
        deleted_count = 0
        current_time = datetime.utcnow()

        # Check all metadata files
        metadata_dir = self.upload_dir / "metadata"
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                async with aiofiles.open(metadata_file, "r") as f:
                    content = await f.read()
                    metadata = json.loads(content)

                expires_at = datetime.fromisoformat(metadata.get("expires_at"))
                if current_time > expires_at:
                    file_id = metadata_file.stem
                    if await self.delete_file(file_id):
                        deleted_count += 1
            except Exception:
                # Skip files with errors
                continue

        return deleted_count

    def _generate_file_id(self, filename: str) -> str:
        """Generate a unique file ID."""
        timestamp = str(time.time()).encode()
        filename_bytes = filename.encode()
        return hashlib.sha256(timestamp + filename_bytes).hexdigest()[:16]

    async def _get_file_size(self, file: UploadFile) -> int:
        """Get the size of an uploaded file."""
        # Try to get size from content-length header
        if hasattr(file, "size") and file.size:
            return file.size

        # Otherwise, read and count
        size = 0
        async for chunk in file:
            size += len(chunk)

        # Reset file position
        await file.seek(0)
        return size

    async def _validate_pdf(self, file_path: Path) -> Dict[str, any]:
        """Validate PDF file and extract basic information."""
        try:
            # Use existing PDFHandler for validation
            pdf_handler = PDFHandler()

            # This will raise an exception if PDF is invalid
            with pdf_handler.load_pdf(str(file_path)) as pdf_doc:
                page_count = len(pdf_doc)

                # Check if PDF has text
                has_text = False
                if page_count > 0:
                    page = pdf_doc[0]
                    text = page.get_text()
                    has_text = bool(text.strip())

                return {
                    "page_count": page_count,
                    "version": pdf_doc.metadata.get("format", "Unknown"),
                    "is_encrypted": pdf_doc.is_encrypted,
                    "has_text": has_text,
                    "metadata": dict(pdf_doc.metadata),
                }
        except Exception as e:
            raise ValueError(f"Invalid PDF file: {str(e)}")

    async def _save_metadata(self, file_id: str, metadata: Dict[str, any]):
        """Save file metadata."""
        metadata_path = self.upload_dir / "metadata" / f"{file_id}.json"
        async with aiofiles.open(metadata_path, "w") as f:
            await f.write(json.dumps(metadata, indent=2))
