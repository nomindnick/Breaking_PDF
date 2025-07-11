"""Shared dependencies for API endpoints.

This module provides dependency injection functions for FastAPI endpoints.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import Depends, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.core.logging import get_logger
from pdf_splitter.preprocessing.pdf_handler import PDFHandler
from pdf_splitter.splitting.pdf_splitter import PDFSplitter
from pdf_splitter.splitting.session_manager import SplitSessionManager

logger = get_logger(__name__)

# Optional security scheme
security = HTTPBearer(auto_error=False)

# Singleton instances
_pdf_config: Optional[PDFConfig] = None
_session_manager: Optional[SplitSessionManager] = None
_upload_dir: Optional[Path] = None
_test_override_session_manager: Optional[SplitSessionManager] = None


def get_pdf_config() -> PDFConfig:
    """Get PDF configuration singleton."""
    global _pdf_config
    if _pdf_config is None:
        _pdf_config = PDFConfig()
    return _pdf_config


def get_session_manager() -> SplitSessionManager:
    """Get session manager singleton."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SplitSessionManager(config=get_pdf_config())
    return _session_manager


def get_upload_directory() -> Path:
    """Get upload directory, creating if necessary."""
    global _upload_dir
    if _upload_dir is None:
        # Use system temp directory or configured directory
        base_dir = Path(os.environ.get("PDF_UPLOAD_DIR", tempfile.gettempdir()))
        _upload_dir = base_dir / "pdf_splitter_uploads"
        _upload_dir.mkdir(parents=True, exist_ok=True)
    return _upload_dir


def get_pdf_handler(config: PDFConfig = Depends(get_pdf_config)) -> PDFHandler:
    """Get PDF handler instance."""
    return PDFHandler(config=config)


def get_pdf_splitter(config: PDFConfig = Depends(get_pdf_config)) -> PDFSplitter:
    """Get PDF splitter instance."""
    return PDFSplitter(config=config)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """Get current user from auth token (optional).

    This is a placeholder for authentication logic.
    Returns None if no auth is configured.
    """
    if not credentials:
        return None

    # TODO: Implement actual authentication logic here
    # For now, just return a mock user
    return {"user_id": "anonymous", "token": credentials.credentials}


async def require_auth(
    user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Require authentication for protected endpoints."""
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


class UploadManager:
    """Manages file uploads and temporary storage."""

    def __init__(self, upload_dir: Path = None):
        """Initialize upload manager.

        Args:
            upload_dir: Directory for storing uploads
        """
        self.upload_dir = upload_dir or get_upload_directory()
        self.upload_metadata = {}  # In-memory metadata storage

    async def save_upload(self, file: UploadFile) -> tuple[str, Path]:
        """Save uploaded file and return upload ID and path.

        Args:
            file: FastAPI UploadFile object

        Returns:
            Tuple of (upload_id, file_path)
        """
        # Generate unique upload ID
        upload_id = str(uuid4())

        # Create upload subdirectory
        upload_subdir = self.upload_dir / upload_id
        upload_subdir.mkdir(parents=True, exist_ok=True)

        # Save file with original name
        file_path = upload_subdir / file.filename

        try:
            # Save file in chunks
            chunk_size = 1024 * 1024  # 1MB chunks
            with open(file_path, "wb") as f:
                while chunk := await file.read(chunk_size):
                    f.write(chunk)

            # Store metadata
            self.upload_metadata[upload_id] = {
                "filename": file.filename,
                "size": file_path.stat().st_size,
                "content_type": file.content_type,
                "path": str(file_path),
            }

            logger.info(f"Saved upload {upload_id}: {file.filename}")
            return upload_id, file_path

        except Exception:
            # Clean up on error
            if file_path.exists():
                file_path.unlink()
            if upload_subdir.exists():
                upload_subdir.rmdir()
            raise

    def get_upload_path(self, upload_id: str) -> Optional[Path]:
        """Get path for uploaded file.

        Args:
            upload_id: Upload identifier

        Returns:
            Path to uploaded file or None if not found
        """
        metadata = self.upload_metadata.get(upload_id)
        if metadata:
            # Check both "path" and "file_path" for compatibility
            path_str = metadata.get("file_path") or metadata.get("path")
            if path_str:
                path = Path(path_str)
                if path.exists():
                    return path
        return None

    def get_upload_metadata(self, upload_id: str) -> Optional[dict]:
        """Get metadata for uploaded file.

        Args:
            upload_id: Upload identifier

        Returns:
            Upload metadata or None if not found
        """
        return self.upload_metadata.get(upload_id)

    def delete_upload(self, upload_id: str) -> bool:
        """Delete uploaded file and metadata.

        Args:
            upload_id: Upload identifier

        Returns:
            True if deleted, False if not found
        """
        metadata = self.upload_metadata.get(upload_id)
        if not metadata:
            return False

        # Delete file and directory
        path = Path(metadata["path"])
        if path.exists():
            path.unlink()

        upload_dir = self.upload_dir / upload_id
        if upload_dir.exists():
            try:
                upload_dir.rmdir()
            except OSError:
                # Directory not empty, ignore
                pass

        # Remove metadata
        del self.upload_metadata[upload_id]
        return True

    def cleanup_old_uploads(self, max_age_hours: int = 24):
        """Clean up uploads older than specified age.

        Args:
            max_age_hours: Maximum age in hours
        """
        import time

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for upload_id in list(self.upload_metadata.keys()):
            path = Path(self.upload_metadata[upload_id]["path"])
            if path.exists():
                file_age = current_time - path.stat().st_mtime
                if file_age > max_age_seconds:
                    logger.info(f"Cleaning up old upload: {upload_id}")
                    self.delete_upload(upload_id)


# Singleton upload manager
_upload_manager: Optional[UploadManager] = None


def get_upload_manager() -> UploadManager:
    """Get upload manager singleton."""
    global _upload_manager
    if _upload_manager is None:
        _upload_manager = UploadManager()
    return _upload_manager
