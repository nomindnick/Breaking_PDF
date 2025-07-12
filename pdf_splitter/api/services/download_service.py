"""
Download Service

Handles secure file downloads with streaming, progress tracking, and validation.
"""
import asyncio
import io
import os
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Dict, List, Optional

import aiofiles
import jwt
from fastapi import HTTPException, status
from starlette.responses import StreamingResponse

from pdf_splitter.api.config import config
from pdf_splitter.api.models.results import (
    DownloadProgress,
    DownloadToken,
    FileType,
    OutputFileInfo,
)
from pdf_splitter.api.models.websocket import ProcessingStage
from pdf_splitter.api.services.results_service import ResultsService
from pdf_splitter.api.services.websocket_enhanced import enhanced_websocket_manager
from pdf_splitter.api.utils.exceptions import (
    FileNotFoundError,
    SecurityError,
    SessionNotFoundError,
)


class DownloadService:
    """Service for secure file downloads with streaming support."""

    def __init__(self):
        self.results_service = ResultsService()
        self.active_downloads: Dict[str, DownloadProgress] = {}
        self.download_tokens: Dict[str, DownloadToken] = {}

        # Security settings
        self.secret_key = config.secret_key
        self.token_expiry = timedelta(hours=24)
        self.chunk_size = 64 * 1024  # 64KB chunks
        self.max_concurrent_downloads = 10

    def create_download_token(
        self,
        session_id: str,
        allowed_files: Optional[List[str]] = None,
        expires_in: Optional[timedelta] = None,
    ) -> DownloadToken:
        """
        Create a secure download token.

        Args:
            session_id: Session ID
            allowed_files: Specific files allowed (None = all files)
            expires_in: Token expiry duration

        Returns:
            Download token
        """
        # Validate session exists
        results = self.results_service.get_session_results(session_id)

        # Generate token
        expires_at = datetime.utcnow() + (expires_in or self.token_expiry)

        token_data = {
            "session_id": session_id,
            "exp": expires_at,
            "iat": datetime.utcnow(),
            "type": "download",
        }

        if allowed_files:
            token_data["files"] = allowed_files

        token_str = jwt.encode(token_data, self.secret_key, algorithm="HS256")

        # Create token object
        download_token = DownloadToken(
            token=token_str,
            session_id=session_id,
            expires_at=expires_at,
            allowed_files=allowed_files or [],
            max_downloads=100,  # Configurable
        )

        # Store token
        self.download_tokens[token_str] = download_token

        return download_token

    def validate_download_token(self, token: str) -> DownloadToken:
        """
        Validate a download token.

        Args:
            token: Token string

        Returns:
            Valid download token

        Raises:
            SecurityError: If token is invalid
        """
        try:
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])

            # Check if token is in active tokens
            if token not in self.download_tokens:
                raise SecurityError("Token not found")

            download_token = self.download_tokens[token]

            # Check expiry
            if datetime.utcnow() > download_token.expires_at:
                del self.download_tokens[token]
                raise SecurityError("Token expired")

            # Check download limit
            if download_token.download_count >= download_token.max_downloads:
                raise SecurityError("Download limit exceeded")

            return download_token

        except jwt.ExpiredSignatureError:
            raise SecurityError("Token expired")
        except jwt.InvalidTokenError:
            raise SecurityError("Invalid token")

    def validate_file_access(
        self, session_id: str, filename: str, token: Optional[str] = None
    ) -> OutputFileInfo:
        """
        Validate file access permissions.

        Args:
            session_id: Session ID
            filename: Filename to access
            token: Optional download token

        Returns:
            File information if valid

        Raises:
            SecurityError: If access denied
            FileNotFoundError: If file not found
        """
        # Get file info
        file_info = self.results_service.get_file_info(session_id, filename)

        # Validate file path is within output directory
        file_path = Path(file_info.path).resolve()
        output_dir = config.output_dir.resolve()

        try:
            file_path.relative_to(output_dir)
        except ValueError:
            raise SecurityError("Invalid file path")

        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        # Validate token if provided
        if token:
            download_token = self.validate_download_token(token)

            # Check session matches
            if download_token.session_id != session_id:
                raise SecurityError("Token session mismatch")

            # Check file is allowed
            if (
                download_token.allowed_files
                and filename not in download_token.allowed_files
            ):
                raise SecurityError("File not allowed by token")

            # Increment download count
            download_token.download_count += 1

        return file_info

    async def stream_file(
        self,
        session_id: str,
        filename: str,
        token: Optional[str] = None,
        track_progress: bool = True,
    ) -> StreamingResponse:
        """
        Stream a file for download.

        Args:
            session_id: Session ID
            filename: Filename
            token: Optional download token
            track_progress: Track download progress

        Returns:
            Streaming response
        """
        # Validate access
        file_info = self.validate_file_access(session_id, filename, token)

        # Create download progress if tracking
        download_id = None
        if track_progress:
            download_id = f"dl_{session_id}_{int(time.time())}"
            progress = DownloadProgress(
                session_id=session_id,
                download_id=download_id,
                filename=filename,
                total_bytes=file_info.size,
                bytes_sent=0,
                progress=0,
                started_at=datetime.utcnow(),
            )
            self.active_downloads[download_id] = progress

        # Stream file
        async def file_streamer():
            bytes_sent = 0
            start_time = time.time()

            try:
                async with aiofiles.open(file_info.path, "rb") as f:
                    while chunk := await f.read(self.chunk_size):
                        yield chunk
                        bytes_sent += len(chunk)

                        # Update progress
                        if download_id and download_id in self.active_downloads:
                            progress = self.active_downloads[download_id]
                            progress.update_progress(bytes_sent)

                            # Send WebSocket update periodically
                            if bytes_sent % (1024 * 1024) == 0:  # Every MB
                                await self._send_download_progress(progress)

                # Record successful download
                download_time = time.time() - start_time
                await self._record_download(session_id, filename, download_time, True)

            except Exception as e:
                # Record failed download
                download_time = time.time() - start_time
                await self._record_download(
                    session_id, filename, download_time, False, str(e)
                )
                raise

            finally:
                # Clean up progress tracking
                if download_id:
                    self.active_downloads.pop(download_id, None)

        # Determine content type
        content_type = self._get_content_type(file_info.file_type)

        return StreamingResponse(
            file_streamer(),
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(file_info.size),
                "Cache-Control": "no-cache",
                "X-Session-ID": session_id,
                "X-File-Checksum": file_info.checksum or "",
            },
        )

    async def stream_zip(
        self,
        session_id: str,
        filenames: Optional[List[str]] = None,
        token: Optional[str] = None,
        track_progress: bool = True,
    ) -> StreamingResponse:
        """
        Stream multiple files as a ZIP archive.

        Args:
            session_id: Session ID
            filenames: Specific files to include (None = all)
            token: Optional download token
            track_progress: Track download progress

        Returns:
            Streaming ZIP response
        """
        # Get session results
        results = self.results_service.get_session_results(session_id)

        # Filter files
        files_to_zip = results.output_files
        if filenames:
            files_to_zip = [f for f in files_to_zip if f.filename in filenames]

        # Validate all files
        for file_info in files_to_zip:
            self.validate_file_access(session_id, file_info.filename, token)

        # Calculate total size (approximate)
        total_size = sum(f.size for f in files_to_zip)

        # Create download progress if tracking
        download_id = None
        if track_progress:
            download_id = f"zip_{session_id}_{int(time.time())}"
            progress = DownloadProgress(
                session_id=session_id,
                download_id=download_id,
                filename=f"session_{session_id}.zip",
                total_bytes=total_size,
                bytes_sent=0,
                progress=0,
                started_at=datetime.utcnow(),
            )
            self.active_downloads[download_id] = progress

        # Stream ZIP
        async def zip_streamer():
            bytes_sent = 0
            buffer = io.BytesIO()

            try:
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for file_info in files_to_zip:
                        # Add file to ZIP
                        zf.write(file_info.path, file_info.filename)

                        # Update progress
                        bytes_sent += file_info.size
                        if download_id:
                            progress = self.active_downloads[download_id]
                            progress.update_progress(bytes_sent)
                            await self._send_download_progress(progress)

                # Yield ZIP content
                buffer.seek(0)
                while chunk := buffer.read(self.chunk_size):
                    yield chunk

            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error creating ZIP: {str(e)}",
                )

            finally:
                if download_id:
                    self.active_downloads.pop(download_id, None)

        return StreamingResponse(
            zip_streamer(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="session_{session_id}.zip"',
                "Cache-Control": "no-cache",
                "X-Session-ID": session_id,
            },
        )

    async def create_download_link(
        self, session_id: str, filename: str, expires_in: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Create a temporary download link.

        Args:
            session_id: Session ID
            filename: Filename
            expires_in: Link expiry duration

        Returns:
            Download link information
        """
        # Create token for specific file
        token = self.create_download_token(
            session_id,
            allowed_files=[filename],
            expires_in=expires_in or timedelta(hours=1),
        )

        # Build download URL
        base_url = config.api_url or "http://localhost:8000"
        download_url = (
            f"{base_url}/api/download/{session_id}/{filename}?token={token.token}"
        )

        return {
            "url": download_url,
            "token": token.token,
            "expires_at": token.expires_at.isoformat(),
            "filename": filename,
            "session_id": session_id,
        }

    def get_active_downloads(self) -> List[DownloadProgress]:
        """Get list of active downloads."""
        return list(self.active_downloads.values())

    def cancel_download(self, download_id: str) -> bool:
        """
        Cancel an active download.

        Args:
            download_id: Download ID

        Returns:
            True if cancelled, False if not found
        """
        if download_id in self.active_downloads:
            del self.active_downloads[download_id]
            return True
        return False

    async def _send_download_progress(self, progress: DownloadProgress):
        """Send download progress via WebSocket."""
        await enhanced_websocket_manager.send_progress_update(
            session_id=progress.session_id,
            stage=ProcessingStage.COMPLETE,
            progress=progress.progress,
            message=f"Downloading {progress.filename}",
            details={
                "download_id": progress.download_id,
                "bytes_sent": progress.bytes_sent,
                "total_bytes": progress.total_bytes,
                "speed_mbps": progress.speed_mbps,
                "eta_seconds": progress.estimated_time_remaining,
            },
        )

    async def _record_download(
        self,
        session_id: str,
        filename: str,
        download_time: float,
        completed: bool,
        error: Optional[str] = None,
    ):
        """Record download in history."""
        # Get client info from context (would be set by middleware)
        client_info = {"ip": "unknown", "user_agent": "unknown"}

        self.results_service.record_download(
            session_id, filename, client_info, download_time, completed, error
        )

    def _get_content_type(self, file_type: FileType) -> str:
        """Get content type for file type."""
        content_types = {
            FileType.PDF: "application/pdf",
            FileType.ZIP: "application/zip",
            FileType.MANIFEST: "application/json",
            FileType.LOG: "text/plain",
            FileType.PREVIEW: "image/png",
        }
        return content_types.get(file_type, "application/octet-stream")

    def cleanup_expired_tokens(self):
        """Clean up expired download tokens."""
        now = datetime.utcnow()
        expired_tokens = [
            token
            for token, data in self.download_tokens.items()
            if data.expires_at < now
        ]

        for token in expired_tokens:
            del self.download_tokens[token]

        return len(expired_tokens)
