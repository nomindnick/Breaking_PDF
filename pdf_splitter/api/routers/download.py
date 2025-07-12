"""
Download API Endpoints

Handles secure file downloads with streaming and progress tracking.
"""
from datetime import timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from pdf_splitter.api.models.responses import APIResponse
from pdf_splitter.api.models.results import DownloadProgress, DownloadToken
from pdf_splitter.api.services.download_service import DownloadService
from pdf_splitter.api.utils.exceptions import (
    FileNotFoundError,
    SecurityError,
    SessionNotFoundError,
)

router = APIRouter(prefix="/api/download", tags=["download"])


def get_download_service() -> DownloadService:
    """Dependency to get download service."""
    return DownloadService()


def get_client_info(request: Request) -> dict:
    """Extract client information from request."""
    return {
        "ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
    }


@router.get("/{session_id}/{filename}")
async def download_file(
    session_id: str,
    filename: str,
    token: Optional[str] = Query(None, description="Download token"),
    download_service: DownloadService = Depends(get_download_service),
) -> StreamingResponse:
    """
    Download a specific file.

    Streams the file with progress tracking. Optionally requires a
    download token for enhanced security.

    Args:
        session_id: Session ID
        filename: Filename to download
        token: Optional download token

    Returns:
        Streaming file response

    Raises:
        404: File not found
        401: Unauthorized (invalid token)
    """
    try:
        return await download_service.stream_file(
            session_id, filename, token, track_progress=True
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except SecurityError as e:
        raise HTTPException(401, str(e))


@router.get("/{session_id}/zip")
async def download_zip(
    session_id: str,
    files: Optional[List[str]] = Query(None, description="Specific files to include"),
    token: Optional[str] = Query(None, description="Download token"),
    download_service: DownloadService = Depends(get_download_service),
) -> StreamingResponse:
    """
    Download multiple files as a ZIP archive.

    Streams a ZIP file containing all session files (or specified subset).
    Efficient streaming prevents memory issues with large files.

    Args:
        session_id: Session ID
        files: Optional list of filenames to include
        token: Optional download token

    Returns:
        Streaming ZIP response

    Raises:
        404: Session not found
        401: Unauthorized (invalid token)
    """
    try:
        return await download_service.stream_zip(
            session_id, files, token, track_progress=True
        )
    except SessionNotFoundError as e:
        raise HTTPException(404, str(e))
    except SecurityError as e:
        raise HTTPException(401, str(e))


@router.post("/token/{session_id}", response_model=DownloadToken)
async def create_download_token(
    session_id: str,
    allowed_files: Optional[List[str]] = None,
    expires_in_hours: int = Query(
        24, ge=1, le=168, description="Token expiry in hours"
    ),
    download_service: DownloadService = Depends(get_download_service),
) -> DownloadToken:
    """
    Create a secure download token.

    Tokens can be used to share download links securely. They can be
    restricted to specific files and have configurable expiry.

    Args:
        session_id: Session ID
        allowed_files: Specific files allowed (None = all)
        expires_in_hours: Token expiry in hours

    Returns:
        Download token

    Raises:
        404: Session not found
    """
    try:
        return download_service.create_download_token(
            session_id, allowed_files, timedelta(hours=expires_in_hours)
        )
    except SessionNotFoundError:
        raise HTTPException(404, f"Session {session_id} not found")


@router.post("/link/{session_id}/{filename}", response_model=dict)
async def create_download_link(
    session_id: str,
    filename: str,
    expires_in_hours: int = Query(1, ge=1, le=24, description="Link expiry in hours"),
    download_service: DownloadService = Depends(get_download_service),
) -> dict:
    """
    Create a temporary download link for a file.

    Generates a secure, time-limited download link that can be shared.

    Args:
        session_id: Session ID
        filename: Filename
        expires_in_hours: Link expiry in hours

    Returns:
        Download link information

    Raises:
        404: File not found
    """
    try:
        return await download_service.create_download_link(
            session_id, filename, timedelta(hours=expires_in_hours)
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.get("/progress/active", response_model=List[DownloadProgress])
async def get_active_downloads(
    session_id: Optional[str] = Query(None, description="Filter by session"),
    download_service: DownloadService = Depends(get_download_service),
) -> List[DownloadProgress]:
    """
    Get list of active downloads with progress.

    Returns real-time information about ongoing downloads including
    progress, speed, and estimated completion time.

    Args:
        session_id: Optional session filter

    Returns:
        List of active downloads
    """
    downloads = download_service.get_active_downloads()

    if session_id:
        downloads = [d for d in downloads if d.session_id == session_id]

    return downloads


@router.delete("/progress/{download_id}", response_model=APIResponse)
async def cancel_download(
    download_id: str, download_service: DownloadService = Depends(get_download_service)
) -> APIResponse:
    """
    Cancel an active download.

    Args:
        download_id: Download ID to cancel

    Returns:
        Cancellation result
    """
    cancelled = download_service.cancel_download(download_id)

    if cancelled:
        return APIResponse(success=True, message=f"Download {download_id} cancelled")
    else:
        return APIResponse(success=False, message=f"Download {download_id} not found")


@router.post("/validate-token", response_model=dict)
async def validate_token(
    token: str, download_service: DownloadService = Depends(get_download_service)
) -> dict:
    """
    Validate a download token.

    Checks if a token is valid and returns its properties.

    Args:
        token: Token to validate

    Returns:
        Token information

    Raises:
        401: Invalid token
    """
    try:
        download_token = download_service.validate_download_token(token)
        return {
            "valid": True,
            "session_id": download_token.session_id,
            "expires_at": download_token.expires_at.isoformat(),
            "allowed_files": download_token.allowed_files,
            "download_count": download_token.download_count,
            "remaining_downloads": download_token.max_downloads
            - download_token.download_count,
        }
    except SecurityError as e:
        raise HTTPException(401, str(e))


@router.post("/cleanup-tokens", response_model=APIResponse)
async def cleanup_expired_tokens(
    download_service: DownloadService = Depends(get_download_service),
) -> APIResponse:
    """
    Clean up expired download tokens.

    Removes tokens that have passed their expiry time.

    Returns:
        Cleanup summary
    """
    cleaned = download_service.cleanup_expired_tokens()

    return APIResponse(success=True, message=f"Cleaned up {cleaned} expired tokens")


# Add download tracking middleware
@router.middleware("http")
async def track_downloads(request: Request, call_next):
    """Middleware to track download metrics."""
    # Only track actual file downloads
    if request.url.path.startswith("/api/download/") and request.method == "GET":
        # Could add metrics tracking here
        pass

    response = await call_next(request)
    return response
