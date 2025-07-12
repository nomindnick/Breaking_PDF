"""
Results API Endpoints

Provides comprehensive results viewing, filtering, and analytics.
"""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from pdf_splitter.api.models.responses import APIResponse
from pdf_splitter.api.models.results import (
    DownloadManifest,
    FilePreview,
    ResultsFilter,
    ResultsPage,
    SplitResultDetailed,
)
from pdf_splitter.api.services.download_service import DownloadService
from pdf_splitter.api.services.results_service import ResultsService
from pdf_splitter.api.utils.exceptions import (
    FileNotFoundError,
    ProcessingError,
    SessionNotFoundError,
)

router = APIRouter(prefix="/api/results", tags=["results"])


def get_results_service() -> ResultsService:
    """Dependency to get results service."""
    return ResultsService()


def get_download_service() -> DownloadService:
    """Dependency to get download service."""
    return DownloadService()


@router.get("/{session_id}", response_model=SplitResultDetailed)
async def get_session_results(
    session_id: str,
    include_files: bool = Query(True, description="Include file information"),
    results_service: ResultsService = Depends(get_results_service),
) -> SplitResultDetailed:
    """
    Get detailed results for a session.

    Returns comprehensive information about the split operation including:
    - Processing statistics
    - Output file information
    - Performance metrics
    - Any errors or warnings

    Args:
        session_id: Session ID
        include_files: Include detailed file information

    Returns:
        Detailed split results

    Raises:
        404: Session not found
        422: Results not available
    """
    try:
        return results_service.get_session_results(session_id, include_files)
    except SessionNotFoundError:
        raise HTTPException(404, f"Session {session_id} not found")
    except ProcessingError as e:
        raise HTTPException(422, str(e))


@router.post("/search", response_model=ResultsPage)
async def search_results(
    filter_criteria: ResultsFilter,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    results_service: ResultsService = Depends(get_results_service),
) -> ResultsPage:
    """
    Search and filter results with pagination.

    Allows filtering by:
    - Session IDs
    - Status
    - Date range
    - File count
    - Document types

    Args:
        filter_criteria: Filter criteria
        page: Page number (1-based)
        page_size: Results per page

    Returns:
        Paginated results
    """
    return results_service.search_results(filter_criteria, page, page_size)


@router.get("/{session_id}/files/{filename}", response_model=dict)
async def get_file_info(
    session_id: str,
    filename: str,
    results_service: ResultsService = Depends(get_results_service),
) -> dict:
    """
    Get detailed information about a specific output file.

    Args:
        session_id: Session ID
        filename: Filename

    Returns:
        File information including size, checksum, metadata

    Raises:
        404: File not found
    """
    try:
        file_info = results_service.get_file_info(session_id, filename)
        return file_info.dict()
    except FileNotFoundError:
        raise HTTPException(404, f"File {filename} not found")


@router.get("/{session_id}/preview/{filename}", response_model=FilePreview)
async def preview_file(
    session_id: str,
    filename: str,
    preview_type: str = Query(
        "auto", description="Preview type: text, image, metadata, auto"
    ),
    max_pages: int = Query(3, ge=1, le=10, description="Maximum pages to preview"),
    results_service: ResultsService = Depends(get_results_service),
) -> FilePreview:
    """
    Generate a preview of an output file.

    Preview types:
    - text: Extract text content
    - image: Generate thumbnail images
    - metadata: File metadata only
    - auto: Automatically determine best preview

    Args:
        session_id: Session ID
        filename: Filename
        preview_type: Type of preview
        max_pages: Maximum pages to preview

    Returns:
        File preview

    Raises:
        404: File not found
    """
    try:
        return await results_service.generate_file_preview(
            session_id, filename, preview_type, max_pages
        )
    except FileNotFoundError:
        raise HTTPException(404, f"File {filename} not found")


@router.post("/{session_id}/manifest", response_model=DownloadManifest)
async def create_download_manifest(
    session_id: str,
    file_filter: Optional[dict] = None,
    results_service: ResultsService = Depends(get_results_service),
) -> DownloadManifest:
    """
    Create a download manifest for batch downloads.

    The manifest includes all files (or filtered subset) with metadata
    and can be used to generate download links or track downloads.

    Args:
        session_id: Session ID
        file_filter: Optional filter criteria

    Returns:
        Download manifest

    Raises:
        404: Session not found
    """
    try:
        return results_service.create_download_manifest(session_id, file_filter)
    except SessionNotFoundError:
        raise HTTPException(404, f"Session {session_id} not found")


@router.get("/analytics/downloads", response_model=dict)
async def get_download_analytics(
    session_id: Optional[str] = Query(None, description="Filter by session"),
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    results_service: ResultsService = Depends(get_results_service),
) -> dict:
    """
    Get download analytics and statistics.

    Returns analytics including:
    - Total downloads
    - Success rate
    - Popular files
    - Download patterns

    Args:
        session_id: Optional session filter
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Analytics data
    """
    time_range = None
    if start_date and end_date:
        time_range = (start_date, end_date)

    return results_service.get_download_analytics(session_id, time_range)


@router.post("/cleanup", response_model=APIResponse)
async def cleanup_old_results(
    days_to_keep: int = Query(7, ge=1, description="Days to keep results"),
    dry_run: bool = Query(True, description="Preview without deleting"),
    results_service: ResultsService = Depends(get_results_service),
) -> APIResponse:
    """
    Clean up old results and files.

    Removes results and files older than specified days.
    Use dry_run=true to preview what would be deleted.

    Args:
        days_to_keep: Number of days to keep
        dry_run: Preview mode

    Returns:
        Cleanup summary
    """
    if dry_run:
        # TODO: Implement dry run logic
        return APIResponse(
            success=True,
            message=f"Dry run: Would clean up results older than {days_to_keep} days",
        )

    cleaned_count = results_service.cleanup_old_results(days_to_keep)

    return APIResponse(
        success=True,
        message=f"Cleaned up {cleaned_count} old sessions",
        data={"sessions_cleaned": cleaned_count},
    )


@router.get("/stats/summary", response_model=dict)
async def get_results_summary(
    results_service: ResultsService = Depends(get_results_service),
) -> dict:
    """
    Get summary statistics across all results.

    Returns:
        Summary statistics including totals, averages, and trends
    """
    # Get all completed sessions
    filter_criteria = ResultsFilter(status=["complete", "confirmed"])
    all_results = results_service.search_results(filter_criteria, 1, 1000)

    if not all_results.results:
        return {
            "total_sessions": 0,
            "total_files": 0,
            "total_size": 0,
            "average_files_per_session": 0,
            "average_processing_time": 0,
        }

    # Calculate statistics
    total_files = sum(r.files_created for r in all_results.results)
    total_size = sum(r.total_output_size for r in all_results.results)
    total_time = sum(r.processing_time for r in all_results.results)

    return {
        "total_sessions": all_results.total,
        "total_files": total_files,
        "total_size": total_size,
        "total_size_mb": total_size / (1024 * 1024),
        "average_files_per_session": total_files / all_results.total
        if all_results.total > 0
        else 0,
        "average_processing_time": total_time / all_results.total
        if all_results.total > 0
        else 0,
        "average_file_size": total_size / total_files if total_files > 0 else 0,
        "date_range": {
            "earliest": min(r.created_at for r in all_results.results).isoformat(),
            "latest": max(r.created_at for r in all_results.results).isoformat(),
        }
        if all_results.results
        else None,
    }
