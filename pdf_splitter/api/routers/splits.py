"""
Split Management Endpoints

Handles split proposal management, modifications, and execution.
"""
import io
import os
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from pdf_splitter.api.config import config
from pdf_splitter.api.models.responses import (
    APIResponse,
    DocumentSegmentResponse,
    PreviewResponse,
    SplitExecuteResponse,
    SplitProposalResponse,
    SplitResultResponse,
)
from pdf_splitter.api.services.session_service import SessionService
from pdf_splitter.api.services.split_service import SplitService
from pdf_splitter.api.utils.exceptions import ProcessingError, SessionNotFoundError
from pdf_splitter.splitting.models import DocumentSegment

router = APIRouter(prefix="/api/splits", tags=["splits"])


class ProposalUpdateRequest(BaseModel):
    """Request model for updating split proposal."""

    segments: Optional[List[Dict[str, Any]]] = Field(
        None, description="Full segment replacement"
    )
    merge_segments: Optional[List[str]] = Field(
        None, description="Segment IDs to merge"
    )
    split_at_page: Optional[Dict[str, Any]] = Field(
        None, description="Split segment at page"
    )
    update_segment: Optional[Dict[str, Any]] = Field(
        None, description="Update segment metadata"
    )


class SegmentMergeRequest(BaseModel):
    """Request model for merging segments."""

    segment_ids: List[str] = Field(..., min_items=2, description="Segment IDs to merge")


class SegmentSplitRequest(BaseModel):
    """Request model for splitting a segment."""

    segment_id: str = Field(..., description="Segment ID to split")
    page_number: int = Field(..., description="Page number to split at")


class SegmentUpdateRequest(BaseModel):
    """Request model for updating segment metadata."""

    document_type: Optional[str] = Field(None, description="New document type")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata updates")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score"
    )


def get_split_service() -> SplitService:
    """Dependency to get split service instance."""
    return SplitService()


@router.get("/{session_id}/proposal", response_model=SplitProposalResponse)
async def get_split_proposal(
    session_id: str, split_service: SplitService = Depends(get_split_service)
) -> SplitProposalResponse:
    """
    Get the current split proposal for a session.

    Returns the proposed document segments with boundaries and metadata.

    Args:
        session_id: Session ID

    Returns:
        SplitProposalResponse with segments

    Raises:
        404: Session or proposal not found
    """
    try:
        proposal = split_service.get_proposal(session_id)
        if not proposal:
            raise HTTPException(404, "No proposal found for session")

        # Convert to response model
        segments = []
        for segment in proposal.segments:
            segments.append(
                DocumentSegmentResponse(
                    segment_id=segment.id,
                    start_page=segment.start_page,
                    end_page=segment.end_page,
                    page_count=segment.end_page - segment.start_page + 1,
                    document_type=segment.document_type,
                    suggested_filename=segment.suggested_filename
                    or f"document_{segment.id}.pdf",
                    confidence=segment.confidence,
                    summary=segment.metadata.get("summary"),
                    metadata=segment.metadata,
                    is_user_defined=segment.metadata.get("user_defined", False),
                    preview_available=True,
                )
            )

        return SplitProposalResponse(
            success=True,
            session_id=session_id,
            pdf_path=proposal.pdf_path,
            total_pages=proposal.total_pages,
            segments=segments,
            created_at=proposal.created_at,
            modified_at=proposal.modified_at,
            total_segments=len(segments),
        )

    except SessionNotFoundError as e:
        raise e


@router.put("/{session_id}/proposal", response_model=SplitProposalResponse)
async def update_split_proposal(
    session_id: str,
    request: ProposalUpdateRequest,
    split_service: SplitService = Depends(get_split_service),
) -> SplitProposalResponse:
    """
    Update the split proposal with modifications.

    Supports various update operations:
    - Full segment replacement
    - Merging segments
    - Splitting segments
    - Updating segment metadata

    Args:
        session_id: Session ID
        request: Update request

    Returns:
        Updated split proposal

    Raises:
        404: Session not found
        422: Invalid update request
    """
    try:
        # Build updates dictionary
        updates = {}

        if request.segments is not None:
            # Convert segment dicts to DocumentSegment objects
            segments = []
            for seg_data in request.segments:
                segment = DocumentSegment(
                    start_page=seg_data["start_page"],
                    end_page=seg_data["end_page"],
                    document_type=seg_data.get("document_type", "unknown"),
                    confidence=seg_data.get("confidence", 0.5),
                    metadata=seg_data.get("metadata", {}),
                )
                segments.append(segment)
            updates["segments"] = segments

        if request.merge_segments:
            updates["merge_segments"] = request.merge_segments

        if request.split_at_page:
            updates["split_at_page"] = request.split_at_page

        if request.update_segment:
            updates["update_segment"] = request.update_segment

        # Apply updates
        proposal = split_service.update_proposal(session_id, updates)

        # Return updated proposal
        return await get_split_proposal(session_id, split_service)

    except (SessionNotFoundError, ProcessingError) as e:
        raise e


@router.post("/{session_id}/merge", response_model=SplitProposalResponse)
async def merge_segments(
    session_id: str,
    request: SegmentMergeRequest,
    split_service: SplitService = Depends(get_split_service),
) -> SplitProposalResponse:
    """
    Merge multiple segments into one.

    Args:
        session_id: Session ID
        request: Merge request with segment IDs

    Returns:
        Updated split proposal

    Raises:
        404: Session not found
        422: Invalid merge request
    """
    try:
        updates = {"merge_segments": request.segment_ids}
        proposal = split_service.update_proposal(session_id, updates)
        return await get_split_proposal(session_id, split_service)

    except (SessionNotFoundError, ProcessingError) as e:
        raise e


@router.post("/{session_id}/split", response_model=SplitProposalResponse)
async def split_segment(
    session_id: str,
    request: SegmentSplitRequest,
    split_service: SplitService = Depends(get_split_service),
) -> SplitProposalResponse:
    """
    Split a segment at the specified page.

    Args:
        session_id: Session ID
        request: Split request with segment ID and page number

    Returns:
        Updated split proposal

    Raises:
        404: Session not found
        422: Invalid split request
    """
    try:
        updates = {
            "split_at_page": {
                "segment_id": request.segment_id,
                "page_number": request.page_number,
            }
        }
        proposal = split_service.update_proposal(session_id, updates)
        return await get_split_proposal(session_id, split_service)

    except (SessionNotFoundError, ProcessingError) as e:
        raise e


@router.patch("/{session_id}/segments/{segment_id}", response_model=APIResponse)
async def update_segment(
    session_id: str,
    segment_id: str,
    request: SegmentUpdateRequest,
    split_service: SplitService = Depends(get_split_service),
) -> APIResponse:
    """
    Update segment metadata.

    Args:
        session_id: Session ID
        segment_id: Segment ID to update
        request: Update request

    Returns:
        Success confirmation

    Raises:
        404: Session or segment not found
    """
    try:
        changes = {}
        if request.document_type is not None:
            changes["document_type"] = request.document_type
        if request.metadata is not None:
            changes["metadata"] = request.metadata
        if request.confidence is not None:
            changes["confidence"] = request.confidence

        updates = {"update_segment": {"segment_id": segment_id, "changes": changes}}

        split_service.update_proposal(session_id, updates)

        return APIResponse(
            success=True, message=f"Segment {segment_id} updated successfully"
        )

    except (SessionNotFoundError, ProcessingError) as e:
        raise e


@router.get("/{session_id}/preview/{segment_id}", response_model=PreviewResponse)
async def get_segment_preview(
    session_id: str,
    segment_id: str,
    max_pages: int = 3,
    split_service: SplitService = Depends(get_split_service),
) -> PreviewResponse:
    """
    Get preview images for a document segment.

    Args:
        session_id: Session ID
        segment_id: Segment ID
        max_pages: Maximum pages to preview (1-5)

    Returns:
        Preview images as base64 encoded strings

    Raises:
        404: Session or segment not found
    """
    try:
        # Limit max pages
        max_pages = min(max(1, max_pages), 5)

        # Generate previews
        preview_images = await split_service.generate_preview(
            session_id, segment_id, max_pages
        )

        return PreviewResponse(
            success=True,
            session_id=session_id,
            segment_id=segment_id,
            preview_type="image",
            pages_included=len(preview_images),
            images=preview_images,
            metadata={
                "max_pages_requested": max_pages,
                "format": "png",
                "encoding": "base64",
            },
        )

    except (SessionNotFoundError, ProcessingError) as e:
        raise e


@router.post("/{session_id}/execute", response_model=SplitExecuteResponse)
async def execute_split(
    session_id: str,
    background_tasks: BackgroundTasks,
    split_service: SplitService = Depends(get_split_service),
) -> SplitExecuteResponse:
    """
    Execute the split operation for the current proposal.

    This starts a background task to split the PDF according to the
    current proposal. Progress updates are sent via WebSocket.

    Args:
        session_id: Session ID

    Returns:
        Split execution response with split ID

    Raises:
        404: Session not found
        422: No proposal found
    """
    try:
        # Start split execution
        split_id = await split_service.execute_split(session_id)

        return SplitExecuteResponse(
            success=True,
            session_id=session_id,
            split_id=split_id,
            status="started",
            estimated_time=None,  # Could calculate based on file size
            message="Split operation started successfully",
        )

    except (SessionNotFoundError, ProcessingError) as e:
        raise e


@router.get("/{session_id}/results", response_model=SplitResultResponse)
async def get_split_results(
    session_id: str, session_service: SessionService = Depends(SessionService)
) -> SplitResultResponse:
    """
    Get results of a completed split operation.

    Args:
        session_id: Session ID

    Returns:
        Split results with file information

    Raises:
        404: Session not found
        422: Split not complete
    """
    try:
        # Get session details
        details = session_service.get_session_details(session_id)

        if details["status"] != "complete":
            raise ProcessingError("Split operation not complete")

        metadata = details["metadata"]

        # Build output files list
        output_files = []
        if "output_files" in metadata:
            for file_path in metadata["output_files"]:
                if os.path.exists(file_path):
                    file_stat = os.stat(file_path)
                    output_files.append(
                        {
                            "filename": os.path.basename(file_path),
                            "path": file_path,
                            "size": file_stat.st_size,
                            "created_at": datetime.fromtimestamp(
                                file_stat.st_ctime
                            ).isoformat(),
                        }
                    )

        return SplitResultResponse(
            success=True,
            session_id=session_id,
            split_id=metadata.get("split_id", ""),
            status="completed",
            files_created=len(output_files),
            output_files=output_files,
            zip_file=None,  # Could generate on demand
            manifest_file=None,
            processing_time=metadata.get("processing_time", 0),
            output_size_bytes=metadata.get("output_size", 0),
        )

    except SessionNotFoundError as e:
        raise e


@router.get("/{session_id}/download/{filename}")
async def download_file(
    session_id: str,
    filename: str,
    session_service: SessionService = Depends(SessionService),
):
    """
    Download a specific split output file.

    Args:
        session_id: Session ID
        filename: Filename to download

    Returns:
        File download response

    Raises:
        404: File not found
    """
    # Validate session
    try:
        details = session_service.get_session_details(session_id)
    except SessionNotFoundError:
        raise HTTPException(404, "Session not found")

    # Build file path
    file_path = config.output_dir / session_id / filename

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    # Security check - ensure file is within output directory
    try:
        file_path.resolve().relative_to(config.output_dir.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")

    return FileResponse(
        path=str(file_path), filename=filename, media_type="application/pdf"
    )


@router.get("/{session_id}/download/zip")
async def download_all_as_zip(
    session_id: str, session_service: SessionService = Depends(SessionService)
):
    """
    Download all split files as a ZIP archive.

    Args:
        session_id: Session ID

    Returns:
        ZIP file stream

    Raises:
        404: Session not found
        422: No files to download
    """
    try:
        # Get session details
        details = session_service.get_session_details(session_id)

        if details["status"] != "complete":
            raise ProcessingError("Split operation not complete")

        # Get output files
        output_dir = config.output_dir / session_id
        if not output_dir.exists():
            raise HTTPException(404, "Output directory not found")

        pdf_files = list(output_dir.glob("*.pdf"))
        if not pdf_files:
            raise ProcessingError("No PDF files found")

        # Create ZIP in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for pdf_file in pdf_files:
                zip_file.write(pdf_file, arcname=pdf_file.name)

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=split_results_{session_id}.zip"
            },
        )

    except SessionNotFoundError as e:
        raise e
