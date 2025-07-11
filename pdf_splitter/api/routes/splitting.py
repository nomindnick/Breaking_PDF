"""Split management endpoints.

This module provides API endpoints for managing PDF splits.
"""

from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from pdf_splitter.api.dependencies import (
    get_pdf_config,
    get_session_manager,
    get_upload_directory,
)
from pdf_splitter.api.exceptions import (
    ResourceNotFoundError,
    SessionExpiredError,
    SessionNotFoundError,
    SessionStateError,
    SplitError,
    ValidationError,
)
from pdf_splitter.api.models.requests import (
    PreviewRequest,
    SegmentCreateRequest,
    SegmentUpdateRequest,
    SplitExecuteRequest,
)
from pdf_splitter.api.models.responses import (
    DocumentSegmentResponse,
    PreviewResponse,
    SegmentUpdateResponse,
    SplitExecuteResponse,
    SplitProgressResponse,
    SplitProposalResponse,
    SplitResultResponse,
)
from pdf_splitter.api.services.progress_service import (
    ProcessingStage,
    get_progress_service,
)
from pdf_splitter.api.services.splitting_service import SplittingService
from pdf_splitter.core.config import PDFConfig
from pdf_splitter.core.logging import get_logger
from pdf_splitter.splitting.models import DocumentSegment, UserModification
from pdf_splitter.splitting.session_manager import SplitSessionManager

logger = get_logger(__name__)

router = APIRouter(prefix="/api/splits", tags=["splitting"])

# Service instances
_splitting_service = None


def get_splitting_service(
    config: PDFConfig = Depends(get_pdf_config),
) -> SplittingService:
    """Get splitting service instance."""
    global _splitting_service
    if _splitting_service is None:
        output_dir = get_upload_directory().parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        _splitting_service = SplittingService(config, output_dir)
    return _splitting_service


def _segment_to_response(segment: DocumentSegment) -> DocumentSegmentResponse:
    """Convert DocumentSegment to response model."""
    return DocumentSegmentResponse(
        segment_id=segment.segment_id,
        start_page=segment.start_page,
        end_page=segment.end_page,
        page_count=segment.page_count,
        document_type=segment.document_type,
        suggested_filename=segment.suggested_filename,
        confidence=segment.confidence,
        summary=segment.summary,
        metadata=segment.metadata,
        is_user_defined=segment.is_user_defined,
        preview_available=False,  # Will be updated when preview is generated
    )


@router.get("/{session_id}/proposal", response_model=SplitProposalResponse)
async def get_split_proposal(
    session_id: str,
    session_manager: SplitSessionManager = Depends(get_session_manager),
):
    """Get current split proposal for a session.

    Args:
        session_id: Session identifier
        session_manager: Session manager instance

    Returns:
        Split proposal

    Raises:
        HTTPException: If session not found
    """
    try:
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        # Check if expired
        if session.is_expired:
            raise SessionExpiredError(session_id)

        # Convert segments to responses
        segment_responses = [
            _segment_to_response(segment) for segment in session.proposal.segments
        ]

        return SplitProposalResponse(
            session_id=session_id,
            pdf_path=str(session.proposal.pdf_path),
            total_pages=session.proposal.total_pages,
            segments=segment_responses,
            created_at=session.proposal.created_at,
            modified_at=session.proposal.modified_at,
            total_segments=len(segment_responses),
            message=f"Proposal has {len(segment_responses)} segments",
        )

    except (SessionNotFoundError, SessionExpiredError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to get proposal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get proposal: {str(e)}")


@router.put("/{session_id}/segments/{segment_id}", response_model=SegmentUpdateResponse)
async def update_segment(
    session_id: str,
    segment_id: str,
    request: SegmentUpdateRequest,
    session_manager: SplitSessionManager = Depends(get_session_manager),
):
    """Update a document segment.

    Args:
        session_id: Session identifier
        segment_id: Segment identifier
        request: Update request
        session_manager: Session manager instance

    Returns:
        Updated segment

    Raises:
        HTTPException: If session/segment not found
    """
    try:
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        # Check state
        if session.status not in ["pending", "modified"]:
            raise SessionStateError(session_id, session.status, "pending or modified")

        # Update segment
        update_fields = request.model_dump(exclude_unset=True)
        if not session.proposal.update_segment(segment_id, **update_fields):
            raise ResourceNotFoundError("segment", segment_id)

        # Track modification
        modification = UserModification(
            modification_type="update",
            segment_id=segment_id,
            details=update_fields,
        )
        session.add_modification(modification)

        # Update session
        session_manager.update_session(session)

        # Get updated segment
        segment = session.proposal.get_segment(segment_id)

        return SegmentUpdateResponse(
            session_id=session_id,
            segment_id=segment_id,
            updated_fields=list(update_fields.keys()),
            segment=_segment_to_response(segment),
            message=f"Updated {len(update_fields)} fields",
        )

    except (SessionNotFoundError, SessionStateError, ResourceNotFoundError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to update segment: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update segment: {str(e)}"
        )


@router.post("/{session_id}/segments", response_model=SegmentUpdateResponse)
async def create_segment(
    session_id: str,
    request: SegmentCreateRequest,
    session_manager: SplitSessionManager = Depends(get_session_manager),
):
    """Create a new document segment.

    Args:
        session_id: Session identifier
        request: Segment creation request
        session_manager: Session manager instance

    Returns:
        Created segment

    Raises:
        HTTPException: If session not found or validation fails
    """
    try:
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        # Check state
        if session.status not in ["pending", "modified"]:
            raise SessionStateError(session_id, session.status, "pending or modified")

        # Create segment
        segment = DocumentSegment(
            start_page=request.start_page,
            end_page=request.end_page,
            document_type=request.document_type,
            suggested_filename=request.suggested_filename
            or f"segment_{request.start_page + 1}_{request.end_page + 1}.pdf",
            confidence=1.0,  # User-defined segments have full confidence
            summary=request.summary,
            metadata=request.metadata or {},
            is_user_defined=True,
        )

        # Add to proposal
        session.proposal.add_segment(segment)

        # Track modification
        modification = UserModification(
            modification_type="add",
            segment_id=segment.segment_id,
            details=request.model_dump(),
        )
        session.add_modification(modification)

        # Update session
        session_manager.update_session(session)

        return SegmentUpdateResponse(
            session_id=session_id,
            segment_id=segment.segment_id,
            updated_fields=["created"],
            segment=_segment_to_response(segment),
            message="Segment created successfully",
        )

    except (SessionNotFoundError, SessionStateError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create segment: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create segment: {str(e)}"
        )


@router.delete("/{session_id}/segments/{segment_id}")
async def delete_segment(
    session_id: str,
    segment_id: str,
    session_manager: SplitSessionManager = Depends(get_session_manager),
):
    """Delete a document segment.

    Args:
        session_id: Session identifier
        segment_id: Segment identifier
        session_manager: Session manager instance

    Returns:
        Success message

    Raises:
        HTTPException: If session/segment not found
    """
    try:
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        # Check state
        if session.status not in ["pending", "modified"]:
            raise SessionStateError(session_id, session.status, "pending or modified")

        # Remove segment
        if not session.proposal.remove_segment(segment_id):
            raise ResourceNotFoundError("segment", segment_id)

        # Track modification
        modification = UserModification(
            modification_type="remove",
            segment_id=segment_id,
        )
        session.add_modification(modification)

        # Update session
        session_manager.update_session(session)

        return JSONResponse(
            content={
                "success": True,
                "message": f"Segment {segment_id} deleted successfully",
            }
        )

    except (SessionNotFoundError, SessionStateError, ResourceNotFoundError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to delete segment: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete segment: {str(e)}"
        )


@router.post("/{session_id}/preview/{segment_id}", response_model=PreviewResponse)
async def generate_preview(
    session_id: str,
    segment_id: str,
    request: PreviewRequest = Body(default=PreviewRequest()),
    session_manager: SplitSessionManager = Depends(get_session_manager),
    splitting_service: SplittingService = Depends(get_splitting_service),
):
    """Generate preview for a segment.

    Args:
        session_id: Session identifier
        segment_id: Segment identifier
        request: Preview request parameters
        session_manager: Session manager instance
        splitting_service: Splitting service instance

    Returns:
        Preview data with base64 encoded images

    Raises:
        HTTPException: If session/segment not found
    """
    try:
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        # Generate preview
        preview_data = await splitting_service.generate_preview(
            session,
            segment_id,
            max_pages=request.max_pages,
            resolution=request.resolution,
            format=request.format,
        )

        return PreviewResponse(
            session_id=session_id,
            segment_id=segment_id,
            preview_type=preview_data["preview_type"],
            pages_included=preview_data["pages_included"],
            images=preview_data["images"],
            metadata=preview_data["metadata"],
            message=f"Generated preview with {preview_data['pages_included']} pages",
        )

    except SessionNotFoundError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except SplitError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to generate preview: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate preview: {str(e)}"
        )


@router.post("/{session_id}/execute", response_model=SplitExecuteResponse)
async def execute_split(
    session_id: str,
    request: SplitExecuteRequest = Body(default=SplitExecuteRequest()),
    session_manager: SplitSessionManager = Depends(get_session_manager),
    splitting_service: SplittingService = Depends(get_splitting_service),
    progress_service=Depends(get_progress_service),
):
    """Execute PDF split operation.

    Args:
        session_id: Session identifier
        request: Split execution parameters
        session_manager: Session manager instance
        splitting_service: Splitting service instance

    Returns:
        Split execution response

    Raises:
        HTTPException: If session not found or execution fails
    """
    try:
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        # Create progress callback
        progress_callback = progress_service.create_progress_callback(
            session.session_id, ProcessingStage.SPLITTING
        )

        # Start split
        split_id = await splitting_service.execute_split(
            session,
            output_format=request.output_format,
            compress=request.compress,
            create_zip=request.create_zip,
            preserve_metadata=request.preserve_metadata,
            generate_manifest=request.generate_manifest,
            progress_callback=progress_callback,
        )

        # Estimate time
        estimated_time = len(session.proposal.segments) * 0.5  # 0.5s per segment

        return SplitExecuteResponse(
            session_id=session_id,
            split_id=split_id,
            status="started",
            estimated_time=estimated_time,
            message=f"Started splitting {len(session.proposal.segments)} segments",
        )

    except SessionNotFoundError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except (SessionStateError, SplitError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to execute split: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to execute split: {str(e)}"
        )


@router.get("/{split_id}/progress", response_model=SplitProgressResponse)
async def get_split_progress(
    split_id: str,
    splitting_service: SplittingService = Depends(get_splitting_service),
):
    """Get progress of split operation.

    Args:
        split_id: Split identifier
        splitting_service: Splitting service instance

    Returns:
        Split progress

    Raises:
        HTTPException: If split not found
    """
    try:
        status = splitting_service.get_split_status(split_id)

        return SplitProgressResponse(
            session_id=status["session_id"],
            split_id=split_id,
            status=status["status"],
            progress=status["progress"],
            current_segment=status.get("current_segment"),
            total_segments=status["total_segments"],
            files_created=status["files_created"],
            elapsed_time=status["elapsed_time"],
            message=f"Split {status['status']}",
        )

    except ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to get split progress: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get split progress: {str(e)}"
        )


@router.get("/{split_id}/results", response_model=SplitResultResponse)
async def get_split_results(
    split_id: str,
    splitting_service: SplittingService = Depends(get_splitting_service),
):
    """Get results of completed split.

    Args:
        split_id: Split identifier
        splitting_service: Splitting service instance

    Returns:
        Split results

    Raises:
        HTTPException: If split not found or not completed
    """
    try:
        # Get status first
        status = splitting_service.get_split_status(split_id)

        # Get results
        results = splitting_service.get_split_results(split_id)

        # Build file info
        output_files = []
        total_size = 0
        for file_path in results.get("output_files", []):
            path = Path(file_path)
            if path.exists():
                size = path.stat().st_size
                total_size += size
                output_files.append(
                    {
                        "filename": path.name,
                        "size": size,
                        "path": str(path),
                    }
                )

        return SplitResultResponse(
            session_id=status["session_id"],
            split_id=split_id,
            status="completed",
            files_created=len(output_files),
            output_files=output_files,
            zip_file=results.get("zip_file"),
            manifest_file=results.get("manifest_file"),
            processing_time=status["elapsed_time"],
            output_size_bytes=total_size,
            message=f"Split completed. Created {len(output_files)} files.",
        )

    except ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to get split results: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get split results: {str(e)}"
        )


@router.get("/{split_id}/download/{filename}")
async def download_file(
    split_id: str,
    filename: str,
    splitting_service: SplittingService = Depends(get_splitting_service),
):
    """Download a split output file.

    Args:
        split_id: Split identifier
        filename: Filename to download
        splitting_service: Splitting service instance

    Returns:
        File download

    Raises:
        HTTPException: If file not found
    """
    try:
        # Get results to find file
        results = splitting_service.get_split_results(split_id)

        # Find file
        for file_path in results.get("output_files", []):
            path = Path(file_path)
            if path.name == filename and path.exists():
                return FileResponse(
                    path,
                    media_type="application/pdf",
                    filename=filename,
                )

        # Check for zip file
        if results.get("zip_file"):
            zip_path = Path(results["zip_file"])
            if zip_path.name == filename and zip_path.exists():
                return FileResponse(
                    zip_path,
                    media_type="application/zip",
                    filename=filename,
                )

        # Check for manifest
        if results.get("manifest_file"):
            manifest_path = Path(results["manifest_file"])
            if manifest_path.name == filename and manifest_path.exists():
                return FileResponse(
                    manifest_path,
                    media_type="application/json",
                    filename=filename,
                )

        raise ResourceNotFoundError("file", filename)

    except ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to download file: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to download file: {str(e)}"
        )
