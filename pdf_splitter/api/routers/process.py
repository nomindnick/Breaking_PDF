"""
PDF Processing Endpoints

Handles PDF processing initiation and status tracking.
"""
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from pdf_splitter.api.config import config
from pdf_splitter.api.models.responses import APIResponse
from pdf_splitter.api.services.file_service import FileService
from pdf_splitter.api.services.process_service import ProcessingService
from pdf_splitter.api.services.session_service import SessionService
from pdf_splitter.api.utils.exceptions import (
    FileNotFoundError,
    ProcessingError,
    SessionNotFoundError,
)

router = APIRouter(prefix="/api", tags=["process"])


class ProcessRequest(BaseModel):
    """Request model for starting PDF processing."""

    file_id: str = Field(..., description="ID of the uploaded file to process")
    detector_type: Optional[str] = Field(
        "embeddings", description="Type of detector to use"
    )


class ProcessResponse(APIResponse):
    """Response model for process initiation."""

    session_id: str = Field(..., description="Session ID for tracking processing")
    status: str = Field(..., description="Current processing status")
    message: str = Field(..., description="Status message")


class ProcessStatusResponse(APIResponse):
    """Response model for processing status."""

    session_id: str
    status: str
    stage: Optional[str] = None
    progress: Optional[float] = Field(None, ge=0.0, le=100.0)
    message: Optional[str] = None
    is_active: bool = False
    error: Optional[str] = None


def get_process_service() -> ProcessingService:
    """Dependency to get process service instance."""
    return ProcessingService()


def get_session_service() -> SessionService:
    """Dependency to get session service instance."""
    return SessionService()


@router.post("/process", response_model=ProcessResponse)
async def start_processing(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    process_service: ProcessingService = Depends(get_process_service),
    file_service: FileService = Depends(FileService),
) -> ProcessResponse:
    """
    Start PDF processing for an uploaded file.

    This endpoint initiates the complete PDF processing pipeline:
    1. Load and validate the PDF
    2. Extract text from all pages
    3. Detect document boundaries
    4. Generate split proposal

    The processing happens asynchronously in the background.

    Args:
        request: Processing request with file ID
        background_tasks: FastAPI background tasks

    Returns:
        ProcessResponse with session ID for tracking

    Raises:
        404: File not found
        422: Processing error
    """
    try:
        # Verify file exists
        await file_service.get_file_path(request.file_id)

        # Start processing and get session ID
        session_id = await process_service.start_processing(
            file_id=request.file_id,
            progress_callback=None,  # WebSocket will handle real-time updates
        )

        return ProcessResponse(
            success=True,
            session_id=session_id,
            status="processing",
            message="PDF processing started successfully",
        )

    except FileNotFoundError as e:
        raise e
    except Exception as e:
        raise ProcessingError(f"Failed to start processing: {str(e)}")


@router.get("/process/{session_id}/status", response_model=ProcessStatusResponse)
async def get_processing_status(
    session_id: str, process_service: ProcessingService = Depends(get_process_service)
) -> ProcessStatusResponse:
    """
    Get the current status of PDF processing.

    Args:
        session_id: Session ID to check

    Returns:
        ProcessStatusResponse with current status and progress

    Raises:
        404: Session not found
    """
    try:
        status_data = await process_service.get_processing_status(session_id)

        return ProcessStatusResponse(
            success=True,
            session_id=session_id,
            status=status_data["status"],
            stage=status_data.get("stage"),
            progress=status_data.get("progress"),
            message=status_data.get("message"),
            is_active=status_data.get("is_active", False),
            error=status_data.get("error"),
        )

    except SessionNotFoundError as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving processing status: {str(e)}"
        )


@router.post("/process/{session_id}/cancel")
async def cancel_processing(
    session_id: str, process_service: ProcessingService = Depends(get_process_service)
) -> APIResponse:
    """
    Cancel active PDF processing.

    Args:
        session_id: Session ID to cancel

    Returns:
        Success confirmation

    Raises:
        404: Session not found
    """
    try:
        cancelled = await process_service.cancel_processing(session_id)

        if cancelled:
            return APIResponse(
                success=True,
                message=f"Processing for session {session_id} has been cancelled",
            )
        else:
            return APIResponse(
                success=True, message=f"Session {session_id} is not actively processing"
            )

    except SessionNotFoundError as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error cancelling processing: {str(e)}"
        )
