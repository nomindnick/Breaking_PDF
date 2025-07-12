"""Detection endpoints for boundary detection.

This module provides API endpoints for document boundary detection.
"""

from fastapi import APIRouter, Depends, HTTPException

from pdf_splitter.api.dependencies import get_pdf_config, get_session_manager
from pdf_splitter.api.exceptions import (
    DetectionError,
    SessionStateError,
    ValidationError,
)
from pdf_splitter.api.models.requests import DetectionStartRequest
from pdf_splitter.api.models.responses import (
    BoundaryResultResponse,
    DetectionResultsResponse,
    DetectionStartResponse,
    DetectionStatusResponse,
)
from pdf_splitter.api.routes.sessions import get_session_by_upload_id
from pdf_splitter.api.services.detection_service import DetectionService
from pdf_splitter.api.services.process_service import ProcessingStage
from pdf_splitter.api.services.progress_service import get_progress_service
from pdf_splitter.core.config import PDFConfig
from pdf_splitter.core.logging import get_logger
from pdf_splitter.splitting.exceptions import SessionNotFoundError
from pdf_splitter.splitting.session_manager import SplitSessionManager

logger = get_logger(__name__)

router = APIRouter(prefix="/api/detection", tags=["detection"])

# Service instances
_detection_service = None


def get_detection_service(
    config: PDFConfig = Depends(get_pdf_config),
) -> DetectionService:
    """Get detection service instance."""
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService(config)
    return _detection_service


@router.post("/start", response_model=DetectionStartResponse)
async def start_detection(
    request: DetectionStartRequest,
    session_manager: SplitSessionManager = Depends(get_session_manager),
    detection_service: DetectionService = Depends(get_detection_service),
    progress_service=Depends(get_progress_service),
):
    """Start boundary detection for a PDF.

    Args:
        request: Detection start request
        session_manager: Session manager instance
        detection_service: Detection service instance

    Returns:
        Detection start response

    Raises:
        HTTPException: On detection start failure
    """
    try:
        # Get session from upload ID
        session = get_session_by_upload_id(request.upload_id, session_manager)

        if not session:
            raise SessionNotFoundError(request.upload_id)

        # Check session state
        if session.status not in ["pending", "modified"]:
            raise SessionStateError(
                session.session_id,
                session.status,
                "pending or modified",
            )

        # Create progress callback
        progress_callback = progress_service.create_progress_callback(
            session.session_id, ProcessingStage.DETECTING_BOUNDARIES
        )

        # Start detection
        detection_id = await detection_service.start_detection(
            session_id=session.session_id,
            pdf_path=session.proposal.pdf_path,
            detector_type=request.detector_type,
            confidence_threshold=request.confidence_threshold,
            progress_callback=progress_callback,
        )

        # Estimate processing time
        total_pages = session.proposal.total_pages
        estimated_time = total_pages * 0.1  # ~0.1 seconds per page

        return DetectionStartResponse(
            session_id=session.session_id,
            detection_id=detection_id,
            status="started",
            detector_type=request.detector_type,
            estimated_time=estimated_time,
            message=f"Started {request.detector_type} detection for {total_pages} pages",
        )

    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except SessionStateError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to start detection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start detection: {str(e)}",
        )


@router.get("/{detection_id}/status", response_model=DetectionStatusResponse)
async def get_detection_status(
    detection_id: str,
    detection_service: DetectionService = Depends(get_detection_service),
):
    """Get status of a running detection.

    Args:
        detection_id: Detection identifier
        detection_service: Detection service instance

    Returns:
        Detection status

    Raises:
        HTTPException: If detection not found
    """
    try:
        status = detection_service.get_detection_status(detection_id)

        return DetectionStatusResponse(
            session_id=status["session_id"],
            detection_id=detection_id,
            status=status["status"],
            progress=status["progress"],
            current_page=status.get("current_page"),
            total_pages=status["total_pages"],
            elapsed_time=status["elapsed_time"],
            estimated_remaining=status.get("estimated_remaining"),
            message=f"Detection {status['status']}",
        )

    except ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to get detection status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get detection status: {str(e)}",
        )


@router.get("/{detection_id}/results", response_model=DetectionResultsResponse)
async def get_detection_results(
    detection_id: str,
    detection_service: DetectionService = Depends(get_detection_service),
    session_manager: SplitSessionManager = Depends(get_session_manager),
):
    """Get results of completed detection.

    Args:
        detection_id: Detection identifier
        detection_service: Detection service instance
        session_manager: Session manager instance

    Returns:
        Detection results

    Raises:
        HTTPException: If detection not found or not completed
    """
    try:
        # Get detection results
        boundaries, proposal = detection_service.get_detection_results(detection_id)

        # Get session ID from detection status
        status = detection_service.get_detection_status(detection_id)
        session_id = status["session_id"]

        # Update session with new proposal
        session = session_manager.get_session(session_id)
        if session:
            session.proposal = proposal
            session_manager.update_session(session)

        # Convert boundaries to response format
        boundary_responses = [
            BoundaryResultResponse(
                page_number=b.page_number,
                boundary_type=b.boundary_type,
                confidence=b.confidence,
                detector_type=b.detector_type,
                reasoning=b.reasoning,
                is_user_defined=False,
            )
            for b in boundaries
        ]

        return DetectionResultsResponse(
            session_id=session_id,
            detection_id=detection_id,
            total_pages=proposal.total_pages,
            boundaries_found=len(boundaries),
            processing_time=status["elapsed_time"],
            boundaries=boundary_responses,
            segments_proposed=len(proposal.segments),
            message=f"Detection completed. Found {len(boundaries)} boundaries, "
            f"proposed {len(proposal.segments)} segments.",
        )

    except ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except DetectionError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to get detection results: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get detection results: {str(e)}",
        )


@router.post("/{session_id}/rerun")
async def rerun_detection(
    session_id: str,
    detector_type: str = "embeddings",
    confidence_threshold: float = 0.5,
    session_manager: SplitSessionManager = Depends(get_session_manager),
    detection_service: DetectionService = Depends(get_detection_service),
):
    """Rerun detection with different parameters.

    Args:
        session_id: Session identifier
        detector_type: Type of detector to use
        confidence_threshold: Confidence threshold
        session_manager: Session manager instance
        detection_service: Detection service instance

    Returns:
        New detection start response

    Raises:
        HTTPException: On detection start failure
    """
    try:
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        # Start new detection
        detection_id = await detection_service.start_detection(
            session_id=session_id,
            pdf_path=session.proposal.pdf_path,
            detector_type=detector_type,
            confidence_threshold=confidence_threshold,
        )

        # Estimate processing time
        total_pages = session.proposal.total_pages
        estimated_time = total_pages * 0.1

        return DetectionStartResponse(
            session_id=session_id,
            detection_id=detection_id,
            status="started",
            detector_type=detector_type,
            estimated_time=estimated_time,
            message=f"Rerunning {detector_type} detection for {total_pages} pages",
        )

    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to rerun detection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rerun detection: {str(e)}",
        )
