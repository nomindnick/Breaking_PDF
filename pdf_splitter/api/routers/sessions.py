"""
Session Management Endpoints

Handles session lifecycle operations including listing, retrieval, extension, and deletion.
"""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from pdf_splitter.api.models.responses import APIResponse
from pdf_splitter.api.services.session_service import SessionService
from pdf_splitter.api.utils.exceptions import SessionNotFoundError
from pdf_splitter.splitting.models import SessionStatus

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class SessionResponse(BaseModel):
    """Response model for session information."""

    session_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    pdf_path: str
    has_proposal: bool
    modifications_count: int
    metadata: dict = Field(default_factory=dict)


class SessionListResponse(APIResponse):
    """Response model for session listing."""

    sessions: List[SessionResponse]
    total_count: int
    active_count: int
    limit: int
    offset: int


class SessionDetailsResponse(SessionResponse):
    """Extended response model for session details."""

    proposal_summary: Optional[dict] = None


class ExtendSessionRequest(BaseModel):
    """Request model for extending session expiration."""

    hours: int = Field(24, gt=0, le=168, description="Hours to extend (max 7 days)")


def get_session_service() -> SessionService:
    """Dependency to get session service instance."""
    return SessionService()


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    status: Optional[SessionStatus] = Query(
        None, description="Filter by session status"
    ),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    order_by: str = Query("created_at", regex="^(created_at|updated_at)$"),
    order_desc: bool = Query(True, description="Order descending"),
    session_service: SessionService = Depends(get_session_service),
) -> SessionListResponse:
    """
    List all sessions with optional filtering and pagination.

    Args:
        status: Optional status filter
        limit: Maximum number of results (1-100)
        offset: Number of results to skip
        order_by: Field to order by (created_at or updated_at)
        order_desc: Order descending if True

    Returns:
        SessionListResponse with paginated sessions
    """
    result = session_service.list_sessions(
        status=status,
        limit=limit,
        offset=offset,
        order_by=order_by,
        order_desc=order_desc,
    )

    # Convert sessions to response models
    session_responses = []
    for session in result["sessions"]:
        # Get additional details
        details = session_service.get_session_details(session.session_id)

        session_responses.append(
            SessionResponse(
                session_id=session.session_id,
                status=session.status.value,
                created_at=session.created_at,
                updated_at=session.updated_at,
                expires_at=session.expires_at,
                pdf_path=session.pdf_path,
                has_proposal=details["has_proposal"],
                modifications_count=details["modifications_count"],
                metadata=session.metadata or {},
            )
        )

    return SessionListResponse(
        success=True,
        sessions=session_responses,
        total_count=result["total_count"],
        active_count=result["active_count"],
        limit=limit,
        offset=offset,
    )


@router.get("/{session_id}", response_model=SessionDetailsResponse)
async def get_session(
    session_id: str, session_service: SessionService = Depends(get_session_service)
) -> SessionDetailsResponse:
    """
    Get detailed information about a specific session.

    Args:
        session_id: Session ID to retrieve

    Returns:
        SessionDetailsResponse with full session details

    Raises:
        404: Session not found
    """
    try:
        details = session_service.get_session_details(session_id)

        return SessionDetailsResponse(
            session_id=details["session_id"],
            status=details["status"],
            created_at=details["created_at"],
            updated_at=details["updated_at"],
            expires_at=details["expires_at"],
            pdf_path=details["pdf_path"],
            has_proposal=details["has_proposal"],
            modifications_count=details["modifications_count"],
            metadata=details["metadata"],
            proposal_summary=details.get("proposal_summary"),
        )

    except SessionNotFoundError as e:
        raise e


@router.post("/{session_id}/extend", response_model=SessionResponse)
async def extend_session(
    session_id: str,
    request: ExtendSessionRequest,
    session_service: SessionService = Depends(get_session_service),
) -> SessionResponse:
    """
    Extend the expiration time of a session.

    Args:
        session_id: Session ID to extend
        request: Extension request with hours

    Returns:
        Updated session information

    Raises:
        404: Session not found
    """
    try:
        session = session_service.extend_session(session_id, request.hours)
        details = session_service.get_session_details(session_id)

        return SessionResponse(
            session_id=session.session_id,
            status=session.status.value,
            created_at=session.created_at,
            updated_at=session.updated_at,
            expires_at=session.expires_at,
            pdf_path=session.pdf_path,
            has_proposal=details["has_proposal"],
            modifications_count=details["modifications_count"],
            metadata=session.metadata or {},
        )

    except SessionNotFoundError as e:
        raise e


@router.delete("/{session_id}")
async def delete_session(
    session_id: str, session_service: SessionService = Depends(get_session_service)
) -> APIResponse:
    """
    Delete a session and all associated data.

    This will remove:
    - Session from database
    - Any generated output files
    - Associated proposals and modifications

    Args:
        session_id: Session ID to delete

    Returns:
        Success confirmation

    Raises:
        404: Session not found
    """
    deleted = session_service.delete_session(session_id)

    if deleted:
        return APIResponse(
            success=True, message=f"Session {session_id} deleted successfully"
        )
    else:
        raise SessionNotFoundError(session_id)


@router.post("/cleanup")
async def cleanup_expired_sessions(
    session_service: SessionService = Depends(get_session_service),
) -> APIResponse:
    """
    Clean up all expired sessions.

    This endpoint triggers cleanup of sessions that have passed their expiration time.

    Returns:
        Number of sessions cleaned up
    """
    cleaned_count = session_service.cleanup_expired_sessions()

    return APIResponse(
        success=True, message=f"Cleaned up {cleaned_count} expired sessions"
    )


@router.get("/stats/summary")
async def get_session_statistics(
    session_service: SessionService = Depends(get_session_service),
) -> dict:
    """
    Get overall session statistics.

    Returns:
        Statistics including counts by status, averages, etc.
    """
    stats = session_service.get_session_statistics()

    return {"success": True, "statistics": stats}
