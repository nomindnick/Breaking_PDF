"""Session management endpoints.

This module provides API endpoints for managing split sessions.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from pdf_splitter.api.dependencies import get_session_manager, get_upload_manager
from pdf_splitter.api.exceptions import SessionExpiredError as APISessionExpiredError
from pdf_splitter.api.exceptions import SessionNotFoundError as APISessionNotFoundError
from pdf_splitter.api.exceptions import ValidationError
from pdf_splitter.api.models.requests import SessionCreateRequest
from pdf_splitter.api.models.responses import SessionListResponse, SessionResponse
from pdf_splitter.core.logging import get_logger
from pdf_splitter.splitting.exceptions import SessionExpiredError, SessionNotFoundError
from pdf_splitter.splitting.models import SplitProposal, SplitSession
from pdf_splitter.splitting.session_manager import SplitSessionManager

logger = get_logger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def get_session_by_upload_id(
    upload_id: str,
    session_manager: SplitSessionManager,
) -> Optional[SplitSession]:
    """Find session by upload ID.

    Args:
        upload_id: Upload identifier
        session_manager: Session manager instance

    Returns:
        Session if found, None otherwise
    """
    # Get all active sessions
    sessions = session_manager.list_active_sessions()

    # Look for session with matching upload_id in metadata
    for session in sessions:
        # Check metadata first (new sessions)
        if session.metadata.get("upload_id") == upload_id:
            return session
        # Fallback: check if the session's PDF path contains the upload ID (legacy)
        elif upload_id in str(session.proposal.pdf_path):
            return session

    return None


def _session_to_response(
    session: SplitSession, upload_info: Optional[dict] = None
) -> SessionResponse:
    """Convert SplitSession to SessionResponse.

    Args:
        session: Split session object
        upload_info: Optional upload information

    Returns:
        Session response object
    """
    return SessionResponse(
        session_id=session.session_id,
        status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
        expires_at=session.expires_at,
        upload_id=session.metadata.get(
            "upload_id", str(session.proposal.pdf_path)
        ),  # Use metadata or fallback
        file_name=upload_info.get("file_name") or upload_info.get("filename")
        if upload_info
        else session.proposal.pdf_path.name,
        total_pages=session.proposal.total_pages,
        has_proposal=True,
        modifications_count=len(session.user_modifications),
    )


@router.post("/create", response_model=SessionResponse)
async def create_session(
    request: SessionCreateRequest,
    session_manager: SplitSessionManager = Depends(get_session_manager),
    upload_manager=Depends(get_upload_manager),
):
    """Create a new split session.

    Args:
        request: Session creation request
        session_manager: Session manager instance
        upload_manager: Upload manager instance

    Returns:
        Created session details

    Raises:
        HTTPException: On session creation failure
    """
    try:
        # Get upload information
        upload_info = upload_manager.get_upload_metadata(request.upload_id)
        if not upload_info:
            raise ValidationError(
                f"Upload {request.upload_id} not found", field="upload_id"
            )

        # Get upload path
        upload_path = upload_manager.get_upload_path(request.upload_id)
        if not upload_path or not upload_path.exists():
            raise ValidationError(
                f"Upload file not found for {request.upload_id}", field="upload_id"
            )

        # Create a basic proposal for now (detection will update it)
        from pdf_splitter.splitting.models import DocumentSegment

        # Create single segment for entire document as default
        default_segment = DocumentSegment(
            start_page=0,
            end_page=upload_info.get("total_pages", 1) - 1,
            document_type="Document",
            suggested_filename=upload_path.stem + "_complete.pdf",
            confidence=0.0,  # No detection yet
            summary="Complete document (no detection performed yet)",
        )

        proposal = SplitProposal(
            pdf_path=upload_path,
            total_pages=upload_info.get("total_pages", 1),
            segments=[default_segment],
            detection_results=[],  # Will be populated by detection
        )

        # Set lifetime
        lifetime = timedelta(hours=request.expires_in_hours)

        # Create session
        session = session_manager.create_session(proposal, lifetime=lifetime)

        if not session:
            raise RuntimeError("Failed to create session")

        # Store upload_id in session metadata
        session.metadata["upload_id"] = request.upload_id
        session_manager.update_session(session)

        logger.info(
            f"Created session {session.session_id} for upload {request.upload_id}"
        )

        return _session_to_response(session, upload_info)

    except ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        import traceback

        logger.error(f"Failed to create session: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create session: {str(e)}"
        )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    session_manager: SplitSessionManager = Depends(get_session_manager),
    upload_manager=Depends(get_upload_manager),
):
    """Get session details.

    Args:
        session_id: Session identifier
        session_manager: Session manager instance
        upload_manager: Upload manager instance

    Returns:
        Session details

    Raises:
        HTTPException: If session not found
    """
    try:
        # Get session - this will raise SessionNotFoundError if not found
        session = session_manager.get_session(session_id)

        # Check if expired
        if session.is_expired:
            raise APISessionExpiredError(session_id)

        # Try to get upload info (may not exist if upload was cleaned up)
        upload_info = None
        if session.proposal.pdf_path:
            # Extract upload ID from path (if stored in standard location)
            path_parts = session.proposal.pdf_path.parts
            for i, part in enumerate(path_parts):
                if part == "pdf_splitter_uploads" and i + 1 < len(path_parts):
                    potential_upload_id = path_parts[i + 1]
                    upload_info = upload_manager.get_upload_metadata(
                        potential_upload_id
                    )
                    break

        return _session_to_response(session, upload_info)

    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except SessionExpiredError:
        raise HTTPException(status_code=410, detail=f"Session {session_id} has expired")
    except (APISessionNotFoundError, APISessionExpiredError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        logger.error(f"Failed to get session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@router.get("/", response_model=SessionListResponse)
async def list_sessions(
    status: Optional[str] = Query(
        None, pattern="^(pending|modified|confirmed|completed|cancelled)$"
    ),
    active_only: bool = Query(True, description="Only show active sessions"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session_manager: SplitSessionManager = Depends(get_session_manager),
    upload_manager=Depends(get_upload_manager),
):
    """List all sessions with optional filtering.

    Args:
        status: Filter by session status
        active_only: Only show active (non-expired) sessions
        page: Page number
        page_size: Items per page
        session_manager: Session manager instance
        upload_manager: Upload manager instance

    Returns:
        List of sessions

    Raises:
        HTTPException: On listing failure
    """
    try:
        # Get all sessions
        if active_only:
            all_sessions = session_manager.list_active_sessions()
        else:
            # For now, we only support active sessions
            # TODO: Add method to list all sessions including completed/expired
            all_sessions = session_manager.list_active_sessions()

        # Filter by status if provided
        if status:
            all_sessions = [s for s in all_sessions if s.status == status]

        # Calculate pagination
        total_count = len(all_sessions)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_sessions = all_sessions[start_idx:end_idx]

        # Convert to responses
        session_responses = []
        for session in paginated_sessions:
            # Try to get upload info
            upload_info = None
            if session.proposal.pdf_path:
                path_parts = session.proposal.pdf_path.parts
                for i, part in enumerate(path_parts):
                    if part == "pdf_splitter_uploads" and i + 1 < len(path_parts):
                        potential_upload_id = path_parts[i + 1]
                        upload_info = upload_manager.get_upload_metadata(
                            potential_upload_id
                        )
                        break

            session_responses.append(_session_to_response(session, upload_info))

        # Count active sessions
        active_count = sum(1 for s in all_sessions if s.is_active)

        return SessionListResponse(
            sessions=session_responses,
            total_count=total_count,
            active_count=active_count,
            page=page,
            page_size=page_size,
            message=f"Found {total_count} sessions ({active_count} active)",
        )

    except Exception as e:
        logger.error(f"Failed to list sessions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list sessions: {str(e)}"
        )


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    session_manager: SplitSessionManager = Depends(get_session_manager),
):
    """Cancel/delete a session.

    Args:
        session_id: Session identifier
        session_manager: Session manager instance

    Returns:
        Success message

    Raises:
        HTTPException: If session not found or deletion fails
    """
    try:
        # Get session - this will raise SessionNotFoundError if not found
        session = session_manager.get_session(session_id)

        # Cancel session
        session.cancel()
        session_manager.update_session(session)

        # Don't clean up immediately - let scheduled cleanup handle it
        # session_manager.cleanup_expired_sessions()

        return JSONResponse(
            content={
                "success": True,
                "message": f"Session {session_id} cancelled successfully",
            }
        )

    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except APISessionNotFoundError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        logger.error(f"Failed to delete session: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete session: {str(e)}"
        )


@router.post("/{session_id}/extend")
async def extend_session(
    session_id: str,
    hours: int = Query(24, ge=1, le=72),
    session_manager: SplitSessionManager = Depends(get_session_manager),
):
    """Extend session expiration time.

    Args:
        session_id: Session identifier
        hours: Hours to extend by
        session_manager: Session manager instance

    Returns:
        Updated session details

    Raises:
        HTTPException: If session not found or extension fails
    """
    try:
        # Get session - this will raise SessionNotFoundError if not found
        session = session_manager.get_session(session_id)

        # Check if already expired
        if session.is_expired:
            raise APISessionExpiredError(session_id)

        # Extend expiration
        new_expires_at = datetime.now() + timedelta(hours=hours)
        if session.expires_at:
            new_expires_at = max(new_expires_at, session.expires_at)

        session.expires_at = new_expires_at
        session.updated_at = datetime.now()

        # Update session
        session_manager.update_session(session)

        logger.info(f"Extended session {session_id} by {hours} hours")

        return JSONResponse(
            content={
                "success": True,
                "message": f"Session extended by {hours} hours",
                "new_expires_at": new_expires_at.isoformat(),
            }
        )

    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except SessionExpiredError:
        raise HTTPException(status_code=410, detail=f"Session {session_id} has expired")
    except (APISessionNotFoundError, APISessionExpiredError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        logger.error(f"Failed to extend session: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to extend session: {str(e)}"
        )
