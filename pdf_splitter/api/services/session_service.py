"""
Session Management Service

Provides high-level session management operations for the API.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from pdf_splitter.api.config import config
from pdf_splitter.api.utils.exceptions import SessionNotFoundError
from pdf_splitter.splitting.models import SplitProposal, SplitSession, UserModification
from pdf_splitter.splitting.session_manager import SplitSessionManager


class SessionService:
    """Service for managing processing sessions."""

    def __init__(self, session_manager: SplitSessionManager = None):
        self.session_manager = session_manager or SplitSessionManager(
            str(config.session_db_path)
        )

    def list_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        order_by: str = "created_at",
        order_desc: bool = True,
    ) -> Dict[str, Any]:
        """
        List sessions with filtering and pagination.

        Args:
            status: Filter by session status
            limit: Maximum number of results
            offset: Number of results to skip
            order_by: Field to order by
            order_desc: Order descending if True

        Returns:
            Dictionary with sessions and metadata
        """
        # Get all sessions
        all_sessions = self.session_manager.list_sessions()

        # Filter by status if specified
        if status:
            filtered_sessions = [s for s in all_sessions if s.status == status]
        else:
            filtered_sessions = all_sessions

        # Sort sessions
        if order_by == "created_at":
            filtered_sessions.sort(key=lambda s: s.created_at, reverse=order_desc)
        elif order_by == "updated_at":
            filtered_sessions.sort(key=lambda s: s.updated_at, reverse=order_desc)

        # Calculate totals
        total_count = len(filtered_sessions)
        active_count = sum(
            1 for s in filtered_sessions if s.status in ["processing", "confirmed"]
        )

        # Apply pagination
        paginated_sessions = filtered_sessions[offset : offset + limit]

        return {
            "sessions": paginated_sessions,
            "total_count": total_count,
            "active_count": active_count,
            "limit": limit,
            "offset": offset,
        }

    def get_session_details(self, session_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a session.

        Args:
            session_id: Session ID

        Returns:
            Session details including metadata and proposal

        Raises:
            SessionNotFoundError: If session not found
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        details = {
            "session_id": session.session_id,
            "status": session.status.value,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "expires_at": session.expires_at,
            "pdf_path": session.pdf_path,
            "metadata": session.metadata or {},
        }

        # Add proposal info if available
        proposal = self.session_manager.get_proposal(session_id)
        if proposal:
            details["has_proposal"] = True
            details["proposal_summary"] = {
                "total_pages": proposal.total_pages,
                "segments_count": len(proposal.segments),
                "created_at": proposal.created_at,
                "modified_at": proposal.modified_at,
            }
        else:
            details["has_proposal"] = False

        # Add modification history
        modifications = self.session_manager.get_modifications(session_id)
        details["modifications_count"] = len(modifications)

        return details

    def extend_session(self, session_id: str, hours: int = 24) -> SplitSession:
        """
        Extend session expiration time.

        Args:
            session_id: Session ID to extend
            hours: Number of hours to extend by

        Returns:
            Updated session

        Raises:
            SessionNotFoundError: If session not found
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        # Calculate new expiration
        new_expiration = datetime.utcnow() + timedelta(hours=hours)

        # Update session
        session.expires_at = new_expiration
        session.updated_at = datetime.utcnow()

        # Save to database
        self.session_manager._save_session_to_db(session)

        return session

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and associated data.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return False

        # Delete associated files
        output_dir = Path(config.output_dir) / session_id
        if output_dir.exists():
            import shutil

            shutil.rmtree(output_dir)

        # Delete from database
        return self.session_manager.delete_session(session_id)

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        expired_sessions = self.session_manager.get_expired_sessions()
        cleaned_count = 0

        for session in expired_sessions:
            if self.delete_session(session.session_id):
                cleaned_count += 1

        return cleaned_count

    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get overall session statistics.

        Returns:
            Statistics dictionary
        """
        all_sessions = self.session_manager.list_sessions()

        # Count by status
        status_counts = {}
        for status in ["pending", "processing", "confirmed", "completed", "cancelled"]:
            status_counts[status] = sum(
                1 for s in all_sessions if s.status == status
            )

        # Calculate averages
        completed_sessions = [s for s in all_sessions if s.status == "completed"]

        avg_processing_time = None
        if completed_sessions:
            processing_times = []
            for session in completed_sessions:
                if session.metadata and "processing_time" in session.metadata:
                    processing_times.append(session.metadata["processing_time"])

            if processing_times:
                avg_processing_time = sum(processing_times) / len(processing_times)

        return {
            "total_sessions": len(all_sessions),
            "status_counts": status_counts,
            "active_sessions": status_counts.get("processing", 0),
            "completed_sessions": status_counts.get("completed", 0),
            "average_processing_time": avg_processing_time,
            "oldest_session": min(all_sessions, key=lambda s: s.created_at).created_at
            if all_sessions
            else None,
            "newest_session": max(all_sessions, key=lambda s: s.created_at).created_at
            if all_sessions
            else None,
        }
