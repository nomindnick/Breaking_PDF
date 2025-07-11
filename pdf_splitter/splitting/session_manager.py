"""Session management for stateful PDF split operations.

This module provides session management to track the lifecycle of split
operations from proposal generation through user modifications to final execution.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.splitting.exceptions import (
    InvalidSessionStateError,
    SessionExpiredError,
    SessionNotFoundError,
)
from pdf_splitter.splitting.models import (
    DocumentSegment,
    SplitProposal,
    SplitSession,
    UserModification,
)

logger = logging.getLogger(__name__)


class SplitSessionManager:
    """Manages stateful split operations with persistent storage.

    Uses SQLite for session persistence to handle:
    - Session creation and retrieval
    - User modifications tracking
    - Session expiration
    - Cleanup of old sessions
    """

    def __init__(
        self, config: Optional[PDFConfig] = None, db_path: Optional[Path] = None
    ):
        """Initialize session manager.

        Args:
            config: PDF processing configuration
            db_path: Path to SQLite database (defaults to temp location)
        """
        self.config = config or PDFConfig()

        # Set database path
        if db_path:
            self.db_path = db_path
        else:
            # Use project's data directory
            self.db_path = Path.home() / ".cache" / "pdf_splitter" / "sessions.db"
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Session lifetime configuration
        self.session_lifetime = timedelta(hours=24)  # Sessions expire after 24 hours
        self.cleanup_interval = timedelta(hours=1)  # Clean up every hour

        # Initialize database
        self._init_database()

        # Track last cleanup time
        self._last_cleanup = datetime.now()

    def _init_database(self):
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    proposal_json TEXT NOT NULL,
                    modifications_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    output_directory TEXT
                )
            """
            )

            # Create indexes for efficient queries
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_status
                ON sessions(status)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_expires
                ON sessions(expires_at)
            """
            )

            conn.commit()

    def create_session(
        self, proposal: SplitProposal, lifetime: Optional[timedelta] = None
    ) -> SplitSession:
        """Create a new split session.

        Args:
            proposal: The split proposal to track
            lifetime: Optional custom session lifetime

        Returns:
            New SplitSession instance
        """
        # Generate session ID
        session_id = str(uuid4())

        # Set expiration
        expires_at = datetime.now() + (lifetime or self.session_lifetime)

        # Create session
        session = SplitSession(
            session_id=session_id, proposal=proposal, expires_at=expires_at
        )

        # Persist to database
        self._save_session(session)

        logger.info(f"Created session {session_id} for {proposal.pdf_path}")

        # Periodic cleanup
        self._maybe_cleanup()

        return session

    def get_session(self, session_id: str) -> SplitSession:
        """Retrieve a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            SplitSession instance

        Raises:
            SessionNotFoundError: If session doesn't exist
            SessionExpiredError: If session has expired
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = cursor.fetchone()

        if not row:
            raise SessionNotFoundError(f"Session {session_id} not found")

        # Deserialize session
        session = self._deserialize_session(dict(row))

        # Check expiration
        if session.is_expired:
            raise SessionExpiredError(f"Session {session_id} has expired")

        return session

    def update_session(
        self,
        session_id: str,
        modifications: Optional[List[UserModification]] = None,
        status: Optional[str] = None,
        output_directory: Optional[Path] = None,
    ) -> SplitSession:
        """Update an existing session.

        Args:
            session_id: Session identifier
            modifications: New modifications to add
            status: New status
            output_directory: Output directory for confirmed sessions

        Returns:
            Updated SplitSession

        Raises:
            SessionNotFoundError: If session doesn't exist
            InvalidSessionStateError: If update is invalid for current state
        """
        # Get existing session
        session = self.get_session(session_id)

        # Validate state transitions
        if status:
            self._validate_status_transition(session.status, status)
            session.status = status

        # Add modifications
        if modifications:
            for mod in modifications:
                session.add_modification(mod)

        # Set output directory
        if output_directory:
            session.output_directory = output_directory

        # Update timestamp
        session.updated_at = datetime.now()

        # Persist changes
        self._save_session(session)

        logger.info(f"Updated session {session_id}")

        return session

    def list_active_sessions(self) -> List[SplitSession]:
        """List all active (non-expired, non-completed) sessions.

        Returns:
            List of active sessions
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM sessions
                WHERE status IN ('pending', 'modified', 'confirmed')
                AND expires_at > ?
                ORDER BY updated_at DESC
            """,
                (datetime.now().isoformat(),),
            )

            rows = cursor.fetchall()

        sessions = []
        for row in rows:
            try:
                session = self._deserialize_session(dict(row))
                sessions.append(session)
            except Exception as e:
                logger.warning(f"Failed to deserialize session: {e}")
                continue

        return sessions

    def delete_session(self, session_id: str):
        """Delete a session.

        Args:
            session_id: Session identifier
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

        logger.info(f"Deleted session {session_id}")

    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions from database.

        Returns:
            Number of sessions cleaned up
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM sessions
                WHERE expires_at < ?
                OR status IN ('completed', 'cancelled')
            """,
                (datetime.now().isoformat(),),
            )

            count = cursor.rowcount
            conn.commit()

        if count > 0:
            logger.info(f"Cleaned up {count} expired/completed sessions")

        return count

    def _save_session(self, session: SplitSession):
        """Persist session to database."""
        # Serialize complex objects
        proposal_json = self._serialize_proposal(session.proposal)
        modifications_json = self._serialize_modifications(session.user_modifications)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                (session_id, proposal_json, modifications_json, status,
                 created_at, updated_at, expires_at, output_directory)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session.session_id,
                    proposal_json,
                    modifications_json,
                    session.status,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    session.expires_at.isoformat() if session.expires_at else None,
                    str(session.output_directory) if session.output_directory else None,
                ),
            )
            conn.commit()

    def _deserialize_session(self, row: Dict) -> SplitSession:
        """Deserialize session from database row."""
        # Deserialize proposal
        proposal = self._deserialize_proposal(row["proposal_json"])

        # Deserialize modifications
        modifications = self._deserialize_modifications(row["modifications_json"])

        # Create session
        session = SplitSession(
            session_id=row["session_id"],
            proposal=proposal,
            user_modifications=modifications,
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"])
            if row["expires_at"]
            else None,
            output_directory=Path(row["output_directory"])
            if row["output_directory"]
            else None,
        )

        return session

    def _serialize_proposal(self, proposal: SplitProposal) -> str:
        """Serialize proposal to JSON."""
        data = {
            "pdf_path": str(proposal.pdf_path),
            "total_pages": proposal.total_pages,
            "segments": [
                {
                    "start_page": s.start_page,
                    "end_page": s.end_page,
                    "document_type": s.document_type,
                    "suggested_filename": s.suggested_filename,
                    "confidence": s.confidence,
                    "summary": s.summary,
                    "metadata": s.metadata,
                    "segment_id": s.segment_id,
                    "is_user_defined": s.is_user_defined,
                }
                for s in proposal.segments
            ],
            "proposal_id": proposal.proposal_id,
            "created_at": proposal.created_at.isoformat(),
            "modified_at": proposal.modified_at.isoformat()
            if proposal.modified_at
            else None,
        }
        return json.dumps(data)

    def _deserialize_proposal(self, json_str: str) -> SplitProposal:
        """Deserialize proposal from JSON."""
        data = json.loads(json_str)

        # Recreate segments
        segments = [DocumentSegment(**seg_data) for seg_data in data["segments"]]

        # Create proposal with minimal validation
        proposal = SplitProposal.__new__(SplitProposal)
        proposal.pdf_path = Path(data["pdf_path"])
        proposal.total_pages = data["total_pages"]
        proposal.segments = segments
        proposal.detection_results = []  # Not persisted
        proposal.proposal_id = data["proposal_id"]
        proposal.created_at = datetime.fromisoformat(data["created_at"])
        proposal.modified_at = (
            datetime.fromisoformat(data["modified_at"]) if data["modified_at"] else None
        )

        return proposal

    def _serialize_modifications(self, modifications: List[UserModification]) -> str:
        """Serialize modifications to JSON."""
        data = [
            {
                "modification_type": m.modification_type,
                "segment_id": m.segment_id,
                "timestamp": m.timestamp.isoformat(),
                "details": m.details,
            }
            for m in modifications
        ]
        return json.dumps(data)

    def _deserialize_modifications(self, json_str: str) -> List[UserModification]:
        """Deserialize modifications from JSON."""
        data = json.loads(json_str)
        return [
            UserModification(
                modification_type=m["modification_type"],
                segment_id=m["segment_id"],
                timestamp=datetime.fromisoformat(m["timestamp"]),
                details=m["details"],
            )
            for m in data
        ]

    def _validate_status_transition(self, current: str, new: str):
        """Validate status transitions."""
        valid_transitions = {
            "pending": ["modified", "confirmed", "cancelled"],
            "modified": ["confirmed", "cancelled"],
            "confirmed": ["completed", "cancelled"],
            "completed": [],
            "cancelled": [],
        }

        if new not in valid_transitions.get(current, []):
            raise InvalidSessionStateError(
                f"Invalid status transition from '{current}' to '{new}'"
            )

    def _maybe_cleanup(self):
        """Perform cleanup if interval has passed."""
        if datetime.now() - self._last_cleanup > self.cleanup_interval:
            self.cleanup_expired_sessions()
            self._last_cleanup = datetime.now()
