"""Tests for session management functionality."""

import sqlite3
from datetime import datetime, timedelta

import pytest

from pdf_splitter.detection.base_detector import (
    BoundaryResult,
    BoundaryType,
    DetectorType,
)
from pdf_splitter.splitting.exceptions import (
    InvalidSessionStateError,
    SessionExpiredError,
    SessionNotFoundError,
)
from pdf_splitter.splitting.models import (
    DocumentSegment,
    SplitProposal,
    UserModification,
)
from pdf_splitter.splitting.session_manager import SplitSessionManager


class TestSplitSessionManager:
    """Test SplitSessionManager functionality."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create temporary database path."""
        return tmp_path / "test_sessions.db"

    @pytest.fixture
    def manager(self, temp_db):
        """Create session manager with temp database."""
        return SplitSessionManager(db_path=temp_db)

    @pytest.fixture
    def sample_proposal(self, tmp_path):
        """Create sample proposal for testing."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf")

        segments = [
            DocumentSegment(
                start_page=0,
                end_page=5,
                document_type="Invoice",
                suggested_filename="invoice_001.pdf",
                confidence=0.95,
            ),
            DocumentSegment(
                start_page=6,
                end_page=10,
                document_type="Email",
                suggested_filename="email_002.pdf",
                confidence=0.88,
            ),
        ]

        boundaries = [
            BoundaryResult(
                page_number=0,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.95,
                detector_type=DetectorType.EMBEDDINGS,
            ),
            BoundaryResult(
                page_number=6,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.88,
                detector_type=DetectorType.EMBEDDINGS,
            ),
        ]

        return SplitProposal(
            pdf_path=pdf_path,
            total_pages=15,
            segments=segments,
            detection_results=boundaries,
        )

    def test_database_initialization(self, temp_db):
        """Test database is properly initialized."""
        SplitSessionManager(db_path=temp_db)

        # Check database exists
        assert temp_db.exists()

        # Check schema
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert "sessions" in tables

    def test_create_session(self, manager, sample_proposal):
        """Test session creation."""
        session = manager.create_session(sample_proposal)

        assert session.session_id
        assert session.proposal == sample_proposal
        assert session.status == "pending"
        assert session.expires_at > datetime.now()
        assert len(session.user_modifications) == 0

    def test_get_session(self, manager, sample_proposal):
        """Test session retrieval."""
        # Create session
        created_session = manager.create_session(sample_proposal)

        # Retrieve session
        retrieved_session = manager.get_session(created_session.session_id)

        assert retrieved_session.session_id == created_session.session_id
        assert retrieved_session.status == created_session.status
        assert len(retrieved_session.proposal.segments) == 2

    def test_get_nonexistent_session(self, manager):
        """Test retrieving non-existent session."""
        with pytest.raises(SessionNotFoundError):
            manager.get_session("nonexistent-id")

    def test_get_expired_session(self, manager, sample_proposal):
        """Test retrieving expired session."""
        # Create session with short lifetime
        session = manager.create_session(
            sample_proposal, lifetime=timedelta(seconds=-1)  # Already expired
        )

        with pytest.raises(SessionExpiredError):
            manager.get_session(session.session_id)

    def test_update_session_status(self, manager, sample_proposal):
        """Test updating session status."""
        session = manager.create_session(sample_proposal)

        # Update to modified
        updated = manager.update_session(session.session_id, status="modified")

        assert updated.status == "modified"
        assert updated.updated_at > session.created_at

    def test_update_session_modifications(self, manager, sample_proposal):
        """Test adding modifications to session."""
        session = manager.create_session(sample_proposal)

        # Add modification
        mod = UserModification(
            modification_type="rename",
            segment_id=sample_proposal.segments[0].segment_id,
            details={"new_filename": "custom_invoice.pdf"},
        )

        updated = manager.update_session(session.session_id, modifications=[mod])

        assert len(updated.user_modifications) == 1
        assert updated.user_modifications[0].modification_type == "rename"
        assert updated.status == "modified"

    def test_invalid_status_transition(self, manager, sample_proposal):
        """Test invalid status transitions are rejected."""
        session = manager.create_session(sample_proposal)

        # Try invalid transition from pending to completed
        with pytest.raises(InvalidSessionStateError):
            manager.update_session(session.session_id, status="completed")

    def test_list_active_sessions(self, manager, sample_proposal):
        """Test listing active sessions."""
        # Create multiple sessions
        session1 = manager.create_session(sample_proposal)
        session2 = manager.create_session(sample_proposal)

        # Complete one session
        manager.update_session(session1.session_id, status="confirmed")
        manager.update_session(session1.session_id, status="completed")

        # List active sessions
        active = manager.list_active_sessions()

        assert len(active) == 1
        assert active[0].session_id == session2.session_id

    def test_delete_session(self, manager, sample_proposal):
        """Test session deletion."""
        session = manager.create_session(sample_proposal)

        # Delete session
        manager.delete_session(session.session_id)

        # Try to retrieve
        with pytest.raises(SessionNotFoundError):
            manager.get_session(session.session_id)

    def test_cleanup_expired_sessions(self, manager, sample_proposal):
        """Test cleanup of expired sessions."""
        # Create expired session
        expired = manager.create_session(
            sample_proposal, lifetime=timedelta(seconds=-1)
        )

        # Create active session
        active = manager.create_session(sample_proposal)

        # Run cleanup
        count = manager.cleanup_expired_sessions()

        assert count == 1

        # Verify expired is gone
        with pytest.raises(SessionNotFoundError):
            manager.get_session(expired.session_id)

        # Verify active still exists
        retrieved = manager.get_session(active.session_id)
        assert retrieved.session_id == active.session_id

    def test_session_confirmation_flow(self, manager, sample_proposal, tmp_path):
        """Test full session workflow from creation to confirmation."""
        # Create session
        session = manager.create_session(sample_proposal)

        # Add user modifications
        mod1 = UserModification(
            modification_type="rename",
            segment_id=sample_proposal.segments[0].segment_id,
            details={"new_filename": "invoice_renamed.pdf"},
        )

        mod2 = UserModification(
            modification_type="remove",
            segment_id=sample_proposal.segments[1].segment_id,
        )

        # Update with modifications
        session = manager.update_session(session.session_id, modifications=[mod1, mod2])

        assert len(session.user_modifications) == 2
        assert session.status == "modified"

        # Confirm session
        output_dir = tmp_path / "output"
        session = manager.update_session(
            session.session_id, status="confirmed", output_directory=output_dir
        )

        assert session.status == "confirmed"
        assert session.output_directory == output_dir

    def test_proposal_serialization_roundtrip(self, manager, sample_proposal):
        """Test proposal can be serialized and deserialized correctly."""
        # Serialize
        json_str = manager._serialize_proposal(sample_proposal)

        # Deserialize
        deserialized = manager._deserialize_proposal(json_str)

        assert deserialized.pdf_path == sample_proposal.pdf_path
        assert deserialized.total_pages == sample_proposal.total_pages
        assert len(deserialized.segments) == len(sample_proposal.segments)
        assert (
            deserialized.segments[0].suggested_filename
            == sample_proposal.segments[0].suggested_filename
        )

    def test_modifications_serialization_roundtrip(self, manager):
        """Test modifications can be serialized and deserialized correctly."""
        mods = [
            UserModification(
                modification_type="rename",
                segment_id="seg-1",
                details={"new_filename": "renamed.pdf"},
            ),
            UserModification(
                modification_type="add",
                segment_id="seg-2",
                details={"start_page": 10, "end_page": 15},
            ),
        ]

        # Serialize
        json_str = manager._serialize_modifications(mods)

        # Deserialize
        deserialized = manager._deserialize_modifications(json_str)

        assert len(deserialized) == 2
        assert deserialized[0].modification_type == "rename"
        assert deserialized[1].segment_id == "seg-2"

    def test_concurrent_session_management(self, manager, sample_proposal):
        """Test managing multiple concurrent sessions."""
        # Create multiple sessions
        sessions = []
        for i in range(5):
            session = manager.create_session(sample_proposal)
            sessions.append(session)

        # Verify all can be retrieved
        for session in sessions:
            retrieved = manager.get_session(session.session_id)
            assert retrieved.session_id == session.session_id

        # List should show all 5
        active = manager.list_active_sessions()
        assert len(active) == 5
