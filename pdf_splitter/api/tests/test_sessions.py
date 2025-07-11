"""Tests for session management endpoints.

This module tests session creation, retrieval, listing, and lifecycle management.
"""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from pdf_splitter.api.tests.conftest import assert_api_response


class TestSessionEndpoints:
    """Test session management API endpoints."""

    def test_create_session(self, client: TestClient, uploaded_file: dict):
        """Test creating a new session."""
        response = client.post(
            "/api/sessions/create",
            json={
                "upload_id": uploaded_file["upload_id"],
                "session_name": "Test Session",
                "expires_in_hours": 24,
            },
        )

        data = assert_api_response(response)
        assert "session_id" in data
        assert data["status"] == "pending"
        assert data["file_name"] == uploaded_file["file_name"]
        assert data["total_pages"] == uploaded_file["total_pages"]
        assert data["has_proposal"] is True
        assert data["modifications_count"] == 0

        # Check timestamps
        assert "created_at" in data
        assert "updated_at" in data
        assert "expires_at" in data

        # Verify expiration is approximately 24 hours from now
        expires_at = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
        expected_expiry = datetime.now() + timedelta(hours=24)
        assert (
            abs((expires_at - expected_expiry).total_seconds()) < 60
        )  # Within 1 minute

    def test_create_session_with_custom_expiry(
        self, client: TestClient, uploaded_file: dict
    ):
        """Test creating session with custom expiration."""
        response = client.post(
            "/api/sessions/create",
            json={
                "upload_id": uploaded_file["upload_id"],
                "expires_in_hours": 48,
            },
        )

        data = assert_api_response(response)
        expires_at = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
        expected_expiry = datetime.now() + timedelta(hours=48)
        assert abs((expires_at - expected_expiry).total_seconds()) < 60

    def test_create_session_invalid_upload(self, client: TestClient):
        """Test creating session with invalid upload ID."""
        response = client.post(
            "/api/sessions/create",
            json={
                "upload_id": "00000000-0000-0000-0000-000000000000",
            },
        )

        assert response.status_code == 422
        assert "not found" in response.json()["detail"]

    def test_get_session(self, client: TestClient, test_session: dict):
        """Test retrieving session details."""
        session_id = test_session["session_id"]

        response = client.get(f"/api/sessions/{session_id}")
        data = assert_api_response(response)

        assert data["session_id"] == session_id
        assert data["status"] == test_session["status"]
        assert data["file_name"] == test_session["file_name"]
        assert data["total_pages"] == test_session["total_pages"]

    def test_get_session_not_found(self, client: TestClient):
        """Test retrieving non-existent session."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        response = client.get(f"/api/sessions/{fake_id}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_list_sessions(self, client: TestClient, test_session: dict):
        """Test listing all sessions."""
        response = client.get("/api/sessions/")
        data = assert_api_response(response)

        assert "sessions" in data
        assert len(data["sessions"]) >= 1
        assert data["total_count"] >= 1
        assert data["active_count"] >= 1
        assert data["page"] == 1
        assert data["page_size"] == 20

        # Find our test session
        session_found = False
        for session in data["sessions"]:
            if session["session_id"] == test_session["session_id"]:
                session_found = True
                break
        assert session_found

    def test_list_sessions_with_filters(self, client: TestClient, test_session: dict):
        """Test listing sessions with filters."""
        # Filter by status
        response = client.get("/api/sessions/?status=pending")
        data = assert_api_response(response)

        for session in data["sessions"]:
            assert session["status"] == "pending"

        # Filter active only
        response = client.get("/api/sessions/?active_only=false")
        data = assert_api_response(response)
        assert data["total_count"] >= data["active_count"]

    def test_list_sessions_pagination(self, client: TestClient, uploaded_file: dict):
        """Test session listing pagination."""
        # Create multiple sessions
        for i in range(5):
            client.post(
                "/api/sessions/create",
                json={
                    "upload_id": uploaded_file["upload_id"],
                    "session_name": f"Test Session {i}",
                },
            )

        # Test pagination
        response = client.get("/api/sessions/?page=1&page_size=2")
        data = assert_api_response(response)
        assert len(data["sessions"]) <= 2
        assert data["page"] == 1
        assert data["page_size"] == 2

        # Get second page
        response = client.get("/api/sessions/?page=2&page_size=2")
        data = assert_api_response(response)
        assert data["page"] == 2

    def test_delete_session(self, client: TestClient, test_session: dict):
        """Test deleting/cancelling a session."""
        session_id = test_session["session_id"]

        # Delete session
        response = client.delete(f"/api/sessions/{session_id}")
        data = assert_api_response(response)
        assert data["success"] is True

        # Verify it's cancelled
        response = client.get(f"/api/sessions/{session_id}")
        data = assert_api_response(response)
        assert data["status"] == "cancelled"

    def test_delete_session_not_found(self, client: TestClient):
        """Test deleting non-existent session."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        response = client.delete(f"/api/sessions/{fake_id}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_extend_session(self, client: TestClient, test_session: dict):
        """Test extending session expiration."""
        session_id = test_session["session_id"]

        # Get original expiration
        response = client.get(f"/api/sessions/{session_id}")
        original_expires = datetime.fromisoformat(
            response.json()["expires_at"].replace("Z", "+00:00")
        )

        # Extend by 24 hours
        response = client.post(f"/api/sessions/{session_id}/extend?hours=24")
        data = assert_api_response(response)
        assert data["success"] is True

        new_expires = datetime.fromisoformat(
            data["new_expires_at"].replace("Z", "+00:00")
        )

        # Should be extended
        assert new_expires > original_expires

    def test_extend_session_not_found(self, client: TestClient):
        """Test extending non-existent session."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        response = client.post(f"/api/sessions/{fake_id}/extend")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_session_expiration(self, client: TestClient, uploaded_file: dict):
        """Test session expiration handling."""
        # Create session with very short expiration
        response = client.post(
            "/api/sessions/create",
            json={
                "upload_id": uploaded_file["upload_id"],
                "expires_in_hours": 1,  # Minimum allowed
            },
        )
        data = assert_api_response(response)
        _ = data["session_id"]

        # Manually expire the session by modifying its expiration time
        # (In real tests, we'd mock time or wait)
        # For now, just verify the session was created with correct expiry
        expires_at = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
        expected_expiry = datetime.now() + timedelta(hours=1)
        assert abs((expires_at - expected_expiry).total_seconds()) < 60

    @pytest.mark.parametrize("invalid_expiry", [0, -1, 73, 100])
    def test_create_session_invalid_expiry(
        self, client: TestClient, uploaded_file: dict, invalid_expiry: int
    ):
        """Test creating session with invalid expiration hours."""
        response = client.post(
            "/api/sessions/create",
            json={
                "upload_id": uploaded_file["upload_id"],
                "expires_in_hours": invalid_expiry,
            },
        )

        assert response.status_code == 422
