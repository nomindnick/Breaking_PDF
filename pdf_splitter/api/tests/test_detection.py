"""Tests for detection endpoints.

This module tests boundary detection functionality including starting detection,
checking progress, and retrieving results.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from pdf_splitter.api.services.detection_service import DetectionService
from pdf_splitter.api.tests.conftest import assert_api_response


class TestDetectionEndpoints:
    """Test detection API endpoints."""

    def test_start_detection(self, client: TestClient, uploaded_file: dict):
        """Test starting boundary detection."""
        # Need to create a session first to match the detection logic
        session_response = client.post(
            "/api/sessions/create",
            json={"upload_id": uploaded_file["upload_id"]},
        )
        session_data = assert_api_response(session_response)

        # Mock the detection service
        with patch.object(DetectionService, "start_detection") as mock_start:
            mock_start.return_value = "test-detection-id"

            response = client.post(
                "/api/detection/start",
                json={
                    "upload_id": uploaded_file["upload_id"],
                    "detector_type": "embeddings",
                    "confidence_threshold": 0.5,
                },
            )

            data = assert_api_response(response)
            assert "detection_id" in data
            assert data["session_id"] == session_data["session_id"]
            assert data["status"] == "started"
            assert data["detector_type"] == "embeddings"
            assert "estimated_time" in data

    def test_start_detection_with_options(
        self, client: TestClient, uploaded_file: dict
    ):
        """Test starting detection with different options."""
        # Create session
        session_response = client.post(
            "/api/sessions/create",
            json={"upload_id": uploaded_file["upload_id"]},
        )
        assert_api_response(session_response)

        with patch.object(DetectionService, "start_detection") as mock_start:
            mock_start.return_value = "test-detection-id"

            # Test different detector types
            for detector_type in ["embeddings", "heuristic", "visual", "llm"]:
                response = client.post(
                    "/api/detection/start",
                    json={
                        "upload_id": uploaded_file["upload_id"],
                        "detector_type": detector_type,
                        "confidence_threshold": 0.7,
                        "max_processing_time": 600,
                    },
                )

                data = assert_api_response(response)
                assert data["detector_type"] == detector_type

    def test_start_detection_invalid_detector(
        self, client: TestClient, uploaded_file: dict
    ):
        """Test starting detection with invalid detector type."""
        # Create session
        client.post(
            "/api/sessions/create",
            json={"upload_id": uploaded_file["upload_id"]},
        )

        response = client.post(
            "/api/detection/start",
            json={
                "upload_id": uploaded_file["upload_id"],
                "detector_type": "invalid_detector",
            },
        )

        assert response.status_code == 422

    def test_start_detection_no_session(self, client: TestClient):
        """Test starting detection without a session."""
        response = client.post(
            "/api/detection/start",
            json={
                "upload_id": "00000000-0000-0000-0000-000000000000",
                "detector_type": "embeddings",
            },
        )

        assert response.status_code == 404
        # The error message should contain the session/upload ID that wasn't found
        assert "00000000-0000-0000-0000-000000000000" in response.json()["detail"]

    def test_get_detection_status(self, client: TestClient):
        """Test getting detection status."""
        detection_id = "test-detection-id"

        # Mock detection service
        with patch.object(DetectionService, "get_detection_status") as mock_status:
            mock_status.return_value = {
                "detection_id": detection_id,
                "session_id": "test-session-id",
                "status": "detecting_boundaries",
                "progress": 0.45,
                "detector_type": "embeddings",
                "current_page": 45,
                "total_pages": 100,
                "elapsed_time": 12.5,
                "estimated_remaining": 15.0,
            }

            response = client.get(f"/api/detection/{detection_id}/status")
            data = assert_api_response(response)

            assert data["detection_id"] == detection_id
            assert data["status"] == "detecting_boundaries"
            assert data["progress"] == 0.45
            assert data["current_page"] == 45
            assert data["total_pages"] == 100
            assert data["elapsed_time"] == 12.5
            assert data["estimated_remaining"] == 15.0

    def test_get_detection_status_not_found(self, client: TestClient):
        """Test getting status for non-existent detection."""
        with patch.object(DetectionService, "get_detection_status") as mock_status:
            from pdf_splitter.api.exceptions import ValidationError

            mock_status.side_effect = ValidationError(
                "Detection not found", "detection_id"
            )

            response = client.get("/api/detection/fake-id/status")
            assert response.status_code == 422

    def test_get_detection_results(
        self, client: TestClient, mock_boundaries, mock_proposal
    ):
        """Test getting detection results."""
        detection_id = "test-detection-id"
        session_id = "test-session-id"

        # Mock detection service
        with patch.object(DetectionService, "get_detection_results") as mock_results:
            mock_results.return_value = (mock_boundaries, mock_proposal)

            with patch.object(DetectionService, "get_detection_status") as mock_status:
                mock_status.return_value = {
                    "detection_id": detection_id,
                    "session_id": session_id,
                    "status": "completed",
                    "elapsed_time": 15.0,
                }

                # Mock session manager using dependency override
                from pdf_splitter.api.dependencies import get_session_manager

                mock_manager = MagicMock()
                mock_session = MagicMock()
                mock_session.proposal = mock_proposal
                mock_manager.get_session.return_value = mock_session
                mock_manager.update_session.return_value = mock_session

                # Override dependency in test client
                client.app.dependency_overrides[
                    get_session_manager
                ] = lambda: mock_manager

                try:
                    response = client.get(f"/api/detection/{detection_id}/results")
                    data = assert_api_response(response)

                    assert data["detection_id"] == detection_id
                    assert data["session_id"] == session_id
                    assert data["boundaries_found"] == len(mock_boundaries)
                    assert data["segments_proposed"] == len(mock_proposal.segments)
                    assert data["processing_time"] == 15.0
                    assert len(data["boundaries"]) == len(mock_boundaries)

                    # Check boundary data
                    for i, boundary in enumerate(data["boundaries"]):
                        assert boundary["page_number"] == mock_boundaries[i].page_number
                        assert boundary["confidence"] == mock_boundaries[i].confidence
                finally:
                    # Clean up override
                    client.app.dependency_overrides.pop(get_session_manager, None)

    def test_get_detection_results_not_completed(self, client: TestClient):
        """Test getting results for incomplete detection."""
        detection_id = "test-detection-id"

        with patch.object(DetectionService, "get_detection_status") as mock_status:
            mock_status.return_value = {"status": "detecting_boundaries"}

            with patch.object(
                DetectionService, "get_detection_results"
            ) as mock_results:
                from pdf_splitter.api.exceptions import ValidationError

                mock_results.side_effect = ValidationError(
                    "Detection not completed", "detection_id"
                )

                response = client.get(f"/api/detection/{detection_id}/results")
                assert response.status_code == 422

    def test_rerun_detection(self, client: TestClient, test_session: dict):
        """Test rerunning detection with different parameters."""
        session_id = test_session["session_id"]

        with patch.object(DetectionService, "start_detection") as mock_start:
            mock_start.return_value = "new-detection-id"

            response = client.post(
                f"/api/detection/{session_id}/rerun",
                params={
                    "detector_type": "heuristic",
                    "confidence_threshold": 0.6,
                },
            )

            data = assert_api_response(response)
            assert "detection_id" in data
            assert data["session_id"] == session_id
            assert data["detector_type"] == "heuristic"
            assert data["status"] == "started"

    def test_rerun_detection_invalid_session(self, client: TestClient):
        """Test rerunning detection for invalid session."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        response = client.post(f"/api/detection/{fake_id}/rerun")
        assert response.status_code == 404

    @pytest.mark.parametrize("confidence", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_detection_confidence_thresholds(
        self, client: TestClient, uploaded_file: dict, confidence: float
    ):
        """Test detection with various confidence thresholds."""
        # Create session
        session_response = client.post(
            "/api/sessions/create",
            json={"upload_id": uploaded_file["upload_id"]},
        )
        assert_api_response(session_response)

        with patch.object(DetectionService, "start_detection") as mock_start:
            mock_start.return_value = f"detection-{confidence}"

            response = client.post(
                "/api/detection/start",
                json={
                    "upload_id": uploaded_file["upload_id"],
                    "confidence_threshold": confidence,
                },
            )

            data = assert_api_response(response)
            assert data["detection_id"] == f"detection-{confidence}"

    def test_detection_progress_tracking(self, client: TestClient):
        """Test detection progress updates."""
        detection_id = "test-detection-id"

        # Simulate progress updates
        progress_states = [
            {"status": "loading_pdf", "progress": 0.0},
            {"status": "extracting_text", "progress": 0.3},
            {"status": "detecting_boundaries", "progress": 0.6},
            {"status": "generating_proposal", "progress": 0.9},
            {"status": "completed", "progress": 1.0},
        ]

        for state in progress_states:
            with patch.object(DetectionService, "get_detection_status") as mock_status:
                mock_status.return_value = {
                    "detection_id": detection_id,
                    "session_id": "test-session-id",
                    "detector_type": "embeddings",
                    "total_pages": 100,
                    "elapsed_time": 10.0,
                    **state,
                }

                response = client.get(f"/api/detection/{detection_id}/status")
                data = assert_api_response(response)
                assert data["status"] == state["status"]
                assert data["progress"] == state["progress"]
