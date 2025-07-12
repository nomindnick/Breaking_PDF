"""Integration tests for the complete API workflow.

This module tests the full end-to-end workflow from upload through
detection to splitting and download.
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from pdf_splitter.api.services.detection_service import DetectionService
from pdf_splitter.api.services.splitting_service import SplittingService
from pdf_splitter.api.tests.conftest import assert_api_response
from pdf_splitter.test_utils import create_test_pdf


class TestAPIIntegration:
    """Test complete API workflow integration."""

    @pytest.fixture
    def multi_doc_pdf(self, temp_dir: Path) -> Path:
        """Create a multi-document PDF for testing."""
        pdf_path = temp_dir / "multi_document.pdf"
        create_test_pdf(
            num_pages=10,
            output_path=pdf_path,
            page_size=(612, 792),
            include_text=True,
        )
        return pdf_path

    def test_complete_workflow(self, client: TestClient, multi_doc_pdf: Path):
        """Test the complete workflow from upload to download."""
        # Step 1: Upload PDF
        with open(multi_doc_pdf, "rb") as f:
            upload_response = client.post(
                "/api/upload/file",
                files={"file": ("multi_document.pdf", f, "application/pdf")},
            )

        upload_data = assert_api_response(upload_response)
        upload_id = upload_data["upload_id"]
        assert upload_data["total_pages"] == 10

        # Step 2: Create session
        session_response = client.post(
            "/api/sessions/create",
            json={
                "upload_id": upload_id,
                "session_name": "Integration Test Session",
            },
        )

        session_data = assert_api_response(session_response)
        session_id = session_data["session_id"]

        # Step 3: Start detection (mocked)
        with patch.object(DetectionService, "start_detection") as mock_start:
            mock_start.return_value = "detection-123"

            detection_response = client.post(
                "/api/detection/start",
                json={
                    "upload_id": upload_id,
                    "detector_type": "embeddings",
                    "confidence_threshold": 0.5,
                },
            )

            detection_data = assert_api_response(detection_response)
            detection_id = detection_data["detection_id"]

        # Step 4: Check detection status (mocked as complete)
        with patch.object(DetectionService, "get_detection_status") as mock_status:
            mock_status.return_value = {
                "detection_id": detection_id,
                "session_id": session_id,
                "status": "completed",
                "progress": 1.0,
                "elapsed_time": 5.0,
                "detector_type": "embeddings",
                "total_pages": 10,
            }

            status_response = client.get(f"/api/detection/{detection_id}/status")
            status_data = assert_api_response(status_response)
            assert status_data["status"] == "completed"

        # Step 5: Get proposal
        proposal_response = client.get(f"/api/splits/{session_id}/proposal")
        proposal_data = assert_api_response(proposal_response)
        assert proposal_data["total_pages"] == 10
        assert len(proposal_data["segments"]) >= 1

        # Step 6: Modify proposal - add custom segments
        # Delete default segment
        default_segment_id = proposal_data["segments"][0]["segment_id"]
        delete_response = client.delete(
            f"/api/splits/{session_id}/segments/{default_segment_id}"
        )
        assert_api_response(delete_response)

        # Add custom segments
        segments = [
            {
                "start_page": 0,
                "end_page": 2,
                "document_type": "Invoice",
                "suggested_filename": "Invoice_2024_001.pdf",
            },
            {
                "start_page": 3,
                "end_page": 5,
                "document_type": "Letter",
                "suggested_filename": "Letter_to_Smith.pdf",
            },
            {
                "start_page": 6,
                "end_page": 9,
                "document_type": "Report",
                "suggested_filename": "Annual_Report_2024.pdf",
            },
        ]

        created_segments = []
        for segment in segments:
            create_response = client.post(
                f"/api/splits/{session_id}/segments",
                json=segment,
            )
            create_data = assert_api_response(create_response)
            created_segments.append(create_data["segment_id"])

        # Step 7: Generate preview for first segment
        with patch.object(SplittingService, "generate_preview") as mock_preview:
            mock_preview.return_value = {
                "segment_id": created_segments[0],
                "preview_type": "png",
                "pages_included": 2,
                "images": [
                    "data:image/png;base64,mock1",
                    "data:image/png;base64,mock2",
                ],
                "metadata": {"total_pages": 3},
            }

            preview_response = client.post(
                f"/api/splits/{session_id}/preview/{created_segments[0]}",
                json={"max_pages": 2},
            )
            preview_data = assert_api_response(preview_response)
            assert preview_data["pages_included"] == 2

        # Step 8: Execute split
        with patch.object(SplittingService, "execute_split") as mock_execute:
            mock_execute.return_value = "split-456"

            split_response = client.post(
                f"/api/splits/{session_id}/execute",
                json={
                    "create_zip": True,
                    "generate_manifest": True,
                },
            )

            split_data = assert_api_response(split_response)
            split_id = split_data["split_id"]

        # Step 9: Check split progress
        with patch.object(SplittingService, "get_split_status") as mock_status:
            mock_status.return_value = {
                "split_id": split_id,
                "session_id": session_id,
                "status": "completed",
                "progress": 1.0,
                "files_created": 3,
                "elapsed_time": 3.0,
                "total_segments": 3,
            }

            progress_response = client.get(f"/api/splits/{split_id}/progress")
            progress_data = assert_api_response(progress_response)
            assert progress_data["status"] == "completed"
            assert progress_data["files_created"] == 3

        # Step 10: Get results
        temp_dir = Path(tempfile.gettempdir())
        output_files = [
            temp_dir / "Invoice_2024_001.pdf",
            temp_dir / "Letter_to_Smith.pdf",
            temp_dir / "Annual_Report_2024.pdf",
        ]

        # Create mock files
        for f in output_files:
            f.write_bytes(b"Mock PDF content")

        with patch.object(SplittingService, "get_split_status") as mock_status:
            mock_status.return_value = {
                "split_id": split_id,
                "session_id": session_id,
                "status": "completed",
                "elapsed_time": 5.0,
            }

            with patch.object(SplittingService, "get_split_results") as mock_results:
                mock_results.return_value = {
                    "output_files": [str(f) for f in output_files],
                    "zip_file": str(temp_dir / "output.zip"),
                    "manifest_file": str(temp_dir / "manifest.json"),
                }

                results_response = client.get(f"/api/splits/{split_id}/results")
                results_data = assert_api_response(results_response)
            assert results_data["files_created"] == 3
            assert len(results_data["output_files"]) == 3

        # Cleanup
        for f in output_files:
            if f.exists():
                f.unlink()

    def test_workflow_with_websocket(self, client: TestClient, test_pdf_path: Path):
        """Test workflow with WebSocket progress monitoring."""
        # Upload file
        with open(test_pdf_path, "rb") as f:
            upload_response = client.post(
                "/api/upload/file",
                files={"file": ("test.pdf", f, "application/pdf")},
            )

        upload_data = assert_api_response(upload_response)
        upload_id = upload_data["upload_id"]

        # Create session
        session_response = client.post(
            "/api/sessions/create",
            json={"upload_id": upload_id},
        )

        session_data = assert_api_response(session_response)
        session_id = session_data["session_id"]

        # Connect WebSocket and monitor progress
        with client.websocket_connect("/ws") as websocket:
            # Get connected message
            connected_msg = websocket.receive_json()
            assert connected_msg["type"] == "connected"

            # Subscribe to session
            websocket.send_json(
                {
                    "type": "subscribe",
                    "session_id": session_id,
                    "include_previews": True,
                }
            )

            # Start detection (would trigger progress updates in real scenario)
            with patch.object(DetectionService, "start_detection") as mock_start:
                mock_start.return_value = "detection-789"

                client.post(
                    "/api/detection/start",
                    json={
                        "upload_id": upload_id,
                        "detector_type": "embeddings",
                    },
                )

    def test_error_handling_workflow(self, client: TestClient):
        """Test error handling throughout the workflow."""
        # Test upload error - invalid file
        response = client.post(
            "/api/upload/file",
            files={"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")},
        )
        assert response.status_code == 415

        # Test session creation with invalid upload
        response = client.post(
            "/api/sessions/create",
            json={"upload_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert response.status_code == 422

        # Test detection with no session
        response = client.post(
            "/api/detection/start",
            json={
                "upload_id": "00000000-0000-0000-0000-000000000000",
                "detector_type": "embeddings",
            },
        )
        assert response.status_code == 404

    def test_concurrent_operations(self, client: TestClient, test_pdf_path: Path):
        """Test handling concurrent operations."""
        # Upload multiple files
        upload_ids = []
        for i in range(3):
            with open(test_pdf_path, "rb") as f:
                response = client.post(
                    "/api/upload/file",
                    files={"file": (f"test_{i}.pdf", f, "application/pdf")},
                )
                data = assert_api_response(response)
                upload_ids.append(data["upload_id"])

        # Create sessions for each
        session_ids = []
        for upload_id in upload_ids:
            response = client.post(
                "/api/sessions/create",
                json={"upload_id": upload_id},
            )
            data = assert_api_response(response)
            session_ids.append(data["session_id"])

        # Verify all sessions exist
        response = client.get("/api/sessions/")
        data = assert_api_response(response)
        assert data["total_count"] >= 3

        found_sessions = [s["session_id"] for s in data["sessions"]]
        for session_id in session_ids:
            assert session_id in found_sessions

    @pytest.mark.parametrize("detector_type", ["embeddings", "heuristic"])
    def test_different_detectors(
        self, client: TestClient, uploaded_file: dict, detector_type: str
    ):
        """Test workflow with different detector types."""
        # Create session
        session_response = client.post(
            "/api/sessions/create",
            json={"upload_id": uploaded_file["upload_id"]},
        )
        assert_api_response(session_response)

        # Start detection with specific detector
        with patch.object(DetectionService, "start_detection") as mock_start:
            mock_start.return_value = f"detection-{detector_type}"

            response = client.post(
                "/api/detection/start",
                json={
                    "upload_id": uploaded_file["upload_id"],
                    "detector_type": detector_type,
                },
            )

            data = assert_api_response(response)
            assert data["detector_type"] == detector_type

    def test_session_expiration_handling(self, client: TestClient, uploaded_file: dict):
        """Test handling of session expiration."""
        # Create session with short expiration
        response = client.post(
            "/api/sessions/create",
            json={
                "upload_id": uploaded_file["upload_id"],
                "expires_in_hours": 1,  # Minimum
            },
        )

        session_data = assert_api_response(response)
        session_id = session_data["session_id"]

        # Extend session
        response = client.post(f"/api/sessions/{session_id}/extend?hours=24")
        data = assert_api_response(response)
        assert data["success"] is True

    def test_large_pdf_handling(self, client: TestClient, temp_dir: Path):
        """Test handling of large PDFs."""
        # Create a larger PDF
        large_pdf = temp_dir / "large.pdf"
        create_test_pdf(
            num_pages=50,  # 50 pages
            output_path=large_pdf,
            include_text=True,
        )

        # Upload
        with open(large_pdf, "rb") as f:
            response = client.post(
                "/api/upload/file",
                files={"file": ("large.pdf", f, "application/pdf")},
            )

        data = assert_api_response(response)
        assert data["total_pages"] == 50

        # The processing would take longer but the API should handle it

    def test_api_versioning(self, client: TestClient):
        """Test API versioning and compatibility."""
        # Current API endpoints should work
        response = client.get("/api/sessions/")
        assert response.status_code == 200

        # Health check should work
        response = client.get("/health")
        assert response.status_code == 200

    def test_cleanup_operations(
        self, client: TestClient, uploaded_file: dict, test_session: dict
    ):
        """Test cleanup and deletion operations."""
        upload_id = uploaded_file["upload_id"]
        session_id = test_session["session_id"]

        # Delete session
        response = client.delete(f"/api/sessions/{session_id}")
        assert_api_response(response)

        # Verify session is cancelled
        response = client.get(f"/api/sessions/{session_id}")
        data = assert_api_response(response)
        assert data["status"] == "cancelled"

        # Delete upload
        response = client.delete(f"/api/upload/{upload_id}")
        assert_api_response(response)

        # Verify upload is gone
        response = client.get(f"/api/upload/{upload_id}/status")
        assert response.status_code == 404
