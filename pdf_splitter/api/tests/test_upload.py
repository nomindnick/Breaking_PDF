"""Tests for upload endpoints.

This module tests the file upload functionality including validation,
storage, and status checking.
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pdf_splitter.api.tests.conftest import assert_api_response, create_mock_file


class TestUploadEndpoints:
    """Test upload API endpoints."""

    def test_upload_file_success(self, client: TestClient, test_pdf_path: Path):
        """Test successful file upload."""
        with open(test_pdf_path, "rb") as f:
            response = client.post(
                "/api/upload/file",
                files={"file": ("test_document.pdf", f, "application/pdf")},
            )

        data = assert_api_response(response)
        assert "upload_id" in data
        assert data["file_name"] == "test_document.pdf"
        assert data["file_size"] > 0
        assert data["total_pages"] == 5
        assert data["status"] == "uploaded"
        assert data["processing_time"] > 0

    def test_upload_file_validate_only(self, client: TestClient, test_pdf_path: Path):
        """Test file validation without storage."""
        with open(test_pdf_path, "rb") as f:
            response = client.post(
                "/api/upload/file?validate_only=true",
                files={"file": ("test_document.pdf", f, "application/pdf")},
            )

        data = assert_api_response(response)
        assert data["status"] == "validated"
        assert data["upload_id"] is not None  # ID generated but file not stored

    def test_upload_file_invalid_type(self, client: TestClient):
        """Test upload with invalid file type."""
        content = b"Not a PDF file"
        file = io.BytesIO(content)

        response = client.post(
            "/api/upload/file",
            files={"file": ("test.txt", file, "text/plain")},
        )

        assert response.status_code == 415
        assert "not allowed" in response.json()["detail"]

    def test_upload_file_invalid_extension(self, client: TestClient):
        """Test upload with wrong file extension."""
        content = create_mock_file()
        file = io.BytesIO(content)

        response = client.post(
            "/api/upload/file",
            files={"file": ("test.txt", file, "application/pdf")},
        )

        assert response.status_code == 415
        assert "not allowed" in response.json()["detail"]

    def test_upload_file_too_large(self, client: TestClient, pdf_config):
        """Test upload with file exceeding size limit."""
        # Create large mock content
        max_size = int(pdf_config.max_file_size_mb * 1024 * 1024)
        content = b"0" * (max_size + 1024)  # Exceed by 1KB
        file = io.BytesIO(content)

        response = client.post(
            "/api/upload/file",
            files={"file": ("large.pdf", file, "application/pdf")},
        )

        assert response.status_code == 413
        assert "exceeds maximum" in response.json()["detail"]

    def test_validate_request(self, client: TestClient):
        """Test file validation endpoint."""
        response = client.post(
            "/api/upload/validate",
            json={
                "file_name": "test.pdf",
                "file_size": 1024 * 1024,  # 1MB
                "content_type": "application/pdf",
            },
        )

        data = assert_api_response(response)
        assert data["status"] == "validated"
        assert data["file_name"] == "test.pdf"

    def test_validate_request_invalid_size(self, client: TestClient, pdf_config):
        """Test validation with invalid file size."""
        max_size = int(pdf_config.max_file_size_mb * 1024 * 1024)

        response = client.post(
            "/api/upload/validate",
            json={
                "file_name": "test.pdf",
                "file_size": max_size + 1024,
                "content_type": "application/pdf",
            },
        )

        assert response.status_code == 413
        assert "exceeds maximum" in response.json()["detail"]

    def test_validate_request_invalid_type(self, client: TestClient):
        """Test validation with invalid content type."""
        response = client.post(
            "/api/upload/validate",
            json={
                "file_name": "test.pdf",
                "file_size": 1024,
                "content_type": "image/jpeg",
            },
        )

        assert response.status_code == 415
        assert "not allowed" in response.json()["detail"]

    def test_get_upload_status(self, client: TestClient, uploaded_file: dict):
        """Test getting upload status."""
        upload_id = uploaded_file["upload_id"]

        response = client.get(f"/api/upload/{upload_id}/status")
        data = assert_api_response(response)

        assert data["upload_id"] == upload_id
        assert data["status"] == "uploaded"
        assert data["file_name"] == uploaded_file["file_name"]
        assert data["file_size"] == uploaded_file["file_size"]
        assert "created_at" in data
        assert "expires_at" in data

    def test_get_upload_status_not_found(self, client: TestClient):
        """Test getting status for non-existent upload."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        response = client.get(f"/api/upload/{fake_id}/status")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_delete_upload(self, client: TestClient, uploaded_file: dict):
        """Test deleting an upload."""
        upload_id = uploaded_file["upload_id"]

        # Delete upload
        response = client.delete(f"/api/upload/{upload_id}")
        data = assert_api_response(response)
        assert data["success"] is True

        # Verify it's deleted
        response = client.get(f"/api/upload/{upload_id}/status")
        assert response.status_code == 404

    def test_delete_upload_not_found(self, client: TestClient):
        """Test deleting non-existent upload."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        response = client.delete(f"/api/upload/{fake_id}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.parametrize(
        "content_type",
        [
            "application/pdf",
            "application/x-pdf",
            "application/acrobat",
            "text/pdf",
        ],
    )
    def test_upload_allowed_content_types(
        self, client: TestClient, test_pdf_path: Path, content_type: str
    ):
        """Test upload with various allowed content types."""
        with open(test_pdf_path, "rb") as f:
            response = client.post(
                "/api/upload/file",
                files={"file": ("test.pdf", f, content_type)},
            )

        assert_api_response(response)

    def test_upload_corrupt_pdf(self, client: TestClient):
        """Test upload with corrupt PDF content."""
        # Create invalid PDF content
        content = b"Invalid PDF content that is not a real PDF"
        file = io.BytesIO(content)

        response = client.post(
            "/api/upload/file",
            files={"file": ("corrupt.pdf", file, "application/pdf")},
        )

        # Should fail during processing
        assert response.status_code == 400
        assert "Failed to process PDF" in response.json()["detail"]
