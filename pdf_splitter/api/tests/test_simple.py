"""Simple test to verify API setup."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from pdf_splitter.api.tests.conftest import assert_api_response


def test_health_check(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint(client: TestClient):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert data["status"] == "operational"


def test_upload_with_mocked_handler(client: TestClient, temp_dir: Path):
    """Test upload with fully mocked PDF handler."""
    # Create a mock file
    test_file = temp_dir / "test.pdf"
    test_file.write_bytes(b"Mock PDF content")

    # Mock the entire PDFHandler
    with patch("pdf_splitter.api.services.upload_service.PDFHandler") as MockHandler:
        # Configure the mock
        mock_handler = MagicMock()

        # Create a mock context manager for load_pdf
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_handler)
        mock_context.__exit__ = MagicMock(return_value=None)
        mock_handler.load_pdf.return_value = mock_context

        # Set up page analysis
        mock_handler.page_count = 5
        mock_handler.get_page_type = MagicMock()

        # Create a mock PageType enum
        from enum import Enum

        class MockPageType(Enum):
            searchable = "searchable"
            image_based = "image_based"
            empty = "empty"
            mixed = "mixed"

        # Return searchable pages
        mock_handler.get_page_type.return_value = MockPageType.searchable

        MockHandler.return_value = mock_handler

        # Upload file
        with open(test_file, "rb") as f:
            response = client.post(
                "/api/upload/file",
                files={"file": ("test.pdf", f, "application/pdf")},
            )

        data = assert_api_response(response)
        assert data["upload_id"] is not None
        assert data["file_name"] == "test.pdf"
        assert data["total_pages"] == 5
        assert data["status"] == "uploaded"
