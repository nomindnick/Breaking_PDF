"""Tests for splitting endpoints.

This module tests split management functionality including proposal viewing,
segment manipulation, preview generation, and split execution.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from pdf_splitter.api.services.splitting_service import SplittingService
from pdf_splitter.api.tests.conftest import assert_api_response


class TestSplittingEndpoints:
    """Test splitting API endpoints."""

    def test_get_split_proposal(self, client: TestClient, test_session: dict):
        """Test getting split proposal."""
        session_id = test_session["session_id"]

        response = client.get(f"/api/splits/{session_id}/proposal")
        data = assert_api_response(response)

        assert data["session_id"] == session_id
        assert "pdf_path" in data
        assert data["total_pages"] == test_session["total_pages"]
        assert "segments" in data
        assert len(data["segments"]) >= 1  # Default single segment
        assert data["total_segments"] == len(data["segments"])

        # Check segment structure
        segment = data["segments"][0]
        assert "segment_id" in segment
        assert "start_page" in segment
        assert "end_page" in segment
        assert "document_type" in segment
        assert "suggested_filename" in segment
        assert "confidence" in segment
        assert "is_user_defined" in segment

    def test_update_segment(self, client: TestClient, test_session: dict):
        """Test updating a segment."""
        session_id = test_session["session_id"]

        # Get proposal to find segment ID
        proposal_response = client.get(f"/api/splits/{session_id}/proposal")
        proposal_data = assert_api_response(proposal_response)
        segment_id = proposal_data["segments"][0]["segment_id"]

        # Update segment
        update_data = {
            "suggested_filename": "Updated_Document.pdf",
            "document_type": "Report",
            "summary": "Updated summary",
        }

        response = client.put(
            f"/api/splits/{session_id}/segments/{segment_id}",
            json=update_data,
        )

        data = assert_api_response(response)
        assert data["session_id"] == session_id
        assert data["segment_id"] == segment_id
        assert set(data["updated_fields"]) == set(update_data.keys())
        assert data["segment"]["suggested_filename"] == "Updated_Document.pdf"
        assert data["segment"]["document_type"] == "Report"

    def test_update_segment_page_range(self, client: TestClient, test_session: dict):
        """Test updating segment page range."""
        session_id = test_session["session_id"]

        # Get proposal
        proposal_response = client.get(f"/api/splits/{session_id}/proposal")
        proposal_data = assert_api_response(proposal_response)
        segment_id = proposal_data["segments"][0]["segment_id"]

        # Update page range
        response = client.put(
            f"/api/splits/{session_id}/segments/{segment_id}",
            json={
                "start_page": 0,
                "end_page": 2,
            },
        )

        data = assert_api_response(response)
        assert data["segment"]["start_page"] == 0
        assert data["segment"]["end_page"] == 2
        assert data["segment"]["page_count"] == 3

    def test_update_segment_invalid_range(self, client: TestClient, test_session: dict):
        """Test updating segment with invalid page range."""
        session_id = test_session["session_id"]

        # Get proposal
        proposal_response = client.get(f"/api/splits/{session_id}/proposal")
        proposal_data = assert_api_response(proposal_response)
        segment_id = proposal_data["segments"][0]["segment_id"]

        # Try invalid range (end < start)
        response = client.put(
            f"/api/splits/{session_id}/segments/{segment_id}",
            json={
                "start_page": 3,
                "end_page": 1,
            },
        )

        assert response.status_code == 422

    def test_create_segment(self, client: TestClient, test_session: dict):
        """Test creating a new segment."""
        session_id = test_session["session_id"]

        # Create new segment
        segment_data = {
            "start_page": 1,
            "end_page": 2,
            "document_type": "Invoice",
            "suggested_filename": "New_Invoice.pdf",
            "summary": "New invoice document",
        }

        response = client.post(
            f"/api/splits/{session_id}/segments",
            json=segment_data,
        )

        data = assert_api_response(response)
        assert data["session_id"] == session_id
        assert "segment_id" in data
        assert data["segment"]["start_page"] == 1
        assert data["segment"]["end_page"] == 2
        assert data["segment"]["document_type"] == "Invoice"
        assert data["segment"]["is_user_defined"] is True
        assert data["segment"]["confidence"] == 1.0  # User-defined

    def test_create_segment_overlapping(self, client: TestClient, test_session: dict):
        """Test creating overlapping segments."""
        session_id = test_session["session_id"]

        # First segment: pages 0-2
        response1 = client.post(
            f"/api/splits/{session_id}/segments",
            json={
                "start_page": 0,
                "end_page": 2,
                "document_type": "Document1",
            },
        )
        assert_api_response(response1)

        # Try to create overlapping segment: pages 1-3
        response2 = client.post(
            f"/api/splits/{session_id}/segments",
            json={
                "start_page": 1,
                "end_page": 3,
                "document_type": "Document2",
            },
        )

        # Should fail due to overlap
        assert response2.status_code == 422

    def test_delete_segment(self, client: TestClient, test_session: dict):
        """Test deleting a segment."""
        session_id = test_session["session_id"]

        # Create a segment first
        create_response = client.post(
            f"/api/splits/{session_id}/segments",
            json={
                "start_page": 1,
                "end_page": 2,
                "document_type": "ToDelete",
            },
        )
        create_data = assert_api_response(create_response)
        segment_id = create_data["segment_id"]

        # Delete it
        response = client.delete(f"/api/splits/{session_id}/segments/{segment_id}")
        data = assert_api_response(response)
        assert data["success"] is True

        # Verify it's gone
        proposal_response = client.get(f"/api/splits/{session_id}/proposal")
        proposal_data = assert_api_response(proposal_response)

        segment_ids = [s["segment_id"] for s in proposal_data["segments"]]
        assert segment_id not in segment_ids

    def test_generate_preview(self, client: TestClient, test_session: dict):
        """Test generating segment preview."""
        session_id = test_session["session_id"]

        # Get segment ID
        proposal_response = client.get(f"/api/splits/{session_id}/proposal")
        proposal_data = assert_api_response(proposal_response)
        segment_id = proposal_data["segments"][0]["segment_id"]

        # Mock preview generation
        with patch.object(SplittingService, "generate_preview") as mock_preview:
            mock_preview.return_value = {
                "segment_id": segment_id,
                "preview_type": "png",
                "pages_included": 2,
                "images": [
                    "data:image/png;base64,iVBORw0KGgoAAAANS...",
                    "data:image/png;base64,iVBORw0KGgoAAAANS...",
                ],
                "metadata": {
                    "total_pages": 3,
                    "document_type": "Document",
                },
            }

            response = client.post(
                f"/api/splits/{session_id}/preview/{segment_id}",
                json={
                    "max_pages": 2,
                    "resolution": 150,
                    "format": "png",
                },
            )

            data = assert_api_response(response)
            assert data["segment_id"] == segment_id
            assert data["preview_type"] == "png"
            assert data["pages_included"] == 2
            assert len(data["images"]) == 2
            assert all(
                img.startswith("data:image/png;base64,") for img in data["images"]
            )

    def test_execute_split(self, client: TestClient, test_session: dict):
        """Test executing split operation."""
        session_id = test_session["session_id"]

        # Mock split execution
        with patch.object(SplittingService, "execute_split") as mock_execute:
            mock_execute.return_value = "split-id-123"

            response = client.post(
                f"/api/splits/{session_id}/execute",
                json={
                    "output_format": "pdf",
                    "compress": False,
                    "create_zip": True,
                    "preserve_metadata": True,
                    "generate_manifest": True,
                },
            )

            data = assert_api_response(response)
            assert data["session_id"] == session_id
            assert data["split_id"] == "split-id-123"
            assert data["status"] == "started"
            assert "estimated_time" in data

    def test_get_split_progress(self, client: TestClient):
        """Test getting split progress."""
        split_id = "split-id-123"

        # Mock progress
        with patch.object(SplittingService, "get_split_status") as mock_status:
            mock_status.return_value = {
                "split_id": split_id,
                "session_id": "test-session",
                "status": "splitting",
                "progress": 0.5,
                "current_segment": 2,
                "total_segments": 4,
                "files_created": 1,
                "elapsed_time": 5.0,
            }

            response = client.get(f"/api/splits/{split_id}/progress")
            data = assert_api_response(response)

            assert data["split_id"] == split_id
            assert data["status"] == "splitting"
            assert data["progress"] == 0.5
            assert data["current_segment"] == 2
            assert data["total_segments"] == 4
            assert data["files_created"] == 1

    def test_get_split_results(self, client: TestClient, temp_dir: Path):
        """Test getting split results."""
        split_id = "split-id-123"

        # Create mock output files
        output_files = [
            temp_dir / "Document1.pdf",
            temp_dir / "Document2.pdf",
        ]
        for f in output_files:
            f.write_bytes(b"Mock PDF content")

        # Mock results
        with patch.object(SplittingService, "get_split_status") as mock_status:
            mock_status.return_value = {
                "split_id": split_id,
                "session_id": "test-session",
                "status": "completed",
                "elapsed_time": 10.0,
            }

            with patch.object(SplittingService, "get_split_results") as mock_results:
                mock_results.return_value = {
                    "output_files": [str(f) for f in output_files],
                    "zip_file": str(temp_dir / "output.zip"),
                    "manifest_file": str(temp_dir / "manifest.json"),
                }

                response = client.get(f"/api/splits/{split_id}/results")
                data = assert_api_response(response)

                assert data["split_id"] == split_id
                assert data["status"] == "completed"
                assert data["files_created"] == 2
                assert len(data["output_files"]) == 2
                assert data["processing_time"] == 10.0
                assert "output_size_bytes" in data

    def test_download_file(self, client: TestClient, temp_dir: Path):
        """Test downloading split output file."""
        split_id = "split-id-123"
        filename = "Document1.pdf"

        # Create mock file
        output_file = temp_dir / filename
        output_file.write_bytes(b"Mock PDF content for download")

        # Mock results
        with patch.object(SplittingService, "get_split_results") as mock_results:
            mock_results.return_value = {
                "output_files": [str(output_file)],
            }

            response = client.get(f"/api/splits/{split_id}/download/{filename}")

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert response.content == b"Mock PDF content for download"

    def test_download_file_not_found(self, client: TestClient):
        """Test downloading non-existent file."""
        split_id = "split-id-123"

        with patch.object(SplittingService, "get_split_results") as mock_results:
            mock_results.return_value = {"output_files": []}

            response = client.get(f"/api/splits/{split_id}/download/nonexistent.pdf")
            assert response.status_code == 404

    @pytest.mark.parametrize("preview_format", ["png", "jpeg", "webp"])
    def test_preview_formats(
        self, client: TestClient, test_session: dict, preview_format: str
    ):
        """Test preview generation with different formats."""
        session_id = test_session["session_id"]

        # Get segment
        proposal_response = client.get(f"/api/splits/{session_id}/proposal")
        proposal_data = assert_api_response(proposal_response)
        segment_id = proposal_data["segments"][0]["segment_id"]

        with patch.object(SplittingService, "generate_preview") as mock_preview:
            mock_preview.return_value = {
                "segment_id": segment_id,
                "preview_type": preview_format,
                "pages_included": 1,
                "images": [f"data:image/{preview_format};base64,mock"],
                "metadata": {},
            }

            response = client.post(
                f"/api/splits/{session_id}/preview/{segment_id}",
                json={"format": preview_format},
            )

            data = assert_api_response(response)
            assert data["preview_type"] == preview_format
            assert data["images"][0].startswith(f"data:image/{preview_format}")

    def test_split_with_options(self, client: TestClient, test_session: dict):
        """Test split execution with various options."""
        session_id = test_session["session_id"]

        test_cases = [
            {"compress": True, "create_zip": False},
            {"compress": False, "create_zip": True},
            {"preserve_metadata": False, "generate_manifest": False},
            {"output_format": "pdf/a"},
        ]

        for options in test_cases:
            with patch.object(SplittingService, "execute_split") as mock_execute:
                mock_execute.return_value = "split-id"

                response = client.post(
                    f"/api/splits/{session_id}/execute",
                    json=options,
                )

                assert_api_response(response)

                # Verify options were passed
                call_args = mock_execute.call_args[1]
                for key, value in options.items():
                    assert call_args.get(key) == value
