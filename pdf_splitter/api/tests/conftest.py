"""Test configuration and fixtures for API tests.

This module provides shared fixtures and utilities for testing the API module.
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncGenerator, Generator, List
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

from pdf_splitter.api.dependencies import (
    UploadManager,
    get_pdf_config,
    get_session_manager,
    get_upload_manager,
)
from pdf_splitter.api.router import api_router
from pdf_splitter.api.services.detection_service import DetectionService
from pdf_splitter.api.services.progress_service import ProgressService
from pdf_splitter.api.services.splitting_service import SplittingService
from pdf_splitter.api.services.upload_service import UploadService
from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.base_detector import (
    BoundaryResult,
    BoundaryType,
    DetectorType,
    ProcessedPage,
)
from pdf_splitter.splitting.models import DocumentSegment, SplitProposal, SplitSession
from pdf_splitter.splitting.session_manager import SplitSessionManager
from pdf_splitter.test_utils import create_test_pdf


@pytest.fixture
def pdf_config() -> PDFConfig:
    """Create test PDF configuration."""
    return PDFConfig(
        max_file_size_mb=10,
        max_pages=100,
        default_dpi=150,
    )


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_app(pdf_config: PDFConfig, temp_dir: Path) -> FastAPI:
    """Create test FastAPI application."""
    app = FastAPI()

    # Override dependencies
    app.dependency_overrides[get_pdf_config] = lambda: pdf_config

    # Create test upload manager
    upload_manager = UploadManager(upload_dir=temp_dir / "uploads")
    app.dependency_overrides[get_upload_manager] = lambda: upload_manager

    # Create test session manager
    session_manager = SplitSessionManager(
        config=pdf_config, db_path=temp_dir / "sessions.db"
    )
    app.dependency_overrides[get_session_manager] = lambda: session_manager

    # Include routes
    app.include_router(api_router)

    # Add basic endpoints for testing
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "PDF Splitter Test",
            "version": "0.1.0",
            "status": "operational",
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


@pytest.fixture
def client(test_app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
async def async_client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


@pytest.fixture
def test_pdf_path(temp_dir: Path) -> Path:
    """Create test PDF file."""
    pdf_path = temp_dir / "test_document.pdf"
    create_test_pdf(
        num_pages=5,
        output_path=pdf_path,
        page_size=(612, 792),  # Letter size
        include_text=True,
    )
    return pdf_path


@pytest.fixture
def mock_processed_pages() -> list[ProcessedPage]:
    """Create mock processed pages."""
    return [
        ProcessedPage(
            page_number=0,
            text="Invoice #12345\nDate: 2024-01-01\nBill To: Test Company",
            page_type="SEARCHABLE",
        ),
        ProcessedPage(
            page_number=1,
            text="Page 2 of Invoice\nItem Description\nQuantity: 10",
            page_type="SEARCHABLE",
        ),
        ProcessedPage(
            page_number=2,
            text="Dear John,\nThis is a test letter.\nSincerely, Test",
            page_type="SEARCHABLE",
        ),
        ProcessedPage(
            page_number=3,
            text="Page 4 - Letter continued\nMore content here",
            page_type="SEARCHABLE",
        ),
        ProcessedPage(
            page_number=4,
            text="Report Summary\nExecutive Summary\nConclusions",
            page_type="SEARCHABLE",
        ),
    ]


@pytest.fixture
def mock_boundaries() -> list[BoundaryResult]:
    """Create mock boundary results."""
    return [
        BoundaryResult(
            page_number=0,
            boundary_type=BoundaryType.DOCUMENT_START,
            confidence=0.9,
            detector_type=DetectorType.EMBEDDINGS,
            reasoning="Invoice header detected",
        ),
        BoundaryResult(
            page_number=2,
            boundary_type=BoundaryType.DOCUMENT_START,
            confidence=0.85,
            detector_type=DetectorType.EMBEDDINGS,
            reasoning="Letter greeting detected",
        ),
        BoundaryResult(
            page_number=4,
            boundary_type=BoundaryType.DOCUMENT_START,
            confidence=0.8,
            detector_type=DetectorType.EMBEDDINGS,
            reasoning="Report header detected",
        ),
    ]


@pytest.fixture
def mock_proposal(
    test_pdf_path: Path, mock_boundaries: list[BoundaryResult]
) -> SplitProposal:
    """Create mock split proposal."""
    segments = [
        DocumentSegment(
            start_page=0,
            end_page=1,
            document_type="Invoice",
            suggested_filename="Invoice_12345.pdf",
            confidence=0.9,
            summary="Invoice #12345 dated 2024-01-01",
        ),
        DocumentSegment(
            start_page=2,
            end_page=3,
            document_type="Letter",
            suggested_filename="Letter_to_John.pdf",
            confidence=0.85,
            summary="Letter to John",
        ),
        DocumentSegment(
            start_page=4,
            end_page=4,
            document_type="Report",
            suggested_filename="Report_Summary.pdf",
            confidence=0.8,
            summary="Executive summary report",
        ),
    ]

    return SplitProposal(
        pdf_path=test_pdf_path,
        total_pages=5,
        segments=segments,
        detection_results=mock_boundaries,
    )


@pytest.fixture
def mock_session(mock_proposal: SplitProposal) -> SplitSession:
    """Create mock split session."""
    return SplitSession(
        session_id=str(uuid4()),
        proposal=mock_proposal,
        status="pending",
        expires_at=datetime.now() + timedelta(hours=24),
    )


@pytest.fixture
def mock_upload_service(pdf_config: PDFConfig, temp_dir: Path) -> UploadService:
    """Create mock upload service."""
    service = UploadService(config=pdf_config, upload_dir=temp_dir)
    return service


@pytest.fixture
def mock_detection_service(pdf_config: PDFConfig) -> DetectionService:
    """Create mock detection service."""
    service = DetectionService(config=pdf_config)

    # Mock the detector
    mock_detector = MagicMock()
    mock_detector.detect_boundaries = MagicMock(return_value=[])
    service.get_detector = MagicMock(return_value=mock_detector)

    return service


@pytest.fixture
def mock_splitting_service(pdf_config: PDFConfig, temp_dir: Path) -> SplittingService:
    """Create mock splitting service."""
    service = SplittingService(config=pdf_config, output_base_dir=temp_dir)
    return service


@pytest.fixture
def mock_progress_service() -> ProgressService:
    """Create mock progress service."""
    service = ProgressService()
    service.broadcast_progress = AsyncMock()
    service.broadcast_stage_complete = AsyncMock()
    service.broadcast_error = AsyncMock()
    return service


@pytest.fixture
async def uploaded_file(client: TestClient, test_pdf_path: Path) -> dict:
    """Upload a test file and return upload data."""
    with open(test_pdf_path, "rb") as f:
        response = client.post(
            "/api/upload/file",
            files={"file": ("test_document.pdf", f, "application/pdf")},
        )

    assert response.status_code == 200
    return response.json()


@pytest.fixture
def test_session(client: TestClient, uploaded_file: dict) -> dict:
    """Create a test session."""
    response = client.post(
        "/api/sessions/create",
        json={
            "upload_id": uploaded_file["upload_id"],
            "session_name": "Test Session",
            "expires_in_hours": 24,
        },
    )

    assert response.status_code == 200
    return response.json()


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket for testing."""
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_json = AsyncMock()
    ws.close = AsyncMock()
    ws.client_state = MagicMock()
    return ws


# Utility functions for tests


def assert_api_response(response, expected_status: int = 200):
    """Assert API response is successful."""
    assert response.status_code == expected_status
    data = response.json()
    if expected_status == 200:
        assert data.get("success", True)
    return data


def create_mock_file(filename: str = "test.pdf", size: int = 1024) -> bytes:
    """Create mock file content."""
    return b"Mock PDF content" * (size // 16)


async def wait_for_condition(
    condition_func, timeout: float = 5.0, interval: float = 0.1
):
    """Wait for a condition to become true."""
    start_time = asyncio.get_event_loop().time()
    while not condition_func():
        if asyncio.get_event_loop().time() - start_time > timeout:
            raise TimeoutError("Condition not met within timeout")
        await asyncio.sleep(interval)
