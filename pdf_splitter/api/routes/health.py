"""Health check endpoints.

This module provides health check endpoints for monitoring the API.
"""

from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends

from pdf_splitter.api.dependencies import get_pdf_config
from pdf_splitter.core.config import PDFConfig

router = APIRouter(tags=["health"])


@router.get("/health")
@router.get("/api/health")
async def health_check(config: PDFConfig = Depends(get_pdf_config)) -> Dict:
    """Basic health check endpoint.

    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "pdf-splitter-api",
    }


@router.get("/health/ready")
@router.get("/api/health/ready")
async def readiness_check(config: PDFConfig = Depends(get_pdf_config)) -> Dict:
    """Readiness check endpoint.

    Checks if the service is ready to accept requests.

    Returns:
        Readiness status
    """
    # Check dependencies
    checks = {
        "config": config is not None,
        "upload_dir": config.upload_dir.exists() if config else False,
        "output_dir": config.output_dir.exists() if config else False,
    }

    all_ready = all(checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/live")
@router.get("/api/health/live")
async def liveness_check() -> Dict:
    """Liveness check endpoint.

    Simple check to verify the service is alive.

    Returns:
        Liveness status
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
    }
