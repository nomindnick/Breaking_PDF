"""
Health Check Endpoints

Provides health status and system information endpoints.
"""
import os
import time
from datetime import datetime
from typing import Any, Dict

import psutil
from fastapi import APIRouter, Depends
from sqlalchemy import text

from pdf_splitter.api.config import config
from pdf_splitter.api.models.responses import APIResponse
from pdf_splitter.splitting.session_manager import SessionManager

router = APIRouter(prefix="/api/health", tags=["health"])

# Track application start time
APP_START_TIME = time.time()


@router.get("", response_model=APIResponse)
async def health_check() -> APIResponse:
    """
    Basic health check endpoint.

    Returns:
        Simple health status
    """
    return APIResponse(success=True, message="API is healthy", timestamp=datetime.now())


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with system information.

    Returns:
        Comprehensive system status including:
        - API version and uptime
        - System resources (CPU, memory, disk)
        - Database connectivity
        - File storage status
        - Active sessions count
    """
    current_time = time.time()
    uptime_seconds = current_time - APP_START_TIME

    # Get system information
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(config.upload_dir))

    # Check database connectivity
    db_status = await _check_database()

    # Check file storage
    storage_status = await _check_storage()

    # Get session statistics
    session_stats = await _get_session_stats()

    return {
        "status": "healthy"
        if db_status["connected"] and storage_status["writable"]
        else "degraded",
        "timestamp": datetime.now().isoformat(),
        "api": {
            "version": config.api_version,
            "uptime_seconds": round(uptime_seconds, 2),
            "uptime_human": _format_uptime(uptime_seconds),
            "environment": "development" if config.debug else "production",
        },
        "system": {
            "cpu": {"percent": cpu_percent, "cores": psutil.cpu_count()},
            "memory": {
                "total_mb": round(memory.total / 1024 / 1024),
                "used_mb": round(memory.used / 1024 / 1024),
                "available_mb": round(memory.available / 1024 / 1024),
                "percent": memory.percent,
            },
            "disk": {
                "total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
                "used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
                "free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
                "percent": disk.percent,
            },
        },
        "services": {
            "database": db_status,
            "storage": storage_status,
            "sessions": session_stats,
        },
        "configuration": {
            "max_upload_size_mb": config.max_upload_size / 1024 / 1024,
            "session_timeout_hours": config.session_timeout / 3600,
            "max_concurrent_processes": config.max_concurrent_processes,
        },
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Kubernetes-style readiness probe.

    Checks if the application is ready to serve requests.

    Returns:
        Status code 200 if ready, 503 if not ready
    """
    # Check critical dependencies
    db_status = await _check_database()
    storage_status = await _check_storage()

    is_ready = db_status["connected"] and storage_status["writable"]

    return {
        "ready": is_ready,
        "checks": {
            "database": db_status["connected"],
            "storage": storage_status["writable"],
        },
    }


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Kubernetes-style liveness probe.

    Simple check to see if the application is running.

    Returns:
        Always returns 200 unless the application is completely broken
    """
    return {"alive": True, "timestamp": datetime.now().isoformat()}


async def _check_database() -> Dict[str, Any]:
    """Check database connectivity."""
    try:
        # Try to create a session manager (which connects to DB)
        session_manager = SessionManager(str(config.session_db_path))

        # Try a simple query
        with session_manager.db_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        return {
            "connected": True,
            "type": "sqlite",
            "path": str(config.session_db_path),
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


async def _check_storage() -> Dict[str, Any]:
    """Check file storage accessibility."""
    try:
        # Check if upload directory exists and is writable
        upload_dir = config.upload_dir
        output_dir = config.output_dir

        upload_exists = upload_dir.exists()
        output_exists = output_dir.exists()

        # Try to write a test file
        test_file = upload_dir / ".health_check"
        test_file.write_text("health check")
        test_file.unlink()

        return {
            "writable": True,
            "upload_dir": str(upload_dir),
            "output_dir": str(output_dir),
            "upload_dir_exists": upload_exists,
            "output_dir_exists": output_exists,
        }
    except Exception as e:
        return {"writable": False, "error": str(e)}


async def _get_session_stats() -> Dict[str, Any]:
    """Get session statistics."""
    try:
        session_manager = SessionManager(str(config.session_db_path))

        # Get counts by status
        all_sessions = session_manager.list_sessions()
        active_count = sum(
            1 for s in all_sessions if s.status in ["processing", "confirmed"]
        )
        completed_count = sum(1 for s in all_sessions if s.status == "complete")

        return {
            "total": len(all_sessions),
            "active": active_count,
            "completed": completed_count,
            "available": True,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def _format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format."""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")

    return " ".join(parts) if parts else "< 1m"
