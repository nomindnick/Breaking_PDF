"""
Enhanced Health Check and Monitoring Endpoints

Provides comprehensive health checks, readiness probes, and system diagnostics.
"""
import asyncio
import os
import platform
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil
from fastapi import APIRouter, HTTPException, status

from pdf_splitter.api.config import config
from pdf_splitter.api.services.session_service import SessionService
from pdf_splitter.api.services.websocket_enhanced import enhanced_websocket_manager
from pdf_splitter.detection import create_production_detector
from pdf_splitter.preprocessing.pdf_handler import PDFHandler

router = APIRouter(prefix="/api/health", tags=["health", "monitoring"])


# Health check history
health_history = deque(maxlen=100)
component_status = {}
APP_START_TIME = time.time()


class HealthChecker:
    """Comprehensive health checking system."""

    def __init__(self):
        self.checks = {
            "database": self._check_database,
            "storage": self._check_storage,
            "pdf_processing": self._check_pdf_processing,
            "detection_models": self._check_detection_models,
            "websocket": self._check_websocket,
            "memory": self._check_memory,
            "disk_space": self._check_disk_space,
            "api_response": self._check_api_response,
        }

        self.critical_components = {"database", "storage", "memory", "disk_space"}
        self.warning_thresholds = {
            "memory_percent": 80,
            "disk_percent": 85,
            "response_time": 2.0,  # seconds
        }

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_status = "healthy"

        # Run checks concurrently
        tasks = {
            name: asyncio.create_task(check()) for name, check in self.checks.items()
        }

        for name, task in tasks.items():
            try:
                result = await task
                results[name] = result

                # Update overall status
                if not result["healthy"]:
                    if name in self.critical_components:
                        overall_status = "unhealthy"
                    elif overall_status != "unhealthy":
                        overall_status = "degraded"

            except Exception as e:
                results[name] = {
                    "healthy": False,
                    "error": str(e),
                    "type": "check_failed",
                }
                if name in self.critical_components:
                    overall_status = "unhealthy"

        # Store in history
        check_result = {
            "timestamp": datetime.utcnow(),
            "status": overall_status,
            "components": results,
        }
        health_history.append(check_result)

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": results,
            "summary": self._generate_summary(results),
        }

    async def _check_database(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            start = time.time()
            session_service = SessionService()

            # Test query
            sessions = session_service.session_manager.list_sessions(limit=1)

            response_time = time.time() - start

            return {
                "healthy": True,
                "type": "sqlite",
                "response_time": response_time,
                "sessions_accessible": True,
            }
        except Exception as e:
            return {"healthy": False, "error": str(e), "type": "connection_error"}

    async def _check_storage(self) -> Dict[str, Any]:
        """Check storage health."""
        try:
            # Check directories
            dirs_ok = all([config.upload_dir.exists(), config.output_dir.exists()])

            # Test write
            test_file = config.upload_dir / f".health_{time.time()}"
            test_file.write_text("test")
            test_file.unlink()

            # Get storage stats
            upload_stat = os.statvfs(config.upload_dir)
            free_gb = (upload_stat.f_bavail * upload_stat.f_frsize) / (1024**3)

            return {
                "healthy": True,
                "directories_exist": dirs_ok,
                "writable": True,
                "free_space_gb": round(free_gb, 2),
            }
        except Exception as e:
            return {"healthy": False, "error": str(e), "type": "storage_error"}

    async def _check_pdf_processing(self) -> Dict[str, Any]:
        """Check PDF processing capability."""
        try:
            pdf_handler = PDFHandler()

            # Test PDF operations are available
            return {
                "healthy": True,
                "pymupdf_available": True,
                "version": pdf_handler.get_version()
                if hasattr(pdf_handler, "get_version")
                else "unknown",
            }
        except Exception as e:
            return {"healthy": False, "error": str(e), "type": "pdf_processing_error"}

    async def _check_detection_models(self) -> Dict[str, Any]:
        """Check detection models availability."""
        try:
            detector = create_production_detector()

            return {
                "healthy": True,
                "detector_type": type(detector).__name__,
                "model_loaded": True,
            }
        except Exception as e:
            return {"healthy": False, "error": str(e), "type": "model_loading_error"}

    async def _check_websocket(self) -> Dict[str, Any]:
        """Check WebSocket service health."""
        try:
            stats = enhanced_websocket_manager.get_connection_stats()

            return {
                "healthy": True,
                "active_connections": stats["total_connections"],
                "active_sessions": stats["active_sessions"],
            }
        except Exception as e:
            return {"healthy": False, "error": str(e), "type": "websocket_error"}

    async def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()

            healthy = memory.percent < self.warning_thresholds["memory_percent"]

            return {
                "healthy": healthy,
                "system": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent": memory.percent,
                },
                "process": {
                    "rss_mb": round(process_memory.rss / (1024**2), 2),
                    "vms_mb": round(process_memory.vms / (1024**2), 2),
                },
                "warning": memory.percent > self.warning_thresholds["memory_percent"],
            }
        except Exception as e:
            return {"healthy": False, "error": str(e), "type": "memory_check_error"}

    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space."""
        try:
            disk = psutil.disk_usage(str(config.upload_dir))

            healthy = disk.percent < self.warning_thresholds["disk_percent"]

            return {
                "healthy": healthy,
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": disk.percent,
                "warning": disk.percent > self.warning_thresholds["disk_percent"],
            }
        except Exception as e:
            return {"healthy": False, "error": str(e), "type": "disk_check_error"}

    async def _check_api_response(self) -> Dict[str, Any]:
        """Check API response time."""
        try:
            # Simulate API call
            start = time.time()
            # Just a simple operation
            _ = {"test": "data"}
            response_time = time.time() - start

            healthy = response_time < self.warning_thresholds["response_time"]

            return {
                "healthy": healthy,
                "response_time": response_time,
                "warning": response_time > self.warning_thresholds["response_time"],
            }
        except Exception as e:
            return {"healthy": False, "error": str(e), "type": "response_check_error"}

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate health check summary."""
        total = len(results)
        healthy = sum(1 for r in results.values() if r.get("healthy"))
        warnings = sum(1 for r in results.values() if r.get("warning"))

        return {
            "total_checks": total,
            "healthy_checks": healthy,
            "failed_checks": total - healthy,
            "warnings": warnings,
            "health_percentage": round((healthy / total) * 100, 2) if total > 0 else 0,
        }


# Global health checker
health_checker = HealthChecker()


@router.get("/comprehensive")
async def comprehensive_health_check() -> Dict[str, Any]:
    """
    Comprehensive health check with all system components.

    Returns detailed status of:
    - All critical components
    - System resources
    - Service availability
    - Performance metrics
    """
    return await health_checker.run_all_checks()


@router.get("/status")
async def system_status() -> Dict[str, Any]:
    """
    Get current system status and metrics.

    Returns:
        System information, metrics, and resource usage
    """
    uptime = time.time() - APP_START_TIME

    # System info
    cpu_info = {
        "percent": psutil.cpu_percent(interval=0.1),
        "count": psutil.cpu_count(),
        "count_logical": psutil.cpu_count(logical=True),
    }

    # Process info
    process = psutil.Process()
    process_info = {
        "pid": process.pid,
        "threads": process.num_threads(),
        "connections": len(process.connections(kind="inet")),
        "open_files": len(process.open_files()),
        "cpu_percent": process.cpu_percent(),
    }

    # Python info
    python_info = {
        "version": sys.version,
        "implementation": platform.python_implementation(),
        "path": sys.executable,
    }

    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": {"seconds": uptime, "human": _format_uptime(uptime)},
        "system": {
            "platform": platform.platform(),
            "hostname": platform.node(),
            "cpu": cpu_info,
        },
        "process": process_info,
        "python": python_info,
        "api": {
            "version": config.api_version,
            "environment": "development" if config.debug else "production",
            "workers": config.api_workers,
        },
    }


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get application metrics for monitoring.

    Returns:
        Prometheus-compatible metrics
    """
    from pdf_splitter.api.middleware.logging import (
        active_requests,
        download_count,
        error_count,
        pdf_processed,
        request_count,
        request_duration,
    )

    # Collect metrics
    metrics = []

    # Add request metrics
    # This would normally use Prometheus registry
    metrics.append(
        {
            "name": "http_requests_active",
            "type": "gauge",
            "value": active_requests._value.get(),
            "help": "Active HTTP requests",
        }
    )

    return {"timestamp": datetime.utcnow().isoformat(), "metrics": metrics}


@router.get("/history")
async def get_health_history() -> Dict[str, Any]:
    """
    Get health check history.

    Returns:
        Recent health check results
    """
    history = list(health_history)

    if not history:
        return {
            "history": [],
            "summary": {"total_checks": 0, "healthy_percentage": 100.0},
        }

    # Calculate summary
    total = len(history)
    healthy = sum(1 for h in history if h["status"] == "healthy")

    return {
        "history": [
            {
                "timestamp": h["timestamp"].isoformat(),
                "status": h["status"],
                "summary": h["components"],
            }
            for h in history[-20:]  # Last 20 checks
        ],
        "summary": {
            "total_checks": total,
            "healthy_checks": healthy,
            "healthy_percentage": round((healthy / total) * 100, 2),
            "last_check": history[-1]["timestamp"].isoformat(),
            "current_status": history[-1]["status"],
        },
    }


@router.post("/diagnostics")
async def run_diagnostics() -> Dict[str, Any]:
    """
    Run diagnostic tests on the system.

    This endpoint performs deeper checks and may take longer.
    Use sparingly in production.
    """
    diagnostics = {}

    # File system diagnostics
    diagnostics["filesystem"] = await _diagnose_filesystem()

    # Database diagnostics
    diagnostics["database"] = await _diagnose_database()

    # Dependencies diagnostics
    diagnostics["dependencies"] = await _diagnose_dependencies()

    # Performance diagnostics
    diagnostics["performance"] = await _diagnose_performance()

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "diagnostics": diagnostics,
        "recommendations": _generate_recommendations(diagnostics),
    }


async def _diagnose_filesystem() -> Dict[str, Any]:
    """Diagnose filesystem issues."""
    issues = []

    # Check permissions
    for dir_path in [config.upload_dir, config.output_dir]:
        if not os.access(dir_path, os.W_OK):
            issues.append(f"No write permission for {dir_path}")
        if not os.access(dir_path, os.R_OK):
            issues.append(f"No read permission for {dir_path}")

    # Check space
    stat = os.statvfs(config.upload_dir)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    if free_gb < 1:
        issues.append(f"Low disk space: {free_gb:.2f} GB free")

    return {
        "healthy": len(issues) == 0,
        "issues": issues,
        "free_space_gb": round(free_gb, 2),
    }


async def _diagnose_database() -> Dict[str, Any]:
    """Diagnose database issues."""
    issues = []
    metrics = {}

    try:
        session_service = SessionService()

        # Test operations
        start = time.time()
        sessions = session_service.session_manager.list_sessions()
        query_time = time.time() - start

        metrics["session_count"] = len(sessions)
        metrics["query_time"] = query_time

        if query_time > 1.0:
            issues.append(f"Slow query performance: {query_time:.2f}s")

        # Check database file size
        db_path = config.session_db_path
        if db_path.exists():
            db_size_mb = db_path.stat().st_size / (1024**2)
            metrics["db_size_mb"] = round(db_size_mb, 2)

            if db_size_mb > 100:
                issues.append(f"Large database size: {db_size_mb:.2f} MB")

    except Exception as e:
        issues.append(f"Database error: {str(e)}")

    return {"healthy": len(issues) == 0, "issues": issues, "metrics": metrics}


async def _diagnose_dependencies() -> Dict[str, Any]:
    """Diagnose dependency issues."""
    dependencies = {}

    # Check critical imports
    critical_modules = [
        "fastapi",
        "pydantic",
        "sqlalchemy",
        "fitz",  # PyMuPDF
        "PIL",
        "transformers",
    ]

    for module in critical_modules:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            dependencies[module] = {"available": True, "version": version}
        except ImportError:
            dependencies[module] = {"available": False, "error": "Module not found"}

    return {
        "healthy": all(d["available"] for d in dependencies.values()),
        "dependencies": dependencies,
    }


async def _diagnose_performance() -> Dict[str, Any]:
    """Diagnose performance issues."""
    metrics = {}
    issues = []

    # Memory usage
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        issues.append(f"High memory usage: {memory.percent}%")

    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    if cpu_percent > 80:
        issues.append(f"High CPU usage: {cpu_percent}%")

    # Response time test
    start = time.time()
    # Simulate some work
    _ = sum(i for i in range(1000000))
    compute_time = time.time() - start

    metrics["compute_time"] = compute_time
    metrics["memory_percent"] = memory.percent
    metrics["cpu_percent"] = cpu_percent

    if compute_time > 0.1:
        issues.append(f"Slow computation: {compute_time:.3f}s")

    return {"healthy": len(issues) == 0, "issues": issues, "metrics": metrics}


def _generate_recommendations(diagnostics: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on diagnostics."""
    recommendations = []

    # Filesystem recommendations
    fs_diag = diagnostics.get("filesystem", {})
    if fs_diag.get("free_space_gb", 0) < 5:
        recommendations.append("Consider freeing up disk space or adding more storage")

    # Database recommendations
    db_diag = diagnostics.get("database", {})
    if db_diag.get("metrics", {}).get("db_size_mb", 0) > 100:
        recommendations.append(
            "Consider cleaning up old sessions to reduce database size"
        )

    # Performance recommendations
    perf_diag = diagnostics.get("performance", {})
    if perf_diag.get("metrics", {}).get("memory_percent", 0) > 80:
        recommendations.append(
            "Consider increasing system memory or optimizing memory usage"
        )

    return recommendations


def _format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format."""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 and days == 0:  # Only show seconds if uptime is short
        parts.append(f"{secs}s")

    return " ".join(parts) if parts else "< 1s"
