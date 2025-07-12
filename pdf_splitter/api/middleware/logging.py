"""
Comprehensive Logging and Monitoring Middleware.

Provides structured logging, metrics collection, and monitoring integration.
"""
import logging
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from fastapi import Request, Response
from prometheus_client import Counter, Gauge, Histogram, Summary
from starlette.middleware.base import BaseHTTPMiddleware

from pdf_splitter.api.config import config

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)


# Prometheus metrics
request_count = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

request_duration = Histogram(
    "http_request_duration_seconds", "HTTP request duration", ["method", "endpoint"]
)

active_requests = Gauge("http_requests_active", "Active HTTP requests")

error_count = Counter(
    "http_errors_total", "Total HTTP errors", ["method", "endpoint", "error_type"]
)

# Business metrics
pdf_processed = Counter("pdfs_processed_total", "Total PDFs processed")

pdf_processing_duration = Summary(
    "pdf_processing_duration_seconds", "PDF processing duration"
)

download_count = Counter("file_downloads_total", "Total file downloads", ["file_type"])


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured logging of all requests."""

    def __init__(self, app):
        super().__init__(app)
        self.logger = structlog.get_logger()

        # Performance tracking
        self.slow_request_threshold = 1.0  # seconds
        self.very_slow_request_threshold = 5.0

    async def dispatch(self, request: Request, call_next):
        """Log request and response with structured data."""
        # Generate request ID if not present
        request_id = getattr(request.state, "request_id", None)

        # Start timing
        start_time = time.time()

        # Build request context
        request_context = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }

        # Add authentication info if present
        if hasattr(request.state, "api_key"):
            request_context["api_key_name"] = request.state.api_key.name

        # Log request
        self.logger.info("request_started", **request_context)

        # Track active requests
        active_requests.inc()

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Build response context
            response_context = {
                **request_context,
                "status_code": response.status_code,
                "duration": duration,
                "duration_ms": int(duration * 1000),
            }

            # Check for slow requests
            if duration > self.very_slow_request_threshold:
                self.logger.warning("very_slow_request", **response_context)
            elif duration > self.slow_request_threshold:
                self.logger.warning("slow_request", **response_context)
            else:
                self.logger.info("request_completed", **response_context)

            # Update metrics
            request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
            ).inc()

            request_duration.labels(
                method=request.method, endpoint=request.url.path
            ).observe(duration)

            return response

        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time

            # Log error
            error_context = {
                **request_context,
                "duration": duration,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }

            self.logger.error("request_failed", **error_context)

            # Update error metrics
            error_count.labels(
                method=request.method,
                endpoint=request.url.path,
                error_type=type(e).__name__,
            ).inc()

            raise

        finally:
            # Track active requests
            active_requests.dec()


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and analysis."""

    def __init__(self, app):
        super().__init__(app)

        # Performance tracking
        self.endpoint_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.global_stats = deque(maxlen=10000)

        # Alert thresholds
        self.alert_thresholds = {
            "p95_response_time": 2.0,  # seconds
            "error_rate": 0.05,  # 5%
            "requests_per_second": 100,
        }

    async def dispatch(self, request: Request, call_next):
        """Monitor request performance."""
        endpoint = f"{request.method} {request.url.path}"
        start_time = time.time()

        try:
            response = await call_next(request)
            duration = time.time() - start_time

            # Record stats
            self._record_stats(endpoint, duration, response.status_code)

            # Check for alerts
            self._check_alerts(endpoint)

            # Add performance headers
            response.headers["X-Response-Time"] = f"{duration:.3f}"

            return response

        except Exception as e:
            duration = time.time() - start_time
            self._record_stats(endpoint, duration, 500)
            raise

    def _record_stats(self, endpoint: str, duration: float, status_code: int):
        """Record performance statistics."""
        stat = {
            "timestamp": time.time(),
            "duration": duration,
            "status_code": status_code,
            "is_error": status_code >= 400,
        }

        self.endpoint_stats[endpoint].append(stat)
        self.global_stats.append(stat)

    def _check_alerts(self, endpoint: str):
        """Check for performance alerts."""
        stats = list(self.endpoint_stats[endpoint])

        if len(stats) < 100:
            return

        # Calculate metrics
        durations = [s["duration"] for s in stats]
        durations.sort()

        p95 = durations[int(len(durations) * 0.95)]
        error_rate = sum(1 for s in stats if s["is_error"]) / len(stats)

        # Check thresholds
        if p95 > self.alert_thresholds["p95_response_time"]:
            logger.warning(f"High P95 response time for {endpoint}: {p95:.2f}s")

        if error_rate > self.alert_thresholds["error_rate"]:
            logger.warning(f"High error rate for {endpoint}: {error_rate:.2%}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = {}

        for endpoint, endpoint_stats in self.endpoint_stats.items():
            if not endpoint_stats:
                continue

            durations = [s["duration"] for s in endpoint_stats]
            errors = sum(1 for s in endpoint_stats if s["is_error"])

            stats[endpoint] = {
                "count": len(endpoint_stats),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "p50_duration": sorted(durations)[len(durations) // 2],
                "p95_duration": sorted(durations)[int(len(durations) * 0.95)],
                "error_count": errors,
                "error_rate": errors / len(endpoint_stats),
            }

        return stats


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for security audit logging."""

    def __init__(self, app):
        super().__init__(app)
        self.audit_logger = structlog.get_logger("audit")

        # Actions to audit
        self.audit_actions = {
            "POST /api/upload": "file_upload",
            "POST /api/process": "pdf_process",
            "GET /api/download": "file_download",
            "POST /api/download/token": "token_create",
            "DELETE": "resource_delete",
        }

    async def dispatch(self, request: Request, call_next):
        """Log security-relevant actions."""
        # Check if this is an auditable action
        action = None

        # Check specific endpoints
        endpoint_key = f"{request.method} {request.url.path}"
        for pattern, action_name in self.audit_actions.items():
            if pattern in endpoint_key:
                action = action_name
                break

        # Process request
        response = await call_next(request)

        # Log audit event if applicable
        if action:
            await self._log_audit_event(request, response, action)

        return response

    async def _log_audit_event(self, request: Request, response: Response, action: str):
        """Log an audit event."""
        # Build audit context
        audit_context = {
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, "request_id", None),
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "status_code": response.status_code,
            "success": 200 <= response.status_code < 400,
        }

        # Add authentication info
        if hasattr(request.state, "api_key"):
            audit_context["api_key_name"] = request.state.api_key.name
        elif hasattr(request.state, "user_id"):
            audit_context["user_id"] = request.state.user_id

        # Add specific details based on action
        if action == "file_upload":
            # Could extract filename from request
            pass
        elif action == "file_download":
            # Extract filename from path
            parts = request.url.path.split("/")
            if len(parts) > 3:
                audit_context["filename"] = parts[-1]

        # Log audit event
        self.audit_logger.info("audit_event", **audit_context)


class MetricsExportMiddleware(BaseHTTPMiddleware):
    """Middleware for exporting metrics to external systems."""

    def __init__(self, app, export_interval: int = 60):
        super().__init__(app)
        self.export_interval = export_interval
        self.last_export = time.time()

        # Metrics buffer
        self.metrics_buffer = []

    async def dispatch(self, request: Request, call_next):
        """Collect and export metrics."""
        # Process request
        response = await call_next(request)

        # Check if we should export metrics
        if time.time() - self.last_export > self.export_interval:
            await self._export_metrics()
            self.last_export = time.time()

        return response

    async def _export_metrics(self):
        """Export metrics to external systems."""
        # This would integrate with systems like:
        # - Prometheus Push Gateway
        # - CloudWatch
        # - DataDog
        # - New Relic

        # Example: Export to CloudWatch
        # metrics = self._collect_metrics()
        # await cloudwatch_client.put_metrics(metrics)

        logger.info("Metrics exported")

    def _collect_metrics(self) -> List[Dict[str, Any]]:
        """Collect current metrics."""
        # Would collect from Prometheus registry
        # and format for external system
        return []
