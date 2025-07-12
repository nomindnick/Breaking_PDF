"""
PDF Splitter API Main Application.

FastAPI application initialization and configuration.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pdf_splitter.api.config import config
from pdf_splitter.api.middleware import (
    AdaptiveRateLimitMiddleware,
    APIKeyAuthMiddleware,
    AuditLoggingMiddleware,
    EnhancedErrorHandlingMiddleware,
    ErrorRecoveryMiddleware,
    InputSanitizationMiddleware,
    MetricsExportMiddleware,
    PerformanceMonitoringMiddleware,
    RequestValidationMiddleware,
    SlidingWindowStrategy,
    StructuredLoggingMiddleware,
    TokenBucketStrategy,
)
from pdf_splitter.api.routers import (
    docs,
    download,
    health,
    health_enhanced,
    process,
    results,
    sessions,
    splits,
    upload,
    websocket,
    websocket_enhanced,
)
from pdf_splitter.api.routers.docs import custom_openapi_schema

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()), format=config.log_format
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {config.api_title} v{config.api_version}")
    logger.info(f"Environment: {'Development' if config.debug else 'Production'}")
    logger.info(f"Upload directory: {config.upload_dir}")
    logger.info(f"Output directory: {config.output_dir}")

    # Ensure required directories exist
    config.upload_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.log_file:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    logger.info("Shutting down API")


# Create FastAPI application
app = FastAPI(
    title=config.api_title,
    description=config.api_description,
    version=config.api_version,
    debug=config.debug,
    lifespan=lifespan,
    docs_url=None,  # We'll use custom docs router
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

app.openapi = lambda: custom_openapi_schema(app)

# Add middleware in correct order (executed in reverse order)
# 1. Error handling and recovery (outermost)
app.add_middleware(ErrorRecoveryMiddleware)
app.add_middleware(EnhancedErrorHandlingMiddleware)

# 2. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=config.cors_allow_credentials,
    allow_methods=config.cors_allow_methods,
    allow_headers=config.cors_allow_headers,
)

# 3. Security middleware
app.add_middleware(RequestValidationMiddleware)
app.add_middleware(InputSanitizationMiddleware)

# 4. Authentication (if enabled)
if config.api_key_enabled:
    app.add_middleware(APIKeyAuthMiddleware)

# 5. Rate limiting
if config.rate_limit_enabled:
    # Use different strategies for different endpoints
    upload_strategy = TokenBucketStrategy(
        tokens=10, refill_rate=0.0333
    )  # 10 uploads per 5 min
    download_strategy = SlidingWindowStrategy(
        requests=100, window=3600
    )  # 100 downloads per hour
    default_strategy = SlidingWindowStrategy(
        requests=60, window=60
    )  # 60 requests per minute

    app.add_middleware(
        AdaptiveRateLimitMiddleware,
        default_strategy=default_strategy,
        endpoint_strategies={
            "/api/upload": upload_strategy,
            "/api/download": download_strategy,
        },
    )

# 6. Monitoring and logging
app.add_middleware(MetricsExportMiddleware)
app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(PerformanceMonitoringMiddleware)
app.add_middleware(StructuredLoggingMiddleware)

# Include routers
app.include_router(health.router)
app.include_router(health_enhanced.router)  # Enhanced health checks
app.include_router(upload.router)
app.include_router(process.router)
app.include_router(sessions.router)
app.include_router(splits.router)
app.include_router(results.router)  # Results and analytics
app.include_router(download.router)  # File downloads
app.include_router(websocket.router)
app.include_router(websocket_enhanced.router)  # Enhanced WebSocket endpoints
app.include_router(docs.router)  # API documentation

# Mount static files for frontend
frontend_static = Path(__file__).parent.parent / "frontend" / "static"
if frontend_static.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_static)), name="static")
    logger.info(f"Mounted static files from {frontend_static}")

# Set up templates for frontend
frontend_templates = Path(__file__).parent.parent / "frontend" / "templates"
if frontend_templates.exists():
    templates = Jinja2Templates(directory=str(frontend_templates))
    logger.info(f"Loaded templates from {frontend_templates}")


@app.get("/")
async def root():
    """Root endpoint - redirects to API documentation or returns basic info."""
    return {
        "name": config.api_title,
        "version": config.api_version,
        "status": "running",
        "documentation": "/api/docs" if config.debug else "Disabled in production",
        "health": "/api/health",
    }


# Error handlers for common HTTP errors
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return {
        "error": {
            "type": "not_found",
            "message": "The requested resource was not found",
            "path": str(request.url.path),
        }
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return {
        "error": {
            "type": "internal_error",
            "message": "An internal server error occurred",
            "request_id": getattr(request.state, "request_id", None),
        }
    }


if __name__ == "__main__":
    import uvicorn

    # Run the application
    uvicorn.run(
        "pdf_splitter.api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.reload,
        workers=config.api_workers if not config.reload else 1,
        log_level=config.log_level.lower(),
    )
