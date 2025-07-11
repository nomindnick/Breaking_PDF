#!/usr/bin/env python3
"""PDF Splitter Application - Main Entry Point.

An intelligent PDF splitter that automatically identifies and separates
individual documents within large, multi-document PDF files.
"""
from pathlib import Path

import click
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pdf_splitter.api.middleware import (
    ErrorHandlingMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)
from pdf_splitter.core.config import settings
from pdf_splitter.core.logging import get_logger, setup_logging

# Initialize logger
logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        description="Intelligent PDF document splitter with multi-signal detection",
    )

    # Add custom middleware (order matters - first added is outermost)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=120)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    static_dir = Path(__file__).parent / "pdf_splitter" / "frontend" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Configure templates
    templates_dir = Path(__file__).parent / "pdf_splitter" / "frontend" / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

    # Store templates in app state for use in routes
    app.state.templates = templates

    # Import and include routers
    from pdf_splitter.api.router import api_router

    app.include_router(api_router)

    @app.on_event("startup")
    async def startup_event():
        """Initialize application on startup."""
        logger.info(
            "Starting PDF Splitter application",
            version=settings.app_version,
            debug=settings.debug,
        )

        # Create necessary directories
        settings.create_directories()

        # Start progress service
        from pdf_splitter.api.services.progress_service import get_progress_service

        progress_service = get_progress_service()
        await progress_service.start()

        # TODO: Initialize OCR engines, LLM models, etc.

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on application shutdown."""
        logger.info("Shutting down PDF Splitter application")

        # Stop progress service
        from pdf_splitter.api.services.progress_service import get_progress_service

        progress_service = get_progress_service()
        await progress_service.stop()

        # TODO: Cleanup other resources

    @app.get("/")
    async def root(request: Request):
        """Root endpoint - renders the home page."""
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "app_name": settings.app_name,
                "app_version": settings.app_version,
            },
        )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


@click.command()
@click.option("--host", default=settings.host, help="Host to bind the server to")
@click.option(
    "--port", default=settings.port, type=int, help="Port to bind the server to"
)
@click.option(
    "--reload", is_flag=True, default=False, help="Enable auto-reload for development"
)
@click.option(
    "--workers", default=settings.workers, type=int, help="Number of worker processes"
)
def main(host: str, port: int, reload: bool, workers: int):
    """PDF Splitter Application.

    An intelligent tool for splitting multi-document PDFs into individual files.
    """
    # Setup logging
    setup_logging()

    # Log startup information
    logger.info(
        "Starting PDF Splitter server",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
    )

    # Run the server
    uvicorn.run(
        "main:create_app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        factory=True,
        log_config=None,  # We handle logging ourselves
    )


if __name__ == "__main__":
    main()
