#!/usr/bin/env python3
"""PDF Splitter Application - Main Entry Point.

An intelligent PDF splitter that automatically identifies and separates
individual documents within large, multi-document PDF files.
"""
from pathlib import Path

import click
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

    # Import and include routers
    # TODO: Import routers once api.routes is implemented
    # from pdf_splitter.api import routes
    # app.include_router(routes.router)

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

        # TODO: Initialize OCR engines, LLM models, etc.

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on application shutdown."""
        logger.info("Shutting down PDF Splitter application")
        # TODO: Cleanup resources

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "operational",
        }

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
