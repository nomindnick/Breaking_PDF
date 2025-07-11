"""Main API router configuration.

This module combines all API routes into a single router.
"""

from fastapi import APIRouter

from pdf_splitter.api.routes import (
    detection,
    pages,
    sessions,
    splitting,
    upload,
    websocket,
)

# Create main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(upload.router)
api_router.include_router(sessions.router)
api_router.include_router(detection.router)
api_router.include_router(splitting.router)
api_router.include_router(websocket.router)
api_router.include_router(pages.router)
