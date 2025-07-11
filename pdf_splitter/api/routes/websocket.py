"""WebSocket endpoint for real-time updates.

This module provides the WebSocket endpoint for clients to receive
real-time progress updates.
"""

from uuid import uuid4

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from pdf_splitter.api.services.progress_service import (
    ProgressService,
    get_progress_service,
)
from pdf_splitter.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    progress_service: ProgressService = Depends(get_progress_service),
):
    """Handle WebSocket endpoint for real-time updates.

    Args:
        websocket: FastAPI WebSocket instance
        progress_service: Progress service instance
    """
    # Generate client ID
    client_id = str(uuid4())

    try:
        # Handle connection
        await progress_service.handle_connection(websocket, client_id)
    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")


@router.websocket("/ws/{session_id}")
async def websocket_session_endpoint(
    websocket: WebSocket,
    session_id: str,
    progress_service: ProgressService = Depends(get_progress_service),
):
    """Handle WebSocket endpoint with automatic session subscription.

    Args:
        websocket: FastAPI WebSocket instance
        session_id: Session to automatically subscribe to
        progress_service: Progress service instance
    """
    # Generate client ID
    client_id = str(uuid4())

    try:
        # Accept connection
        await websocket.accept()

        # Create connection and auto-subscribe
        await progress_service.handle_connection(websocket, client_id)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
