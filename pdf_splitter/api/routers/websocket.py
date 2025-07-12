"""
WebSocket Endpoints

Handles WebSocket connections for real-time progress updates.
"""
import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from pdf_splitter.api.services.session_service import SessionService
from pdf_splitter.api.services.websocket_service import websocket_manager

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time session updates.

    Clients can connect to this endpoint to receive real-time updates about
    PDF processing progress, status changes, and errors.

    Messages sent to clients:
    - connection: Initial connection confirmation
    - progress: Processing progress updates
    - status: Status changes
    - error: Error notifications

    Messages received from clients:
    - ping: Keep-alive ping (responds with pong)

    Args:
        websocket: WebSocket connection
        session_id: Session ID to subscribe to
    """
    # Accept connection
    await websocket_manager.connect(websocket, session_id)

    try:
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages with timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=30.0  # 30 second timeout
                )

                # Handle ping/pong for keep-alive
                if data == "ping":
                    await websocket.send_text("pong")

                # Could handle other client messages here

            except asyncio.TimeoutError:
                # Send ping to check if client is still connected
                try:
                    await websocket.send_text("ping")
                except Exception:
                    # Connection is dead
                    break

    except WebSocketDisconnect:
        # Client disconnected normally
        pass
    except Exception as e:
        # Log unexpected errors
        print(f"WebSocket error for session {session_id}: {e}")
    finally:
        # Clean up connection
        websocket_manager.disconnect(websocket, session_id)


@router.get("/api/websocket/stats")
async def get_websocket_stats() -> dict:
    """
    Get WebSocket connection statistics.

    Returns:
        Statistics about active WebSocket connections
    """
    return {
        "total_connections": websocket_manager.get_connection_count(),
        "active_sessions": websocket_manager.get_active_sessions(),
        "sessions_with_connections": [
            {
                "session_id": session_id,
                "connection_count": websocket_manager.get_connection_count(session_id),
            }
            for session_id in websocket_manager.get_active_sessions()
        ],
    }
