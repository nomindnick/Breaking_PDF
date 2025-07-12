"""
WebSocket Service

Manages WebSocket connections and broadcasts progress updates.
"""
import json
from datetime import datetime
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from pdf_splitter.api.services.session_service import SessionService


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        # Store active connections by session_id
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.session_service = SessionService()

    async def connect(self, websocket: WebSocket, session_id: str):
        """
        Accept a new WebSocket connection for a session.

        Args:
            websocket: WebSocket connection
            session_id: Session ID to subscribe to
        """
        await websocket.accept()

        # Add to connections
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        self.active_connections[session_id].add(websocket)

        # Send initial status
        try:
            details = self.session_service.get_session_details(session_id)
            await self.send_personal_message(
                {
                    "type": "connection",
                    "data": {
                        "session_id": session_id,
                        "status": details["status"],
                        "message": "Connected to session updates",
                    },
                },
                websocket,
            )
        except Exception as e:
            await self.send_personal_message(
                {
                    "type": "error",
                    "data": {"message": f"Failed to get session status: {str(e)}"},
                },
                websocket,
            )

    def disconnect(self, websocket: WebSocket, session_id: str):
        """
        Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
            session_id: Session ID
        """
        if session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)

            # Clean up empty session entries
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """
        Send a message to a specific WebSocket connection.

        Args:
            message: Message dictionary to send
            websocket: Target WebSocket connection
        """
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(message)
            except Exception:
                # Connection might be closed
                pass

    async def broadcast_to_session(self, session_id: str, message: dict):
        """
        Broadcast a message to all connections for a session.

        Args:
            session_id: Session ID to broadcast to
            message: Message dictionary to broadcast
        """
        if session_id in self.active_connections:
            # Copy set to avoid modification during iteration
            connections = list(self.active_connections[session_id])
            disconnected = []

            for connection in connections:
                try:
                    await self.send_personal_message(message, connection)
                except Exception:
                    # Mark for removal
                    disconnected.append(connection)

            # Clean up disconnected clients
            for conn in disconnected:
                self.disconnect(conn, session_id)

    async def send_progress_update(
        self,
        session_id: str,
        stage: str,
        progress: float,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Send a progress update to all clients watching a session.

        Args:
            session_id: Session ID
            stage: Current processing stage
            progress: Progress percentage (0-100)
            message: Optional status message
            details: Optional additional details
        """
        update_message = {
            "type": "progress",
            "data": {
                "session_id": session_id,
                "stage": stage,
                "progress": progress,
                "message": message,
                "details": details or {},
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        await self.broadcast_to_session(session_id, update_message)

    async def send_status_update(
        self, session_id: str, status: str, message: Optional[str] = None
    ):
        """
        Send a status update to all clients watching a session.

        Args:
            session_id: Session ID
            status: New status
            message: Optional status message
        """
        update_message = {
            "type": "status",
            "data": {
                "session_id": session_id,
                "status": status,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        await self.broadcast_to_session(session_id, update_message)

    async def send_error(
        self, session_id: str, error: str, details: Optional[Dict[str, Any]] = None
    ):
        """
        Send an error message to all clients watching a session.

        Args:
            session_id: Session ID
            error: Error message
            details: Optional error details
        """
        error_message = {
            "type": "error",
            "data": {
                "session_id": session_id,
                "error": error,
                "details": details or {},
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        await self.broadcast_to_session(session_id, error_message)

    def get_connection_count(self, session_id: Optional[str] = None) -> int:
        """
        Get the number of active connections.

        Args:
            session_id: Optional session ID to get count for

        Returns:
            Number of active connections
        """
        if session_id:
            return len(self.active_connections.get(session_id, []))
        else:
            return sum(len(conns) for conns in self.active_connections.values())

    def get_active_sessions(self) -> list:
        """
        Get list of sessions with active connections.

        Returns:
            List of session IDs with active connections
        """
        return list(self.active_connections.keys())


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


async def websocket_progress_callback(
    session_id: str,
    stage: str,
    progress: float,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """
    Progress callback function for process service integration.

    This function can be passed to ProcessingService to send real-time updates.
    """
    await websocket_manager.send_progress_update(
        session_id=session_id,
        stage=stage,
        progress=progress,
        message=message,
        details=details,
    )
