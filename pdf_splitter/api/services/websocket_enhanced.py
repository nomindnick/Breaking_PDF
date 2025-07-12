"""
Enhanced WebSocket Service

Adds heartbeat, authentication, and advanced connection management.
"""
import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Set

from fastapi import Query, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from pdf_splitter.api.models.websocket import (
    ConnectionMessage,
    ErrorMessage,
    HeartbeatMessage,
    ProcessingStage,
    ProgressMessage,
    StatusMessage,
    WebSocketEventType,
    WebSocketMessage,
)
from pdf_splitter.api.services.session_service import SessionService
from pdf_splitter.api.utils.exceptions import SessionNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""

    websocket: WebSocket
    session_id: str
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    auth_token: Optional[str] = None
    client_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    heartbeat_sequence: int = 0

    @property
    def connection_duration(self) -> timedelta:
        """Get duration of connection."""
        return datetime.utcnow() - self.connected_at

    @property
    def is_alive(self) -> bool:
        """Check if connection is still alive based on heartbeat."""
        heartbeat_timeout = timedelta(seconds=90)  # 3x heartbeat interval
        return datetime.utcnow() - self.last_heartbeat < heartbeat_timeout


class EnhancedWebSocketManager:
    """Enhanced WebSocket manager with heartbeat and authentication."""

    def __init__(self, heartbeat_interval: int = 30):
        # Store connections by session_id and client_id
        self.connections: Dict[str, Dict[str, ConnectionInfo]] = defaultdict(dict)
        self.session_service = SessionService()
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}

        # Event handlers
        self.event_handlers: Dict[WebSocketEventType, List[Callable]] = defaultdict(
            list
        )

        # Connection limits
        self.max_connections_per_session = 10
        self.max_total_connections = 1000

    def register_handler(self, event_type: WebSocketEventType, handler: Callable):
        """Register an event handler for a specific event type."""
        self.event_handlers[event_type].append(handler)

    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
        auth_token: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> ConnectionInfo:
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: WebSocket connection
            session_id: Session ID to subscribe to
            auth_token: Optional authentication token
            client_id: Optional client identifier

        Returns:
            ConnectionInfo object

        Raises:
            ValueError: If connection limits are exceeded
        """
        # Check connection limits
        if len(self.connections[session_id]) >= self.max_connections_per_session:
            await websocket.close(code=1008, reason="Too many connections for session")
            raise ValueError(f"Session {session_id} has reached connection limit")

        total_connections = sum(len(conns) for conns in self.connections.values())
        if total_connections >= self.max_total_connections:
            await websocket.close(code=1008, reason="Server connection limit reached")
            raise ValueError("Server has reached total connection limit")

        # Validate session exists
        try:
            session_details = self.session_service.get_session_details(session_id)
        except SessionNotFoundError:
            await websocket.close(code=1008, reason="Invalid session")
            raise ValueError(f"Session {session_id} not found")

        # Accept connection
        await websocket.accept()

        # Generate client ID if not provided
        if not client_id:
            client_id = f"client_{datetime.utcnow().timestamp()}"

        # Create connection info
        conn_info = ConnectionInfo(
            websocket=websocket,
            session_id=session_id,
            auth_token=auth_token,
            client_id=client_id,
            metadata={
                "user_agent": websocket.headers.get("user-agent", "unknown"),
                "origin": websocket.headers.get("origin", "unknown"),
            },
        )

        # Register connection
        self.connections[session_id][client_id] = conn_info

        # Send connection confirmation
        connection_msg = ConnectionMessage()
        connection_msg.data.update(
            {
                "session_id": session_id,
                "client_id": client_id,
                "session_status": session_details["status"],
                "heartbeat_interval": self.heartbeat_interval,
            }
        )

        await self._send_message(conn_info, connection_msg)

        # Start heartbeat monitoring
        if session_id not in self.heartbeat_tasks:
            self.heartbeat_tasks[session_id] = asyncio.create_task(
                self._monitor_heartbeats(session_id)
            )

        logger.info(f"WebSocket connected: session={session_id}, client={client_id}")

        # Trigger connection event handlers
        await self._trigger_handlers(WebSocketEventType.CONNECTION, conn_info, {})

        return conn_info

    async def disconnect(self, conn_info: ConnectionInfo):
        """
        Disconnect and cleanup a WebSocket connection.

        Args:
            conn_info: Connection information
        """
        session_id = conn_info.session_id
        client_id = conn_info.client_id

        # Remove from connections
        if session_id in self.connections:
            self.connections[session_id].pop(client_id, None)

            # Clean up empty sessions
            if not self.connections[session_id]:
                del self.connections[session_id]

                # Cancel heartbeat task
                if session_id in self.heartbeat_tasks:
                    self.heartbeat_tasks[session_id].cancel()
                    del self.heartbeat_tasks[session_id]

        logger.info(f"WebSocket disconnected: session={session_id}, client={client_id}")

        # Trigger disconnection event handlers
        await self._trigger_handlers(WebSocketEventType.DISCONNECTION, conn_info, {})

    async def handle_message(self, conn_info: ConnectionInfo, message: dict):
        """
        Handle incoming WebSocket message.

        Args:
            conn_info: Connection information
            message: Incoming message dictionary
        """
        msg_type = message.get("type")

        if msg_type == WebSocketEventType.HEARTBEAT:
            # Handle heartbeat
            await self._handle_heartbeat(conn_info, message)
        elif msg_type == WebSocketEventType.AUTH_REQUEST:
            # Handle authentication
            await self._handle_auth(conn_info, message)
        else:
            # Trigger custom handlers
            event_type = WebSocketEventType(msg_type) if msg_type else None
            if event_type:
                await self._trigger_handlers(event_type, conn_info, message)

    async def _handle_heartbeat(self, conn_info: ConnectionInfo, message: dict):
        """Handle heartbeat message."""
        conn_info.last_heartbeat = datetime.utcnow()
        conn_info.heartbeat_sequence += 1

        # Send pong response
        pong_msg = WebSocketMessage(
            type=WebSocketEventType.PONG,
            data={
                "sequence": conn_info.heartbeat_sequence,
                "server_time": datetime.utcnow().isoformat(),
                "client_time": message.get("data", {}).get("client_time"),
            },
        )

        await self._send_message(conn_info, pong_msg)

    async def _handle_auth(self, conn_info: ConnectionInfo, message: dict):
        """Handle authentication request."""
        auth_data = message.get("data", {})
        token = auth_data.get("token")

        # TODO: Implement actual authentication logic
        # For now, just accept all auth requests
        auth_success = bool(token)

        if auth_success:
            conn_info.auth_token = token
            auth_msg = WebSocketMessage(
                type=WebSocketEventType.AUTH_SUCCESS,
                data={"message": "Authentication successful", "authenticated": True},
            )
        else:
            auth_msg = WebSocketMessage(
                type=WebSocketEventType.AUTH_FAILURE,
                data={"message": "Authentication failed", "authenticated": False},
            )

        await self._send_message(conn_info, auth_msg)

    async def _monitor_heartbeats(self, session_id: str):
        """Monitor heartbeats for all connections in a session."""
        while session_id in self.connections:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                # Check all connections
                disconnected = []
                for client_id, conn_info in self.connections[session_id].items():
                    if not conn_info.is_alive:
                        logger.warning(
                            f"Connection timeout: session={session_id}, client={client_id}"
                        )
                        disconnected.append(conn_info)

                # Disconnect stale connections
                for conn_info in disconnected:
                    await self.disconnect(conn_info)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")

    async def _send_message(self, conn_info: ConnectionInfo, message: WebSocketMessage):
        """Send a message to a specific connection."""
        if conn_info.websocket.application_state == WebSocketState.CONNECTED:
            try:
                await conn_info.websocket.send_json(message.dict())
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                await self.disconnect(conn_info)

    async def _trigger_handlers(
        self, event_type: WebSocketEventType, conn_info: ConnectionInfo, data: dict
    ):
        """Trigger registered event handlers."""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                await handler(conn_info, data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")

    async def broadcast_to_session(
        self,
        session_id: str,
        message: WebSocketMessage,
        exclude_clients: Optional[Set[str]] = None,
    ):
        """
        Broadcast a message to all connections for a session.

        Args:
            session_id: Session ID to broadcast to
            message: Message to broadcast
            exclude_clients: Optional set of client IDs to exclude
        """
        if session_id not in self.connections:
            return

        exclude_clients = exclude_clients or set()
        tasks = []

        for client_id, conn_info in self.connections[session_id].items():
            if client_id not in exclude_clients:
                tasks.append(self._send_message(conn_info, message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send_progress_update(
        self,
        session_id: str,
        stage: ProcessingStage,
        progress: float,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Send a progress update to all clients watching a session."""
        progress_msg = ProgressMessage(
            data={
                "session_id": session_id,
                "stage": stage.value,
                "progress": progress,
                "total": 100,
                "message": message or "",
                "details": details or {},
            }
        )

        await self.broadcast_to_session(session_id, progress_msg)

    async def send_status_update(
        self,
        session_id: str,
        status: str,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Send a status update to all clients watching a session."""
        status_msg = StatusMessage(
            data={
                "session_id": session_id,
                "status": status,
                "message": message or "",
                "metadata": metadata or {},
            }
        )

        await self.broadcast_to_session(session_id, status_msg)

    async def send_error(
        self,
        session_id: str,
        error_code: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
    ):
        """Send an error message to all clients watching a session."""
        error_msg = ErrorMessage(
            data={
                "session_id": session_id,
                "error_code": error_code,
                "error_message": error_message,
                "details": details or {},
                "recoverable": recoverable,
            }
        )

        await self.broadcast_to_session(session_id, error_msg)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about active connections."""
        total_connections = sum(len(conns) for conns in self.connections.values())

        session_stats = {}
        for session_id, connections in self.connections.items():
            session_stats[session_id] = {
                "connection_count": len(connections),
                "clients": list(connections.keys()),
                "oldest_connection": min(
                    (c.connected_at for c in connections.values()), default=None
                ),
            }

        return {
            "total_connections": total_connections,
            "active_sessions": len(self.connections),
            "sessions": session_stats,
            "heartbeat_tasks": len(self.heartbeat_tasks),
        }


# Global enhanced WebSocket manager instance
enhanced_websocket_manager = EnhancedWebSocketManager()


async def enhanced_progress_callback(
    session_id: str,
    stage: str,
    progress: float,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """
    Enhanced progress callback for process service integration.
    """
    try:
        stage_enum = ProcessingStage(stage)
    except ValueError:
        stage_enum = ProcessingStage.COMPLETE

    await enhanced_websocket_manager.send_progress_update(
        session_id=session_id,
        stage=stage_enum,
        progress=progress,
        message=message,
        details=details,
    )
