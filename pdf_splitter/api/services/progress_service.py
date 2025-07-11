"""Service for managing progress updates and WebSocket connections.

This module provides business logic for tracking operation progress
and broadcasting updates to connected clients.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Optional, Set

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from pdf_splitter.api.models.websocket import (
    ConnectedMessage,
    DisconnectedMessage,
    ErrorMessage,
    MessageType,
    PongMessage,
    ProcessingStage,
    ProgressUpdate,
    ServerMessage,
    StageCompleteMessage,
    SubscribeMessage,
    UnsubscribeMessage,
)
from pdf_splitter.core.logging import get_logger

logger = get_logger(__name__)


class WebSocketConnection:
    """Manages a single WebSocket connection."""

    def __init__(self, websocket: WebSocket, client_id: str):
        """Initialize WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance
            client_id: Unique client identifier
        """
        self.websocket = websocket
        self.client_id = client_id
        self.subscribed_sessions: Set[str] = set()
        self.connected_at = datetime.now()
        self.last_ping = datetime.now()

    async def send_message(self, message: ServerMessage):
        """Send a message to the client.

        Args:
            message: Server message to send
        """
        try:
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.send_json(message.model_dump())
        except Exception as e:
            logger.error(f"Failed to send message to {self.client_id}: {str(e)}")
            raise

    async def close(self, reason: str = "Connection closed"):
        """Close the WebSocket connection.

        Args:
            reason: Reason for closing
        """
        try:
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.send_message(
                    DisconnectedMessage(
                        reason=reason,
                        reconnect_allowed=True,
                        reconnect_delay_seconds=5,
                    )
                )
                await self.websocket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket for {self.client_id}: {str(e)}")


class ProgressService:
    """Service for managing progress updates across all operations."""

    def __init__(self):
        """Initialize progress service."""
        self._connections: Dict[str, WebSocketConnection] = {}
        self._session_subscribers: Dict[
            str, Set[str]
        ] = {}  # session_id -> set of client_ids
        self._progress_callbacks: Dict[str, callable] = {}  # operation_id -> callback
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the progress service."""
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_disconnected())
        logger.info("Progress service started")

    async def stop(self):
        """Stop the progress service."""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for connection in list(self._connections.values()):
            await connection.close("Server shutting down")

        logger.info("Progress service stopped")

    async def handle_connection(self, websocket: WebSocket, client_id: str):
        """Handle a new WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance
            client_id: Unique client identifier
        """
        # Accept connection
        await websocket.accept()

        # Create connection wrapper
        connection = WebSocketConnection(websocket, client_id)
        self._connections[client_id] = connection

        # Send connected message
        await connection.send_message(
            ConnectedMessage(
                client_id=client_id,
                protocol_version="1.0",
                features=["progress", "previews", "errors", "session_updates"],
            )
        )

        logger.info(f"WebSocket connected: {client_id}")

        try:
            # Handle messages
            while True:
                try:
                    # Receive message with timeout
                    data = await asyncio.wait_for(
                        websocket.receive_json(), timeout=60.0
                    )

                    # Parse message type
                    message_type = data.get("type")

                    if message_type == MessageType.SUBSCRIBE:
                        await self._handle_subscribe(
                            connection, SubscribeMessage(**data)
                        )
                    elif message_type == MessageType.UNSUBSCRIBE:
                        await self._handle_unsubscribe(
                            connection, UnsubscribeMessage(**data)
                        )
                    elif message_type == MessageType.PING:
                        await self._handle_ping(connection)
                    else:
                        await connection.send_message(
                            ErrorMessage(
                                error_code="INVALID_MESSAGE_TYPE",
                                message=f"Unknown message type: {message_type}",
                                recoverable=True,
                            )
                        )

                except asyncio.TimeoutError:
                    # Send ping on timeout
                    await self._handle_ping(connection)
                except json.JSONDecodeError as e:
                    await connection.send_message(
                        ErrorMessage(
                            error_code="INVALID_JSON",
                            message=f"Invalid JSON: {str(e)}",
                            recoverable=True,
                        )
                    )

        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {str(e)}")
        finally:
            # Clean up connection
            await self._disconnect_client(client_id)

    async def _handle_subscribe(
        self, connection: WebSocketConnection, message: SubscribeMessage
    ):
        """Handle subscribe message.

        Args:
            connection: WebSocket connection
            message: Subscribe message
        """
        session_id = message.session_id

        # Add to subscriptions
        connection.subscribed_sessions.add(session_id)

        # Track session subscribers
        if session_id not in self._session_subscribers:
            self._session_subscribers[session_id] = set()
        self._session_subscribers[session_id].add(connection.client_id)

        logger.info(f"Client {connection.client_id} subscribed to session {session_id}")

    async def _handle_unsubscribe(
        self, connection: WebSocketConnection, message: UnsubscribeMessage
    ):
        """Handle unsubscribe message.

        Args:
            connection: WebSocket connection
            message: Unsubscribe message
        """
        session_id = message.session_id

        # Remove from subscriptions
        connection.subscribed_sessions.discard(session_id)

        # Update session subscribers
        if session_id in self._session_subscribers:
            self._session_subscribers[session_id].discard(connection.client_id)
            if not self._session_subscribers[session_id]:
                del self._session_subscribers[session_id]

        logger.info(
            f"Client {connection.client_id} unsubscribed from session {session_id}"
        )

    async def _handle_ping(self, connection: WebSocketConnection):
        """Handle ping message.

        Args:
            connection: WebSocket connection
        """
        connection.last_ping = datetime.now()
        await connection.send_message(PongMessage())

    async def _disconnect_client(self, client_id: str):
        """Disconnect a client and clean up.

        Args:
            client_id: Client identifier
        """
        if client_id not in self._connections:
            return

        connection = self._connections[client_id]

        # Remove from all session subscriptions
        for session_id in list(connection.subscribed_sessions):
            if session_id in self._session_subscribers:
                self._session_subscribers[session_id].discard(client_id)
                if not self._session_subscribers[session_id]:
                    del self._session_subscribers[session_id]

        # Close connection
        await connection.close()

        # Remove from connections
        del self._connections[client_id]

        logger.info(f"WebSocket disconnected: {client_id}")

    async def _cleanup_disconnected(self):
        """Periodically clean up disconnected clients."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Find disconnected clients
                disconnected = []
                for client_id, connection in self._connections.items():
                    if connection.websocket.client_state != WebSocketState.CONNECTED:
                        disconnected.append(client_id)

                # Clean up
                for client_id in disconnected:
                    await self._disconnect_client(client_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")

    async def broadcast_progress(
        self,
        session_id: str,
        stage: ProcessingStage,
        progress: float,
        message: str,
        current_item: Optional[int] = None,
        total_items: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        """Broadcast progress update to all subscribers of a session.

        Args:
            session_id: Session identifier
            stage: Current processing stage
            progress: Progress value (0.0 to 1.0)
            message: Progress message
            current_item: Current item being processed
            total_items: Total items to process
            details: Additional details
        """
        if session_id not in self._session_subscribers:
            return

        update = ProgressUpdate(
            session_id=session_id,
            stage=stage,
            progress=progress,
            message=message,
            current_item=current_item,
            total_items=total_items,
            details=details or {},
        )

        # Send to all subscribers
        for client_id in list(self._session_subscribers[session_id]):
            if client_id in self._connections:
                try:
                    await self._connections[client_id].send_message(update)
                except Exception as e:
                    logger.error(f"Failed to send progress to {client_id}: {str(e)}")
                    await self._disconnect_client(client_id)

    async def broadcast_stage_complete(
        self,
        session_id: str,
        stage: ProcessingStage,
        success: bool,
        message: str,
        duration_seconds: float,
        next_stage: Optional[ProcessingStage] = None,
        results: Optional[dict] = None,
    ):
        """Broadcast stage completion to all subscribers.

        Args:
            session_id: Session identifier
            stage: Completed stage
            success: Whether stage completed successfully
            message: Completion message
            duration_seconds: Stage duration
            next_stage: Next stage if any
            results: Stage results
        """
        if session_id not in self._session_subscribers:
            return

        update = StageCompleteMessage(
            session_id=session_id,
            stage=stage,
            success=success,
            message=message,
            duration_seconds=duration_seconds,
            next_stage=next_stage,
            results=results or {},
        )

        # Send to all subscribers
        for client_id in list(self._session_subscribers[session_id]):
            if client_id in self._connections:
                try:
                    await self._connections[client_id].send_message(update)
                except Exception:
                    await self._disconnect_client(client_id)

    async def broadcast_error(
        self,
        session_id: str,
        error_code: str,
        message: str,
        details: Optional[dict] = None,
        recoverable: bool = False,
    ):
        """Broadcast error to all subscribers.

        Args:
            session_id: Session identifier
            error_code: Error code
            message: Error message
            details: Error details
            recoverable: Whether error is recoverable
        """
        if session_id not in self._session_subscribers:
            return

        update = ErrorMessage(
            session_id=session_id,
            error_code=error_code,
            message=message,
            details=details or {},
            recoverable=recoverable,
        )

        # Send to all subscribers
        for client_id in list(self._session_subscribers[session_id]):
            if client_id in self._connections:
                try:
                    await self._connections[client_id].send_message(update)
                except Exception:
                    await self._disconnect_client(client_id)

    def create_progress_callback(
        self, session_id: str, stage: ProcessingStage
    ) -> callable:
        """Create a progress callback function for an operation.

        Args:
            session_id: Session identifier
            stage: Processing stage

        Returns:
            Async callback function
        """

        async def callback(substage: str, progress: float, message: str, **kwargs):
            """Progress callback.

            Args:
                substage: Sub-stage name
                progress: Progress value
                message: Progress message
                **kwargs: Additional parameters
            """
            await self.broadcast_progress(
                session_id=session_id,
                stage=stage,
                progress=progress,
                message=message,
                current_item=kwargs.get("current_item"),
                total_items=kwargs.get("total_items"),
                details=kwargs.get("details", {}),
            )

        return callback


# Global progress service instance
_progress_service: Optional[ProgressService] = None


def get_progress_service() -> ProgressService:
    """Get progress service singleton."""
    global _progress_service
    if _progress_service is None:
        _progress_service = ProgressService()
    return _progress_service
