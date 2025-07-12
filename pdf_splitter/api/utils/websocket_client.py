"""
WebSocket Client Utilities

Provides auto-reconnecting WebSocket client with heartbeat support.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import websockets
from websockets.exceptions import WebSocketException

from pdf_splitter.api.models.websocket import (
    HeartbeatMessage,
    WebSocketClientConfig,
    WebSocketEventType,
    WebSocketMessage,
)

logger = logging.getLogger(__name__)


class WebSocketClient:
    """Auto-reconnecting WebSocket client with heartbeat support."""

    def __init__(self, config: WebSocketClientConfig):
        """
        Initialize WebSocket client.

        Args:
            config: Client configuration
        """
        self.config = config
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.reconnect_attempts = 0
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.receive_task: Optional[asyncio.Task] = None
        self.heartbeat_sequence = 0

        # Event handlers
        self.handlers: Dict[WebSocketEventType, Callable] = {}

        # Default handlers
        self.register_handler(WebSocketEventType.CONNECTION, self._on_connection)
        self.register_handler(WebSocketEventType.PONG, self._on_pong)
        self.register_handler(WebSocketEventType.ERROR, self._on_error)

    def register_handler(self, event_type: WebSocketEventType, handler: Callable):
        """
        Register an event handler.

        Args:
            event_type: Event type to handle
            handler: Async function to handle the event
        """
        self.handlers[event_type] = handler

    async def connect(self):
        """Connect to WebSocket server with automatic retry."""
        while self.reconnect_attempts < self.config.max_reconnect_attempts:
            try:
                # Build connection URL
                url = self.config.url
                if self.config.session_id:
                    url = f"{url}?session_id={self.config.session_id}"
                if self.config.auth_token:
                    url = f"{url}&token={self.config.auth_token}"

                logger.info(f"Connecting to WebSocket: {self.config.url}")

                # Connect with timeout
                self.websocket = await asyncio.wait_for(
                    websockets.connect(url), timeout=self.config.timeout
                )

                self.connected = True
                self.reconnect_attempts = 0

                logger.info("WebSocket connected successfully")

                # Start tasks
                self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                self.receive_task = asyncio.create_task(self._receive_loop())

                # Wait for tasks
                await asyncio.gather(
                    self.heartbeat_task, self.receive_task, return_exceptions=True
                )

            except asyncio.TimeoutError:
                logger.error("WebSocket connection timeout")
                await self._handle_disconnect()

            except WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
                await self._handle_disconnect()

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await self._handle_disconnect()

            # Wait before reconnecting
            if (
                self.config.reconnect
                and self.reconnect_attempts < self.config.max_reconnect_attempts
            ):
                wait_time = self.config.reconnect_interval * (
                    2 ** min(self.reconnect_attempts, 5)
                )
                logger.info(f"Reconnecting in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                self.reconnect_attempts += 1
            else:
                break

    async def disconnect(self):
        """Disconnect from WebSocket server."""
        self.connected = False

        # Cancel tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.receive_task:
            self.receive_task.cancel()

        # Close connection
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        logger.info("WebSocket disconnected")

    async def send_message(self, message: WebSocketMessage):
        """
        Send a message to the server.

        Args:
            message: Message to send
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket not connected")

        try:
            await self.websocket.send(json.dumps(message.dict()))
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            await self._handle_disconnect()
            raise

    async def send_data(self, event_type: WebSocketEventType, data: Dict[str, Any]):
        """
        Send data with a specific event type.

        Args:
            event_type: Event type
            data: Data to send
        """
        message = WebSocketMessage(type=event_type, data=data)
        await self.send_message(message)

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to keep connection alive."""
        while self.connected:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                # Send heartbeat
                self.heartbeat_sequence += 1
                heartbeat = HeartbeatMessage(
                    data={
                        "sequence": self.heartbeat_sequence,
                        "client_time": datetime.utcnow().isoformat(),
                    }
                )

                await self.send_message(heartbeat)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await self._handle_disconnect()
                break

    async def _receive_loop(self):
        """Receive and process messages from server."""
        while self.connected and self.websocket:
            try:
                message = await self.websocket.recv()

                # Parse message
                data = json.loads(message)
                event_type = data.get("type")

                # Handle message
                if event_type in self.handlers:
                    handler = self.handlers[event_type]
                    await handler(data)
                else:
                    logger.debug(f"Unhandled message type: {event_type}")

            except asyncio.CancelledError:
                break
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
            except WebSocketException as e:
                logger.error(f"WebSocket error in receive loop: {e}")
                await self._handle_disconnect()
                break
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                await self._handle_disconnect()
                break

    async def _handle_disconnect(self):
        """Handle disconnection and cleanup."""
        self.connected = False

        # Cancel tasks
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
        if self.receive_task and not self.receive_task.done():
            self.receive_task.cancel()

        # Close websocket
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None

    async def _on_connection(self, data: dict):
        """Handle connection event."""
        logger.info(f"Connection established: {data}")

    async def _on_pong(self, data: dict):
        """Handle pong response."""
        logger.debug(f"Pong received: sequence={data.get('data', {}).get('sequence')}")

    async def _on_error(self, data: dict):
        """Handle error event."""
        error_data = data.get("data", {})
        logger.error(
            f"Server error: {error_data.get('error_message')} "
            f"(code: {error_data.get('error_code')})"
        )


class ProgressTracker:
    """Utility class to track progress updates from WebSocket."""

    def __init__(self):
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.current_stage: Optional[str] = None
        self.overall_progress: float = 0.0
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.error: Optional[str] = None

    async def on_progress(self, data: dict):
        """Handle progress update."""
        msg_data = data.get("data", {})
        stage = msg_data.get("stage")
        progress = msg_data.get("progress", 0)
        message = msg_data.get("message", "")

        self.current_stage = stage
        self.stages[stage] = {
            "progress": progress,
            "message": message,
            "timestamp": datetime.utcnow(),
            "details": msg_data.get("details", {}),
        }

        # Calculate overall progress
        self._calculate_overall_progress()

        logger.info(f"Progress: {stage} - {progress}% - {message}")

    async def on_status(self, data: dict):
        """Handle status update."""
        msg_data = data.get("data", {})
        status = msg_data.get("status")

        logger.info(f"Status: {status}")

        if status in ["complete", "confirmed"]:
            self.end_time = datetime.utcnow()
            self.overall_progress = 100.0
        elif status == "cancelled":
            self.error = "Processing cancelled"

    async def on_error(self, data: dict):
        """Handle error event."""
        msg_data = data.get("data", {})
        self.error = msg_data.get("error_message", "Unknown error")
        self.end_time = datetime.utcnow()

    def _calculate_overall_progress(self):
        """Calculate overall progress from stage progress."""
        if not self.stages:
            self.overall_progress = 0.0
            return

        # Simple average for now
        total_progress = sum(s["progress"] for s in self.stages.values())
        self.overall_progress = total_progress / len(self.stages)

    @property
    def duration(self) -> Optional[timedelta]:
        """Get processing duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return datetime.utcnow() - self.start_time

    @property
    def is_complete(self) -> bool:
        """Check if processing is complete."""
        return self.end_time is not None

    @property
    def is_error(self) -> bool:
        """Check if processing ended with error."""
        return self.error is not None

    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        return {
            "overall_progress": self.overall_progress,
            "current_stage": self.current_stage,
            "stages": self.stages,
            "duration": str(self.duration) if self.duration else None,
            "is_complete": self.is_complete,
            "is_error": self.is_error,
            "error": self.error,
        }


async def track_session_progress(
    websocket_url: str, session_id: str, auth_token: Optional[str] = None
) -> ProgressTracker:
    """
    Convenience function to track progress for a session.

    Args:
        websocket_url: WebSocket server URL
        session_id: Session ID to track
        auth_token: Optional authentication token

    Returns:
        ProgressTracker instance
    """
    # Create client config
    config = WebSocketClientConfig(
        url=websocket_url, session_id=session_id, auth_token=auth_token, reconnect=True
    )

    # Create client and tracker
    client = WebSocketClient(config)
    tracker = ProgressTracker()

    # Register handlers
    client.register_handler(WebSocketEventType.PROGRESS, tracker.on_progress)
    client.register_handler(WebSocketEventType.STATUS, tracker.on_status)
    client.register_handler(WebSocketEventType.ERROR, tracker.on_error)

    # Connect and track
    try:
        await client.connect()
    finally:
        await client.disconnect()

    return tracker
