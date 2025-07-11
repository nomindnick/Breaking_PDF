"""Tests for WebSocket functionality.

This module tests WebSocket connections, message handling, and real-time
progress updates.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from pdf_splitter.api.models.websocket import MessageType, ProcessingStage
from pdf_splitter.api.services.progress_service import (
    ProgressService,
    WebSocketConnection,
)


class TestWebSocket:
    """Test WebSocket functionality."""

    def test_websocket_connection(self, client: TestClient):
        """Test basic WebSocket connection."""
        with client.websocket_connect("/ws") as websocket:
            # Should receive connected message
            data = websocket.receive_json()
            assert data["type"] == MessageType.CONNECTED
            assert "client_id" in data
            assert data["protocol_version"] == "1.0"
            assert "progress" in data["features"]

    def test_websocket_subscribe(self, client: TestClient):
        """Test subscribing to session updates."""
        with client.websocket_connect("/ws") as websocket:
            # Receive connected message
            _ = websocket.receive_json()

            # Subscribe to session
            subscribe_msg = {
                "type": "subscribe",
                "session_id": "test-session-id",
                "include_previews": True,
            }
            websocket.send_json(subscribe_msg)

            # Should not receive immediate response for subscribe
            # (unless there's an error)

    def test_websocket_unsubscribe(self, client: TestClient):
        """Test unsubscribing from session updates."""
        with client.websocket_connect("/ws") as websocket:
            # Connect and subscribe
            websocket.receive_json()  # Connected message

            session_id = "test-session-id"

            # Subscribe
            websocket.send_json(
                {
                    "type": "subscribe",
                    "session_id": session_id,
                }
            )

            # Unsubscribe
            websocket.send_json(
                {
                    "type": "unsubscribe",
                    "session_id": session_id,
                }
            )

    def test_websocket_ping_pong(self, client: TestClient):
        """Test ping/pong messages."""
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # Connected message

            # Send ping
            websocket.send_json({"type": "ping"})

            # Should receive pong
            response = websocket.receive_json()
            assert response["type"] == MessageType.PONG

    def test_websocket_invalid_message(self, client: TestClient):
        """Test handling of invalid messages."""
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # Connected message

            # Send invalid message type
            websocket.send_json({"type": "invalid_type"})

            # Should receive error
            response = websocket.receive_json()
            assert response["type"] == MessageType.ERROR
            assert response["error_code"] == "INVALID_MESSAGE_TYPE"
            assert response["recoverable"] is True

    def test_websocket_invalid_json(self, client: TestClient):
        """Test handling of invalid JSON."""
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # Connected message

            # Send invalid JSON
            websocket.send_text("invalid json {")

            # Should receive error
            response = websocket.receive_json()
            assert response["type"] == MessageType.ERROR
            assert response["error_code"] == "INVALID_JSON"

    def test_websocket_auto_subscribe(self, client: TestClient):
        """Test WebSocket with automatic session subscription."""
        session_id = "test-session-id"

        with client.websocket_connect(f"/ws/{session_id}") as websocket:
            # Should receive connected message
            data = websocket.receive_json()
            assert data["type"] == MessageType.CONNECTED

            # The session subscription happens automatically in the background


class TestProgressService:
    """Test progress service functionality."""

    @pytest.fixture
    def progress_service(self):
        """Create progress service instance."""
        return ProgressService()

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket."""
        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        ws.receive_json = AsyncMock()
        ws.close = AsyncMock()
        ws.client_state = MagicMock()
        return ws

    @pytest.mark.asyncio
    async def test_broadcast_progress(self, progress_service: ProgressService):
        """Test broadcasting progress updates."""
        session_id = "test-session"
        client_id = "test-client"

        # Create mock connection
        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.client_state = MagicMock()

        connection = WebSocketConnection(mock_ws, client_id)
        connection.subscribed_sessions.add(session_id)

        progress_service._connections[client_id] = connection
        progress_service._session_subscribers[session_id] = {client_id}

        # Broadcast progress
        await progress_service.broadcast_progress(
            session_id=session_id,
            stage=ProcessingStage.DETECTION,
            progress=0.5,
            message="Processing page 50 of 100",
            current_item=50,
            total_items=100,
        )

        # Verify message sent
        mock_ws.send_json.assert_called_once()
        sent_data = mock_ws.send_json.call_args[0][0]
        assert sent_data["type"] == MessageType.PROGRESS
        assert sent_data["session_id"] == session_id
        assert sent_data["stage"] == ProcessingStage.DETECTION
        assert sent_data["progress"] == 0.5
        assert sent_data["current_item"] == 50

    @pytest.mark.asyncio
    async def test_broadcast_stage_complete(self, progress_service: ProgressService):
        """Test broadcasting stage completion."""
        session_id = "test-session"
        client_id = "test-client"

        # Setup mock connection
        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.client_state = MagicMock()

        connection = WebSocketConnection(mock_ws, client_id)
        connection.subscribed_sessions.add(session_id)

        progress_service._connections[client_id] = connection
        progress_service._session_subscribers[session_id] = {client_id}

        # Broadcast stage complete
        await progress_service.broadcast_stage_complete(
            session_id=session_id,
            stage=ProcessingStage.DETECTION,
            success=True,
            message="Detection completed successfully",
            duration_seconds=15.5,
            next_stage=ProcessingStage.SPLITTING,
            results={"boundaries_found": 5},
        )

        # Verify message
        mock_ws.send_json.assert_called_once()
        sent_data = mock_ws.send_json.call_args[0][0]
        assert sent_data["type"] == MessageType.STAGE_COMPLETE
        assert sent_data["success"] is True
        assert sent_data["duration_seconds"] == 15.5
        assert sent_data["next_stage"] == ProcessingStage.SPLITTING

    @pytest.mark.asyncio
    async def test_broadcast_error(self, progress_service: ProgressService):
        """Test broadcasting errors."""
        session_id = "test-session"
        client_id = "test-client"

        # Setup mock connection
        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.client_state = MagicMock()

        connection = WebSocketConnection(mock_ws, client_id)
        connection.subscribed_sessions.add(session_id)

        progress_service._connections[client_id] = connection
        progress_service._session_subscribers[session_id] = {client_id}

        # Broadcast error
        await progress_service.broadcast_error(
            session_id=session_id,
            error_code="DETECTION_FAILED",
            message="Failed to detect boundaries",
            details={"reason": "Insufficient text"},
            recoverable=False,
        )

        # Verify message
        mock_ws.send_json.assert_called_once()
        sent_data = mock_ws.send_json.call_args[0][0]
        assert sent_data["type"] == MessageType.ERROR
        assert sent_data["error_code"] == "DETECTION_FAILED"
        assert sent_data["recoverable"] is False

    @pytest.mark.asyncio
    async def test_progress_callback(self, progress_service: ProgressService):
        """Test progress callback creation."""
        session_id = "test-session"

        # Create callback
        callback = progress_service.create_progress_callback(
            session_id, ProcessingStage.SPLITTING
        )

        # Mock broadcasting
        progress_service.broadcast_progress = AsyncMock()

        # Call the callback
        await callback(
            "processing_segment",
            0.25,
            "Processing segment 1 of 4",
            current_item=1,
            total_items=4,
        )

        # Verify broadcast was called
        progress_service.broadcast_progress.assert_called_once_with(
            session_id=session_id,
            stage=ProcessingStage.SPLITTING,
            progress=0.25,
            message="Processing segment 1 of 4",
            current_item=1,
            total_items=4,
            details={},
        )

    @pytest.mark.asyncio
    async def test_connection_cleanup(self, progress_service: ProgressService):
        """Test connection cleanup on disconnect."""
        client_id = "test-client"
        session_id = "test-session"

        # Create connection
        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()
        mock_ws.client_state = MagicMock()

        connection = WebSocketConnection(mock_ws, client_id)
        connection.subscribed_sessions.add(session_id)

        progress_service._connections[client_id] = connection
        progress_service._session_subscribers[session_id] = {client_id}

        # Disconnect client
        await progress_service._disconnect_client(client_id)

        # Verify cleanup
        assert client_id not in progress_service._connections
        assert session_id not in progress_service._session_subscribers
        mock_ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, progress_service: ProgressService):
        """Test broadcasting to multiple subscribers."""
        session_id = "test-session"
        clients = ["client1", "client2", "client3"]

        # Create multiple connections
        for client_id in clients:
            mock_ws = MagicMock()
            mock_ws.send_json = AsyncMock()
            mock_ws.client_state = MagicMock()

            connection = WebSocketConnection(mock_ws, client_id)
            connection.subscribed_sessions.add(session_id)

            progress_service._connections[client_id] = connection

            if session_id not in progress_service._session_subscribers:
                progress_service._session_subscribers[session_id] = set()
            progress_service._session_subscribers[session_id].add(client_id)

        # Broadcast to all
        await progress_service.broadcast_progress(
            session_id=session_id,
            stage=ProcessingStage.DETECTION,
            progress=1.0,
            message="Complete",
        )

        # Verify all received message
        for client_id in clients:
            connection = progress_service._connections[client_id]
            connection.websocket.send_json.assert_called_once()

    def test_websocket_connection_model(self):
        """Test WebSocketConnection model."""
        mock_ws = MagicMock()
        client_id = "test-client"

        connection = WebSocketConnection(mock_ws, client_id)

        assert connection.client_id == client_id
        assert connection.websocket == mock_ws
        assert len(connection.subscribed_sessions) == 0
        assert connection.connected_at is not None
        assert connection.last_ping is not None
