"""
Enhanced WebSocket Endpoints

Provides advanced WebSocket functionality with heartbeat and authentication.
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Header, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from pdf_splitter.api.models.websocket import WebSocketEventType, WebSocketMessage
from pdf_splitter.api.services.websocket_enhanced import (
    ConnectionInfo,
    enhanced_websocket_manager,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/enhanced/{session_id}")
async def enhanced_websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = Query(None),
    client_id: Optional[str] = Query(None),
):
    """
    Enhanced WebSocket endpoint with heartbeat and authentication support.

    Features:
    - Automatic heartbeat/pong for connection health
    - Optional authentication token support
    - Client ID tracking for multiple connections
    - Structured message protocol
    - Connection statistics

    Message Protocol:

    Server -> Client:
    - connection: Initial connection with config
    - heartbeat: Periodic heartbeat (client should respond)
    - pong: Response to client heartbeat
    - progress: Processing progress updates
    - status: Status changes
    - error: Error notifications
    - auth_success/auth_failure: Authentication results

    Client -> Server:
    - heartbeat: Keep-alive with sequence number
    - auth_request: Authentication with token
    - (custom message types can be added)

    Query Parameters:
    - token: Optional authentication token
    - client_id: Optional client identifier

    Args:
        websocket: WebSocket connection
        session_id: Session ID to subscribe to
        token: Optional authentication token
        client_id: Optional client identifier
    """
    conn_info = None

    try:
        # Connect with enhanced manager
        conn_info = await enhanced_websocket_manager.connect(
            websocket=websocket,
            session_id=session_id,
            auth_token=token,
            client_id=client_id,
        )

        logger.info(
            f"Enhanced WebSocket connected: session={session_id}, "
            f"client={conn_info.client_id}"
        )

        # Handle incoming messages
        while True:
            try:
                # Receive message
                raw_message = await websocket.receive_json()

                # Parse and validate message
                if isinstance(raw_message, dict):
                    message = WebSocketMessage(**raw_message)

                    # Handle message
                    await enhanced_websocket_manager.handle_message(
                        conn_info, raw_message
                    )
                else:
                    logger.warning(f"Invalid message format: {raw_message}")

            except asyncio.TimeoutError:
                # This is normal - we use timeouts for periodic checks
                continue

            except WebSocketDisconnect:
                # Client disconnected normally
                logger.info(
                    f"Client disconnected: session={session_id}, "
                    f"client={conn_info.client_id}"
                )
                break

            except Exception as e:
                logger.error(
                    f"Error handling message: {e}, session={session_id}, "
                    f"client={conn_info.client_id}"
                )
                # Send error to client
                await enhanced_websocket_manager.send_error(
                    session_id=session_id,
                    error_code="MESSAGE_ERROR",
                    error_message=str(e),
                    recoverable=True,
                )

    except ValueError as e:
        # Connection rejected
        logger.warning(f"Connection rejected: {e}")

    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected WebSocket error: {e}")

    finally:
        # Clean up connection
        if conn_info:
            await enhanced_websocket_manager.disconnect(conn_info)


@router.get("/api/websocket/enhanced/stats")
async def get_enhanced_websocket_stats() -> dict:
    """
    Get enhanced WebSocket connection statistics.

    Returns detailed statistics including:
    - Total connections
    - Active sessions
    - Connection durations
    - Client information
    - Heartbeat status

    Returns:
        Detailed statistics dictionary
    """
    stats = enhanced_websocket_manager.get_connection_stats()

    # Add timestamp
    stats["timestamp"] = datetime.utcnow().isoformat()

    # Add configuration info
    stats["config"] = {
        "heartbeat_interval": enhanced_websocket_manager.heartbeat_interval,
        "max_connections_per_session": enhanced_websocket_manager.max_connections_per_session,
        "max_total_connections": enhanced_websocket_manager.max_total_connections,
    }

    return stats


@router.post("/api/websocket/enhanced/broadcast/{session_id}")
async def broadcast_message(
    session_id: str,
    message: dict,
    event_type: str = "info",
    exclude_clients: Optional[list] = None,
):
    """
    Broadcast a message to all clients connected to a session.

    This endpoint is useful for sending custom notifications or updates
    to all connected clients without going through the normal processing flow.

    Args:
        session_id: Session ID to broadcast to
        message: Message data to send
        event_type: Type of message (default: "info")
        exclude_clients: Optional list of client IDs to exclude

    Returns:
        Broadcast result
    """
    try:
        # Create WebSocket message
        ws_message = WebSocketMessage(type=WebSocketEventType(event_type), data=message)

        # Broadcast to session
        await enhanced_websocket_manager.broadcast_to_session(
            session_id=session_id,
            message=ws_message,
            exclude_clients=set(exclude_clients) if exclude_clients else None,
        )

        # Get connection count
        session_connections = enhanced_websocket_manager.connections.get(session_id, {})

        return {
            "success": True,
            "session_id": session_id,
            "clients_notified": len(session_connections),
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
        }


@router.delete("/api/websocket/enhanced/disconnect/{session_id}/{client_id}")
async def disconnect_client(
    session_id: str, client_id: str, reason: Optional[str] = None
):
    """
    Forcefully disconnect a specific client.

    Useful for administrative purposes or when a client needs to be
    disconnected due to security or resource concerns.

    Args:
        session_id: Session ID
        client_id: Client ID to disconnect
        reason: Optional reason for disconnection

    Returns:
        Disconnection result
    """
    try:
        # Find connection
        conn_info = enhanced_websocket_manager.connections.get(session_id, {}).get(
            client_id
        )

        if not conn_info:
            return {
                "success": False,
                "error": "Client not found",
                "session_id": session_id,
                "client_id": client_id,
            }

        # Send disconnection notice
        if conn_info.websocket.application_state == WebSocketState.CONNECTED:
            disconnect_msg = WebSocketMessage(
                type=WebSocketEventType.DISCONNECTION,
                data={
                    "reason": reason or "Administrative disconnection",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            try:
                await conn_info.websocket.send_json(disconnect_msg.dict())
            except:
                pass

        # Disconnect
        await enhanced_websocket_manager.disconnect(conn_info)

        return {
            "success": True,
            "session_id": session_id,
            "client_id": client_id,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error disconnecting client: {e}")
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id,
            "client_id": client_id,
        }
