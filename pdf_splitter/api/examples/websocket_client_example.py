#!/usr/bin/env python3
"""
WebSocket Client Example.

Demonstrates how to use the PDF Splitter WebSocket API for real-time progress tracking.
"""
import asyncio

from pdf_splitter.api.models.websocket import WebSocketEventType
from pdf_splitter.api.utils.websocket_client import (
    ProgressTracker,
    WebSocketClient,
    WebSocketClientConfig,
    track_session_progress,
)


async def basic_client_example():
    """Show basic WebSocket client usage."""
    print("=== Basic WebSocket Client Example ===\n")

    # Configure client
    config = WebSocketClientConfig(
        url="ws://localhost:8000/ws/enhanced",
        session_id="your-session-id",
        reconnect=True,
        heartbeat_interval=30,
    )

    # Create client
    client = WebSocketClient(config)

    # Register custom handlers
    async def on_progress(data: dict):
        msg_data = data.get("data", {})
        print(f"Progress: {msg_data.get('stage')} - {msg_data.get('progress'):.1f}%")

    async def on_status(data: dict):
        msg_data = data.get("data", {})
        print(f"Status changed to: {msg_data.get('status')}")

    client.register_handler(WebSocketEventType.PROGRESS, on_progress)
    client.register_handler(WebSocketEventType.STATUS, on_status)

    # Connect and monitor
    try:
        print("Connecting to WebSocket...")
        await client.connect()
    except KeyboardInterrupt:
        print("\nDisconnecting...")
    finally:
        await client.disconnect()


async def progress_tracker_example():
    """Show usage of the ProgressTracker utility."""
    print("=== Progress Tracker Example ===\n")

    # Track progress for a session
    tracker = await track_session_progress(
        websocket_url="ws://localhost:8000/ws/enhanced", session_id="your-session-id"
    )

    # Print summary
    summary = tracker.get_summary()
    print("\nProgress Summary:")
    print(f"Overall Progress: {summary['overall_progress']:.1f}%")
    print(f"Duration: {summary['duration']}")
    print(f"Complete: {summary['is_complete']}")

    if summary["stages"]:
        print("\nStages:")
        for stage, info in summary["stages"].items():
            print(f"  - {stage}: {info['progress']:.1f}% - {info['message']}")


async def custom_message_example():
    """Show how to send custom messages."""
    print("=== Custom Message Example ===\n")

    config = WebSocketClientConfig(
        url="ws://localhost:8000/ws/enhanced", session_id="your-session-id"
    )

    client = WebSocketClient(config)

    try:
        await client.connect()

        # Send authentication request
        await client.send_data(
            WebSocketEventType.AUTH_REQUEST, {"token": "your-auth-token"}
        )

        # Wait for response
        await asyncio.sleep(2)

    finally:
        await client.disconnect()


def sync_example():
    """Show synchronous wrapper usage."""
    print("=== Synchronous Example ===\n")

    # For synchronous code, use asyncio.run()
    asyncio.run(basic_client_example())


# Example: Integrate with existing code
class PDFProcessorWithProgress:
    """Example of integrating WebSocket progress into existing code."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        """Initialize the PDF processor."""
        self.api_url = api_url
        self.ws_url = api_url.replace("http", "ws")

    async def process_pdf_with_progress(self, pdf_path: str):
        """Process PDF with real-time progress tracking."""
        import aiohttp

        # Step 1: Upload file
        async with aiohttp.ClientSession() as session:
            with open(pdf_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename=pdf_path)

                async with session.post(
                    f"{self.api_url}/api/upload", data=data
                ) as resp:
                    upload_result = await resp.json()
                    file_id = upload_result["upload_id"]

        # Step 2: Start processing
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/api/process", json={"file_id": file_id}
            ) as resp:
                process_result = await resp.json()
                session_id = process_result["session_id"]

        # Step 3: Track progress via WebSocket
        print(f"Processing started, session: {session_id}")
        print("Tracking progress...\n")

        tracker = ProgressTracker()

        config = WebSocketClientConfig(
            url=f"{self.ws_url}/ws/enhanced", session_id=session_id, reconnect=True
        )

        client = WebSocketClient(config)
        client.register_handler(WebSocketEventType.PROGRESS, tracker.on_progress)
        client.register_handler(WebSocketEventType.STATUS, tracker.on_status)
        client.register_handler(WebSocketEventType.ERROR, tracker.on_error)

        # Connect and wait for completion
        await client.connect()

        while not tracker.is_complete:
            await asyncio.sleep(1)
            if tracker.overall_progress > 0:
                print(
                    f"\rProgress: {tracker.overall_progress:.1f}%", end="", flush=True
                )

        print("\n")
        await client.disconnect()

        # Return results
        if tracker.is_error:
            raise Exception(f"Processing failed: {tracker.error}")

        return {
            "session_id": session_id,
            "duration": str(tracker.duration),
            "stages": tracker.stages,
        }


# React/JavaScript Example
JAVASCRIPT_EXAMPLE = """
// JavaScript/React WebSocket Example

class PDFSplitterWebSocket {
    constructor(sessionId, onProgress, onComplete) {
        this.sessionId = sessionId;
        this.onProgress = onProgress;
        this.onComplete = onComplete;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.heartbeatInterval = null;
    }

    connect() {
        const wsUrl = `ws://localhost:8000/ws/enhanced/${this.sessionId}`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
            this.startHeartbeat();
        };

        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.stopHeartbeat();
            this.attemptReconnect();
        };
    }

    handleMessage(message) {
        switch (message.type) {
            case 'progress':
                if (this.onProgress) {
                    this.onProgress(message.data);
                }
                break;

            case 'status':
                if (message.data.status === 'complete' && this.onComplete) {
                    this.onComplete(message.data);
                }
                break;

            case 'heartbeat':
                // Respond to heartbeat
                this.ws.send(JSON.stringify({
                    type: 'heartbeat',
                    data: {
                        sequence: message.data.sequence,
                        client_time: new Date().toISOString()
                    }
                }));
                break;
        }
    }

    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'heartbeat',
                    data: {
                        client_time: new Date().toISOString()
                    }
                }));
            }
        }, 30000); // 30 seconds
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < 5) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
            console.log(`Reconnecting in ${delay}ms...`);
            setTimeout(() => this.connect(), delay);
        }
    }

    disconnect() {
        this.stopHeartbeat();
        if (this.ws) {
            this.ws.close();
        }
    }
}

// React Hook Example
function useWebSocketProgress(sessionId) {
    const [progress, setProgress] = useState(0);
    const [stage, setStage] = useState('');
    const [isComplete, setIsComplete] = useState(false);
    const wsRef = useRef(null);

    useEffect(() => {
        if (sessionId) {
            wsRef.current = new PDFSplitterWebSocket(
                sessionId,
                (data) => {
                    setProgress(data.progress);
                    setStage(data.stage);
                },
                (data) => {
                    setIsComplete(true);
                }
            );

            wsRef.current.connect();

            return () => {
                if (wsRef.current) {
                    wsRef.current.disconnect();
                }
            };
        }
    }, [sessionId]);

    return { progress, stage, isComplete };
}
"""


if __name__ == "__main__":
    print("PDF Splitter WebSocket Client Examples")
    print("=" * 50)
    print("\nAvailable examples:")
    print("1. Basic client connection")
    print("2. Progress tracker")
    print("3. Custom messages")
    print("4. Full processing with progress")
    print("\nJavaScript example also included in source.")

    # Run basic example
    asyncio.run(basic_client_example())
