#!/usr/bin/env python3
"""
Test script for Sprint 2 API implementation.

Tests processing endpoints, session management, and WebSocket connections.
"""
import asyncio
import json
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Optional

import requests
import websocket

# API base URL
BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"


def test_process_initiation():
    """Test PDF processing initiation."""
    print("Testing PDF processing initiation...")

    # First, upload a file
    test_file = None
    for f in [Path("Test_PDF_Set_1.pdf"), Path("Test_PDF_Set_2_ocr.pdf")]:
        if f.exists():
            test_file = f
            break

    if not test_file:
        print("❌ No test PDF file found")
        return None, None

    # Upload file
    print(f"Uploading {test_file.name}...")
    with open(test_file, "rb") as f:
        files = {"file": (test_file.name, f, "application/pdf")}
        response = requests.post(f"{BASE_URL}/api/upload", files=files)

    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return None, None

    file_id = response.json()["upload_id"]
    print(f"✅ File uploaded: {file_id}")

    # Start processing
    print("Starting processing...")
    response = requests.post(f"{BASE_URL}/api/process", json={"file_id": file_id})

    if response.status_code == 200:
        data = response.json()
        session_id = data["session_id"]
        print(f"✅ Processing started: {session_id}")
        print(f"Status: {data['status']}")
        print(f"Message: {data['message']}")
        return session_id, file_id
    else:
        print(f"❌ Processing failed: {response.text}")
        return None, None


def test_processing_status(session_id: str):
    """Test processing status endpoint."""
    print(f"\nTesting processing status for session {session_id}...")

    # Check status multiple times
    for i in range(5):
        response = requests.get(f"{BASE_URL}/api/process/{session_id}/status")

        if response.status_code == 200:
            data = response.json()
            print(f"Status check {i+1}:")
            print(f"  Status: {data['status']}")
            print(f"  Stage: {data.get('stage', 'N/A')}")
            print(f"  Progress: {data.get('progress', 'N/A')}%")
            print(f"  Active: {data.get('is_active', False)}")

            if data["status"] == "confirmed":
                print("✅ Processing completed successfully")
                return True
            elif data.get("error"):
                print(f"❌ Processing error: {data['error']}")
                return False
        else:
            print(f"❌ Status check failed: {response.text}")
            return False

        time.sleep(2)

    print("⏳ Processing still in progress after 10 seconds")
    return True


def test_session_listing():
    """Test session listing endpoint."""
    print("\nTesting session listing...")

    response = requests.get(f"{BASE_URL}/api/sessions")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Sessions retrieved:")
        print(f"  Total: {data['total_count']}")
        print(f"  Active: {data['active_count']}")
        print(f"  Retrieved: {len(data['sessions'])}")

        if data["sessions"]:
            print("  Recent sessions:")
            for session in data["sessions"][:3]:
                print(f"    - {session['session_id']}: {session['status']}")

        return True
    else:
        print(f"❌ Session listing failed: {response.text}")
        return False


def test_session_details(session_id: str):
    """Test session details endpoint."""
    print(f"\nTesting session details for {session_id}...")

    response = requests.get(f"{BASE_URL}/api/sessions/{session_id}")

    if response.status_code == 200:
        data = response.json()
        print("✅ Session details retrieved:")
        print(f"  Status: {data['status']}")
        print(f"  Created: {data['created_at']}")
        print(f"  Has proposal: {data['has_proposal']}")
        print(f"  Modifications: {data['modifications_count']}")
        return True
    else:
        print(f"❌ Session details failed: {response.text}")
        return False


def test_session_extension(session_id: str):
    """Test session extension endpoint."""
    print(f"\nTesting session extension for {session_id}...")

    response = requests.post(
        f"{BASE_URL}/api/sessions/{session_id}/extend", json={"hours": 48}
    )

    if response.status_code == 200:
        data = response.json()
        print("✅ Session extended:")
        print(f"  New expiration: {data['expires_at']}")
        return True
    else:
        print(f"❌ Session extension failed: {response.text}")
        return False


def test_websocket_connection(session_id: str):
    """Test WebSocket connection for real-time updates."""
    print(f"\nTesting WebSocket connection for session {session_id}...")

    received_messages = []

    def on_message(ws, message):
        data = json.loads(message)
        received_messages.append(data)
        print(
            f"  WS Message: {data['type']} - {data.get('data', {}).get('message', 'N/A')}"
        )

    def on_error(ws, error):
        print(f"  WS Error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print(f"  WS Closed: {close_status_code} - {close_msg}")

    def on_open(ws):
        print("  WS Connected")
        # Send ping
        ws.send("ping")

    try:
        ws = websocket.WebSocketApp(
            f"{WS_URL}/ws/{session_id}",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        # Run WebSocket in thread for 5 seconds
        ws_thread = Thread(target=lambda: ws.run_forever())
        ws_thread.daemon = True
        ws_thread.start()

        time.sleep(5)
        ws.close()

        if received_messages:
            print(f"✅ WebSocket received {len(received_messages)} messages")
            return True
        else:
            print("❌ No WebSocket messages received")
            return False

    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False


def test_session_statistics():
    """Test session statistics endpoint."""
    print("\nTesting session statistics...")

    response = requests.get(f"{BASE_URL}/api/sessions/stats/summary")

    if response.status_code == 200:
        data = response.json()
        stats = data["statistics"]
        print("✅ Statistics retrieved:")
        print(f"  Total sessions: {stats['total_sessions']}")
        print(f"  Active: {stats['active_sessions']}")
        print(f"  Completed: {stats['completed_sessions']}")

        if stats["average_processing_time"]:
            print(f"  Avg processing time: {stats['average_processing_time']:.2f}s")

        return True
    else:
        print(f"❌ Statistics retrieval failed: {response.text}")
        return False


def main():
    """Run all Sprint 2 tests."""
    print("=" * 60)
    print("Sprint 2 API Tests")
    print("=" * 60)
    print()

    # Check if API is running
    try:
        response = requests.get(BASE_URL)
        print(f"API is running at {BASE_URL}")
        print()
    except requests.ConnectionError:
        print(f"❌ Cannot connect to API at {BASE_URL}")
        print("Please start the API with: python run_api.py")
        return 1

    # Install websocket-client if needed
    try:
        import websocket
    except ImportError:
        print("Installing websocket-client...")
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "websocket-client"]
        )
        import websocket

    # Run tests
    all_passed = True
    session_id = None

    # Test process initiation
    session_id, file_id = test_process_initiation()
    if not session_id:
        all_passed = False
    else:
        # Test processing status
        if not test_processing_status(session_id):
            all_passed = False

        # Test WebSocket (do this early to catch progress updates)
        if not test_websocket_connection(session_id):
            all_passed = False

    # Test session management
    if not test_session_listing():
        all_passed = False

    if session_id:
        if not test_session_details(session_id):
            all_passed = False

        if not test_session_extension(session_id):
            all_passed = False

    if not test_session_statistics():
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All Sprint 2 tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
