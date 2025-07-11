# PDF Splitter API Module

## Overview

The API module provides a RESTful API and WebSocket support for the PDF Splitter application. It handles file uploads, document boundary detection, session management, split operations, and real-time progress updates.

## Architecture

### Directory Structure

```
api/
├── __init__.py         # Module initialization
├── dependencies.py     # Shared dependencies and DI
├── exceptions.py       # API-specific exceptions
├── middleware.py       # Custom middleware
├── router.py          # Main API router
├── models/            # Pydantic models
│   ├── requests.py    # Request validation models
│   ├── responses.py   # Response models
│   └── websocket.py   # WebSocket message models
├── routes/            # API endpoints
│   ├── upload.py      # File upload endpoints
│   ├── sessions.py    # Session management
│   ├── detection.py   # Boundary detection
│   ├── splitting.py   # Split operations
│   └── websocket.py   # WebSocket endpoint
└── services/          # Business logic
    ├── upload_service.py     # Upload handling
    ├── detection_service.py  # Detection coordination
    ├── splitting_service.py  # Split execution
    └── progress_service.py   # Progress tracking
```

## API Endpoints

### Upload Endpoints

#### `POST /api/upload/file`
Upload a PDF file for processing.

**Request:**
- Body: multipart/form-data with PDF file
- Query params:
  - `validate_only` (bool): Only validate without storing

**Response:**
```json
{
  "success": true,
  "upload_id": "uuid",
  "file_name": "document.pdf",
  "file_size": 1048576,
  "total_pages": 10,
  "status": "uploaded",
  "processing_time": 0.5
}
```

#### `GET /api/upload/{upload_id}/status`
Check upload status.

### Session Management

#### `POST /api/sessions/create`
Create a new split session.

**Request:**
```json
{
  "upload_id": "uuid",
  "session_name": "My Split Session",
  "expires_in_hours": 24
}
```

#### `GET /api/sessions/{session_id}`
Get session details.

#### `GET /api/sessions`
List all sessions with optional filtering.

### Detection Endpoints

#### `POST /api/detection/start`
Start boundary detection.

**Request:**
```json
{
  "upload_id": "uuid",
  "detector_type": "embeddings",
  "confidence_threshold": 0.5
}
```

#### `GET /api/detection/{detection_id}/status`
Get detection progress.

#### `GET /api/detection/{detection_id}/results`
Get detection results.

### Split Management

#### `GET /api/splits/{session_id}/proposal`
Get current split proposal.

#### `PUT /api/splits/{session_id}/segments/{segment_id}`
Update a segment.

**Request:**
```json
{
  "suggested_filename": "Invoice_2024.pdf",
  "document_type": "Invoice",
  "start_page": 0,
  "end_page": 2
}
```

#### `POST /api/splits/{session_id}/segments`
Add a new segment.

#### `DELETE /api/splits/{session_id}/segments/{segment_id}`
Remove a segment.

#### `POST /api/splits/{session_id}/preview/{segment_id}`
Generate segment preview.

#### `POST /api/splits/{session_id}/execute`
Execute split operation.

**Request:**
```json
{
  "output_format": "pdf",
  "compress": false,
  "create_zip": true,
  "preserve_metadata": true,
  "generate_manifest": true
}
```

#### `GET /api/splits/{split_id}/download/{filename}`
Download split output file.

## WebSocket API

### Connection

Connect to `ws://localhost:8000/ws` or `ws://localhost:8000/ws/{session_id}` for auto-subscription.

### Client Messages

#### Subscribe
```json
{
  "type": "subscribe",
  "session_id": "uuid",
  "include_previews": true
}
```

#### Unsubscribe
```json
{
  "type": "unsubscribe",
  "session_id": "uuid"
}
```

#### Ping
```json
{
  "type": "ping"
}
```

### Server Messages

#### Progress Update
```json
{
  "type": "progress",
  "session_id": "uuid",
  "stage": "detection",
  "progress": 0.45,
  "message": "Processing page 45 of 100",
  "current_item": 45,
  "total_items": 100
}
```

#### Stage Complete
```json
{
  "type": "stage_complete",
  "session_id": "uuid",
  "stage": "detection",
  "success": true,
  "message": "Detection completed",
  "duration_seconds": 12.5,
  "next_stage": "splitting"
}
```

#### Error
```json
{
  "type": "error",
  "session_id": "uuid",
  "error_code": "DETECTION_FAILED",
  "message": "Failed to detect boundaries",
  "recoverable": false
}
```

## Error Handling

### Custom Exceptions

- `UploadError`: File upload failures
- `ValidationError`: Request validation errors
- `SessionNotFoundError`: Session doesn't exist
- `SessionExpiredError`: Session has expired
- `SessionStateError`: Invalid session state
- `DetectionError`: Detection failures
- `SplitError`: Split operation failures

### HTTP Status Codes

- `200`: Success
- `201`: Created
- `400`: Bad Request
- `404`: Not Found
- `409`: Conflict
- `410`: Gone (expired)
- `413`: Payload Too Large
- `415`: Unsupported Media Type
- `422`: Unprocessable Entity
- `429`: Too Many Requests
- `500`: Internal Server Error

## Middleware

### Request Logging
Logs all requests with unique request IDs.

### Error Handling
Converts exceptions to JSON responses.

### Rate Limiting
Limits requests per IP address (120/minute default).

### CORS
Handles cross-origin requests.

## Usage Example

```python
import asyncio
import httpx
import websockets
import json

async def example_workflow():
    # 1. Upload PDF
    async with httpx.AsyncClient() as client:
        files = {"file": open("document.pdf", "rb")}
        response = await client.post(
            "http://localhost:8000/api/upload/file",
            files=files
        )
        upload_data = response.json()
        upload_id = upload_data["upload_id"]

    # 2. Create session
    session_response = await client.post(
        "http://localhost:8000/api/sessions/create",
        json={"upload_id": upload_id}
    )
    session_data = session_response.json()
    session_id = session_data["session_id"]

    # 3. Connect WebSocket for progress
    async with websockets.connect(f"ws://localhost:8000/ws") as ws:
        # Subscribe to session
        await ws.send(json.dumps({
            "type": "subscribe",
            "session_id": session_id
        }))

        # 4. Start detection
        detection_response = await client.post(
            "http://localhost:8000/api/detection/start",
            json={
                "upload_id": upload_id,
                "detector_type": "embeddings"
            }
        )

        # 5. Listen for progress
        while True:
            message = json.loads(await ws.recv())
            print(f"Progress: {message}")

            if message["type"] == "stage_complete":
                if message["stage"] == "detection":
                    break

        # 6. Review and modify proposal
        proposal_response = await client.get(
            f"http://localhost:8000/api/splits/{session_id}/proposal"
        )
        proposal = proposal_response.json()

        # 7. Execute split
        split_response = await client.post(
            f"http://localhost:8000/api/splits/{session_id}/execute"
        )

        # 8. Wait for completion
        while True:
            message = json.loads(await ws.recv())
            if message["type"] == "split_complete":
                break

        # 9. Download results
        # ... download files ...

if __name__ == "__main__":
    asyncio.run(example_workflow())
```

## Configuration

Environment variables:
- `PDF_UPLOAD_DIR`: Directory for uploads
- `PDF_MAX_FILE_SIZE_MB`: Maximum file size
- `CORS_ORIGINS`: Allowed CORS origins

## Testing

See `tests/` directory for comprehensive test suite including:
- Unit tests for all endpoints
- Integration tests for workflows
- WebSocket connection tests
- Error handling tests

## Performance Considerations

- File uploads are streamed in chunks
- Detection runs asynchronously with progress tracking
- WebSocket connections are managed efficiently
- Rate limiting prevents abuse
- Sessions expire after 24 hours by default
