# API Implementation Plan for PDF Splitter

## Overview
This document outlines a detailed implementation plan for the FastAPI backend that will connect the frontend to the existing PDF processing modules. The API will handle file uploads, manage processing sessions, provide real-time updates, and serve results.

## Technology Stack
- **Framework**: FastAPI (for async support, automatic OpenAPI docs, WebSocket support)
- **Database**: SQLite for session management (already implemented in splitting module)
- **File Storage**: Local filesystem with configurable paths
- **WebSocket**: For real-time progress updates
- **Background Tasks**: FastAPI BackgroundTasks for async processing
- **Validation**: Pydantic models for request/response validation

## API Architecture

### Core Components
```
pdf_splitter/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app initialization
│   ├── config.py            # API configuration
│   ├── dependencies.py      # Shared dependencies
│   ├── middleware.py        # CORS, error handling
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── upload.py        # File upload endpoints
│   │   ├── process.py       # Processing endpoints
│   │   ├── splits.py        # Split management endpoints
│   │   ├── sessions.py      # Session management endpoints
│   │   ├── websocket.py     # WebSocket connections
│   │   └── health.py        # Health check endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py      # Request models
│   │   ├── responses.py     # Response models
│   │   └── events.py        # WebSocket event models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── file_service.py  # File handling
│   │   ├── process_service.py # PDF processing orchestration
│   │   ├── session_service.py # Session management
│   │   └── websocket_service.py # WebSocket management
│   └── utils/
│       ├── __init__.py
│       ├── file_utils.py    # File validation, storage
│       ├── auth.py          # Optional authentication
│       └── exceptions.py    # Custom exceptions
```

## Implementation Sprints

---

## Sprint 1: Core API Setup & File Upload (3 hours)

### Goals
- Set up FastAPI application structure
- Implement file upload with validation
- Create file storage service
- Add health check endpoints

### Tasks
1. Create FastAPI application structure
2. Implement file upload endpoint with validation
3. Create file storage service
4. Add CORS middleware and error handling
5. Implement health check endpoints

### Endpoints
```python
# POST /api/upload
# - Accept PDF files up to 100MB
# - Validate file type and size
# - Store file and return upload ID
# - Support for chunked uploads

# GET /api/health
# - Basic health check

# GET /api/health/detailed
# - Detailed system status (DB, storage, etc.)
```

### Key Implementation Details
```python
# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="PDF Splitter API",
    description="API for intelligent PDF document splitting",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# api/routers/upload.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict
import aiofiles
import hashlib

router = APIRouter(prefix="/api", tags=["upload"])

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_number: int = 0,
    total_chunks: int = 1
) -> Dict[str, str]:
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are allowed")

    # Validate file size (100MB limit)
    if file.size > 100 * 1024 * 1024:
        raise HTTPException(400, "File size exceeds 100MB limit")

    # Generate unique file ID
    file_id = hashlib.sha256(f"{file.filename}{time.time()}".encode()).hexdigest()[:16]

    # Handle chunked upload
    if total_chunks > 1:
        return await handle_chunked_upload(file, file_id, chunk_number, total_chunks)

    # Save file
    file_path = await save_uploaded_file(file, file_id)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "size": file.size,
        "status": "uploaded"
    }
```

---

## Sprint 2: Processing Endpoints & Session Management (3 hours)

### Goals
- Implement PDF processing initiation
- Create session management endpoints
- Integrate with existing processing modules
- Add progress tracking infrastructure

### Tasks
1. Create process initiation endpoint
2. Implement session CRUD operations
3. Integrate with preprocessing and detection modules
4. Add background task processing
5. Create progress tracking service

### Endpoints
```python
# POST /api/process
# - Initiate PDF processing for uploaded file
# - Create session and return session ID
# - Start background processing task

# GET /api/sessions
# - List all sessions with filtering/pagination
# - Support status, date range filters

# GET /api/sessions/{session_id}
# - Get session details and current status

# POST /api/sessions/{session_id}/extend
# - Extend session expiration time

# DELETE /api/sessions/{session_id}
# - Delete session and associated data
```

### Key Implementation Details
```python
# api/routers/process.py
from fastapi import APIRouter, BackgroundTasks
from pdf_splitter.preprocessing import PDFHandler
from pdf_splitter.detection import create_production_detector
from pdf_splitter.splitting import PDFSplitter, SessionManager

router = APIRouter(prefix="/api", tags=["process"])

@router.post("/process")
async def process_pdf(
    file_id: str,
    background_tasks: BackgroundTasks,
    session_manager: SessionManager = Depends(get_session_manager)
) -> Dict[str, str]:
    # Create new session
    session = session_manager.create_session({
        "file_id": file_id,
        "status": "processing",
        "created_at": datetime.now()
    })

    # Start background processing
    background_tasks.add_task(
        process_pdf_task,
        session.session_id,
        file_id
    )

    return {
        "session_id": session.session_id,
        "status": "processing_started",
        "message": "PDF processing initiated"
    }

async def process_pdf_task(session_id: str, file_id: str):
    try:
        # Update progress: Loading PDF
        await update_progress(session_id, "loading", 10)

        # Load and preprocess PDF
        pdf_handler = PDFHandler()
        pdf_path = get_file_path(file_id)
        pdf_doc = pdf_handler.load_pdf(pdf_path)

        # Update progress: Extracting text
        await update_progress(session_id, "extracting", 30)

        # Process pages
        pages = await pdf_handler.process_all_pages(pdf_doc)

        # Update progress: Detecting boundaries
        await update_progress(session_id, "detecting", 60)

        # Detect boundaries
        detector = create_production_detector()
        boundaries = detector.detect_boundaries(pages)

        # Update progress: Generating proposal
        await update_progress(session_id, "proposing", 80)

        # Generate split proposal
        splitter = PDFSplitter()
        proposal = splitter.generate_proposal(pdf_path, boundaries)

        # Save proposal to session
        await save_proposal(session_id, proposal)

        # Update progress: Complete
        await update_progress(session_id, "complete", 100)

    except Exception as e:
        await update_progress(session_id, "error", 0, str(e))
```

---

## Sprint 3: Split Management & Execution (3 hours)

### Goals
- Implement split proposal endpoints
- Add boundary modification capabilities
- Create split execution endpoint
- Implement preview generation

### Tasks
1. Create proposal retrieval endpoint
2. Implement boundary modification endpoints
3. Add split execution with progress tracking
4. Create preview image generation
5. Handle document metadata updates

### Endpoints
```python
# GET /api/splits/{session_id}/proposal
# - Get current split proposal with segments

# PUT /api/splits/{session_id}/proposal
# - Update split proposal (modify boundaries)

# POST /api/splits/{session_id}/merge
# - Merge specified segments

# POST /api/splits/{session_id}/split
# - Split segment at specified page

# GET /api/splits/{session_id}/preview/{segment_id}
# - Get preview image for segment

# POST /api/splits/{session_id}/execute
# - Execute the split operation
```

### Key Implementation Details
```python
# api/routers/splits.py
@router.get("/splits/{session_id}/proposal")
async def get_split_proposal(
    session_id: str,
    session_service: SessionService = Depends()
) -> SplitProposalResponse:
    proposal = await session_service.get_proposal(session_id)
    if not proposal:
        raise HTTPException(404, "Proposal not found")

    return SplitProposalResponse(
        session_id=session_id,
        segments=[
            DocumentSegmentResponse(
                id=seg.id,
                start_page=seg.start_page,
                end_page=seg.end_page,
                title=seg.title,
                document_type=seg.document_type,
                confidence=seg.confidence,
                preview_text=seg.preview_text[:200]
            )
            for seg in proposal.segments
        ],
        total_pages=proposal.total_pages
    )

@router.post("/splits/{session_id}/execute")
async def execute_split(
    session_id: str,
    background_tasks: BackgroundTasks,
    splitter: PDFSplitter = Depends()
) -> Dict[str, str]:
    # Get current proposal
    proposal = await get_proposal(session_id)

    # Start split execution in background
    background_tasks.add_task(
        execute_split_task,
        session_id,
        proposal
    )

    return {
        "session_id": session_id,
        "status": "splitting_started",
        "message": "PDF split operation initiated"
    }

async def execute_split_task(session_id: str, proposal: SplitProposal):
    try:
        # Update progress
        await update_progress(session_id, "splitting", 0)

        # Execute split
        output_dir = get_output_dir(session_id)
        result = await splitter.split_pdf(proposal, output_dir)

        # Save results
        await save_results(session_id, result)

        # Update progress
        await update_progress(session_id, "split_complete", 100)

    except Exception as e:
        await update_progress(session_id, "split_error", 0, str(e))
```

---

## Sprint 4: WebSocket & Real-time Updates (2 hours)

### Goals
- Implement WebSocket endpoint for real-time updates
- Create progress broadcasting system
- Handle connection management
- Add reconnection support

### Tasks
1. Create WebSocket endpoint
2. Implement connection manager
3. Add progress event broadcasting
4. Handle reconnection logic
5. Create typed event system

### Implementation
```python
# api/routers/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        self.active_connections[session_id].add(websocket)

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

    async def broadcast_progress(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(data)
                except:
                    disconnected.add(connection)

            # Clean up disconnected clients
            for conn in disconnected:
                self.disconnect(conn, session_id)

manager = ConnectionManager()

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        # Send initial status
        current_status = await get_session_status(session_id)
        await websocket.send_json({
            "type": "status",
            "data": current_status
        })

        # Keep connection alive
        while True:
            # Wait for client messages (ping/pong)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)

# Progress update function used by background tasks
async def update_progress(
    session_id: str,
    stage: str,
    progress: int,
    message: str = None
):
    data = {
        "type": "progress",
        "data": {
            "stage": stage,
            "progress": progress,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
    }

    # Save to database
    await save_progress_to_db(session_id, data["data"])

    # Broadcast to connected clients
    await manager.broadcast_progress(session_id, data)
```

---

## Sprint 5: Results & File Downloads (2 hours)

### Goals
- Implement results retrieval endpoints
- Add file download capabilities
- Create ZIP download for bulk files
- Implement cleanup and retention

### Tasks
1. Create results listing endpoint
2. Implement individual file downloads
3. Add bulk ZIP download
4. Implement file cleanup service
5. Add download progress tracking

### Endpoints
```python
# GET /api/splits/{session_id}/results
# - Get list of split files with metadata

# GET /api/splits/{session_id}/download/{filename}
# - Download individual split file

# GET /api/splits/{session_id}/download/zip
# - Download all files as ZIP

# POST /api/splits/{session_id}/save
# - Save session to permanent storage
```

### Implementation
```python
# api/routers/splits.py (continued)
from fastapi.responses import FileResponse, StreamingResponse
import zipfile
import io

@router.get("/splits/{session_id}/results")
async def get_split_results(
    session_id: str,
    session_service: SessionService = Depends()
) -> SplitResultsResponse:
    results = await session_service.get_results(session_id)
    if not results:
        raise HTTPException(404, "Results not found")

    files = []
    for file_path in results.output_files:
        stat = os.stat(file_path)
        files.append({
            "filename": os.path.basename(file_path),
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime),
            "document_type": extract_doc_type(file_path),
            "page_count": get_pdf_page_count(file_path)
        })

    return SplitResultsResponse(
        session_id=session_id,
        files=files,
        total_documents=len(files),
        processing_time=results.processing_time
    )

@router.get("/splits/{session_id}/download/{filename}")
async def download_file(
    session_id: str,
    filename: str,
    session_service: SessionService = Depends()
):
    # Validate session and filename
    file_path = await session_service.get_file_path(session_id, filename)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(404, "File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )

@router.get("/splits/{session_id}/download/zip")
async def download_all_as_zip(
    session_id: str,
    session_service: SessionService = Depends()
):
    results = await session_service.get_results(session_id)
    if not results:
        raise HTTPException(404, "Results not found")

    # Create ZIP file in memory
    zip_io = io.BytesIO()

    with zipfile.ZipFile(zip_io, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in results.output_files:
            if os.path.exists(file_path):
                arcname = os.path.basename(file_path)
                zip_file.write(file_path, arcname)

    zip_io.seek(0)

    return StreamingResponse(
        zip_io,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=split_results_{session_id}.zip"
        }
    )
```

---

## Sprint 6: Error Handling & Production Features (2 hours)

### Goals
- Implement comprehensive error handling
- Add request validation and rate limiting
- Create API documentation
- Add monitoring and logging
- Implement cleanup tasks

### Tasks
1. Create custom exception handlers
2. Add request validation middleware
3. Implement rate limiting
4. Add comprehensive logging
5. Create background cleanup tasks
6. Generate OpenAPI documentation

### Implementation
```python
# api/middleware.py
from fastapi import Request, status
from fastapi.responses import JSONResponse
import time
import logging

logger = logging.getLogger(__name__)

async def error_handler_middleware(request: Request, call_next):
    try:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)

        # Log request
        logger.info(
            f"{request.method} {request.url.path} "
            f"completed in {process_time:.3f}s"
        )

        return response

    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "type": "internal_error",
                "timestamp": datetime.now().isoformat()
            }
        )

# api/utils/exceptions.py
from fastapi import HTTPException

class PDFProcessingError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=422,
            detail={
                "message": detail,
                "type": "processing_error",
                "timestamp": datetime.now().isoformat()
            }
        )

class SessionNotFoundError(HTTPException):
    def __init__(self, session_id: str):
        super().__init__(
            status_code=404,
            detail={
                "message": f"Session {session_id} not found",
                "type": "session_not_found",
                "timestamp": datetime.now().isoformat()
            }
        )

# api/dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
import time

# Rate limiting
redis_client = redis.Redis(host='localhost', port=6379, db=0)

async def rate_limit(
    request: Request,
    calls: int = 10,
    period: int = 60
):
    # Get client IP
    client_ip = request.client.host
    key = f"rate_limit:{client_ip}:{request.url.path}"

    try:
        current = redis_client.incr(key)
        if current == 1:
            redis_client.expire(key, period)

        if current > calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "message": "Rate limit exceeded",
                    "retry_after": redis_client.ttl(key)
                }
            )
    except redis.RedisError:
        # Don't fail if Redis is down
        logger.warning("Redis unavailable for rate limiting")

# Background cleanup task
async def cleanup_old_sessions():
    """Run periodically to clean up old sessions and files"""
    while True:
        try:
            # Clean up sessions older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            old_sessions = await get_old_sessions(cutoff_time)

            for session in old_sessions:
                # Delete files
                await delete_session_files(session.session_id)
                # Delete session
                await delete_session(session.session_id)

            logger.info(f"Cleaned up {len(old_sessions)} old sessions")

        except Exception as e:
            logger.error(f"Cleanup task error: {e}")

        # Wait 1 hour before next cleanup
        await asyncio.sleep(3600)
```

---

## Testing Strategy

### Unit Tests
```python
# tests/api/test_upload.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_upload_valid_pdf():
    with open("test_files/sample.pdf", "rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    assert response.status_code == 200
    assert "file_id" in response.json()

def test_upload_invalid_file_type():
    response = client.post(
        "/api/upload",
        files={"file": ("test.txt", b"content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only PDF files" in response.json()["detail"]

# tests/api/test_websocket.py
from fastapi.testclient import TestClient

def test_websocket_connection():
    with client.websocket_connect("/ws/test-session") as websocket:
        # Should receive initial status
        data = websocket.receive_json()
        assert data["type"] == "status"

        # Test ping/pong
        websocket.send_text("ping")
        response = websocket.receive_text()
        assert response == "pong"
```

### Integration Tests
```python
# tests/api/test_integration.py
@pytest.mark.asyncio
async def test_full_processing_workflow():
    # 1. Upload file
    upload_response = await upload_test_pdf()
    file_id = upload_response["file_id"]

    # 2. Start processing
    process_response = await start_processing(file_id)
    session_id = process_response["session_id"]

    # 3. Wait for completion (with timeout)
    await wait_for_completion(session_id, timeout=60)

    # 4. Get results
    results = await get_results(session_id)
    assert len(results["files"]) > 0

    # 5. Download file
    file_content = await download_file(
        session_id,
        results["files"][0]["filename"]
    )
    assert len(file_content) > 0
```

---

## Deployment Configuration

### Docker Setup
```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p uploads outputs logs

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Environment Variables
```bash
# .env.production
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# File Storage
UPLOAD_DIR=/app/uploads
OUTPUT_DIR=/app/outputs
MAX_UPLOAD_SIZE=104857600  # 100MB

# Session Management
SESSION_TIMEOUT=86400  # 24 hours
CLEANUP_INTERVAL=3600  # 1 hour

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Rate Limiting
RATE_LIMIT_CALLS=10
RATE_LIMIT_PERIOD=60

# Processing Configuration
MAX_CONCURRENT_PROCESSES=4
PROCESS_TIMEOUT=300  # 5 minutes

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/api.log
```

---

## Monitoring & Observability

### Health Checks
```python
# api/routers/health.py
@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.get("/health/detailed")
async def detailed_health_check(
    session_manager: SessionManager = Depends()
):
    checks = {
        "database": await check_database_health(),
        "storage": check_storage_health(),
        "redis": await check_redis_health(),
        "processing_queue": await check_queue_health()
    }

    overall_status = "healthy" if all(
        check["status"] == "healthy" for check in checks.values()
    ) else "degraded"

    return {
        "status": overall_status,
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }
```

### Metrics
```python
# api/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
pdf_uploads_total = Counter('pdf_uploads_total', 'Total PDF uploads')
pdf_processing_duration = Histogram('pdf_processing_duration_seconds', 'PDF processing duration')
active_sessions = Gauge('active_sessions', 'Number of active sessions')
websocket_connections = Gauge('websocket_connections', 'Active WebSocket connections')

# Usage in endpoints
@router.post("/upload")
async def upload_pdf(...):
    pdf_uploads_total.inc()
    # ... rest of implementation
```

---

## Security Considerations

1. **Input Validation**
   - File type validation (PDF only)
   - File size limits
   - Filename sanitization
   - Path traversal prevention

2. **Rate Limiting**
   - Per-IP rate limiting
   - Endpoint-specific limits
   - WebSocket connection limits

3. **Authentication (Optional)**
   - JWT token support
   - API key authentication
   - Session-based auth for web UI

4. **Data Security**
   - Temporary file cleanup
   - Secure file storage
   - No sensitive data in logs

---

## Performance Optimizations

1. **Async Processing**
   - Background tasks for heavy operations
   - Non-blocking file I/O
   - Concurrent request handling

2. **Caching**
   - Redis for session data
   - File system cache for previews
   - Response caching for static data

3. **Resource Management**
   - Connection pooling
   - Memory limits for processing
   - Automatic cleanup tasks

4. **Scalability**
   - Horizontal scaling with multiple workers
   - Queue-based processing for large files
   - Load balancer ready

---

## API Documentation

FastAPI automatically generates OpenAPI (Swagger) documentation available at:
- `/docs` - Interactive API documentation
- `/redoc` - Alternative API documentation
- `/openapi.json` - OpenAPI schema

---

## Success Criteria

1. **Functional Requirements**
   - ✓ All frontend endpoints implemented
   - ✓ Real-time progress updates via WebSocket
   - ✓ File upload/download capabilities
   - ✓ Session management with persistence
   - ✓ Error handling and recovery

2. **Performance Requirements**
   - ✓ < 5 second response time for API calls
   - ✓ Support for 100MB PDF files
   - ✓ Concurrent processing support
   - ✓ Efficient file streaming

3. **Quality Requirements**
   - ✓ 80%+ test coverage
   - ✓ Comprehensive error handling
   - ✓ API documentation
   - ✓ Logging and monitoring

---

## Timeline

- **Sprint 1**: Core Setup & Upload (3 hours)
- **Sprint 2**: Processing & Sessions (3 hours)
- **Sprint 3**: Split Management (3 hours)
- **Sprint 4**: WebSocket Updates (2 hours)
- **Sprint 5**: Results & Downloads (2 hours)
- **Sprint 6**: Production Features (2 hours)

**Total Estimated Time**: 15 hours (2-3 days)

---

## Next Steps

1. Review and approve this implementation plan
2. Set up development environment for API
3. Begin Sprint 1 implementation
4. Iterate based on testing and feedback
5. Deploy to production environment
