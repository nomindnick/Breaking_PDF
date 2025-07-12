"""
API Documentation Router

Provides enhanced OpenAPI documentation with examples and custom styling.
"""
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse

from pdf_splitter.api.config import config

router = APIRouter(tags=["documentation"])


# Custom OpenAPI schema modifications
def custom_openapi_schema(app) -> Dict[str, Any]:
    """Generate custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title=config.api_title,
        version=config.api_version,
        description="""
# PDF Splitter API

An intelligent PDF document splitter that automatically identifies and separates
individual documents within large, multi-document PDF files.

## Features

- 🚀 **High Performance**: Process PDFs at < 5 seconds per page
- 🤖 **AI-Powered**: Multi-signal detection using embeddings and heuristics
- 📄 **Smart Splitting**: Automatic document boundary detection
- 🔄 **Real-time Updates**: WebSocket support for progress tracking
- 🔒 **Secure**: API key authentication and rate limiting
- 📊 **Analytics**: Comprehensive metrics and monitoring

## Getting Started

1. **Authentication**: Include your API key in the `X-API-Key` header
2. **Upload**: Upload your PDF file to `/api/upload`
3. **Process**: Start processing with `/api/process`
4. **Monitor**: Connect to WebSocket for real-time progress
5. **Review**: Get results and adjust boundaries if needed
6. **Download**: Download split PDFs individually or as ZIP

## Rate Limits

- **Default**: 60 requests per minute
- **Upload**: 10 uploads per 5 minutes
- **Download**: 100 downloads per hour

## WebSocket Events

Connect to `/ws/enhanced/{session_id}` for real-time updates:

- `progress`: Processing progress updates
- `status`: Status changes
- `error`: Error notifications
- `heartbeat`: Keep-alive mechanism

## Error Handling

All errors follow a consistent format:

```json
{
  "error": {
    "type": "error_type",
    "message": "Human-readable message",
    "details": {},
    "request_id": "unique-request-id"
  }
}
```

## Support

For issues or questions, please contact support or check our documentation.
        """,
        routes=app.routes,
        tags=[
            {"name": "upload", "description": "File upload operations"},
            {"name": "process", "description": "PDF processing operations"},
            {"name": "sessions", "description": "Session management"},
            {"name": "splits", "description": "Split management and execution"},
            {"name": "results", "description": "Results viewing and analytics"},
            {"name": "download", "description": "File download operations"},
            {"name": "health", "description": "Health checks and monitoring"},
            {"name": "websocket", "description": "WebSocket connections"},
        ],
        servers=[
            {
                "url": f"http://localhost:{config.api_port}",
                "description": "Local development server",
            },
            {"url": "https://api.pdfsplitter.com", "description": "Production server"},
        ],
        components={
            "securitySchemes": {
                "APIKeyHeader": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "API key for authentication",
                },
                "APIKeyQuery": {
                    "type": "apiKey",
                    "in": "query",
                    "name": "api_key",
                    "description": "API key as query parameter (not recommended)",
                },
                "BearerToken": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": "JWT token for authentication",
                },
            }
        },
    )

    # Add security to all endpoints
    for path in openapi_schema["paths"].values():
        for operation in path.values():
            if isinstance(operation, dict):
                operation["security"] = [{"APIKeyHeader": []}, {"BearerToken": []}]

    # Add example responses
    _add_example_responses(openapi_schema)

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def _add_example_responses(schema: Dict[str, Any]):
    """Add example responses to OpenAPI schema."""
    examples = {
        "/api/upload": {
            "post": {
                "200": {
                    "description": "Successful upload",
                    "content": {
                        "application/json": {
                            "example": {
                                "upload_id": "upload_1234567890",
                                "filename": "document.pdf",
                                "size": 1048576,
                                "pages": 32,
                                "message": "File uploaded successfully",
                            }
                        }
                    },
                }
            }
        },
        "/api/process": {
            "post": {
                "200": {
                    "description": "Processing started",
                    "content": {
                        "application/json": {
                            "example": {
                                "session_id": "session_abc123",
                                "status": "processing",
                                "message": "Processing started successfully",
                                "websocket_url": "ws://localhost:8000/ws/enhanced/session_abc123",
                            }
                        }
                    },
                }
            }
        },
    }

    # Merge examples into schema
    for path, methods in examples.items():
        if path in schema["paths"]:
            for method, responses in methods.items():
                if method in schema["paths"][path]:
                    if "responses" not in schema["paths"][path][method]:
                        schema["paths"][path][method]["responses"] = {}
                    schema["paths"][path][method]["responses"].update(responses)


@router.get("/docs", response_class=HTMLResponse, include_in_schema=False)
async def custom_swagger_ui(request: Request):
    """Custom Swagger UI with enhanced styling."""
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title=f"{config.api_title} - Documentation",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        swagger_ui_parameters={
            "persistAuthorization": True,
            "displayRequestDuration": True,
            "filter": True,
            "showExtensions": True,
            "showCommonExtensions": True,
            "defaultModelsExpandDepth": 2,
            "defaultModelExpandDepth": 2,
            "displayOperationId": False,
            "tryItOutEnabled": True,
            "theme": "dark",  # Dark theme
        },
    )


@router.get("/redoc", response_class=HTMLResponse, include_in_schema=False)
async def custom_redoc(request: Request):
    """Custom ReDoc documentation."""
    return get_redoc_html(
        openapi_url="/api/openapi.json",
        title=f"{config.api_title} - Documentation",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js",
        redoc_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        with_google_fonts=True,
    )


@router.get("/postman", response_model=Dict[str, Any])
async def get_postman_collection(request: Request):
    """
    Generate Postman collection for easy API testing.

    Returns a Postman collection JSON that can be imported into Postman.
    """
    base_url = str(request.base_url).rstrip("/")

    collection = {
        "info": {
            "name": config.api_title,
            "description": config.api_description,
            "version": config.api_version,
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "auth": {
            "type": "apikey",
            "apikey": [
                {"key": "key", "value": "X-API-Key"},
                {"key": "value", "value": "{{api_key}}"},
                {"key": "in", "value": "header"},
            ],
        },
        "variable": [
            {"key": "base_url", "value": base_url},
            {"key": "api_key", "value": "your-api-key-here"},
            {"key": "session_id", "value": ""},
        ],
        "item": [
            {
                "name": "Upload",
                "item": [
                    {
                        "name": "Upload PDF",
                        "request": {
                            "method": "POST",
                            "url": "{{base_url}}/api/upload",
                            "body": {
                                "mode": "formdata",
                                "formdata": [
                                    {
                                        "key": "file",
                                        "type": "file",
                                        "src": "/path/to/your.pdf",
                                    }
                                ],
                            },
                        },
                    }
                ],
            },
            {
                "name": "Process",
                "item": [
                    {
                        "name": "Start Processing",
                        "request": {
                            "method": "POST",
                            "url": "{{base_url}}/api/process",
                            "header": [
                                {"key": "Content-Type", "value": "application/json"}
                            ],
                            "body": {
                                "mode": "raw",
                                "raw": '{\n  "file_id": "upload_1234567890"\n}',
                            },
                        },
                    },
                    {
                        "name": "Get Processing Status",
                        "request": {
                            "method": "GET",
                            "url": "{{base_url}}/api/process/{{session_id}}/status",
                        },
                    },
                ],
            },
            {
                "name": "Results",
                "item": [
                    {
                        "name": "Get Results",
                        "request": {
                            "method": "GET",
                            "url": "{{base_url}}/api/results/{{session_id}}",
                        },
                    }
                ],
            },
            {
                "name": "Download",
                "item": [
                    {
                        "name": "Download File",
                        "request": {
                            "method": "GET",
                            "url": "{{base_url}}/api/download/{{session_id}}/document.pdf",
                        },
                    },
                    {
                        "name": "Download ZIP",
                        "request": {
                            "method": "GET",
                            "url": "{{base_url}}/api/download/{{session_id}}/zip",
                        },
                    },
                ],
            },
            {
                "name": "Health",
                "item": [
                    {
                        "name": "Health Check",
                        "request": {"method": "GET", "url": "{{base_url}}/api/health"},
                    }
                ],
            },
        ],
    }

    return collection


@router.get("/examples", response_model=Dict[str, Any])
async def get_api_examples():
    """
    Get code examples for using the API.

    Returns examples in multiple programming languages.
    """
    return {
        "python": {
            "description": "Python example using requests library",
            "code": """import requests
import time

# Configuration
API_KEY = "your-api-key"
BASE_URL = "http://localhost:8000"

headers = {"X-API-Key": API_KEY}

# Step 1: Upload PDF
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/api/upload", headers=headers, files=files)
    upload_data = response.json()
    file_id = upload_data["upload_id"]

# Step 2: Start processing
response = requests.post(
    f"{BASE_URL}/api/process",
    headers=headers,
    json={"file_id": file_id}
)
process_data = response.json()
session_id = process_data["session_id"]

# Step 3: Wait for completion
while True:
    response = requests.get(f"{BASE_URL}/api/process/{session_id}/status", headers=headers)
    status = response.json()["status"]

    if status == "confirmed":
        break
    elif status == "cancelled":
        print("Processing failed")
        exit(1)

    time.sleep(2)

# Step 4: Get results
response = requests.get(f"{BASE_URL}/api/results/{session_id}", headers=headers)
results = response.json()
print(f"Created {results['files_created']} files")

# Step 5: Download files
response = requests.get(f"{BASE_URL}/api/download/{session_id}/zip", headers=headers)
with open("output.zip", "wb") as f:
    f.write(response.content)
""",
        },
        "javascript": {
            "description": "JavaScript example using fetch API",
            "code": """const API_KEY = 'your-api-key';
const BASE_URL = 'http://localhost:8000';

async function processPDF(file) {
    const headers = { 'X-API-Key': API_KEY };

    // Step 1: Upload PDF
    const formData = new FormData();
    formData.append('file', file);

    const uploadResponse = await fetch(`${BASE_URL}/api/upload`, {
        method: 'POST',
        headers: headers,
        body: formData
    });
    const { upload_id } = await uploadResponse.json();

    // Step 2: Start processing
    const processResponse = await fetch(`${BASE_URL}/api/process`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_id: upload_id })
    });
    const { session_id } = await processResponse.json();

    // Step 3: Monitor progress via WebSocket
    const ws = new WebSocket(`ws://localhost:8000/ws/enhanced/${session_id}`);

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if (message.type === 'progress') {
            console.log(`Progress: ${message.data.progress}%`);
        } else if (message.type === 'status' && message.data.status === 'complete') {
            console.log('Processing complete!');
            ws.close();
        }
    };

    return session_id;
}
""",
        },
        "curl": {
            "description": "Command line example using curl",
            "code": """# Set your API key
API_KEY="your-api-key"
BASE_URL="http://localhost:8000"

# Upload PDF
UPLOAD_RESPONSE=$(curl -X POST \\
  -H "X-API-Key: $API_KEY" \\
  -F "file=@document.pdf" \\
  $BASE_URL/api/upload)

FILE_ID=$(echo $UPLOAD_RESPONSE | jq -r '.upload_id')

# Start processing
PROCESS_RESPONSE=$(curl -X POST \\
  -H "X-API-Key: $API_KEY" \\
  -H "Content-Type: application/json" \\
  -d "{\\"file_id\\": \\"$FILE_ID\\"}" \\
  $BASE_URL/api/process)

SESSION_ID=$(echo $PROCESS_RESPONSE | jq -r '.session_id')

# Check status
curl -H "X-API-Key: $API_KEY" \\
  $BASE_URL/api/process/$SESSION_ID/status

# Download results as ZIP
curl -H "X-API-Key: $API_KEY" \\
  -o output.zip \\
  $BASE_URL/api/download/$SESSION_ID/zip
""",
        },
    }
