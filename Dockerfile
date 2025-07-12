# PDF Splitter API Dockerfile
# Multi-stage build for optimized production image

# Stage 1: Builder
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    wget \
    # Additional dependencies for PDF processing
    libpoppler-cpp-dev \
    libpoppler-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    # Required for OpenCV and image processing
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    # Required for PDF processing
    libpoppler-cpp0v5 \
    poppler-utils \
    # Health check and monitoring
    curl \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and necessary directories
RUN useradd -m -u 1000 pdfuser && \
    mkdir -p /app /data/uploads /data/outputs /data/sessions /data/cache /data/logs /data/db && \
    chown -R pdfuser:pdfuser /app /data

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=pdfuser:pdfuser . .

# Ensure scripts are executable
RUN chmod +x /app/scripts/*.sh 2>/dev/null || true

# Switch to non-root user
USER pdfuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_THREAD_LIMIT=1 \
    UPLOAD_DIR=/data/uploads \
    OUTPUT_DIR=/data/outputs \
    SESSION_DB_PATH=/data/sessions/sessions.db \
    LOG_LEVEL=INFO

# Expose port
EXPOSE 8000

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting PDF Splitter API..."\n\
echo "Environment: ${DEBUG:-false}"\n\
echo "Workers: ${API_WORKERS:-4}"\n\
echo "Upload dir: ${UPLOAD_DIR:-/data/uploads}"\n\
echo "Output dir: ${OUTPUT_DIR:-/data/outputs}"\n\
\n\
# Run migrations if needed (placeholder for future)\n\
# python -m pdf_splitter.db.migrate\n\
\n\
# Start the application\n\
exec uvicorn pdf_splitter.api.main:app \\\n\
    --host 0.0.0.0 \\\n\
    --port ${API_PORT:-8000} \\\n\
    --workers ${API_WORKERS:-4} \\\n\
    --log-level ${LOG_LEVEL:-info} \\\n\
    --access-log \\\n\
    --use-colors\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/api/health || exit 1

# Metadata
LABEL maintainer="PDF Splitter Team" \
      version="1.0.0" \
      description="Intelligent PDF document splitter with ML-powered boundary detection"

# Run the application via entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
