"""
API Configuration Module.

Manages all configuration settings for the PDF Splitter API.
"""
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, validator


class APIConfig(BaseSettings):
    """API Configuration using Pydantic BaseSettings for environment variable support."""

    # API Settings
    api_title: str = "PDF Splitter API"
    api_description: str = "API for intelligent PDF document splitting"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # CORS Settings
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    # File Upload Settings
    upload_dir: Path = Path("uploads")
    output_dir: Path = Path("outputs")
    max_upload_size: int = 100 * 1024 * 1024  # 100MB in bytes
    allowed_extensions: set[str] = {".pdf"}
    chunk_size: int = 1024 * 1024  # 1MB chunks for reading

    # Session Settings
    session_timeout: int = 86400  # 24 hours in seconds
    session_db_path: Path = Path("sessions.db")
    cleanup_interval: int = 3600  # 1 hour in seconds

    # Processing Settings
    max_concurrent_processes: int = 4
    process_timeout: int = 300  # 5 minutes in seconds

    # Security Settings
    secret_key: str = "your-secret-key-here"  # Override in production
    api_key_header: str = "X-API-Key"
    require_api_key: bool = False
    api_key_enabled: bool = False  # Enable API key authentication
    jwt_secret_key: str = "your-jwt-secret-key"  # Override in production
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60

    # WebSocket Settings
    websocket_url: str = "ws://localhost:8000"
    websocket_heartbeat_interval: int = 30  # seconds
    websocket_max_connections_per_session: int = 10
    websocket_max_total_connections: int = 1000
    require_websocket_auth: bool = False  # Set to True in production
    websocket_token_expiry: int = 3600  # 1 hour in seconds

    # Logging Settings
    log_level: str = "INFO"
    log_file: Optional[Path] = Path("logs/api.log")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Development Settings
    debug: bool = False
    reload: bool = False

    # Rate Limiting Settings
    rate_limit_enabled: bool = True
    rate_limit_default: int = 60  # requests per minute
    rate_limit_upload: int = 10  # uploads per 5 minutes
    rate_limit_download: int = 100  # downloads per hour

    # Monitoring Settings
    metrics_enabled: bool = True
    metrics_port: int = 9090
    prometheus_multiproc_dir: str = "/tmp/prometheus"

    # Download Settings
    download_token_expiry: int = 3600  # 1 hour
    download_chunk_size: int = 1024 * 1024  # 1MB

    # Health Check Settings
    health_check_interval: int = 60  # seconds
    health_check_timeout: int = 10  # seconds

    @validator("upload_dir", "output_dir", "log_file", pre=True)
    def create_directories(cls, v):
        """Ensure directories exist."""
        if v:
            path = Path(v)
            if path.suffix:  # It's a file
                path.parent.mkdir(parents=True, exist_ok=True)
            else:  # It's a directory
                path.mkdir(parents=True, exist_ok=True)
            return path
        return v

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_prefix = "API_"
        case_sensitive = False


# Global configuration instance
config = APIConfig()
