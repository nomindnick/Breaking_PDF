"""Configuration management for PDF Splitter application."""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PDFConfig(BaseModel):
    """Configuration for PDF handling and processing."""

    # DPI Settings
    default_dpi: int = Field(
        150, ge=72, le=600, description="Default DPI for rendering"
    )
    max_dpi: int = Field(300, ge=72, le=600, description="Maximum allowed DPI")

    # Processing Limits
    max_file_size_mb: float = Field(
        500.0, gt=0, description="Maximum PDF file size in MB"
    )
    max_pages: int = Field(10000, gt=0, description="Maximum pages to process")
    page_cache_size: int = Field(
        10, ge=0, description="Number of pages to cache in memory"
    )
    stream_batch_size: int = Field(
        5, ge=1, le=50, description="Default batch size for streaming"
    )

    # Quality Thresholds
    min_text_confidence: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum confidence for text extraction"
    )
    min_text_coverage_percent: float = Field(
        5.0,
        ge=0.0,
        le=100.0,
        description="Minimum text coverage to consider page searchable",
    )

    # Performance Settings
    analysis_threads: int = Field(
        4, ge=1, le=16, description="Threads for parallel page analysis"
    )
    timeout_per_page: float = Field(
        30.0, gt=0, description="Timeout in seconds per page"
    )
    enable_repair: bool = Field(True, description="Enable automatic PDF repair")

    # Cache Settings
    render_cache_memory_mb: int = Field(
        100, ge=10, le=1000, description="Memory limit for render cache in MB"
    )
    text_cache_memory_mb: int = Field(
        50, ge=10, le=500, description="Memory limit for text cache in MB"
    )
    cache_ttl_seconds: int = Field(
        3600, ge=60, description="Cache entry time-to-live in seconds"
    )
    memory_pressure_threshold: float = Field(
        0.8,
        ge=0.5,
        le=0.95,
        description="System memory usage threshold for aggressive eviction",
    )
    enable_cache_metrics: bool = Field(
        True, description="Enable cache performance metrics collection"
    )
    cache_warmup_pages: int = Field(
        10, ge=0, le=50, description="Number of pages to pre-cache during warmup"
    )

    # Time Estimates (seconds)
    ocr_time_per_page: float = Field(
        1.5, gt=0, description="Estimated OCR time per page"
    )
    extraction_time_per_page: float = Field(
        0.1, gt=0, description="Estimated text extraction time per page"
    )

    # Processing Parameters
    table_detection_tolerance: float = Field(
        5.0, gt=0, description="Pixel tolerance for table column detection"
    )
    header_footer_threshold: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Page ratio threshold for header/footer detection",
    )
    reading_order_tolerance: float = Field(
        10.0, gt=0, description="Pixel tolerance for reading order detection"
    )
    memory_estimation_per_page_mb: float = Field(
        1.5, gt=0, description="Estimated memory usage per page in MB"
    )
    cache_aggressive_eviction_ratio: float = Field(
        0.5,
        ge=0.1,
        le=0.9,
        description="Fraction of cache to evict under memory pressure",
    )

    @field_validator("max_dpi")
    @classmethod
    def validate_max_dpi(cls, v: int, info) -> int:
        """Ensure max_dpi is greater than or equal to default_dpi."""
        if "default_dpi" in info.data and v < info.data["default_dpi"]:
            raise ValueError("max_dpi must be greater than or equal to default_dpi")
        return v


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # Application settings
    app_name: str = "PDF_Splitter"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # File processing
    max_file_size_mb: int = 100
    allowed_extensions: List[str] = [".pdf"]
    temp_dir: Path = Path("/tmp/pdf_splitter")
    output_dir: Path = Path("./output")

    # OCR settings
    ocr_engine: str = "paddleocr"  # Options: paddleocr, easyocr, tesseract
    ocr_lang: str = "en"
    ocr_cache_enabled: bool = True
    ocr_cache_dir: Path = Path("./ocr_cache")

    # LLM settings
    llm_provider: str = "transformers"  # Options: transformers, ollama
    llm_model: str = "facebook/bart-large-mnli"
    llm_max_context: int = 512
    llm_temperature: float = 0.1

    # Performance settings
    max_workers: int = 4
    batch_size: int = 10
    chunk_size: int = 5  # Pages to process at once

    # Security
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    secret_key: str = "your-secret-key-here"

    # Database (for future use)
    database_url: str = "sqlite:///./pdf_splitter.db"

    # External services
    ollama_host: Optional[str] = "http://localhost:11434"

    # PDF configuration
    pdf: PDFConfig = Field(default_factory=PDFConfig)

    @property
    def max_file_size_bytes(self) -> int:
        """Convert max file size from MB to bytes."""
        return self.max_file_size_mb * 1024 * 1024

    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [self.temp_dir, self.output_dir, self.ocr_cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


def get_pdf_config() -> PDFConfig:
    """Get PDF configuration from settings."""
    return settings.pdf
