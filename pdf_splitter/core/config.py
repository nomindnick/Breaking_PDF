"""Configuration management for PDF Splitter application."""

from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


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
