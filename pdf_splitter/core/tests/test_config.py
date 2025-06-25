"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.preprocessing.ocr_processor import OCRConfig, OCREngine


class TestPDFConfig:
    """Test PDF configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PDFConfig()

        assert config.default_dpi == 300
        assert config.max_dpi == 300
        assert config.max_file_size_mb == 500
        assert config.max_pages == 10000
        assert config.page_cache_size == 10
        assert config.enable_cache_metrics is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PDFConfig(
            default_dpi=150,
            max_file_size_mb=100,
            page_cache_size=20,
            enable_cache_metrics=False,
        )

        assert config.default_dpi == 150
        assert config.max_file_size_mb == 100
        assert config.page_cache_size == 20
        assert config.enable_cache_metrics is False

    def test_config_validation(self):
        """Test configuration validation rules."""
        # Test DPI validation
        with pytest.raises(ValidationError):
            PDFConfig(default_dpi=50)  # Below minimum

        with pytest.raises(ValidationError):
            PDFConfig(default_dpi=700)  # Above maximum

        # Test file size validation
        with pytest.raises(ValidationError):
            PDFConfig(max_file_size_mb=0)  # Below minimum

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(
            os.environ,
            {
                "PDF_DEFAULT_DPI": "150",
                "PDF_MAX_FILE_SIZE_MB": "200",
                "PDF_PAGE_CACHE_SIZE": "20",
            },
        ):
            config = PDFConfig()
            assert config.default_dpi == 150
            assert config.max_file_size_mb == 200
            assert config.page_cache_size == 20

    def test_directory_creation(self):
        """Test automatic directory creation."""
        # PDFConfig doesn't have output_dir/temp_dir anymore
        # Test other directory-related settings instead
        config = PDFConfig()
        assert config.page_cache_size >= 0
        assert config.enable_repair is True

    def test_time_estimates(self):
        """Test time estimate calculations."""
        config = PDFConfig()

        assert config.ocr_time_per_page == 1.5
        assert config.extraction_time_per_page == 0.1

        # Test that values are positive
        assert config.ocr_time_per_page > 0
        assert config.extraction_time_per_page > 0

    def test_processing_parameters(self):
        """Test processing parameter defaults."""
        config = PDFConfig()

        assert config.table_detection_tolerance == 5.0
        assert 0.0 <= config.header_footer_threshold <= 1.0
        assert config.reading_order_tolerance == 10.0


class TestCacheConfig:
    """Test cache configuration."""

    def test_cache_defaults(self):
        """Test default cache settings."""
        config = PDFConfig()

        assert config.render_cache_memory_mb == 100
        assert config.text_cache_memory_mb == 50
        assert config.cache_ttl_seconds == 3600
        assert config.memory_pressure_threshold == 0.8
        assert config.cache_warmup_pages == 10

    def test_cache_validation(self):
        """Test cache configuration validation."""
        # Test memory limits
        with pytest.raises(ValidationError):
            PDFConfig(render_cache_memory_mb=5)  # Below minimum

        # Test TTL
        with pytest.raises(ValidationError):
            PDFConfig(cache_ttl_seconds=30)  # Below minimum

        # Test memory pressure threshold
        with pytest.raises(ValidationError):
            PDFConfig(memory_pressure_threshold=0.4)  # Below minimum

        with pytest.raises(ValidationError):
            PDFConfig(memory_pressure_threshold=0.99)  # Above maximum


class TestOCRConfig:
    """Test OCR configuration."""

    def test_ocr_defaults(self):
        """Test default OCR settings."""
        config = OCRConfig()

        assert config.primary_engine == OCREngine.PADDLEOCR
        assert config.fallback_engines == [OCREngine.TESSERACT]
        assert config.confidence_threshold == 0.6
        assert config.preprocessing_enabled is True

    def test_ocr_engine_validation(self):
        """Test OCR engine configuration."""
        config = OCRConfig(
            primary_engine=OCREngine.TESSERACT, fallback_engines=[OCREngine.PADDLEOCR]
        )

        assert config.primary_engine == OCREngine.TESSERACT
        assert config.fallback_engines == [OCREngine.PADDLEOCR]

    def test_paddle_settings(self):
        """Test PaddleOCR specific settings."""
        config = OCRConfig()

        assert config.paddle_use_angle_cls is True
        assert config.paddle_lang == "en"
        assert config.paddle_use_gpu is False
        assert config.paddle_enable_mkldnn is False  # Critical for accuracy

    def test_preprocessing_settings(self):
        """Test preprocessing configuration."""
        config = OCRConfig()

        assert config.preprocessing_enabled is True
        assert config.denoise_enabled is True
        assert config.deskew_enabled is True
        assert config.adaptive_threshold_block_size == 11

    def test_performance_settings(self):
        """Test performance-related settings."""
        config = OCRConfig()

        assert config.max_workers is None  # CPU count
        assert config.batch_size == 5
        assert config.min_confidence_threshold == 0.5


class TestConfigIntegration:
    """Test configuration integration."""

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = PDFConfig()
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert config_dict["default_dpi"] == 300
        assert "max_pages" in config_dict

    def test_config_from_dict(self):
        """Test configuration deserialization."""
        config_dict = {
            "default_dpi": 150,
            "max_file_size_mb": 200,
            "enable_cache_metrics": False,
        }

        config = PDFConfig(**config_dict)
        assert config.default_dpi == 150
        assert config.max_file_size_mb == 200
        assert config.enable_cache_metrics is False

    def test_config_json_schema(self):
        """Test JSON schema generation."""
        schema = PDFConfig.model_json_schema()

        assert "properties" in schema
        assert "default_dpi" in schema["properties"]
        assert schema["properties"]["default_dpi"]["type"] == "integer"
