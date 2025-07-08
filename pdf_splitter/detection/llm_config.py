"""
Configuration system for LLM detector with flexible runtime options.

This module provides a comprehensive configuration system for the LLM detector,
allowing fine-tuning of all parameters without code changes.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LLMDetectorConfig:
    """
    Configuration for LLM-based document boundary detection.

    All parameters can be overridden via environment variables or config files.
    """

    # Model Configuration
    model_name: str = "gemma3:latest"
    ollama_url: str = "http://localhost:11434"

    # Performance Settings
    timeout: int = 45  # seconds
    max_retries: int = 2
    batch_size: int = 5  # for batched detector

    # Text Extraction
    bottom_lines: int = 15  # Original default, though testing showed 10 was optimal
    top_lines: int = 15

    # Caching
    cache_enabled: bool = True
    cache_path: Optional[Path] = None
    cache_max_age_days: int = 30
    cache_max_size_mb: int = 500

    # LLM Generation Parameters
    temperature: float = 0.1
    top_k: int = 10
    top_p: float = 0.9

    # Confidence Thresholds
    boundary_confidence_high: float = 0.95
    boundary_confidence_medium: float = 0.85
    continuation_confidence_high: float = 0.85
    continuation_confidence_medium: float = 0.75
    continuation_confidence_low: float = 0.65

    # Prompt Configuration
    prompt_version: str = "gemma3_optimal_v1"
    prompt_template_path: Optional[Path] = None

    # Advanced Settings
    enable_similarity_search: bool = False
    similarity_threshold: float = 0.9
    max_context_length: int = 2000  # characters

    # Logging and Debug
    debug_mode: bool = False
    log_responses: bool = False
    save_failed_responses: bool = True

    @classmethod
    def from_env(cls) -> "LLMDetectorConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Model settings
        if model := os.getenv("LLM_MODEL_NAME"):
            config.model_name = model
        if url := os.getenv("OLLAMA_URL"):
            config.ollama_url = url

        # Performance
        if timeout := os.getenv("LLM_TIMEOUT"):
            config.timeout = int(timeout)
        if retries := os.getenv("LLM_MAX_RETRIES"):
            config.max_retries = int(retries)
        if batch := os.getenv("LLM_BATCH_SIZE"):
            config.batch_size = int(batch)

        # Text extraction
        if bottom := os.getenv("LLM_BOTTOM_LINES"):
            config.bottom_lines = int(bottom)
        if top := os.getenv("LLM_TOP_LINES"):
            config.top_lines = int(top)

        # Caching
        if cache := os.getenv("LLM_CACHE_ENABLED"):
            config.cache_enabled = cache.lower() in ("true", "1", "yes")
        if cache_path := os.getenv("LLM_CACHE_PATH"):
            config.cache_path = Path(cache_path)

        # LLM parameters
        if temp := os.getenv("LLM_TEMPERATURE"):
            config.temperature = float(temp)
        if top_k := os.getenv("LLM_TOP_K"):
            config.top_k = int(top_k)
        if top_p := os.getenv("LLM_TOP_P"):
            config.top_p = float(top_p)

        # Debug
        if debug := os.getenv("LLM_DEBUG"):
            config.debug_mode = debug.lower() in ("true", "1", "yes")

        return config

    @classmethod
    def from_file(cls, path: Path) -> "LLMDetectorConfig":
        """Load configuration from JSON or YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml

                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config files")
            else:
                data = json.load(f)

        # Convert paths
        if "cache_path" in data and data["cache_path"]:
            data["cache_path"] = Path(data["cache_path"])
        if "prompt_template_path" in data and data["prompt_template_path"]:
            data["prompt_template_path"] = Path(data["prompt_template_path"])

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                data[key] = str(value)
            else:
                data[key] = value
        return data

    def save(self, path: Path):
        """Save configuration to file."""
        data = self.to_dict()

        with open(path, "w") as f:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml

                    yaml.dump(data, f, default_flow_style=False)
                except ImportError:
                    # Fallback to JSON
                    json.dump(data, f, indent=2)
            else:
                json.dump(data, f, indent=2)

    def validate(self) -> Dict[str, str]:
        """Validate configuration settings."""
        errors = {}

        # Validate numeric ranges
        if self.timeout <= 0:
            errors["timeout"] = "Must be positive"
        if self.max_retries < 0:
            errors["max_retries"] = "Cannot be negative"
        if self.batch_size <= 0:
            errors["batch_size"] = "Must be positive"

        # Validate text extraction
        if self.bottom_lines <= 0:
            errors["bottom_lines"] = "Must be positive"
        if self.top_lines <= 0:
            errors["top_lines"] = "Must be positive"

        # Validate LLM parameters
        if not 0 <= self.temperature <= 2:
            errors["temperature"] = "Must be between 0 and 2"
        if not 0 < self.top_p <= 1:
            errors["top_p"] = "Must be between 0 and 1"

        # Validate confidence thresholds
        for field in [
            "boundary_confidence_high",
            "boundary_confidence_medium",
            "continuation_confidence_high",
            "continuation_confidence_medium",
            "continuation_confidence_low",
        ]:
            value = getattr(self, field)
            if not 0 <= value <= 1:
                errors[field] = "Must be between 0 and 1"

        return errors

    # Presets removed for simplicity - use explicit configuration instead


def get_config(
    config_file: Optional[Path] = None, use_env: bool = True, **overrides
) -> LLMDetectorConfig:
    """
    Get LLM detector configuration with multiple sources.

    Priority order:
    1. Keyword overrides
    2. Environment variables (if use_env=True)
    3. Config file (if provided)
    4. Default configuration

    Args:
        config_file: Path to configuration file
        use_env: Whether to read from environment variables
        **overrides: Keyword arguments to override settings

    Returns:
        LLMDetectorConfig instance
    """
    # Start with default config
    config = LLMDetectorConfig()

    # Load from file
    if config_file:
        file_config = LLMDetectorConfig.from_file(config_file)
        # Merge settings
        for key, value in file_config.__dict__.items():
            if value is not None:
                setattr(config, key, value)

    # Apply environment variables
    if use_env:
        env_config = LLMDetectorConfig.from_env()
        for key, value in env_config.__dict__.items():
            if value != getattr(LLMDetectorConfig(), key):  # Only if changed
                setattr(config, key, value)

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")

    # Validate
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {errors}")

    return config
