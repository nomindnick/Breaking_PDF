"""
Global pytest configuration and shared fixtures.

This file provides common test fixtures and utilities that can be used
across all test modules in the project.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.preprocessing.ocr_processor import OCRConfig
from pdf_splitter.preprocessing.pdf_handler import PDFHandler

# --- Path Fixtures ---


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path("test_files")


@pytest.fixture
def test_pdf_paths(test_data_dir):
    """Dictionary of test PDF paths."""
    return {
        "non_ocr": test_data_dir / "Test_PDF_Set_1.pdf",
        "ocr": test_data_dir / "Test_PDF_Set_2_ocr.pdf",
    }


@pytest.fixture
def ground_truth_path(test_data_dir):
    """Path to ground truth JSON file."""
    return test_data_dir / "Test_PDF_Set_Ground_Truth.json"


@pytest.fixture
def ground_truth_data(ground_truth_path):
    """Load ground truth data."""
    if ground_truth_path.exists():
        with open(ground_truth_path) as f:
            return json.load(f)
    return None


# --- Configuration Fixtures ---


@pytest.fixture
def pdf_config():
    """Create default PDF configuration for tests."""
    return PDFConfig(
        default_dpi=150,  # Lower DPI for faster tests
        max_file_size_mb=100,
        enable_cache_metrics=True,
        render_cache_memory_mb=50,
        text_cache_memory_mb=25,
    )


@pytest.fixture
def ocr_config():
    """Create default OCR configuration for tests."""
    return OCRConfig(
        confidence_threshold=0.6,
        preprocessing_enabled=True,
        max_workers=2,  # Limit workers for tests
        batch_size=3,
    )


# --- Handler Fixtures ---


@pytest.fixture
def pdf_handler(pdf_config):
    """Create PDFHandler instance."""
    return PDFHandler(pdf_config)


@pytest.fixture
def loaded_pdf_handler(pdf_handler, test_pdf_paths):
    """Create PDFHandler with a PDF already loaded."""
    pdf_path = test_pdf_paths["non_ocr"]
    if pdf_path.exists():
        with pdf_handler.load_pdf(pdf_path):
            yield pdf_handler
    else:
        pytest.skip("Test PDF not found")


# --- Mock Fixtures ---


@pytest.fixture
def mock_pdf_page():
    """Create a mock PDF page."""
    page = Mock()
    page.rect = Mock(width=612, height=792)  # Letter size
    page.get_text.return_value = "Sample text content"
    page.get_text_blocks.return_value = [
        (10, 10, 100, 30, "First block", 0, 0, 0),
        (10, 40, 100, 60, "Second block", 0, 0, 0),
    ]
    page.get_pixmap.return_value = Mock(
        width=612,
        height=792,
        samples=b"\xff" * (612 * 792 * 3),  # White image
    )
    return page


@pytest.fixture
def mock_pdf_document():
    """Create a mock PDF document."""
    doc = Mock()
    doc.page_count = 5
    doc.metadata = {
        "title": "Test PDF",
        "author": "Test Author",
        "subject": "Test Subject",
    }
    doc.__len__.return_value = 5
    doc.__getitem__ = lambda self, idx: mock_pdf_page()
    return doc


# --- Image Fixtures ---


@pytest.fixture
def test_image_rgb():
    """Create a test RGB image."""
    # Create a simple test image with text-like features
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255  # White background

    # Add some black rectangles to simulate text
    img[50:70, 50:350] = 0  # Horizontal line (like text)
    img[90:110, 50:300] = 0  # Another line
    img[130:150, 50:250] = 0  # Third line

    return img


@pytest.fixture
def test_image_gray():
    """Create a test grayscale image."""
    img = np.ones((200, 400), dtype=np.uint8) * 255  # White background

    # Add some black rectangles to simulate text
    img[50:70, 50:350] = 0
    img[90:110, 50:300] = 0
    img[130:150, 50:250] = 0

    return img


@pytest.fixture
def noisy_test_image():
    """Create a noisy test image."""
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255

    # Add text-like features
    img[50:70, 50:350] = 0

    # Add salt and pepper noise
    noise = np.random.random((200, 400))
    img[noise < 0.05] = 0  # Salt noise
    img[noise > 0.95] = 255  # Pepper noise

    return img


# --- Temporary Directory Fixtures ---


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_pdf_path(temp_dir):
    """Path for a temporary PDF file."""
    return temp_dir / "test_output.pdf"


# --- Utility Fixtures ---


@pytest.fixture
def sample_text_blocks():
    """Sample text blocks for testing."""
    from pdf_splitter.preprocessing.text_extractor import TextBlock

    return [
        TextBlock(
            text="Header Text",
            bbox=(50, 10, 300, 30),
            block_no=0,
            line_no=0,
            span_no=0,
            font_name="Arial-Bold",
            font_size=16.0,
            flags=16,  # Bold
        ),
        TextBlock(
            text="Body paragraph with more content.",
            bbox=(50, 50, 350, 100),
            block_no=1,
            line_no=0,
            span_no=0,
            font_name="Arial",
            font_size=12.0,
            flags=0,
        ),
        TextBlock(
            text="Another paragraph here.",
            bbox=(50, 120, 350, 150),
            block_no=2,
            line_no=0,
            span_no=0,
            font_name="Arial",
            font_size=12.0,
            flags=0,
        ),
    ]


@pytest.fixture
def sample_ocr_result():
    """Sample OCR result for testing."""
    from pdf_splitter.preprocessing.ocr_processor import (
        BoundingBox,
        OCRResult,
        TextLine,
    )

    return OCRResult(
        page_num=0,
        text_lines=[
            TextLine(
                text="Sample OCR Text",
                confidence=0.95,
                bbox=BoundingBox(x1=10, y1=10, x2=200, y2=30),
                angle=0.0,
            ),
            TextLine(
                text="Another line of text",
                confidence=0.92,
                bbox=BoundingBox(x1=10, y1=40, x2=250, y2=60),
                angle=0.0,
            ),
        ],
        full_text="Sample OCR Text\nAnother line of text",
        avg_confidence=0.935,
        processing_time=1.5,
        engine_used="paddleocr",
        preprocessing_applied=["grayscale", "denoise"],
        word_count=7,
        char_count=35,
        quality_score=0.9,
    )


# --- Performance Testing Fixtures ---


@pytest.fixture
def performance_timer():
    """Create simple performance timer context manager."""
    import time

    class Timer:
        def __init__(self):
            self.elapsed = 0

        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            self.elapsed = time.time() - self.start

    return Timer


# --- Cleanup Fixtures ---


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield
    # Force garbage collection to release resources
    import gc

    gc.collect()


# --- Test Skip Conditions ---


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "requires_pdf: marks tests that require PDF files"
    )


# --- Parametrization Helpers ---


def get_test_pdf_pages():
    """Get list of test PDF page numbers for parametrization."""
    return [0, 1, 2, 5, 10, 15]


def get_test_dpi_values():
    """Get list of DPI values for parametrization."""
    return [72, 150, 300]


def get_test_quality_thresholds():
    """Get list of quality thresholds for parametrization."""
    return [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
