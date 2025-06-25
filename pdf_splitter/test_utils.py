"""
Shared testing utilities for PDF Splitter tests.

This module provides helper functions and utilities that can be used
across different test modules.
"""

import io
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

import fitz  # PyMuPDF
import numpy as np
import pytest
from PIL import Image, ImageDraw

from pdf_splitter.preprocessing.pdf_handler import PageType

# --- PDF Generation Utilities ---


def create_test_pdf(
    num_pages: int = 5,
    page_size: Tuple[float, float] = (612, 792),  # Letter size
    include_text: bool = True,
    include_images: bool = False,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Create a test PDF with specified characteristics.

    Args:
        num_pages: Number of pages to create
        page_size: Page size as (width, height) in points
        include_text: Whether to include text content
        include_images: Whether to include images
        output_path: Path to save the PDF (optional)

    Returns:
        Path to created PDF if output_path provided, None otherwise
    """
    doc = fitz.open()

    for page_num in range(num_pages):
        page = doc.new_page(width=page_size[0], height=page_size[1])

        if include_text:
            # Add header
            header_text = f"Test Document - Page {page_num + 1}"
            page.insert_text(
                (50, 50),
                header_text,
                fontsize=16,
                fontname="helv",
                color=(0, 0, 0),
            )

            # Add body text
            body_text = f"This is the content of page {page_num + 1}. " * 10
            text_rect = fitz.Rect(50, 100, page_size[0] - 50, page_size[1] - 100)
            page.insert_textbox(
                text_rect,
                body_text,
                fontsize=12,
                fontname="helv",
                color=(0, 0, 0),
            )

            # Add footer
            footer_text = f"Page {page_num + 1} of {num_pages}"
            page.insert_text(
                (50, page_size[1] - 50),
                footer_text,
                fontsize=10,
                fontname="helv",
                color=(0.5, 0.5, 0.5),
            )

        if include_images:
            # Create a simple test image
            img = create_test_image_bytes(200, 150)
            img_rect = fitz.Rect(200, 300, 400, 450)
            page.insert_image(img_rect, stream=img)

    if output_path:
        doc.save(str(output_path))
        doc.close()
        return output_path
    else:
        doc.close()
        return None


def create_test_image_bytes(width: int = 200, height: int = 150) -> bytes:
    """Create a test image and return as bytes."""
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Draw some shapes
    draw.rectangle([10, 10, width - 10, height - 10], outline="black", width=2)
    draw.text((20, 20), "Test Image", fill="black")

    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()


# --- Text Generation Utilities ---


def generate_random_text(
    min_words: int = 10,
    max_words: int = 100,
    seed: Optional[int] = None,
) -> str:
    """Generate random text for testing."""
    if seed:
        random.seed(seed)

    words = [
        "the",
        "and",
        "to",
        "of",
        "a",
        "in",
        "that",
        "is",
        "was",
        "for",
        "document",
        "page",
        "text",
        "content",
        "section",
        "paragraph",
        "title",
        "header",
        "footer",
        "table",
        "image",
        "figure",
        "data",
        "information",
        "analysis",
        "report",
        "summary",
    ]

    num_words = random.randint(min_words, max_words)
    text = " ".join(random.choices(words, k=num_words))

    # Capitalize first letter and add period
    return text[0].upper() + text[1:] + "."


def create_document_segments(num_segments: int = 3) -> List[Tuple[int, int]]:
    """Create document segment boundaries for testing."""
    segments = []
    start = 0

    for i in range(num_segments):
        length = random.randint(2, 10)
        end = start + length - 1
        segments.append((start, end))
        start = end + 1

    return segments


# --- Mock Creation Utilities ---


def create_mock_pdf_page(
    text: str = "Sample text",
    page_num: int = 0,
    width: float = 612,
    height: float = 792,
) -> Mock:
    """Create a mock PDF page with specified properties."""
    page = Mock()
    page.number = page_num
    page.rect = Mock(width=width, height=height)
    page.get_text.return_value = text

    # Mock text extraction with blocks
    blocks = []
    lines = text.split("\n")
    y_pos = 50

    for i, line in enumerate(lines):
        if line.strip():
            blocks.append(
                {
                    "type": 0,  # Text block
                    "bbox": (50, y_pos, 500, y_pos + 20),
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": line,
                                    "font": "Arial",
                                    "size": 12,
                                    "flags": 0,
                                }
                            ]
                        }
                    ],
                }
            )
            y_pos += 30

    page.get_text.return_value = {"blocks": blocks}

    # Mock pixmap for rendering
    pixmap = Mock()
    pixmap.width = int(width)
    pixmap.height = int(height)
    pixmap.samples = b"\xff" * (pixmap.width * pixmap.height * 3)
    page.get_pixmap.return_value = pixmap

    return page


def create_mock_ocr_result(
    text: str = "OCR extracted text",
    confidence: float = 0.95,
    page_num: int = 0,
) -> Dict[str, Any]:
    """Create a mock OCR result."""
    lines = text.split("\n")
    text_lines = []

    for i, line in enumerate(lines):
        if line.strip():
            text_lines.append(
                {
                    "text": line,
                    "confidence": confidence + random.uniform(-0.05, 0.05),
                    "bbox": {
                        "x1": 50,
                        "y1": 50 + i * 30,
                        "x2": 500,
                        "y2": 70 + i * 30,
                    },
                }
            )

    return {
        "page_num": page_num,
        "text_lines": text_lines,
        "full_text": text,
        "avg_confidence": confidence,
        "processing_time": random.uniform(0.5, 2.0),
        "word_count": len(text.split()),
        "char_count": len(text),
    }


# --- Assertion Helpers ---


def assert_pdf_valid(pdf_path: Path):
    """Assert that a PDF file is valid and can be opened."""
    assert pdf_path.exists(), f"PDF file not found: {pdf_path}"

    try:
        doc = fitz.open(str(pdf_path))
        assert doc.page_count > 0, "PDF has no pages"
        doc.close()
    except Exception as e:
        pytest.fail(f"Failed to open PDF: {e}")


def assert_text_quality(
    text: str,
    min_length: int = 10,
    min_words: int = 2,
    max_error_ratio: float = 0.1,
):
    """Assert text meets quality requirements."""
    assert len(text) >= min_length, f"Text too short: {len(text)} < {min_length}"

    words = text.split()
    assert len(words) >= min_words, f"Too few words: {len(words)} < {min_words}"

    # Check for common OCR errors
    error_chars = sum(1 for c in text if c in "�□▯")
    error_ratio = error_chars / len(text) if text else 0
    assert (
        error_ratio <= max_error_ratio
    ), f"Too many error characters: {error_ratio:.2%}"


def assert_ocr_result_valid(result: Dict[str, Any]):
    """Assert OCR result has valid structure and content."""
    required_fields = [
        "page_num",
        "text_lines",
        "full_text",
        "avg_confidence",
        "processing_time",
        "word_count",
        "char_count",
    ]

    for field in required_fields:
        assert field in result, f"Missing required field: {field}"

    assert isinstance(result["text_lines"], list)
    assert 0 <= result["avg_confidence"] <= 1
    assert result["processing_time"] >= 0
    assert result["word_count"] >= 0
    assert result["char_count"] >= 0


# --- Performance Testing Helpers ---


def measure_performance(func, *args, iterations: int = 3, **kwargs) -> Dict[str, float]:
    """Measure function performance over multiple iterations."""
    import time

    times = []
    for _ in range(iterations):
        start = time.time()
        func(*args, **kwargs)
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        "min": min(times),
        "max": max(times),
        "mean": sum(times) / len(times),
        "total": sum(times),
        "iterations": iterations,
    }


def assert_performance(
    elapsed_time: float,
    max_time: float,
    operation: str = "Operation",
):
    """Assert that performance meets requirements."""
    assert (
        elapsed_time <= max_time
    ), f"{operation} took too long: {elapsed_time:.2f}s > {max_time:.2f}s"


# --- Image Testing Utilities ---


def create_noisy_image(
    width: int = 400,
    height: int = 200,
    noise_level: float = 0.1,
) -> np.ndarray:
    """Create a noisy test image."""
    # Start with white background
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Add text-like black rectangles
    for i in range(3):
        y_start = 50 + i * 50
        y_end = y_start + 20
        x_end = width - 50 - i * 50
        img[y_start:y_end, 50:x_end] = 0

    # Add noise
    noise = np.random.random((height, width, 3))
    noise_mask = noise < noise_level / 2
    img[noise_mask] = 0  # Salt noise
    noise_mask = noise > (1 - noise_level / 2)
    img[noise_mask] = 255  # Pepper noise

    return img


def compare_images(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compare two images and return similarity score (0-1)."""
    if img1.shape != img2.shape:
        return 0.0

    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = np.mean(img1, axis=2)
    if len(img2.shape) == 3:
        img2 = np.mean(img2, axis=2)

    # Calculate normalized difference
    diff = np.abs(img1.astype(float) - img2.astype(float))
    max_diff = 255.0
    similarity = 1.0 - (np.mean(diff) / max_diff)

    return similarity


# --- Data Validation Helpers ---


def validate_page_type(page_type: PageType) -> bool:
    """Validate that page type is valid enum value."""
    return page_type in [
        PageType.SEARCHABLE,
        PageType.IMAGE_BASED,
        PageType.MIXED,
        PageType.EMPTY,
    ]


def validate_confidence_score(score: float) -> bool:
    """Validate confidence score is in valid range."""
    return 0.0 <= score <= 1.0


def validate_bbox(bbox: Tuple[float, float, float, float]) -> bool:
    """Validate bounding box has valid coordinates."""
    x1, y1, x2, y2 = bbox
    return x1 <= x2 and y1 <= y2 and all(v >= 0 for v in bbox)
