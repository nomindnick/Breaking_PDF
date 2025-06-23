"""
PDF preprocessing module for the PDF Splitter application.

This module handles all PDF preprocessing tasks including loading,
validation, rendering, and text extraction.
"""

from .advanced_cache import (
    AdvancedLRUCache,
    CacheEntry,
    CacheMetrics,
    PDFProcessingCache,
)
from .pdf_handler import (
    PageBatch,
    PageInfo,
    PageText,
    PageType,
    PDFHandler,
    PDFMetadata,
    PDFValidationResult,
    ProcessingEstimate,
)
from .text_extractor import ExtractedPage, TextBlock, TextExtractor

__all__ = [
    # PDF Handler
    "PDFHandler",
    "PageType",
    "PDFValidationResult",
    "PageText",
    "PageInfo",
    "PDFMetadata",
    "ProcessingEstimate",
    "PageBatch",
    # Text Extractor
    "TextExtractor",
    "ExtractedPage",
    "TextBlock",
    # Caching
    "AdvancedLRUCache",
    "PDFProcessingCache",
    "CacheMetrics",
    "CacheEntry",
]
