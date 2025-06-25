"""
High-performance PDF document handler for the PDF Splitter application.

This module provides the foundational layer for PDF processing, offering:
- Lightning-fast PDF loading and validation using PyMuPDF
- Intelligent page type detection (searchable vs image-based)
- Memory-efficient streaming for large documents
- Comprehensive metadata extraction and analysis
"""

import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.core.exceptions import (
    PDFHandlerError,
    PDFTextExtractionError,
    PDFValidationError,
)
from pdf_splitter.preprocessing.advanced_cache import PDFProcessingCache

logger = logging.getLogger(__name__)


class PageType(str, Enum):
    """Classification of PDF page content type."""

    SEARCHABLE = "searchable"
    IMAGE_BASED = "image_based"
    MIXED = "mixed"
    EMPTY = "empty"


class PDFValidationResult(BaseModel):
    """Results from PDF validation process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_valid: bool
    page_count: int
    file_size_mb: float
    estimated_memory_mb: float
    is_encrypted: bool
    is_damaged: bool
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    pdf_version: str = ""
    producer: str = ""
    creator: str = ""


class PageText(BaseModel):
    """Text extracted from a PDF page with quality metrics."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_method: str
    char_count: int
    word_count: int
    has_tables: bool = False
    has_images: bool = False
    bbox_count: int = 0
    avg_font_size: float = 0.0


class PageInfo(BaseModel):
    """Comprehensive information about a single PDF page."""

    page_num: int
    width: float
    height: float
    rotation: int
    page_type: PageType
    text_percentage: float = Field(ge=0.0, le=100.0)
    image_count: int = 0
    has_annotations: bool = False


class PDFMetadata(BaseModel):
    """Document-level metadata and statistics."""

    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: int
    file_size_mb: float
    is_linearized: bool = False
    is_encrypted: bool = False
    pdf_version: str = ""
    page_info_summary: Dict[PageType, int] = Field(default_factory=dict)


class ProcessingEstimate(BaseModel):
    """Estimated processing time and resource usage."""

    total_pages: int
    estimated_seconds: float
    estimated_memory_mb: float
    requires_ocr_pages: int
    searchable_pages: int
    mixed_pages: int
    empty_pages: int


class PageBatch(BaseModel):
    """A batch of pages for streaming processing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start_idx: int
    end_idx: int
    pages: List[Dict[str, Any]]
    batch_size: int


class PDFHandler:
    """
    High-performance PDF document handler optimized for CPU-only processing.

    This class provides efficient PDF loading, validation, and page rendering
    using PyMuPDF (fitz) for maximum performance. It includes intelligent
    page type detection and memory-efficient streaming for large documents.
    """

    def __init__(self, config: Optional[PDFConfig] = None):
        """
        Initialize the PDF handler with configuration.

        Args:
            config: Optional configuration object. Uses defaults if not provided.
        """
        self.config = config or PDFConfig()
        self._document: Optional[fitz.Document] = None
        self._pdf_path: Optional[Path] = None
        # Replace simple caches with advanced caching system
        self.cache_manager = PDFProcessingCache(self.config)
        self._metadata: Optional[PDFMetadata] = None

        logger.info(f"PDFHandler initialized with config: {self.config}")

    @property
    def is_loaded(self) -> bool:
        """Check if a PDF is currently loaded."""
        return self._document is not None

    @property
    def page_count(self) -> int:
        """Get the number of pages in the loaded PDF."""
        if not self.is_loaded:
            return 0
        return len(self._document)  # type: ignore[arg-type]

    def validate_pdf(self, pdf_path: Path) -> PDFValidationResult:
        """
        Validate a PDF file for processing suitability.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            PDFValidationResult with validation details

        Raises:
            PDFValidationError: If the file cannot be accessed
        """
        logger.info(f"Validating PDF: {pdf_path}")

        result = PDFValidationResult(
            is_valid=False,
            page_count=0,
            file_size_mb=0,
            estimated_memory_mb=0,
            is_encrypted=False,
            is_damaged=False,
        )

        # Check file exists and is readable
        if not pdf_path.exists():
            result.errors.append(f"File does not exist: {pdf_path}")
            return result

        if not pdf_path.is_file():
            result.errors.append(f"Path is not a file: {pdf_path}")
            return result

        # Get file size
        file_size_bytes = pdf_path.stat().st_size
        result.file_size_mb = file_size_bytes / (1024 * 1024)

        if result.file_size_mb > self.config.max_file_size_mb:
            result.errors.append(
                f"File size ({result.file_size_mb:.1f} MB) exceeds maximum "
                f"({self.config.max_file_size_mb} MB)"
            )

        # Try to open the PDF
        try:
            with fitz.open(pdf_path) as doc:
                result.page_count = len(doc)

                # Check if encrypted
                if doc.is_encrypted:
                    result.is_encrypted = True
                    if not doc.authenticate(""):  # Try empty password
                        result.errors.append("PDF is password protected")
                        return result
                    else:
                        result.warnings.append("PDF was encrypted but had no password")

                # Check for damage
                if doc.is_repaired:
                    result.is_damaged = True
                    result.warnings.append("PDF was damaged and repaired")

                # Get version info
                result.pdf_version = doc.metadata.get("format", "")
                result.producer = doc.metadata.get("producer", "")
                result.creator = doc.metadata.get("creator", "")

                # Estimate memory usage (rough estimate)
                result.estimated_memory_mb = (
                    result.page_count * self.config.memory_estimation_per_page_mb
                )

                if result.page_count == 0:
                    result.errors.append("PDF has no pages")
                elif result.page_count > self.config.max_pages:
                    result.warnings.append(
                        f"PDF has {result.page_count} pages, "
                        f"which may impact performance"
                    )

                result.is_valid = len(result.errors) == 0

        except Exception as e:
            result.errors.append(f"Failed to open PDF: {str(e)}")
            logger.exception(f"Error validating PDF: {pdf_path}")

        logger.info(
            f"Validation complete: valid={result.is_valid}, "
            f"pages={result.page_count}, warnings={len(result.warnings)}, "
            f"errors={len(result.errors)}"
        )

        return result

    @contextmanager
    def load_pdf(self, pdf_path: Path, validate: bool = True):
        """
        Load a PDF document with automatic resource management.

        Args:
            pdf_path: Path to the PDF file
            validate: Whether to validate the PDF first

        Yields:
            Self for method chaining

        Raises:
            PDFValidationError: If validation fails
            PDFHandlerError: If loading fails
        """
        logger.info(f"Loading PDF: {pdf_path}")

        # Validate if requested
        if validate:
            validation = self.validate_pdf(pdf_path)
            if not validation.is_valid:
                raise PDFValidationError(
                    f"PDF validation failed: {'; '.join(validation.errors)}"
                )

        # Close any existing document
        self.close()

        try:
            self._pdf_path = pdf_path
            self._document = fitz.open(pdf_path)

            # Enable repairs if configured
            if self.config.enable_repair and self._document.is_repaired:
                logger.warning(f"PDF was damaged and repaired: {pdf_path}")

            # Extract metadata immediately
            self._extract_metadata()

            logger.info(f"Successfully loaded PDF with {self.page_count} pages")

            yield self

        except Exception as e:
            logger.exception(f"Failed to load PDF: {pdf_path}")
            raise PDFHandlerError(f"Failed to load PDF: {str(e)}")

        finally:
            self.close()

    def close(self):
        """Close the current PDF and free resources."""
        if self._document:
            try:
                self._document.close()
            except Exception as e:
                logger.warning(f"Error closing PDF document: {e}")

        self._document = None
        self._pdf_path = None
        # Clear all caches
        self.cache_manager.render_cache.clear()
        self.cache_manager.text_cache.clear()
        self.cache_manager.analysis_cache.clear()
        self._metadata = None

        # Force garbage collection for large documents
        gc.collect()

    def _extract_metadata(self):
        """Extract and cache document metadata."""
        if not self.is_loaded:
            return

        meta = self._document.metadata

        self._metadata = PDFMetadata(
            title=meta.get("title"),
            author=meta.get("author"),
            subject=meta.get("subject"),
            keywords=meta.get("keywords"),
            creator=meta.get("creator"),
            producer=meta.get("producer"),
            creation_date=meta.get("creationDate"),
            modification_date=meta.get("modDate"),
            page_count=self.page_count,
            file_size_mb=self._pdf_path.stat().st_size / (1024 * 1024)
            if self._pdf_path
            else 0,
            is_linearized=getattr(self._document, "is_fast", False),
            is_encrypted=self._document.is_encrypted,
            pdf_version=meta.get("format", ""),
        )

    def get_metadata(self) -> Optional[PDFMetadata]:
        """Get cached document metadata."""
        return self._metadata

    def get_page_type(self, page_num: int) -> PageType:
        """
        Detect the type of content on a specific page.

        Args:
            page_num: Zero-based page number

        Returns:
            PageType classification

        Raises:
            PDFHandlerError: If page number is invalid
        """
        if not self.is_loaded:
            raise PDFHandlerError("No PDF loaded")

        if page_num < 0 or page_num >= self.page_count:
            raise PDFHandlerError(f"Invalid page number: {page_num}")

        # Check cache first
        cache_key = (str(self._pdf_path), page_num, "page_info")
        cached_info = self.cache_manager.analysis_cache.get(cache_key)
        if cached_info:
            return PageType(cached_info.get("page_type"))

        page = self._document[page_num]  # type: ignore[index]

        # Get text content
        text = page.get_text()
        text_length = len(text.strip())

        # Get image list
        image_list = page.get_images()
        image_count = len(image_list)

        # Calculate text coverage
        page_area = abs(page.rect.width * page.rect.height)
        text_blocks = page.get_text("blocks")
        text_area = sum(
            abs((b[2] - b[0]) * (b[3] - b[1])) for b in text_blocks if b[4].strip()
        )
        text_percentage = (text_area / page_area * 100) if page_area > 0 else 0
        # Clamp to 100% to handle floating point precision issues
        text_percentage = min(text_percentage, 100.0)

        # Determine page type
        if text_length < 10:
            if image_count > 0:
                page_type = PageType.IMAGE_BASED
            else:
                page_type = PageType.EMPTY
        elif (
            text_percentage < self.config.min_text_coverage_percent and image_count > 0
        ):
            page_type = PageType.MIXED
        else:
            page_type = PageType.SEARCHABLE

        # Cache the result in advanced cache
        page_info = PageInfo(
            page_num=page_num,
            width=page.rect.width,
            height=page.rect.height,
            rotation=page.rotation,
            page_type=page_type,
            text_percentage=text_percentage,
            image_count=image_count,
            has_annotations=len(list(page.annots())) > 0,
        )

        # Store in analysis cache
        self.cache_manager.analysis_cache.put(
            (str(self._pdf_path), page_num, "page_info"), page_info.__dict__
        )

        logger.debug(
            f"Page {page_num}: type={page_type}, text%={text_percentage:.1f}, "
            f"images={image_count}"
        )

        return page_type

    def render_page(self, page_num: int, dpi: Optional[int] = None) -> np.ndarray:
        """
        Render a PDF page to a numpy array.

        Args:
            page_num: Zero-based page number
            dpi: Resolution for rendering (uses config default if None)

        Returns:
            Numpy array of the rendered page (RGB format)

        Raises:
            PDFRenderError: If rendering fails
        """
        if not self.is_loaded:
            raise PDFHandlerError("No PDF loaded")

        if page_num < 0 or page_num >= self.page_count:
            raise PDFHandlerError(f"Invalid page number: {page_num}")

        dpi = dpi or self.config.default_dpi

        # Define render function for cache miss
        def render_func():
            try:
                page = self._document[page_num]  # type: ignore[index]

                # Calculate matrix for desired DPI
                # PDF default is 72 DPI
                mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)

                # Render to pixmap
                pix = page.get_pixmap(matrix=mat, alpha=False)

                # Convert to numpy array (RGB format)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8)
                img_array = img_array.reshape(pix.height, pix.width, 3)

                logger.debug(
                    f"Rendered page {page_num} at {dpi} DPI: {img_array.shape}"
                )

                return img_array

            except Exception as e:
                logger.exception(f"Failed to render page {page_num}")
                raise PDFHandlerError(f"Failed to render page {page_num}: {str(e)}")

        # Use advanced cache with metrics tracking
        rendered_array = self.cache_manager.get_rendered_page(
            pdf_path=str(self._pdf_path),
            page_num=page_num,
            dpi=dpi,
            render_func=render_func,
        )

        # Return a copy to prevent cache modification
        return rendered_array.copy() if rendered_array is not None else render_func()

    def extract_text(self, page_num: int) -> PageText:
        """
        Extract text from a PDF page with quality assessment.

        Args:
            page_num: Zero-based page number

        Returns:
            PageText object with extracted text and metrics

        Raises:
            PDFTextExtractionError: If extraction fails
        """
        if not self.is_loaded:
            raise PDFHandlerError("No PDF loaded")

        if page_num < 0 or page_num >= self.page_count:
            raise PDFHandlerError(f"Invalid page number: {page_num}")

        try:
            page = self._document[page_num]  # type: ignore[index]

            # Extract text
            text = page.get_text()

            # Get text blocks for more detailed analysis
            blocks = page.get_text("blocks")
            text_blocks = [b for b in blocks if b[4].strip()]  # Filter empty blocks

            # Calculate metrics
            char_count = len(text)
            word_count = len(text.split())

            # Detect tables and images
            try:
                tables = page.find_tables()
                has_tables = (
                    len(tables.tables) > 0
                    if hasattr(tables, "tables")
                    else bool(tables)
                )
            except Exception:
                has_tables = False
            has_images = len(page.get_images()) > 0

            # Calculate average font size (approximation)
            total_height = sum(b[3] - b[1] for b in text_blocks)
            avg_font_size = (total_height / len(text_blocks)) if text_blocks else 0

            # Estimate confidence based on text quality indicators
            confidence = self._estimate_text_confidence(text, text_blocks)

            return PageText(
                text=text,
                confidence=confidence,
                extraction_method="pdfplumber_direct",
                char_count=char_count,
                word_count=word_count,
                has_tables=has_tables,
                has_images=has_images,
                bbox_count=len(text_blocks),
                avg_font_size=avg_font_size,
            )

        except Exception as e:
            logger.exception(f"Failed to extract text from page {page_num}")
            raise PDFTextExtractionError(
                f"Failed to extract text from page {page_num}: {str(e)}"
            )

    def _estimate_text_confidence(self, text: str, blocks: List[Any]) -> float:
        """
        Estimate confidence of extracted text quality.

        Args:
            text: Extracted text
            blocks: Text blocks from PyMuPDF

        Returns:
            Confidence score between 0 and 1
        """
        if not text.strip():
            return 0.0

        confidence = 1.0

        # Penalize for too many special characters (might indicate garbled text)
        special_char_ratio = sum(
            1 for c in text if not c.isalnum() and not c.isspace()
        ) / len(text)
        if special_char_ratio > 0.3:
            confidence *= 0.8

        # Penalize for very short average word length (might indicate OCR errors)
        words = text.split()
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if avg_word_length < 2:
                confidence *= 0.7

        # Reward for good text block structure
        if len(blocks) > 5:  # Multiple text blocks suggest good structure
            confidence = min(confidence * 1.1, 1.0)

        return round(confidence, 2)

    def analyze_all_pages(self, max_workers: Optional[int] = None) -> List[PageInfo]:
        """
        Analyze all pages in parallel for quick document overview.

        Args:
            max_workers: Number of threads (defaults to CPU count)

        Returns:
            List of PageInfo for all pages
        """
        if not self.is_loaded:
            raise PDFHandlerError("No PDF loaded")

        max_workers = max_workers or self.config.analysis_threads

        logger.info(f"Analyzing {self.page_count} pages with {max_workers} workers")

        def analyze_page(page_num: int) -> PageInfo:
            # This will populate the cache
            self.get_page_type(page_num)
            # Retrieve from cache
            cache_key = (str(self._pdf_path), page_num, "page_info")
            cached_info = self.cache_manager.analysis_cache.get(cache_key)
            if cached_info:
                return PageInfo(**cached_info)
            else:
                # Should never happen since get_page_type populates cache
                raise PDFHandlerError(f"Failed to analyze page {page_num}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            page_infos = list(executor.map(analyze_page, range(self.page_count)))

        # Update metadata with summary
        if self._metadata:
            summary: Dict[PageType, int] = {}
            for info in page_infos:
                summary[info.page_type] = summary.get(info.page_type, 0) + 1
            self._metadata.page_info_summary = summary

            logger.info(f"Analysis complete: {self._metadata.page_info_summary}")

        return page_infos

    def estimate_processing_time(self) -> ProcessingEstimate:
        """
        Estimate processing time for the entire document.

        Returns:
            ProcessingEstimate with time and resource predictions
        """
        if not self.is_loaded:
            raise PDFHandlerError("No PDF loaded")

        # Analyze pages if not already done
        # Check if we have any cached analysis
        has_analysis = False
        for page_num in range(min(1, self.page_count)):  # Check first page
            cache_key = (str(self._pdf_path), page_num, "page_info")
            if self.cache_manager.analysis_cache.get(cache_key):
                has_analysis = True
                break

        if not has_analysis:
            self.analyze_all_pages()

        # Count page types from cache
        page_types = []
        for page_num in range(self.page_count):
            cache_key = (str(self._pdf_path), page_num, "page_info")
            cached_info = self.cache_manager.analysis_cache.get(cache_key)
            if cached_info:
                page_types.append(PageType(cached_info.get("page_type")))

        requires_ocr = sum(
            1 for pt in page_types if pt in [PageType.IMAGE_BASED, PageType.MIXED]
        )
        searchable = sum(1 for pt in page_types if pt == PageType.SEARCHABLE)
        mixed = sum(1 for pt in page_types if pt == PageType.MIXED)
        empty = sum(1 for pt in page_types if pt == PageType.EMPTY)

        # Estimate time (based on targets from docs)
        # OCR: 1-2 seconds per page
        # Direct extraction: 0.1 seconds per page
        ocr_time = requires_ocr * self.config.ocr_time_per_page
        extraction_time = searchable * self.config.extraction_time_per_page

        total_time = ocr_time + extraction_time

        # Estimate memory (1.5 MB per page for rendering)
        memory_mb = self.page_count * 1.5

        return ProcessingEstimate(
            total_pages=self.page_count,
            estimated_seconds=round(total_time, 1),
            estimated_memory_mb=round(memory_mb, 1),
            requires_ocr_pages=requires_ocr,
            searchable_pages=searchable,
            mixed_pages=mixed,
            empty_pages=empty,
        )

    def stream_pages(
        self,
        batch_size: Optional[int] = None,
        start_page: int = 0,
        end_page: Optional[int] = None,
    ) -> Iterator[PageBatch]:
        """
        Stream pages in batches for memory-efficient processing.

        Args:
            batch_size: Number of pages per batch (uses config default if None)
            start_page: Starting page number (inclusive)
            end_page: Ending page number (exclusive, None for all pages)

        Yields:
            PageBatch objects containing rendered pages
        """
        if not self.is_loaded:
            raise PDFHandlerError("No PDF loaded")

        batch_size = batch_size or self.config.stream_batch_size
        end_page = end_page or self.page_count

        if start_page < 0 or end_page > self.page_count:
            raise PDFHandlerError(f"Invalid page range: {start_page}-{end_page}")

        logger.info(
            f"Streaming pages {start_page}-{end_page} in batches of {batch_size}"
        )

        for batch_start in range(start_page, end_page, batch_size):
            batch_end = min(batch_start + batch_size, end_page)

            batch_pages = []
            for page_num in range(batch_start, batch_end):
                try:
                    # Get page info
                    page_type = self.get_page_type(page_num)

                    # Render if needed (skip empty pages)
                    if page_type != PageType.EMPTY:
                        img_array = self.render_page(page_num)
                    else:
                        img_array = None

                    # Extract text if searchable
                    if page_type in [PageType.SEARCHABLE, PageType.MIXED]:
                        page_text = self.extract_text(page_num)
                    else:
                        page_text = None

                    batch_pages.append(
                        {
                            "page_num": page_num,
                            "page_type": page_type,
                            "image": img_array,
                            "text": page_text,
                        }
                    )

                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    batch_pages.append({"page_num": page_num, "error": str(e)})

            yield PageBatch(
                start_idx=batch_start,
                end_idx=batch_end,
                pages=batch_pages,
                batch_size=len(batch_pages),
            )

            # Cache management is handled automatically by PDFProcessingCache
            gc.collect()

    def save_page_image(
        self,
        page_num: int,
        output_path: Path,
        dpi: Optional[int] = None,
        format: str = "PNG",
    ):
        """
        Save a rendered page as an image file.

        Args:
            page_num: Zero-based page number
            output_path: Path to save the image
            dpi: Resolution for rendering
            format: Image format (PNG, JPEG, etc.)
        """
        img_array = self.render_page(page_num, dpi)

        # Use PIL to save the image
        try:
            from PIL import Image

            img = Image.fromarray(img_array)
            img.save(output_path, format=format)
            logger.info(f"Saved page {page_num} to {output_path}")
        except ImportError:
            raise PDFHandlerError("Pillow is required for saving images")

    def warmup_cache(self, page_range: Optional[range] = None):
        """
        Pre-populate cache with pages likely to be accessed.

        Args:
            page_range: Range of pages to warmup. If None, uses config default.
        """
        if not self.is_loaded:
            raise PDFHandlerError("No PDF loaded")

        # Determine pages to warmup
        if page_range is None:
            warmup_count = min(self.config.cache_warmup_pages, self.page_count)
            page_range = range(warmup_count)

        logger.info(f"Warming up cache for {len(page_range)} pages")

        # Warmup render and text caches
        def render_for_warmup(pdf_path, page_num, dpi):
            return self.render_page(page_num, dpi)

        def extract_for_warmup(pdf_path, page_num):
            return self.extract_text(page_num).__dict__

        self.cache_manager.warmup_pages(
            pdf_path=str(self._pdf_path),
            page_range=page_range,
            render_func=render_for_warmup,
            extract_func=extract_for_warmup,
            dpi=self.config.default_dpi,
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary containing cache performance metrics
        """
        stats = self.cache_manager.get_combined_stats()

        # Add PDF-specific context
        stats["pdf_info"] = {
            "current_pdf": str(self._pdf_path) if self._pdf_path else None,
            "page_count": self.page_count if self.is_loaded else 0,
            "cache_enabled": self.config.enable_cache_metrics,
        }

        return stats

    def log_cache_performance(self):
        """Log cache performance metrics."""
        if not self.config.enable_cache_metrics:
            return

        self.cache_manager.log_performance()

        # Log summary stats
        stats = self.get_cache_stats()
        logger.info(
            f"Cache Summary - Total Memory: {stats['total_memory_mb']:.1f}MB, "
            f"Render Hit Rate: {stats['render_cache']['hit_rate']}, "
            f"Text Hit Rate: {stats['text_cache']['hit_rate']}"
        )
