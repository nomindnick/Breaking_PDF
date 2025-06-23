"""
Advanced text extraction module for searchable PDFs.

This module provides sophisticated text extraction capabilities that go beyond
basic text retrieval, including:
- Multi-method text extraction for maximum accuracy
- Layout-aware text ordering and structure preservation
- Quality assessment and confidence scoring
- Coordinate mapping for visual analysis
- Table and structured content detection
"""

import logging
import re
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, Field

from pdf_splitter.core.exceptions import PDFTextExtractionError
from pdf_splitter.preprocessing.pdf_handler import PageText, PageType, PDFHandler

logger = logging.getLogger(__name__)


class TextBlock(BaseModel):
    """Represents a block of text with layout information."""

    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    block_no: int
    line_no: int
    span_no: int
    font_name: str = ""
    font_size: float = 0.0
    flags: int = 0  # Font flags (bold, italic, etc.)

    @property
    def is_bold(self) -> bool:
        """Check if text is bold based on font flags."""
        return bool(self.flags & 2**4)

    @property
    def is_italic(self) -> bool:
        """Check if text is italic based on font flags."""
        return bool(self.flags & 2**1)


class ExtractedPage(BaseModel):
    """Complete extracted content from a single page."""

    page_num: int
    text: str
    blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0)
    word_count: int = 0
    char_count: int = 0
    avg_font_size: float = 0.0
    dominant_font: str = ""
    has_headers: bool = False
    has_footers: bool = False
    reading_order_confidence: float = Field(ge=0.0, le=1.0)


class TextExtractor:
    """
    Advanced text extraction engine for searchable PDFs.

    This class provides sophisticated text extraction that preserves layout,
    identifies document structure, and assesses extraction quality. It uses
    multiple extraction methods to ensure maximum accuracy and completeness.
    """

    def __init__(self, pdf_handler: PDFHandler):
        """
        Initialize the text extractor with a PDF handler.

        Args:
            pdf_handler: An initialized PDFHandler instance
        """
        self.pdf_handler = pdf_handler
        self._font_cache: Dict[str, float] = {}

    def extract_all_pages(self) -> List[ExtractedPage]:
        """
        Extract text from all pages in the PDF.

        Returns:
            List of ExtractedPage objects with text and metadata

        Raises:
            PDFTextExtractionError: If extraction fails
        """
        if not self.pdf_handler.is_loaded:
            raise PDFTextExtractionError("No PDF loaded in handler")

        extracted_pages = []

        for page_num in range(self.pdf_handler.page_count):
            try:
                # Skip image-based pages
                page_type = self.pdf_handler.get_page_type(page_num)
                if page_type == PageType.IMAGE_BASED:
                    logger.info(f"Skipping image-based page {page_num + 1}")
                    continue

                extracted = self.extract_page(page_num)
                extracted_pages.append(extracted)

            except Exception as e:
                logger.error(f"Failed to extract page {page_num + 1}: {e}")
                # Continue with other pages

        return extracted_pages

    def extract_page(self, page_num: int) -> ExtractedPage:
        """
        Extract text from a single page with full analysis.

        Args:
            page_num: Zero-based page number

        Returns:
            ExtractedPage with extracted content and metadata

        Raises:
            PDFTextExtractionError: If extraction fails
        """
        if not self.pdf_handler.is_loaded:
            raise PDFTextExtractionError("No PDF loaded")

        if page_num < 0 or page_num >= self.pdf_handler.page_count:
            raise PDFTextExtractionError(f"Invalid page number: {page_num}")

        try:
            if self.pdf_handler._document is None:
                raise PDFTextExtractionError("PDF document not loaded")
            page = self.pdf_handler._document[page_num]

            # Extract text using multiple methods
            basic_text = self._extract_basic_text(page)
            blocks = self._extract_text_blocks(page)
            detailed_dict = page.get_text("dict")

            # Analyze layout and structure
            structured_blocks = self._structure_text_blocks(blocks, detailed_dict)
            tables = self._detect_tables(page)

            # Calculate quality metrics
            quality_score = self._calculate_quality_score(structured_blocks, basic_text)
            reading_order_conf = self._assess_reading_order(structured_blocks)

            # Get font statistics
            font_stats = self._analyze_fonts(detailed_dict)

            # Detect headers and footers
            has_headers, has_footers = self._detect_headers_footers(
                structured_blocks, page.rect.height
            )

            # Create result
            return ExtractedPage(
                page_num=page_num,
                text=basic_text,
                blocks=structured_blocks,
                tables=tables,
                quality_score=quality_score,
                word_count=len(basic_text.split()),
                char_count=len(basic_text),
                avg_font_size=font_stats["avg_size"],
                dominant_font=font_stats["dominant_font"],
                has_headers=has_headers,
                has_footers=has_footers,
                reading_order_confidence=reading_order_conf,
            )

        except Exception as e:
            logger.exception(f"Error extracting page {page_num}")
            raise PDFTextExtractionError(f"Failed to extract page {page_num}: {str(e)}")

    def extract_page_text(self, page_num: int) -> PageText:
        """
        Extract text from a page and return as PageText model.

        This method provides compatibility with the existing PageText model
        while using the advanced extraction capabilities.

        Args:
            page_num: Zero-based page number

        Returns:
            PageText object with extracted text and metrics
        """
        extracted = self.extract_page(page_num)

        return PageText(
            text=extracted.text,
            confidence=extracted.quality_score,
            extraction_method="pymupdf_advanced",
            char_count=extracted.char_count,
            word_count=extracted.word_count,
            has_tables=len(extracted.tables) > 0,
            has_images=False,  # Would need image detection
            bbox_count=len(extracted.blocks),
            avg_font_size=extracted.avg_font_size,
        )

    def _extract_basic_text(self, page: Any) -> str:
        """Extract basic text content from a page."""
        return page.get_text().strip()

    def _extract_text_blocks(self, page: Any) -> List[Dict[str, Any]]:
        """Extract text blocks with layout information."""
        blocks = page.get_text("blocks")
        return [
            {
                "x0": b[0],
                "y0": b[1],
                "x1": b[2],
                "y1": b[3],
                "text": b[4].strip(),
                "block_no": b[5],
                "block_type": b[6],
            }
            for b in blocks
            if b[4].strip()  # Only non-empty blocks
        ]

    def _structure_text_blocks(
        self, blocks: List[Dict], detailed_dict: Dict
    ) -> List[TextBlock]:
        """Convert raw blocks to structured TextBlock objects with font info."""
        structured = []
        block_fonts = defaultdict(list)

        # Extract font information from detailed dictionary
        for block in detailed_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                block_no = block.get("number", -1)
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_info = {
                            "font": span.get("font", ""),
                            "size": span.get("size", 0),
                            "flags": span.get("flags", 0),
                        }
                        block_fonts[block_no].append(font_info)

        # Create TextBlock objects
        for i, block in enumerate(blocks):
            # Get dominant font for this block
            fonts = block_fonts.get(i, [])
            if fonts:
                # Find most common font and average size
                font_names = [f["font"] for f in fonts if f["font"]]
                font_sizes = [f["size"] for f in fonts if f["size"] > 0]
                flags = [f["flags"] for f in fonts]

                dominant_font = (
                    max(set(font_names), key=font_names.count) if font_names else ""
                )
                avg_size = mean(font_sizes) if font_sizes else 0.0
                dominant_flags = max(set(flags), key=flags.count) if flags else 0
            else:
                dominant_font = ""
                avg_size = 0.0
                dominant_flags = 0

            structured.append(
                TextBlock(
                    text=block["text"],
                    bbox=(block["x0"], block["y0"], block["x1"], block["y1"]),
                    block_no=block["block_no"],
                    line_no=0,  # Would need line-level parsing
                    span_no=0,  # Would need span-level parsing
                    font_name=dominant_font,
                    font_size=avg_size,
                    flags=dominant_flags,
                )
            )

        return structured

    def _detect_tables(self, page: Any) -> List[Dict[str, Any]]:
        """
        Detect potential tables in the page.

        This is a simple heuristic-based approach that looks for
        aligned text blocks that might form a table structure.
        """
        tables: List[Dict[str, Any]] = []

        # Get all text blocks
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]  # Type 0 = text

        if len(text_blocks) < 3:
            return tables

        # Sort blocks by vertical position
        sorted_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))

        # Look for horizontal alignment patterns
        rows: Dict[float, List[Any]] = defaultdict(list)
        tolerance = 5.0  # pixels

        for block in sorted_blocks:
            y_pos = block[1]
            # Find row with similar y position
            matched_row = None
            for row_y in rows:
                if abs(row_y - y_pos) < tolerance:
                    matched_row = row_y
                    break

            if matched_row:
                rows[matched_row].append(block)
            else:
                rows[y_pos].append(block)

        # Check if we have a table-like structure
        row_counts = [len(blocks) for blocks in rows.values()]
        if len(row_counts) > 2 and max(row_counts) > 1:
            # Possible table detected
            avg_cols = mean(row_counts)
            if avg_cols > 1.5:  # Average more than 1.5 columns per row
                tables.append(
                    {
                        "rows": len(rows),
                        "avg_columns": avg_cols,
                        "confidence": min(
                            avg_cols / 3.0, 1.0
                        ),  # Higher column count = higher confidence
                    }
                )

        return tables

    def _calculate_quality_score(
        self, blocks: List[TextBlock], basic_text: str
    ) -> float:
        """
        Calculate a quality score for the extracted text.

        Considers factors like:
        - Text length and content
        - Font consistency
        - Layout preservation
        - Character encoding issues
        """
        if not blocks or not basic_text:
            return 0.0

        scores = []

        # Text content score (longer text = better)
        text_score = min(len(basic_text) / 1000.0, 1.0)
        scores.append(text_score)

        # Font consistency score
        font_sizes = [b.font_size for b in blocks if b.font_size > 0]
        if font_sizes:
            size_variance = np.var(font_sizes)
            font_score = 1.0 / (1.0 + size_variance / 100.0)
            scores.append(font_score)

        # Check for encoding issues
        encoding_issues = len(re.findall(r"[�\ufffd]", basic_text))
        encoding_score = 1.0 / (1.0 + encoding_issues / 10.0)
        scores.append(encoding_score)

        # Layout score (blocks should have reasonable sizes)
        block_sizes = [
            (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]) for b in blocks
        ]
        if block_sizes:
            reasonable_sizes = sum(1 for s in block_sizes if 100 < s < 100000)
            layout_score = reasonable_sizes / len(block_sizes)
            scores.append(layout_score)

        return mean(scores)

    def _assess_reading_order(self, blocks: List[TextBlock]) -> float:
        """Assess how well the text blocks follow a logical reading order."""
        if len(blocks) < 2:
            return 1.0

        # Check if blocks follow top-to-bottom, left-to-right order
        order_violations = 0

        for i in range(1, len(blocks)):
            curr = blocks[i]
            prev = blocks[i - 1]

            # Expected: current block is either below previous or to the right
            if (
                curr.bbox[1] < prev.bbox[1] - 10
            ):  # Current is above previous (with tolerance)
                # Check if it's a column layout (current is to the right)
                if curr.bbox[0] <= prev.bbox[0]:
                    order_violations += 1

        confidence = 1.0 - (order_violations / len(blocks))
        return max(0.0, confidence)

    def _analyze_fonts(self, detailed_dict: Dict) -> Dict[str, Any]:
        """Analyze font usage in the document."""
        fonts = []
        sizes = []

        for block in detailed_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font = span.get("font", "")
                        size = span.get("size", 0)
                        if font:
                            fonts.append(font)
                        if size > 0:
                            sizes.append(size)

        result = {
            "avg_size": mean(sizes) if sizes else 0.0,
            "dominant_font": max(set(fonts), key=fonts.count) if fonts else "",
            "font_variety": len(set(fonts)),
            "size_range": (min(sizes), max(sizes)) if sizes else (0, 0),
        }

        return result

    def _detect_headers_footers(
        self, blocks: List[TextBlock], page_height: float
    ) -> Tuple[bool, bool]:
        """Detect if the page has headers and footers based on position."""
        if not blocks:
            return False, False

        # Header: text in top 10% of page
        # Footer: text in bottom 10% of page
        header_threshold = page_height * 0.1
        footer_threshold = page_height * 0.9

        has_header = any(b.bbox[1] < header_threshold for b in blocks)
        has_footer = any(b.bbox[3] > footer_threshold for b in blocks)

        return has_header, has_footer

    def extract_document_segments(
        self, page_ranges: List[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        """
        Extract text from specific document segments defined by page ranges.

        This is useful for extracting text from individual documents after
        boundary detection has identified separate documents.

        Args:
            page_ranges: List of (start_page, end_page) tuples (1-indexed)

        Returns:
            List of dictionaries containing document text and metadata
        """
        segments = []

        for doc_idx, (start_page, end_page) in enumerate(page_ranges):
            # Convert to 0-indexed
            start_idx = start_page - 1
            end_idx = end_page

            doc_text = []
            doc_pages = []
            total_quality = 0.0

            for page_num in range(start_idx, end_idx):
                try:
                    extracted = self.extract_page(page_num)
                    doc_text.append(extracted.text)
                    doc_pages.append(extracted)
                    total_quality += extracted.quality_score
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num + 1}: {e}")

            if doc_pages:
                segments.append(
                    {
                        "document_index": doc_idx,
                        "page_range": (start_page, end_page),
                        "page_count": end_page - start_page + 1,
                        "full_text": "\n\n".join(doc_text),
                        "pages": doc_pages,
                        "avg_quality_score": total_quality / len(doc_pages),
                        "total_word_count": sum(p.word_count for p in doc_pages),
                        "total_char_count": sum(p.char_count for p in doc_pages),
                    }
                )

        return segments
