"""PDF splitting service for creating individual PDF files from segments.

This module provides the core functionality for splitting PDFs based on
detected boundaries and user-defined segments.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# PyMuPDF - imported if needed
import pikepdf

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.base_detector import (
    BoundaryResult,
    BoundaryType,
    ProcessedPage,
)
from pdf_splitter.splitting.exceptions import PDFSplitError
from pdf_splitter.splitting.models import DocumentSegment, SplitProposal, SplitResult

logger = logging.getLogger(__name__)


class PDFSplitter:
    """Service for splitting PDFs into individual documents.

    This class handles:
    - Converting boundary results to document segments
    - Generating filename suggestions
    - Splitting PDFs using pikepdf (preserves metadata better)
    - Creating previews for segments
    """

    def __init__(self, config: Optional[PDFConfig] = None):
        """Initialize PDFSplitter with configuration.

        Args:
            config: PDF processing configuration
        """
        self.config = config or PDFConfig()
        self._init_document_patterns()

    def _init_document_patterns(self):
        """Initialize patterns for document type detection."""
        self.document_patterns = {
            "Invoice": [
                r"invoice\s*#?\s*\d+",
                r"bill\s*to",
                r"total\s*amount",
                r"invoice\s*date",
                r"payment\s*due",
            ],
            "Email": [
                r"from:\s*\S+@\S+",
                r"to:\s*\S+@\S+",
                r"subject:",
                r"date:\s*\w+",
                r"sent:\s*\w+",
            ],
            "Letter": [
                r"dear\s+\w+",
                r"sincerely",
                r"regards",
                r"yours\s+truly",
                r"to\s+whom\s+it\s+may\s+concern",
            ],
            "Plans": [
                r"drawing\s*#?\s*\d+",
                r"scale:\s*\d+",
                r"project:",
                r"architect:",
                r"engineer:",
            ],
            "Contract": [
                r"agreement",
                r"whereas",
                r"party\s+of\s+the\s+first",
                r"terms\s+and\s+conditions",
                r"signature",
            ],
            "Report": [
                r"executive\s+summary",
                r"table\s+of\s+contents",
                r"conclusion",
                r"recommendations",
                r"findings",
            ],
        }

    def generate_proposal(
        self,
        boundaries: List[BoundaryResult],
        pages: List[ProcessedPage],
        pdf_path: Path,
    ) -> SplitProposal:
        """Generate a split proposal from boundary detection results.

        Args:
            boundaries: List of detected boundaries
            pages: Processed pages with text content
            pdf_path: Path to the source PDF

        Returns:
            SplitProposal with suggested segments
        """
        if not pdf_path.exists():
            raise PDFSplitError(f"PDF file not found: {pdf_path}")

        # Sort boundaries by page number
        sorted_boundaries = sorted(boundaries, key=lambda b: b.page_number)

        # Filter to document start boundaries
        start_boundaries = [
            b
            for b in sorted_boundaries
            if b.boundary_type == BoundaryType.DOCUMENT_START
        ]

        # Generate segments
        segments = []
        for i, boundary in enumerate(start_boundaries):
            start_page = boundary.page_number

            # Find end page (next boundary - 1 or last page)
            if i < len(start_boundaries) - 1:
                end_page = start_boundaries[i + 1].page_number - 1
            else:
                end_page = len(pages) - 1

            # Skip invalid segments
            if start_page > end_page:
                logger.warning(
                    f"Skipping invalid segment: start={start_page}, end={end_page}"
                )
                continue

            # Extract segment pages
            segment_pages = pages[start_page : end_page + 1]

            # Detect document type and generate filename
            document_type = self._detect_document_type(segment_pages)
            suggested_filename = self._suggest_filename(
                segment_pages, document_type, i + 1
            )

            # Extract summary (first few lines of text)
            summary = self._extract_summary(segment_pages)

            segment = DocumentSegment(
                start_page=start_page,
                end_page=end_page,
                document_type=document_type,
                suggested_filename=suggested_filename,
                confidence=boundary.confidence,
                summary=summary,
                metadata={
                    "detector": boundary.detector_type.value,
                    "page_count": end_page - start_page + 1,
                },
            )
            segments.append(segment)

        return SplitProposal(
            pdf_path=pdf_path,
            total_pages=len(pages),
            segments=segments,
            detection_results=boundaries,
        )

    def _detect_document_type(self, pages: List[ProcessedPage]) -> str:
        """Detect document type from page content.

        Args:
            pages: Pages to analyze

        Returns:
            Detected document type
        """
        # Combine text from first few pages
        text = " ".join(p.text.lower() for p in pages[:3])

        # Check patterns
        scores = {}
        for doc_type, patterns in self.document_patterns.items():
            score = sum(
                1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE)
            )
            scores[doc_type] = score

        # Return highest scoring type
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:
                return best_type

        return "Document"  # Default type

    def _suggest_filename(
        self, pages: List[ProcessedPage], document_type: str, segment_number: int
    ) -> str:
        """Generate a suggested filename for a segment.

        Args:
            pages: Pages in the segment
            document_type: Detected document type
            segment_number: Sequential number of this segment

        Returns:
            Suggested filename
        """
        try:
            # Extract date if possible
            date_str = self._extract_date(pages)

            # Extract identifier if possible
            identifier = self._extract_identifier(pages, document_type)

            # Build filename parts
            parts = []

            # Add document type
            parts.append(document_type.lower())

            # Add date if found
            if date_str:
                parts.append(date_str)

            # Add identifier if found
            if identifier:
                parts.append(identifier)
            else:
                # Use segment number as fallback
                parts.append(f"{segment_number:03d}")

            # Join parts and sanitize
            filename = "_".join(parts) + ".pdf"
            return self._sanitize_filename(filename)

        except Exception as e:
            logger.warning(f"Error suggesting filename: {e}")
            # Fallback filename
            return f"{document_type.lower()}_{segment_number:03d}.pdf"

    def _extract_date(self, pages: List[ProcessedPage]) -> Optional[str]:
        """Extract date from pages.

        Args:
            pages: Pages to search

        Returns:
            Date string in YYYY-MM-DD format or None
        """
        text = " ".join(p.text for p in pages[:2])

        # Common date patterns
        date_patterns = [
            r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})",  # MM/DD/YYYY or DD/MM/YYYY
            r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})",  # YYYY-MM-DD
            r"(\w+)\s+(\d{1,2}),?\s+(\d{4})",  # Month DD, YYYY
            r"(\d{1,2})\s+(\w+)\s+(\d{4})",  # DD Month YYYY
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    # Attempt to normalize to YYYY-MM-DD
                    # This is simplified - production would need better parsing
                    groups = match.groups()
                    if len(groups) == 3:
                        # Assume first pattern for now
                        if len(groups[2]) == 4:  # Year is last
                            return f"{groups[2]}-{groups[0]:0>2}-{groups[1]:0>2}"
                        else:  # Year is first
                            return f"{groups[0]}-{groups[1]:0>2}-{groups[2]:0>2}"
                except Exception:
                    continue

        return None

    def _extract_identifier(
        self, pages: List[ProcessedPage], document_type: str
    ) -> Optional[str]:
        """Extract document identifier (invoice number, etc.).

        Args:
            pages: Pages to search
            document_type: Type of document

        Returns:
            Identifier string or None
        """
        text = " ".join(p.text for p in pages[:2])

        # Patterns by document type
        id_patterns = {
            "Invoice": [r"invoice\s*#?\s*(\d+)", r"inv\s*#?\s*(\d+)"],
            "Contract": [r"contract\s*#?\s*([\w\-]+)", r"agreement\s*#?\s*([\w\-]+)"],
            "Plans": [r"drawing\s*#?\s*([\w\-]+)", r"sheet\s*#?\s*([\w\-]+)"],
            "Report": [r"report\s*#?\s*([\w\-]+)", r"document\s*#?\s*([\w\-]+)"],
        }

        patterns = id_patterns.get(document_type, [])

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Get the full match including any hyphens
                identifier = match.group(1)
                # Convert to lowercase and preserve hyphens/underscores
                return re.sub(r"[^\w\-_]", "", identifier).lower()

        return None

    def _extract_summary(self, pages: List[ProcessedPage]) -> str:
        """Extract a brief summary from the first page.

        Args:
            pages: Pages in the segment

        Returns:
            Summary text (max 200 chars)
        """
        if not pages:
            return ""

        # Get first page text
        text = pages[0].text.strip()

        # Get first few lines
        lines = text.split("\n")[:5]
        summary = " ".join(lines).strip()

        # Limit length
        if len(summary) > 200:
            summary = summary[:197] + "..."

        return summary

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility.

        Args:
            filename: Raw filename

        Returns:
            Sanitized filename
        """
        # Replace invalid characters
        invalid_chars = r'<>:"/\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")

        # Remove multiple underscores
        filename = re.sub(r"_+", "_", filename)

        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit(".", 1)
            filename = name[:250] + "." + ext

        return filename

    def split_pdf(
        self,
        proposal: SplitProposal,
        output_dir: Path,
        use_custom_names: Optional[Dict[str, str]] = None,
    ) -> SplitResult:
        """Execute the PDF split based on a proposal.

        Args:
            proposal: The split proposal to execute
            output_dir: Directory to save split PDFs
            use_custom_names: Optional mapping of segment_id to custom filename

        Returns:
            SplitResult with details of created files
        """
        start_time = datetime.now()

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = []

        try:
            # Open source PDF with pikepdf
            with pikepdf.open(proposal.pdf_path) as pdf:
                for segment in proposal.segments:
                    # Determine filename
                    if use_custom_names and segment.segment_id in use_custom_names:
                        filename = use_custom_names[segment.segment_id]
                    else:
                        filename = segment.suggested_filename

                    output_path = output_dir / filename

                    # Ensure unique filename
                    output_path = self._ensure_unique_path(output_path)

                    # Create new PDF with selected pages
                    new_pdf = pikepdf.new()

                    # Copy pages (pikepdf uses 0-based indexing)
                    for page_num in range(segment.start_page, segment.end_page + 1):
                        new_pdf.pages.append(pdf.pages[page_num])

                    # Preserve metadata if possible
                    if pdf.metadata:
                        new_pdf.metadata = pdf.metadata.copy()
                    else:
                        # Initialize metadata if it doesn't exist
                        new_pdf.metadata = {}

                    # Add custom metadata
                    summary_text = (
                        segment.summary[:50] if segment.summary else "No summary"
                    )
                    new_pdf.metadata[
                        "/Title"
                    ] = f"{segment.document_type} - {summary_text}"
                    new_pdf.metadata[
                        "/Subject"
                    ] = f"Split from {proposal.pdf_path.name}"
                    new_pdf.metadata["/Creator"] = "PDF Splitter"

                    # Save the new PDF
                    new_pdf.save(output_path)
                    output_files.append(output_path)

                    logger.info(f"Created: {output_path}")

        except Exception as e:
            raise PDFSplitError(f"Failed to split PDF: {e}")

        duration = (datetime.now() - start_time).total_seconds()

        return SplitResult(
            session_id=proposal.proposal_id,
            input_pdf=proposal.pdf_path,
            output_files=output_files,
            segments=proposal.segments,
            duration_seconds=duration,
        )

    def _ensure_unique_path(self, path: Path) -> Path:
        """Ensure path is unique by adding number suffix if needed.

        Args:
            path: Desired path

        Returns:
            Unique path
        """
        if not path.exists():
            return path

        # Add number suffix
        base = path.stem
        suffix = path.suffix
        parent = path.parent

        counter = 1
        while True:
            new_path = parent / f"{base}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    def generate_preview(
        self, pdf_path: Path, segment: DocumentSegment, max_pages: int = 5
    ) -> bytes:
        """Generate a preview PDF for a segment.

        Args:
            pdf_path: Source PDF path
            segment: Segment to preview
            max_pages: Maximum pages to include in preview

        Returns:
            PDF bytes for the preview
        """
        try:
            import io

            # Limit preview pages
            end_page = min(segment.end_page, segment.start_page + max_pages - 1)

            # Create preview PDF
            with pikepdf.open(pdf_path) as pdf:
                preview_pdf = pikepdf.new()

                # Copy limited pages
                for page_num in range(segment.start_page, end_page + 1):
                    preview_pdf.pages.append(pdf.pages[page_num])

                # Add watermark or indication this is a preview
                preview_pdf.metadata[
                    "/Title"
                ] = f"PREVIEW: {segment.suggested_filename}"

                # Save to bytes
                output = io.BytesIO()
                preview_pdf.save(output)
                return output.getvalue()

        except Exception as e:
            raise PDFSplitError(f"Failed to generate preview: {e}")
