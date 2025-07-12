"""
Results Service

Manages split results, file information, and download history.
"""
import asyncio
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pdf_splitter.api.config import config
from pdf_splitter.api.models.results import (
    DownloadHistory,
    DownloadManifest,
    FilePreview,
    FileType,
    OutputFileInfo,
    ResultsFilter,
    ResultsPage,
    SplitResultDetailed,
)
from pdf_splitter.api.models.websocket import ProcessingStage
from pdf_splitter.api.services.session_service import SessionService
from pdf_splitter.api.services.websocket_enhanced import enhanced_websocket_manager
from pdf_splitter.api.utils.exceptions import (
    FileNotFoundError,
    ProcessingError,
    SessionNotFoundError,
)
from pdf_splitter.preprocessing.pdf_handler import PDFHandler


class ResultsService:
    """Service for managing split results and file information."""

    def __init__(self):
        self.session_service = SessionService()
        self.download_history: List[DownloadHistory] = []
        self._results_cache: Dict[str, SplitResultDetailed] = {}

    def get_session_results(
        self, session_id: str, include_file_info: bool = True
    ) -> SplitResultDetailed:
        """
        Get detailed results for a session.

        Args:
            session_id: Session ID
            include_file_info: Include detailed file information

        Returns:
            Detailed split results

        Raises:
            SessionNotFoundError: If session not found
            ProcessingError: If results not available
        """
        # Check cache first
        if session_id in self._results_cache:
            return self._results_cache[session_id]

        # Get session details
        session = self.session_service.get_session_details(session_id)

        if session["status"] not in ["complete", "confirmed"]:
            raise ProcessingError(f"Session {session_id} is not complete")

        metadata = session.get("metadata", {})

        # Build detailed results
        results = SplitResultDetailed(
            session_id=session_id,
            split_id=metadata.get("split_id", ""),
            status=session["status"],
            created_at=datetime.fromisoformat(session["created_at"]),
            completed_at=datetime.fromisoformat(
                metadata.get("completed_at", session["updated_at"])
            ),
            processing_time=metadata.get("processing_time", 0),
            input_file=session.get("pdf_path", "unknown.pdf"),
            input_size=self._get_file_size(session.get("pdf_path")),
            total_pages=metadata.get("total_pages", 0),
            files_created=metadata.get("files_created", 0),
            total_output_size=metadata.get("output_size", 0),
            segments_processed=metadata.get("segments_processed", 0),
            boundaries_detected=metadata.get("boundaries_detected", 0),
            detection_method=metadata.get("detection_method", "embeddings"),
        )

        # Get output files if requested
        if include_file_info:
            output_files = self._get_output_files(session_id, metadata)
            results.output_files = output_files

            # Update totals
            results.files_created = len(output_files)
            results.total_output_size = sum(f.size for f in output_files)

        # Cache results
        self._results_cache[session_id] = results

        return results

    def get_file_info(self, session_id: str, filename: str) -> OutputFileInfo:
        """
        Get information about a specific output file.

        Args:
            session_id: Session ID
            filename: Filename

        Returns:
            File information

        Raises:
            FileNotFoundError: If file not found
        """
        results = self.get_session_results(session_id)

        for file_info in results.output_files:
            if file_info.filename == filename:
                return file_info

        raise FileNotFoundError(f"File {filename} not found in session {session_id}")

    def search_results(
        self, filter_criteria: ResultsFilter, page: int = 1, page_size: int = 20
    ) -> ResultsPage:
        """
        Search and filter results with pagination.

        Args:
            filter_criteria: Filter criteria
            page: Page number (1-based)
            page_size: Results per page

        Returns:
            Paginated results
        """
        # Get all sessions
        all_sessions = self.session_service.get_all_sessions()

        # Apply filters
        filtered_results = []

        for session_id, session in all_sessions.items():
            # Skip incomplete sessions
            if session.status not in ["complete", "confirmed"]:
                continue

            # Apply session ID filter
            if (
                filter_criteria.session_ids
                and session_id not in filter_criteria.session_ids
            ):
                continue

            # Apply status filter
            if filter_criteria.status and session.status not in filter_criteria.status:
                continue

            # Apply date filters
            created_at = session.created_at
            if (
                filter_criteria.created_after
                and created_at < filter_criteria.created_after
            ):
                continue
            if (
                filter_criteria.created_before
                and created_at > filter_criteria.created_before
            ):
                continue

            # Get detailed results
            try:
                results = self.get_session_results(session_id, include_file_info=False)

                # Apply file count filters
                if (
                    filter_criteria.min_files
                    and results.files_created < filter_criteria.min_files
                ):
                    continue
                if (
                    filter_criteria.max_files
                    and results.files_created > filter_criteria.max_files
                ):
                    continue

                filtered_results.append(results)

            except Exception:
                # Skip sessions with errors
                continue

        # Sort by creation date (newest first)
        filtered_results.sort(key=lambda r: r.created_at, reverse=True)

        # Paginate
        total = len(filtered_results)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_results = filtered_results[start_idx:end_idx]

        return ResultsPage.create(page_results, total, page, page_size)

    def create_download_manifest(
        self, session_id: str, file_filter: Optional[Dict[str, Any]] = None
    ) -> DownloadManifest:
        """
        Create a download manifest for batch downloads.

        Args:
            session_id: Session ID
            file_filter: Optional filter criteria

        Returns:
            Download manifest
        """
        results = self.get_session_results(session_id)

        # Filter files
        files = results.output_files
        if file_filter:
            # Apply filters
            if "file_types" in file_filter:
                files = [f for f in files if f.file_type in file_filter["file_types"]]
            if "min_size" in file_filter:
                files = [f for f in files if f.size >= file_filter["min_size"]]
            if "max_size" in file_filter:
                files = [f for f in files if f.size <= file_filter["max_size"]]

        # Create manifest
        manifest = DownloadManifest(
            manifest_id=f"manifest_{session_id}_{int(datetime.utcnow().timestamp())}",
            session_id=session_id,
            created_at=datetime.utcnow(),
            total_files=len(files),
            total_size=sum(f.size for f in files),
            files=files,
            expires_at=datetime.utcnow() + timedelta(hours=24),
        )

        return manifest

    async def generate_file_preview(
        self,
        session_id: str,
        filename: str,
        preview_type: str = "auto",
        max_pages: int = 3,
    ) -> FilePreview:
        """
        Generate preview for a file.

        Args:
            session_id: Session ID
            filename: Filename
            preview_type: Preview type ('text', 'image', 'metadata', 'auto')
            max_pages: Maximum pages for preview

        Returns:
            File preview
        """
        file_info = self.get_file_info(session_id, filename)

        preview = FilePreview(
            filename=filename, file_type=file_info.file_type, preview_type=preview_type
        )

        if file_info.file_type == FileType.PDF:
            # Generate PDF preview
            if preview_type in ["text", "auto"]:
                preview.content = await self._generate_text_preview(
                    file_info.path, max_pages
                )
                preview.preview_type = "text"
            elif preview_type == "image":
                preview.content = await self._generate_image_preview(
                    file_info.path, max_pages
                )
                preview.preview_type = "image"

            # Add metadata
            preview.metadata = await self._get_pdf_metadata(file_info.path)
            preview.page_count = preview.metadata.get("page_count", 0)

        elif file_info.file_type == FileType.MANIFEST:
            # Read manifest content
            with open(file_info.path, "r") as f:
                preview.content = f.read()
                preview.preview_type = "text"

        return preview

    def record_download(
        self,
        session_id: str,
        filename: str,
        client_info: Dict[str, Any],
        download_time: float,
        completed: bool = True,
        error: Optional[str] = None,
    ) -> DownloadHistory:
        """
        Record a download in history.

        Args:
            session_id: Session ID
            filename: Downloaded filename
            client_info: Client information (IP, user agent, etc.)
            download_time: Download duration in seconds
            completed: Whether download completed successfully
            error: Error message if failed

        Returns:
            Download history entry
        """
        file_info = self.get_file_info(session_id, filename)

        history = DownloadHistory(
            download_id=f"dl_{int(datetime.utcnow().timestamp())}",
            session_id=session_id,
            filename=filename,
            file_size=file_info.size,
            downloaded_at=datetime.utcnow(),
            client_ip=client_info.get("ip"),
            user_agent=client_info.get("user_agent"),
            download_time=download_time,
            completed=completed,
            error=error,
        )

        self.download_history.append(history)

        # Keep only recent history (last 1000 entries)
        if len(self.download_history) > 1000:
            self.download_history = self.download_history[-1000:]

        return history

    def get_download_analytics(
        self,
        session_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> Dict[str, Any]:
        """
        Get download analytics.

        Args:
            session_id: Optional session ID filter
            time_range: Optional time range filter

        Returns:
            Analytics data
        """
        # Filter history
        filtered_history = self.download_history

        if session_id:
            filtered_history = [
                h for h in filtered_history if h.session_id == session_id
            ]

        if time_range:
            start, end = time_range
            filtered_history = [
                h for h in filtered_history if start <= h.downloaded_at <= end
            ]

        if not filtered_history:
            return {
                "total_downloads": 0,
                "total_size": 0,
                "average_download_time": 0,
                "success_rate": 0,
                "popular_files": [],
                "download_by_hour": {},
            }

        # Calculate analytics
        total_downloads = len(filtered_history)
        successful_downloads = sum(1 for h in filtered_history if h.completed)
        total_size = sum(h.file_size for h in filtered_history)
        total_time = sum(h.download_time for h in filtered_history)

        # Popular files
        file_counts = defaultdict(int)
        for h in filtered_history:
            file_counts[h.filename] += 1

        popular_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        # Downloads by hour
        downloads_by_hour = defaultdict(int)
        for h in filtered_history:
            hour = h.downloaded_at.hour
            downloads_by_hour[hour] += 1

        return {
            "total_downloads": total_downloads,
            "successful_downloads": successful_downloads,
            "failed_downloads": total_downloads - successful_downloads,
            "success_rate": (successful_downloads / total_downloads) * 100
            if total_downloads > 0
            else 0,
            "total_size": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "average_download_time": total_time / total_downloads
            if total_downloads > 0
            else 0,
            "average_file_size": total_size / total_downloads
            if total_downloads > 0
            else 0,
            "popular_files": popular_files,
            "downloads_by_hour": dict(downloads_by_hour),
            "time_range": {
                "start": min(h.downloaded_at for h in filtered_history).isoformat(),
                "end": max(h.downloaded_at for h in filtered_history).isoformat(),
            }
            if filtered_history
            else None,
        }

    def cleanup_old_results(self, days_to_keep: int = 7) -> int:
        """
        Clean up old results and files.

        Args:
            days_to_keep: Number of days to keep results

        Returns:
            Number of sessions cleaned up
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        cleaned_count = 0

        # Get all sessions
        all_sessions = self.session_service.get_all_sessions()

        for session_id, session in all_sessions.items():
            if session.created_at < cutoff_date:
                # Delete output files
                output_dir = config.output_dir / session_id
                if output_dir.exists():
                    import shutil

                    shutil.rmtree(output_dir)

                # Remove from cache
                self._results_cache.pop(session_id, None)

                cleaned_count += 1

        return cleaned_count

    def _get_output_files(
        self, session_id: str, metadata: Dict[str, Any]
    ) -> List[OutputFileInfo]:
        """Get output file information."""
        output_files = []
        output_dir = config.output_dir / session_id

        if not output_dir.exists():
            return output_files

        # Get files from metadata or scan directory
        file_paths = metadata.get("output_files", [])
        if not file_paths:
            # Scan directory
            file_paths = [str(f) for f in output_dir.glob("*.pdf")]

        for file_path in file_paths:
            path = Path(file_path)
            if path.exists():
                file_info = OutputFileInfo(
                    filename=path.name,
                    path=str(path),
                    size=path.stat().st_size,
                    file_type=self._determine_file_type(path),
                    created_at=datetime.fromtimestamp(path.stat().st_ctime),
                    checksum=self._calculate_checksum(path),
                    metadata=self._extract_file_metadata(path),
                )

                # Extract document type and page range from filename
                self._parse_filename_info(file_info)

                output_files.append(file_info)

        return output_files

    def _determine_file_type(self, path: Path) -> FileType:
        """Determine file type from extension."""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return FileType.PDF
        elif suffix == ".zip":
            return FileType.ZIP
        elif suffix in [".json", ".txt"]:
            return FileType.MANIFEST
        elif suffix == ".log":
            return FileType.LOG
        else:
            return FileType.PDF

    def _calculate_checksum(self, path: Path, algorithm: str = "md5") -> str:
        """Calculate file checksum."""
        hash_func = hashlib.new(algorithm)

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def _extract_file_metadata(self, path: Path) -> Dict[str, Any]:
        """Extract metadata from file."""
        metadata = {
            "created": datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            "permissions": oct(path.stat().st_mode)[-3:],
        }

        return metadata

    def _parse_filename_info(self, file_info: OutputFileInfo):
        """Parse document type and page range from filename."""
        # Example: "invoice_2024-01-15_p1-5.pdf"
        filename = file_info.filename.replace(".pdf", "")
        parts = filename.split("_")

        if parts:
            # First part is usually document type
            file_info.document_type = parts[0]

            # Look for page range (p1-5 format)
            for part in parts:
                if part.startswith("p") and "-" in part:
                    file_info.page_range = part[1:]
                    break

    def _get_file_size(self, file_path: Optional[str]) -> int:
        """Get file size safely."""
        if not file_path:
            return 0

        path = Path(file_path)
        if path.exists():
            return path.stat().st_size

        return 0

    async def _generate_text_preview(self, file_path: str, max_pages: int) -> str:
        """Generate text preview of PDF."""
        try:
            pdf_handler = PDFHandler()
            text_content = []

            with pdf_handler.load_pdf(file_path) as pdf_doc:
                pages_to_preview = min(max_pages, len(pdf_doc))

                for page_num in range(pages_to_preview):
                    page_data = await pdf_handler.process_page(pdf_doc, page_num)
                    text_content.append(
                        f"--- Page {page_num + 1} ---\n{page_data.text}\n"
                    )

            preview = "\n".join(text_content)

            # Truncate if too long
            max_length = 5000
            if len(preview) > max_length:
                preview = preview[:max_length] + "\n... (truncated)"

            return preview

        except Exception as e:
            return f"Error generating preview: {str(e)}"

    async def _generate_image_preview(self, file_path: str, max_pages: int) -> str:
        """Generate image preview of PDF pages."""
        # This would generate base64 encoded images
        # Implementation depends on requirements
        return "Image preview not implemented"

    async def _get_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get PDF metadata."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)
            metadata = doc.metadata or {}

            # Add additional info
            metadata["page_count"] = len(doc)
            metadata["file_size"] = os.path.getsize(file_path)

            doc.close()

            return metadata

        except Exception:
            return {"error": "Could not read PDF metadata"}
