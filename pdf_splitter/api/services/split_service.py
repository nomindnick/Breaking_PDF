"""
Split Management Service

Handles split proposal management, modifications, and execution.
"""
import asyncio
import base64
import io
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from pdf_splitter.api.config import config
from pdf_splitter.api.models.websocket import ProcessingStage as WSProcessingStage
from pdf_splitter.api.services.websocket_enhanced import enhanced_websocket_manager
from pdf_splitter.api.services.websocket_service import websocket_manager
from pdf_splitter.api.utils.exceptions import ProcessingError, SessionNotFoundError
from pdf_splitter.preprocessing.pdf_handler import PDFHandler
from pdf_splitter.splitting.models import (
    DocumentSegment,
    ModificationType,
    SessionModification,
    SessionStatus,
    SplitProposal,
)
from pdf_splitter.splitting.pdf_splitter import PDFSplitter
from pdf_splitter.splitting.session_manager import SessionManager


class SplitService:
    """Service for managing split proposals and execution."""

    def __init__(
        self, session_manager: SessionManager = None, pdf_splitter: PDFSplitter = None
    ):
        self.session_manager = session_manager or SessionManager(
            str(config.session_db_path)
        )
        self.pdf_splitter = pdf_splitter or PDFSplitter()
        self.active_splits: Dict[str, asyncio.Task] = {}

        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)

    def get_proposal(self, session_id: str) -> Optional[SplitProposal]:
        """
        Get the current split proposal for a session.

        Args:
            session_id: Session ID

        Returns:
            Split proposal or None if not found

        Raises:
            SessionNotFoundError: If session not found
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        return self.session_manager.get_proposal(session_id)

    def update_proposal(
        self, session_id: str, updates: Dict[str, Any]
    ) -> SplitProposal:
        """
        Update split proposal with modifications.

        Args:
            session_id: Session ID
            updates: Dictionary of updates to apply

        Returns:
            Updated proposal

        Raises:
            SessionNotFoundError: If session not found
            ProcessingError: If update fails
        """
        proposal = self.get_proposal(session_id)
        if not proposal:
            raise ProcessingError("No proposal found for session")

        # Track modification
        modification = SessionModification(
            modification_type=ModificationType.BOUNDARY_ADJUSTED,
            details=updates,
            timestamp=datetime.utcnow(),
        )

        # Apply updates based on type
        if "segments" in updates:
            # Full segment replacement
            proposal.segments = updates["segments"]

        if "merge_segments" in updates:
            # Merge specified segments
            proposal = self._merge_segments(proposal, updates["merge_segments"])

        if "split_at_page" in updates:
            # Split a segment at specified page
            proposal = self._split_segment(
                proposal,
                updates["split_at_page"]["segment_id"],
                updates["split_at_page"]["page_number"],
            )

        if "update_segment" in updates:
            # Update specific segment metadata
            proposal = self._update_segment(
                proposal,
                updates["update_segment"]["segment_id"],
                updates["update_segment"]["changes"],
            )

        # Update timestamps
        proposal.modified_at = datetime.utcnow()

        # Save updated proposal
        self.session_manager.save_proposal(session_id, proposal)
        self.session_manager.add_modification(session_id, modification)

        return proposal

    def _merge_segments(
        self, proposal: SplitProposal, segment_ids: List[str]
    ) -> SplitProposal:
        """Merge multiple segments into one."""
        if len(segment_ids) < 2:
            raise ProcessingError("At least 2 segments required for merge")

        # Find segments to merge
        segments_to_merge = []
        other_segments = []

        for segment in proposal.segments:
            if segment.id in segment_ids:
                segments_to_merge.append(segment)
            else:
                other_segments.append(segment)

        if len(segments_to_merge) != len(segment_ids):
            raise ProcessingError("Some segments not found")

        # Sort by page number
        segments_to_merge.sort(key=lambda s: s.start_page)

        # Create merged segment
        merged_segment = DocumentSegment(
            start_page=segments_to_merge[0].start_page,
            end_page=segments_to_merge[-1].end_page,
            document_type=segments_to_merge[0].document_type,
            confidence=min(s.confidence for s in segments_to_merge),
            metadata={
                "merged_from": [s.id for s in segments_to_merge],
                "merged_at": datetime.utcnow().isoformat(),
            },
        )

        # Rebuild segments list maintaining order
        new_segments = []
        merged_added = False

        for segment in proposal.segments:
            if segment.id in segment_ids:
                if not merged_added:
                    new_segments.append(merged_segment)
                    merged_added = True
            else:
                new_segments.append(segment)

        proposal.segments = new_segments
        return proposal

    def _split_segment(
        self, proposal: SplitProposal, segment_id: str, split_page: int
    ) -> SplitProposal:
        """Split a segment at specified page."""
        # Find segment to split
        segment_index = None
        segment_to_split = None

        for i, segment in enumerate(proposal.segments):
            if segment.id == segment_id:
                segment_index = i
                segment_to_split = segment
                break

        if not segment_to_split:
            raise ProcessingError(f"Segment {segment_id} not found")

        # Validate split page
        if (
            split_page <= segment_to_split.start_page
            or split_page > segment_to_split.end_page
        ):
            raise ProcessingError(
                f"Split page {split_page} must be within segment range "
                f"{segment_to_split.start_page + 1} to {segment_to_split.end_page}"
            )

        # Create two new segments
        segment1 = DocumentSegment(
            start_page=segment_to_split.start_page,
            end_page=split_page - 1,
            document_type=segment_to_split.document_type,
            confidence=segment_to_split.confidence,
            metadata={
                **segment_to_split.metadata,
                "split_from": segment_to_split.id,
                "split_at": datetime.utcnow().isoformat(),
            },
        )

        segment2 = DocumentSegment(
            start_page=split_page,
            end_page=segment_to_split.end_page,
            document_type=segment_to_split.document_type,
            confidence=segment_to_split.confidence,
            metadata={
                **segment_to_split.metadata,
                "split_from": segment_to_split.id,
                "split_at": datetime.utcnow().isoformat(),
            },
        )

        # Replace original segment with two new ones
        proposal.segments[segment_index : segment_index + 1] = [segment1, segment2]

        return proposal

    def _update_segment(
        self, proposal: SplitProposal, segment_id: str, changes: Dict[str, Any]
    ) -> SplitProposal:
        """Update segment metadata."""
        # Find segment
        segment = None
        for s in proposal.segments:
            if s.id == segment_id:
                segment = s
                break

        if not segment:
            raise ProcessingError(f"Segment {segment_id} not found")

        # Apply changes
        if "document_type" in changes:
            segment.document_type = changes["document_type"]

        if "metadata" in changes:
            segment.metadata.update(changes["metadata"])

        if "confidence" in changes:
            segment.confidence = changes["confidence"]

        # Update suggested filename if type changed
        if "document_type" in changes:
            # This will be regenerated during split execution
            segment.suggested_filename = None

        return proposal

    async def generate_preview(
        self, session_id: str, segment_id: str, max_pages: int = 3
    ) -> List[str]:
        """
        Generate preview images for a segment.

        Args:
            session_id: Session ID
            segment_id: Segment ID
            max_pages: Maximum pages to preview

        Returns:
            List of base64 encoded preview images

        Raises:
            SessionNotFoundError: If session not found
            ProcessingError: If preview generation fails
        """
        # Get proposal and session
        proposal = self.get_proposal(session_id)
        if not proposal:
            raise ProcessingError("No proposal found")

        session = self.session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        # Find segment
        segment = None
        for s in proposal.segments:
            if s.id == segment_id:
                segment = s
                break

        if not segment:
            raise ProcessingError(f"Segment {segment_id} not found")

        # Generate previews
        try:
            pdf_handler = PDFHandler()
            previews = []

            with pdf_handler.load_pdf(session.pdf_path) as pdf_doc:
                # Determine pages to preview
                pages_to_preview = min(
                    max_pages, segment.end_page - segment.start_page + 1
                )

                for i in range(pages_to_preview):
                    page_num = segment.start_page + i

                    # Render page as image
                    page = pdf_doc[page_num]
                    pix = page.get_pixmap(dpi=150)

                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))

                    # Resize if too large
                    max_width = 800
                    if img.width > max_width:
                        ratio = max_width / img.width
                        new_height = int(img.height * ratio)
                        img = img.resize(
                            (max_width, new_height), Image.Resampling.LANCZOS
                        )

                    # Convert to base64
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG", optimize=True)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    previews.append(f"data:image/png;base64,{img_base64}")

            return previews

        except Exception as e:
            raise ProcessingError(f"Failed to generate preview: {str(e)}")

    async def execute_split(
        self, session_id: str, output_format: Optional[str] = None
    ) -> str:
        """
        Execute the split operation for a session.

        Args:
            session_id: Session ID
            output_format: Optional output format preferences

        Returns:
            Split ID for tracking

        Raises:
            SessionNotFoundError: If session not found
            ProcessingError: If split fails
        """
        # Validate session and proposal
        session = self.session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        proposal = self.get_proposal(session_id)
        if not proposal:
            raise ProcessingError("No proposal found for session")

        # Generate split ID
        split_id = f"split_{session_id}_{int(time.time())}"

        # Create output directory
        output_dir = config.output_dir / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Start split task
        task = asyncio.create_task(
            self._execute_split_task(session_id, split_id, proposal, output_dir)
        )

        # Track active split
        self.active_splits[split_id] = task

        # Clean up when done
        task.add_done_callback(lambda t: self.active_splits.pop(split_id, None))

        return split_id

    async def _execute_split_task(
        self, session_id: str, split_id: str, proposal: SplitProposal, output_dir: Path
    ):
        """Execute split operation in background."""
        start_time = time.time()

        try:
            # Update session status
            self.session_manager.update_session_status(
                session_id,
                SessionStatus.PROCESSING,
                {"split_id": split_id, "stage": "splitting"},
            )

            # Send initial status via enhanced WebSocket
            await enhanced_websocket_manager.send_status_update(
                session_id,
                "splitting",
                "Starting PDF split operation",
                {"split_id": split_id},
            )

            # Also send via legacy WebSocket for backward compatibility
            await websocket_manager.send_status_update(
                session_id, "splitting", "Starting PDF split operation"
            )

            # Progress tracking
            total_segments = len(proposal.segments)

            async def progress_callback(current: int, total: int, message: str):
                progress = (current / total) * 100

                # Send via enhanced WebSocket
                await enhanced_websocket_manager.send_progress_update(
                    session_id,
                    WSProcessingStage.SPLITTING,
                    progress,
                    message,
                    {
                        "split_id": split_id,
                        "current_segment": current,
                        "total_segments": total,
                    },
                )

                # Also send via legacy WebSocket
                await websocket_manager.send_progress_update(
                    session_id,
                    "splitting",
                    progress,
                    message,
                    {
                        "split_id": split_id,
                        "current_segment": current,
                        "total_segments": total,
                    },
                )

            # Execute split
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._split_pdf_sync, proposal, str(output_dir), progress_callback
            )

            # Calculate total output size
            total_size = sum(
                Path(f).stat().st_size for f in result.output_files if Path(f).exists()
            )

            # Update session with results
            processing_time = time.time() - start_time
            self.session_manager.update_session_status(
                session_id,
                SessionStatus.COMPLETE,
                {
                    "split_id": split_id,
                    "completed_at": datetime.utcnow().isoformat(),
                    "processing_time": processing_time,
                    "files_created": len(result.output_files),
                    "output_size": total_size,
                    "output_files": result.output_files,
                },
            )

            # Send completion via enhanced WebSocket
            await enhanced_websocket_manager.send_progress_update(
                session_id,
                WSProcessingStage.COMPLETE,
                100,
                f"Split complete! Created {len(result.output_files)} files",
                {
                    "split_id": split_id,
                    "files_created": len(result.output_files),
                    "processing_time": processing_time,
                    "output_size": total_size,
                },
            )

            # Also send via legacy WebSocket
            await websocket_manager.send_progress_update(
                session_id,
                "complete",
                100,
                f"Split complete! Created {len(result.output_files)} files",
                {
                    "split_id": split_id,
                    "files_created": len(result.output_files),
                    "processing_time": processing_time,
                    "output_size": total_size,
                },
            )

        except Exception as e:
            # Update with error
            error_msg = str(e)
            self.session_manager.update_session_status(
                session_id,
                SessionStatus.CANCELLED,
                {
                    "split_id": split_id,
                    "error": error_msg,
                    "failed_at": datetime.utcnow().isoformat(),
                },
            )

            # Send error via enhanced WebSocket
            await enhanced_websocket_manager.send_error(
                session_id,
                "SPLIT_FAILED",
                f"Split operation failed: {error_msg}",
                {"split_id": split_id},
                recoverable=False,
            )

            # Also send via legacy WebSocket
            await websocket_manager.send_error(
                session_id,
                f"Split operation failed: {error_msg}",
                {"split_id": split_id},
            )

            raise ProcessingError(f"Split operation failed: {error_msg}")

    def _split_pdf_sync(
        self, proposal: SplitProposal, output_dir: str, progress_callback
    ):
        """Synchronous PDF split execution."""

        # Create a sync wrapper for the async callback
        def sync_progress(current: int, total: int, message: str):
            # We'll handle progress in the async layer
            pass

        # Execute split
        result = self.pdf_splitter.split_pdf(
            proposal, output_dir, progress_callback=sync_progress
        )

        return result

    def get_split_status(self, session_id: str, split_id: str) -> Dict[str, Any]:
        """
        Get status of a split operation.

        Args:
            session_id: Session ID
            split_id: Split ID

        Returns:
            Status information
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        # Check if split is active
        is_active = split_id in self.active_splits

        # Get status from session metadata
        status_info = {
            "split_id": split_id,
            "session_id": session_id,
            "is_active": is_active,
            "status": session.status.value,
        }

        if session.metadata:
            if session.metadata.get("split_id") == split_id:
                status_info.update(
                    {
                        "stage": session.metadata.get("stage"),
                        "files_created": session.metadata.get("files_created"),
                        "output_size": session.metadata.get("output_size"),
                        "error": session.metadata.get("error"),
                    }
                )

        return status_info
