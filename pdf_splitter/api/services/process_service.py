"""
PDF Processing Service

Orchestrates the complete PDF processing pipeline including preprocessing,
boundary detection, and split proposal generation.
"""
import asyncio
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pdf_splitter.api.config import config
from pdf_splitter.api.models.websocket import ProcessingStage as WSProcessingStage
from pdf_splitter.api.services.file_service import FileService
from pdf_splitter.api.services.websocket_enhanced import (
    enhanced_progress_callback,
    enhanced_websocket_manager,
)
from pdf_splitter.api.services.websocket_service import websocket_progress_callback
from pdf_splitter.api.utils.exceptions import (
    FileNotFoundError,
    ProcessingError,
    SessionNotFoundError,
)
from pdf_splitter.detection import create_production_detector
from pdf_splitter.detection.base_detector import BoundaryResult
from pdf_splitter.preprocessing.models import ProcessedPage
from pdf_splitter.preprocessing.pdf_handler import PDFHandler
from pdf_splitter.splitting.models import SessionStatus, SplitProposal
from pdf_splitter.splitting.pdf_splitter import PDFSplitter
from pdf_splitter.splitting.session_manager import SessionManager


class ProcessingStage(str, Enum):
    """Processing pipeline stages."""

    INITIALIZING = "initializing"
    LOADING_PDF = "loading_pdf"
    EXTRACTING_TEXT = "extracting_text"
    DETECTING_BOUNDARIES = "detecting_boundaries"
    GENERATING_PROPOSAL = "generating_proposal"
    COMPLETE = "complete"
    ERROR = "error"

    def to_websocket_stage(self) -> WSProcessingStage:
        """Convert to WebSocket processing stage."""
        mapping = {
            self.INITIALIZING: WSProcessingStage.VALIDATION,
            self.LOADING_PDF: WSProcessingStage.VALIDATION,
            self.EXTRACTING_TEXT: WSProcessingStage.TEXT_EXTRACTION,
            self.DETECTING_BOUNDARIES: WSProcessingStage.BOUNDARY_DETECTION,
            self.GENERATING_PROPOSAL: WSProcessingStage.PROPOSAL_GENERATION,
            self.COMPLETE: WSProcessingStage.COMPLETE,
            self.ERROR: WSProcessingStage.COMPLETE,
        }
        return mapping.get(self, WSProcessingStage.COMPLETE)


class ProcessingService:
    """Service for orchestrating PDF processing."""

    def __init__(
        self, file_service: FileService = None, session_manager: SessionManager = None
    ):
        self.file_service = file_service or FileService()
        self.session_manager = session_manager or SessionManager(
            str(config.session_db_path)
        )
        self.active_processes: Dict[str, asyncio.Task] = {}

    async def start_processing(
        self, file_id: str, progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Start PDF processing for an uploaded file.

        Args:
            file_id: ID of the uploaded file
            progress_callback: Optional callback for progress updates

        Returns:
            Session ID for tracking the processing

        Raises:
            FileNotFoundError: If file not found
            ProcessingError: If processing fails
        """
        # Validate file exists
        try:
            file_path = await self.file_service.get_file_path(file_id)
            metadata = await self.file_service.get_file_metadata(file_id)
        except FileNotFoundError:
            raise FileNotFoundError(file_id)

        # Create session
        session = self.session_manager.create_session(
            pdf_path=str(file_path),
            metadata={
                "file_id": file_id,
                "original_filename": metadata["original_filename"],
                "total_pages": metadata["total_pages"],
                "started_at": datetime.utcnow().isoformat(),
            },
        )

        # Start background processing with WebSocket callback
        task = asyncio.create_task(
            self._process_pdf(
                session.session_id,
                file_path,
                progress_callback or websocket_progress_callback,
            )
        )

        # Track active process
        self.active_processes[session.session_id] = task

        # Clean up when done
        task.add_done_callback(
            lambda t: self.active_processes.pop(session.session_id, None)
        )

        return session.session_id

    async def _process_pdf(
        self,
        session_id: str,
        file_path: Path,
        progress_callback: Optional[Callable] = None,
    ):
        """
        Process PDF file through the complete pipeline.

        Args:
            session_id: Session ID for tracking
            file_path: Path to PDF file
            progress_callback: Optional progress callback
        """
        start_time = time.time()

        try:
            # Update progress helper
            async def update_progress(
                stage: ProcessingStage,
                progress: float,
                message: Optional[str] = None,
                details: Optional[Dict] = None,
            ):
                # Update session
                self.session_manager.update_session_status(
                    session_id,
                    SessionStatus.PROCESSING,
                    {
                        "stage": stage.value,
                        "progress": progress,
                        "message": message,
                        "details": details or {},
                    },
                )

                # Send via enhanced WebSocket manager
                await enhanced_websocket_manager.send_progress_update(
                    session_id=session_id,
                    stage=stage.to_websocket_stage(),
                    progress=progress,
                    message=message,
                    details=details,
                )

                # Call callback if provided (for backward compatibility)
                if progress_callback:
                    await progress_callback(
                        session_id=session_id,
                        stage=stage.value,
                        progress=progress,
                        message=message,
                        details=details,
                    )

            # Initialize
            await update_progress(
                ProcessingStage.INITIALIZING, 0, "Starting PDF processing"
            )

            # Step 1: Load PDF
            await update_progress(
                ProcessingStage.LOADING_PDF, 10, "Loading PDF document"
            )

            pdf_handler = PDFHandler()
            pdf_doc = pdf_handler.load_pdf(str(file_path))
            total_pages = len(pdf_doc)

            await update_progress(
                ProcessingStage.LOADING_PDF,
                20,
                f"PDF loaded successfully ({total_pages} pages)",
                {"total_pages": total_pages},
            )

            # Step 2: Extract text and process pages
            await update_progress(
                ProcessingStage.EXTRACTING_TEXT, 30, "Extracting text from pages"
            )

            processed_pages: List[ProcessedPage] = []
            pages_per_update = max(1, total_pages // 10)  # Update every 10%

            for i, page_num in enumerate(range(total_pages)):
                # Process page
                page_data = await pdf_handler.process_page(pdf_doc, page_num)
                processed_pages.append(page_data)

                # Update progress periodically
                if (i + 1) % pages_per_update == 0 or i == total_pages - 1:
                    progress = 30 + (30 * (i + 1) / total_pages)
                    await update_progress(
                        ProcessingStage.EXTRACTING_TEXT,
                        progress,
                        f"Processed {i + 1}/{total_pages} pages",
                    )

            pdf_doc.close()

            # Step 3: Detect boundaries
            await update_progress(
                ProcessingStage.DETECTING_BOUNDARIES,
                60,
                "Detecting document boundaries",
            )

            detector = create_production_detector()
            boundaries = detector.detect_boundaries(processed_pages)

            # Convert boundaries to page numbers
            boundary_pages = [b.page_number for b in boundaries if b.is_boundary]

            await update_progress(
                ProcessingStage.DETECTING_BOUNDARIES,
                80,
                f"Found {len(boundary_pages)} document boundaries",
                {"boundaries_found": len(boundary_pages)},
            )

            # Step 4: Generate split proposal
            await update_progress(
                ProcessingStage.GENERATING_PROPOSAL, 85, "Generating split proposal"
            )

            splitter = PDFSplitter()

            # Convert boundary results to the format expected by splitter
            pages_with_boundaries = []
            for page in processed_pages:
                page_boundary = next(
                    (b for b in boundaries if b.page_number == page.page_number), None
                )
                pages_with_boundaries.append(
                    {
                        "page": page,
                        "is_boundary": page_boundary.is_boundary
                        if page_boundary
                        else False,
                        "confidence": page_boundary.confidence
                        if page_boundary
                        else 0.0,
                    }
                )

            proposal = splitter.generate_proposal(str(file_path), pages_with_boundaries)

            # Save proposal to session
            self.session_manager.save_proposal(session_id, proposal)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Complete
            await update_progress(
                ProcessingStage.COMPLETE,
                100,
                "Processing complete",
                {
                    "processing_time": processing_time,
                    "total_pages": total_pages,
                    "boundaries_found": len(boundary_pages),
                    "segments_created": len(proposal.segments),
                },
            )

            # Update session to confirmed status
            self.session_manager.update_session_status(
                session_id,
                SessionStatus.CONFIRMED,
                {
                    "completed_at": datetime.utcnow().isoformat(),
                    "processing_time": processing_time,
                },
            )

        except Exception as e:
            # Log error
            error_message = str(e)

            # Update progress with error
            if progress_callback:
                await progress_callback(
                    session_id=session_id,
                    stage=ProcessingStage.ERROR.value,
                    progress=0,
                    message=f"Processing failed: {error_message}",
                    details={"error": error_message},
                )

            # Update session status
            self.session_manager.update_session_status(
                session_id,
                SessionStatus.CANCELLED,
                {"error": error_message, "failed_at": datetime.utcnow().isoformat()},
            )

            raise ProcessingError(f"PDF processing failed: {error_message}")

    async def get_processing_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get current processing status for a session.

        Args:
            session_id: Session ID to check

        Returns:
            Status information dictionary

        Raises:
            SessionNotFoundError: If session not found
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        # Get current status
        status_data = {
            "session_id": session_id,
            "status": session.status.value,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }

        # Add processing details if available
        if session.metadata:
            if "stage" in session.metadata:
                status_data["stage"] = session.metadata["stage"]
            if "progress" in session.metadata:
                status_data["progress"] = session.metadata["progress"]
            if "message" in session.metadata:
                status_data["message"] = session.metadata["message"]
            if "error" in session.metadata:
                status_data["error"] = session.metadata["error"]

        # Check if processing is still active
        if session_id in self.active_processes:
            task = self.active_processes[session_id]
            status_data["is_active"] = not task.done()
        else:
            status_data["is_active"] = False

        return status_data

    async def cancel_processing(self, session_id: str) -> bool:
        """
        Cancel active processing for a session.

        Args:
            session_id: Session ID to cancel

        Returns:
            True if cancelled, False if not active
        """
        if session_id in self.active_processes:
            task = self.active_processes[session_id]
            if not task.done():
                task.cancel()

                # Update session status
                self.session_manager.update_session_status(
                    session_id,
                    SessionStatus.CANCELLED,
                    {"cancelled_at": datetime.utcnow().isoformat()},
                )

                return True

        return False
