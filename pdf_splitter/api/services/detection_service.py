"""Service for handling document boundary detection.

This module provides business logic for running boundary detection
and managing detection results.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from pdf_splitter.api.exceptions import DetectionError, ValidationError
from pdf_splitter.core.config import PDFConfig
from pdf_splitter.core.logging import get_logger
from pdf_splitter.detection.base_detector import (
    BaseDetector,
    BoundaryResult,
    DetectionContext,
    ProcessedPage,
)
from pdf_splitter.detection.embeddings_detector.embeddings_detector import (
    EmbeddingsDetector,
)
from pdf_splitter.detection.heuristic_detector.heuristic_detector import (
    HeuristicDetector,
)
from pdf_splitter.detection.llm_detector import LLMDetector
from pdf_splitter.detection.visual_detector.visual_detector import VisualDetector
from pdf_splitter.preprocessing.pdf_handler import PDFHandler
from pdf_splitter.splitting.models import SplitProposal
from pdf_splitter.splitting.pdf_splitter import PDFSplitter

logger = get_logger(__name__)


class DetectionService:
    """Service for managing boundary detection operations."""

    def __init__(self, config: PDFConfig):
        """Initialize detection service.

        Args:
            config: PDF processing configuration
        """
        self.config = config
        self._detectors: Dict[str, BaseDetector] = {}
        self._detection_tasks: Dict[str, dict] = {}  # Track running detections

    def get_detector(self, detector_type: str) -> BaseDetector:
        """Get or create detector instance.

        Args:
            detector_type: Type of detector

        Returns:
            Detector instance

        Raises:
            ValidationError: If detector type is invalid
        """
        if detector_type not in self._detectors:
            try:
                if detector_type == "embeddings":
                    self._detectors[detector_type] = EmbeddingsDetector(self.config)
                elif detector_type == "heuristic":
                    self._detectors[detector_type] = HeuristicDetector(self.config)
                elif detector_type == "visual":
                    self._detectors[detector_type] = VisualDetector(self.config)
                elif detector_type == "llm":
                    self._detectors[detector_type] = LLMDetector(self.config)
                else:
                    raise ValidationError(
                        f"Invalid detector type: {detector_type}",
                        field="detector_type",
                    )
            except Exception as e:
                logger.error(f"Failed to create {detector_type} detector: {str(e)}")
                raise DetectionError(
                    f"Failed to create {detector_type} detector: {str(e)}"
                )

        return self._detectors[detector_type]

    async def start_detection(
        self,
        session_id: str,
        pdf_path: Path,
        detector_type: str = "embeddings",
        confidence_threshold: float = 0.5,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Start boundary detection process.

        Args:
            session_id: Session identifier
            pdf_path: Path to PDF file
            detector_type: Type of detector to use
            confidence_threshold: Minimum confidence threshold
            progress_callback: Optional callback for progress updates

        Returns:
            Detection ID

        Raises:
            DetectionError: If detection fails to start
        """
        detection_id = str(uuid4())

        try:
            # Initialize detection task
            self._detection_tasks[detection_id] = {
                "session_id": session_id,
                "status": "initializing",
                "progress": 0.0,
                "start_time": time.time(),
                "detector_type": detector_type,
                "total_pages": 0,
                "current_page": 0,
            }

            # Start detection in background
            asyncio.create_task(
                self._run_detection(
                    detection_id,
                    session_id,
                    pdf_path,
                    detector_type,
                    confidence_threshold,
                    progress_callback,
                )
            )

            logger.info(
                f"Started {detector_type} detection {detection_id} "
                f"for session {session_id}"
            )

            return detection_id

        except Exception as e:
            logger.error(f"Failed to start detection: {str(e)}")
            if detection_id in self._detection_tasks:
                del self._detection_tasks[detection_id]
            raise DetectionError(f"Failed to start detection: {str(e)}", session_id)

    async def _run_detection(
        self,
        detection_id: str,
        session_id: str,
        pdf_path: Path,
        detector_type: str,
        confidence_threshold: float,
        progress_callback: Optional[callable] = None,
    ):
        """Run detection process asynchronously.

        Args:
            detection_id: Detection identifier
            session_id: Session identifier
            pdf_path: Path to PDF file
            detector_type: Type of detector
            confidence_threshold: Confidence threshold
            progress_callback: Progress callback
        """
        task_info = self._detection_tasks[detection_id]

        try:
            # Update status
            task_info["status"] = "loading_pdf"
            if progress_callback:
                await progress_callback("loading_pdf", 0.0, "Loading PDF file...")

            # Load PDF and extract pages
            pdf_handler = PDFHandler(config=self.config)
            await asyncio.to_thread(pdf_handler.load_pdf, pdf_path)

            total_pages = pdf_handler.page_count
            task_info["total_pages"] = total_pages

            # Extract text from all pages
            task_info["status"] = "extracting_text"
            processed_pages: List[ProcessedPage] = []

            for page_num in range(total_pages):
                # Extract text
                text = await asyncio.to_thread(
                    pdf_handler.extract_text_from_page, page_num
                )

                # Create processed page
                processed_page = ProcessedPage(
                    page_number=page_num,
                    text=text,
                    page_type=pdf_handler.page_types.get(page_num, "unknown"),
                )
                processed_pages.append(processed_page)

                # Update progress
                progress = (page_num + 1) / total_pages * 0.3  # 30% for text extraction
                task_info["progress"] = progress
                task_info["current_page"] = page_num + 1

                if progress_callback:
                    await progress_callback(
                        "extracting_text",
                        progress,
                        f"Extracting text from page {page_num + 1}/{total_pages}",
                    )

            # Run detection
            task_info["status"] = "detecting_boundaries"
            detector = self.get_detector(detector_type)

            # Create detection context
            context = DetectionContext(
                config=self.config,
                total_pages=total_pages,
            )

            # Run detection with progress tracking
            def detection_progress(current_page: int):
                progress = 0.3 + (current_page / total_pages * 0.6)  # 60% for detection
                task_info["progress"] = progress
                task_info["current_page"] = current_page
                if progress_callback:
                    asyncio.create_task(
                        progress_callback(
                            "detecting_boundaries",
                            progress,
                            f"Analyzing page {current_page}/{total_pages}",
                        )
                    )

            # Run detection in thread
            boundaries = await asyncio.to_thread(
                detector.detect_boundaries, processed_pages, context
            )

            # Update progress during detection (simple progress tracking)
            for i in range(10):
                await asyncio.sleep(0.1)  # Simulate progress
                progress = 0.3 + (i / 10 * 0.6)
                task_info["progress"] = progress
                if progress_callback:
                    await progress_callback(
                        "detecting_boundaries",
                        progress,
                        "Analyzing boundaries...",
                    )

            # Filter by confidence
            filtered_boundaries = [
                b for b in boundaries if b.confidence >= confidence_threshold
            ]

            # Generate proposal
            task_info["status"] = "generating_proposal"
            task_info["progress"] = 0.9

            if progress_callback:
                await progress_callback(
                    "generating_proposal",
                    0.9,
                    "Generating split proposal...",
                )

            # Create proposal using PDFSplitter
            pdf_splitter = PDFSplitter(config=self.config)
            proposal = await asyncio.to_thread(
                pdf_splitter.generate_proposal,
                filtered_boundaries,
                processed_pages,
                pdf_path,
            )

            # Store results
            task_info["status"] = "completed"
            task_info["progress"] = 1.0
            task_info["results"] = {
                "boundaries_found": len(filtered_boundaries),
                "segments_proposed": len(proposal.segments),
                "boundaries": filtered_boundaries,
                "proposal": proposal,
            }
            task_info["end_time"] = time.time()
            task_info["duration"] = task_info["end_time"] - task_info["start_time"]

            if progress_callback:
                await progress_callback(
                    "completed",
                    1.0,
                    f"Detection completed. Found {len(filtered_boundaries)} boundaries.",
                )

            logger.info(
                f"Detection {detection_id} completed successfully. "
                f"Found {len(filtered_boundaries)} boundaries in {task_info['duration']:.2f}s"
            )

        except Exception as e:
            logger.error(f"Detection {detection_id} failed: {str(e)}")
            task_info["status"] = "failed"
            task_info["error"] = str(e)
            task_info["end_time"] = time.time()

            if progress_callback:
                await progress_callback(
                    "failed", task_info["progress"], f"Detection failed: {str(e)}"
                )

    def get_detection_status(self, detection_id: str) -> dict:
        """Get status of a detection operation.

        Args:
            detection_id: Detection identifier

        Returns:
            Detection status information

        Raises:
            ValidationError: If detection not found
        """
        task_info = self._detection_tasks.get(detection_id)
        if not task_info:
            raise ValidationError(
                f"Detection {detection_id} not found", field="detection_id"
            )

        # Calculate estimated time remaining
        estimated_remaining = None
        if (
            task_info["status"] in ["extracting_text", "detecting_boundaries"]
            and task_info["progress"] > 0
        ):
            elapsed = time.time() - task_info["start_time"]
            total_estimated = elapsed / task_info["progress"]
            estimated_remaining = max(0, total_estimated - elapsed)

        return {
            "detection_id": detection_id,
            "session_id": task_info["session_id"],
            "status": task_info["status"],
            "progress": task_info["progress"],
            "detector_type": task_info["detector_type"],
            "current_page": task_info.get("current_page", 0),
            "total_pages": task_info.get("total_pages", 0),
            "elapsed_time": time.time() - task_info["start_time"],
            "estimated_remaining": estimated_remaining,
            "error": task_info.get("error"),
        }

    def get_detection_results(
        self, detection_id: str
    ) -> Tuple[List[BoundaryResult], SplitProposal]:
        """Get results of completed detection.

        Args:
            detection_id: Detection identifier

        Returns:
            Tuple of (boundaries, proposal)

        Raises:
            ValidationError: If detection not found or not completed
        """
        task_info = self._detection_tasks.get(detection_id)
        if not task_info:
            raise ValidationError(
                f"Detection {detection_id} not found", field="detection_id"
            )

        if task_info["status"] != "completed":
            raise ValidationError(
                f"Detection {detection_id} is not completed (status: {task_info['status']})",
                field="detection_id",
            )

        results = task_info.get("results", {})
        boundaries = results.get("boundaries", [])
        proposal = results.get("proposal")

        if not proposal:
            raise DetectionError(
                f"No proposal found for detection {detection_id}",
                task_info["session_id"],
            )

        return boundaries, proposal

    def cleanup_old_detections(self, max_age_hours: int = 24):
        """Clean up old detection tasks.

        Args:
            max_age_hours: Maximum age in hours
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for detection_id in list(self._detection_tasks.keys()):
            task_info = self._detection_tasks[detection_id]
            if (
                task_info.get("end_time")
                and current_time - task_info["end_time"] > max_age_seconds
            ):
                logger.info(f"Cleaning up old detection: {detection_id}")
                del self._detection_tasks[detection_id]
