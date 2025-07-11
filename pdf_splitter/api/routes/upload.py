"""Upload endpoints for PDF files.

This module provides API endpoints for uploading and validating PDF files.
"""

import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from pdf_splitter.api.dependencies import (
    get_pdf_config,
    get_upload_directory,
    get_upload_manager,
)
from pdf_splitter.api.exceptions import FileSizeError, FileTypeError, UploadError
from pdf_splitter.api.models.requests import UploadRequest
from pdf_splitter.api.models.responses import UploadResponse, UploadStatusResponse
from pdf_splitter.api.services.upload_service import UploadService
from pdf_splitter.core.config import PDFConfig
from pdf_splitter.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post("/file", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    validate_only: bool = False,
    config: PDFConfig = Depends(get_pdf_config),
    upload_manager=Depends(get_upload_manager),
):
    """Upload a PDF file for processing.

    Args:
        file: PDF file to upload
        validate_only: If true, only validate without storing
        config: PDF configuration
        upload_manager: Upload manager instance

    Returns:
        Upload response with file details

    Raises:
        HTTPException: On upload failure
    """
    start_time = time.time()

    try:
        # Initialize upload service
        upload_service = UploadService(config=config, upload_dir=get_upload_directory())

        # Process upload
        upload_id, upload_info = await upload_service.process_upload(
            file, validate_only=validate_only
        )

        # Store in upload manager if not validate-only
        if not validate_only:
            upload_manager.upload_metadata[upload_id] = upload_info

        processing_time = time.time() - start_time

        return UploadResponse(
            upload_id=upload_id,
            file_name=upload_info["file_name"],
            file_size=upload_info["file_size"],
            total_pages=upload_info.get("total_pages", 0),
            status=upload_info["status"],
            validation_errors=None,
            processing_time=processing_time,
            message=f"Successfully uploaded {upload_info['file_name']}",
        )

    except FileTypeError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except FileSizeError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except UploadError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/validate", response_model=UploadResponse)
async def validate_file(
    request: UploadRequest,
    config: PDFConfig = Depends(get_pdf_config),
):
    """Validate PDF file metadata before upload.

    Args:
        request: Upload validation request
        config: PDF configuration

    Returns:
        Validation response

    Raises:
        HTTPException: On validation failure
    """
    try:
        # Check file size
        max_bytes = int(config.max_file_size_mb * 1024 * 1024)
        if request.file_size > max_bytes:
            raise FileSizeError(file_size=request.file_size, max_size=max_bytes)

        # Check content type
        if request.content_type not in UploadService.ALLOWED_CONTENT_TYPES:
            raise FileTypeError(
                file_type=request.content_type,
                allowed_types=list(UploadService.ALLOWED_CONTENT_TYPES),
            )

        # Check filename
        if not request.file_name.lower().endswith(".pdf"):
            raise FileTypeError(file_type="non-pdf", allowed_types=[".pdf"])

        return UploadResponse(
            upload_id="",  # No ID for validation-only
            file_name=request.file_name,
            file_size=request.file_size,
            total_pages=0,  # Unknown until actual upload
            status="validated",
            processing_time=0,
            message="File validation successful",
        )

    except (FileTypeError, FileSizeError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/{upload_id}/status", response_model=UploadStatusResponse)
async def get_upload_status(
    upload_id: str,
    upload_manager=Depends(get_upload_manager),
):
    """Get status of an uploaded file.

    Args:
        upload_id: Upload identifier
        upload_manager: Upload manager instance

    Returns:
        Upload status information

    Raises:
        HTTPException: If upload not found
    """
    try:
        # Get upload metadata
        upload_info = upload_manager.get_upload_metadata(upload_id)
        if not upload_info:
            raise HTTPException(status_code=404, detail=f"Upload {upload_id} not found")

        # Get file path
        upload_path = upload_manager.get_upload_path(upload_id)
        if not upload_path or not upload_path.exists():
            status = "deleted"
        else:
            status = upload_info.get("status", "uploaded")

        # Calculate expiration (24 hours from creation)
        import datetime

        file_stats = (
            upload_path.stat() if upload_path and upload_path.exists() else None
        )
        created_at = datetime.datetime.fromtimestamp(
            file_stats.st_mtime if file_stats else time.time()
        )
        expires_at = created_at + datetime.timedelta(hours=24)

        return UploadStatusResponse(
            upload_id=upload_id,
            status=status,
            file_name=upload_info["filename"],
            file_size=upload_info["size"],
            total_pages=upload_info.get("total_pages"),
            created_at=created_at,
            expires_at=expires_at,
            message=f"Upload {upload_id} is {status}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get upload status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get upload status: {str(e)}"
        )


@router.delete("/{upload_id}")
async def delete_upload(
    upload_id: str,
    upload_manager=Depends(get_upload_manager),
):
    """Delete an uploaded file.

    Args:
        upload_id: Upload identifier
        upload_manager: Upload manager instance

    Returns:
        Success message

    Raises:
        HTTPException: If upload not found or deletion fails
    """
    try:
        # Delete upload
        deleted = upload_manager.delete_upload(upload_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Upload {upload_id} not found")

        return JSONResponse(
            content={
                "success": True,
                "message": f"Upload {upload_id} deleted successfully",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete upload: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete upload: {str(e)}"
        )
