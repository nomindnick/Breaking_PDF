"""
File Upload Endpoints

Handles PDF file uploads with validation and chunking support.
"""
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from pdf_splitter.api.config import config
from pdf_splitter.api.models.responses import UploadResponse, UploadStatusResponse
from pdf_splitter.api.services.file_service import FileService
from pdf_splitter.api.utils.exceptions import (
    FileNotFoundError,
    FileSizeError,
    FileTypeError,
    FileUploadError,
)

router = APIRouter(prefix="/api", tags=["upload"])


def get_file_service() -> FileService:
    """Dependency to get file service instance."""
    return FileService()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    chunk_number: Optional[int] = Form(
        0, description="Current chunk number for chunked uploads"
    ),
    total_chunks: Optional[int] = Form(1, description="Total number of chunks"),
    file_service: FileService = Depends(get_file_service),
) -> UploadResponse:
    """
    Upload a PDF file for processing.

    Supports both single file upload and chunked upload for large files.

    Args:
        file: The PDF file to upload
        chunk_number: Current chunk number (0-based) for chunked uploads
        total_chunks: Total number of chunks for chunked uploads

    Returns:
        UploadResponse with file information and upload ID

    Raises:
        400: Invalid file format or upload error
        413: File size exceeds limit
        415: Unsupported file type
    """
    start_time = time.time()

    try:
        # Perform upload
        result = await file_service.save_upload(
            file=file, chunk_number=chunk_number, total_chunks=total_chunks
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Prepare response based on upload type
        if result.get("status") == "uploading":
            # Chunked upload in progress
            return UploadResponse(
                success=True,
                message=f"Chunk {chunk_number + 1} of {total_chunks} uploaded successfully",
                upload_id=result["file_id"],
                file_name=file.filename,
                file_size=0,  # Will be set when complete
                total_pages=0,  # Will be set when complete
                status="uploading",
                processing_time=processing_time,
            )
        else:
            # Upload complete
            return UploadResponse(
                success=True,
                message="File uploaded successfully",
                upload_id=result["file_id"],
                file_name=result["filename"],
                file_size=result["size"],
                total_pages=result["total_pages"],
                status="uploaded",
                processing_time=processing_time,
            )

    except FileTypeError as e:
        raise e
    except FileSizeError as e:
        raise e
    except FileUploadError as e:
        raise e
    except Exception as e:
        raise FileUploadError(f"Unexpected error during upload: {str(e)}")


@router.get("/upload/{upload_id}/status", response_model=UploadStatusResponse)
async def get_upload_status(
    upload_id: str, file_service: FileService = Depends(get_file_service)
) -> UploadStatusResponse:
    """
    Get the status of an uploaded file.

    Args:
        upload_id: The unique identifier of the uploaded file

    Returns:
        UploadStatusResponse with file metadata and status

    Raises:
        404: Upload not found
    """
    try:
        # Get file metadata
        metadata = await file_service.get_file_metadata(upload_id)

        # Check if file still exists
        try:
            file_path = await file_service.get_file_path(upload_id)
            status = "ready"
        except FileNotFoundError:
            status = "deleted"

        return UploadStatusResponse(
            success=True,
            message=f"Upload status: {status}",
            upload_id=upload_id,
            status=status,
            file_name=metadata["original_filename"],
            file_size=metadata["file_size"],
            total_pages=metadata.get("total_pages"),
            created_at=metadata["upload_time"],
            expires_at=metadata["expires_at"],
        )

    except FileNotFoundError as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving upload status: {str(e)}"
        )


@router.delete("/upload/{upload_id}")
async def delete_upload(
    upload_id: str, file_service: FileService = Depends(get_file_service)
) -> JSONResponse:
    """
    Delete an uploaded file.

    Args:
        upload_id: The unique identifier of the uploaded file

    Returns:
        Success confirmation

    Raises:
        404: Upload not found
    """
    try:
        deleted = await file_service.delete_file(upload_id)

        if deleted:
            return JSONResponse(
                content={
                    "success": True,
                    "message": f"Upload {upload_id} deleted successfully",
                }
            )
        else:
            raise FileNotFoundError(upload_id)

    except FileNotFoundError as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting upload: {str(e)}")
