"""Page routes for rendering HTML templates."""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from pdf_splitter.core.config import settings

router = APIRouter(tags=["pages"])


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Render the upload page."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "max_file_size_mb": settings.max_file_size_mb,
        },
    )


@router.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    """Render the history page."""
    templates = request.app.state.templates
    return templates.TemplateResponse("history.html", {"request": request})


@router.get("/progress/{session_id}", response_class=HTMLResponse)
async def progress_page(request: Request, session_id: str):
    """Render the progress tracking page for a session."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "progress.html",
        {
            "request": request,
            "session_id": session_id,
        },
    )


@router.get("/review/{session_id}", response_class=HTMLResponse)
async def review_page(request: Request, session_id: str):
    """Render the review page for a specific session."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "review.html",
        {
            "request": request,
            "session_id": session_id,
        },
    )


@router.get("/results/{split_id}", response_class=HTMLResponse)
async def results_page(request: Request, split_id: str):
    """Render the results page for a completed split."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "split_id": split_id,
        },
    )
