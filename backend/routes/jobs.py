"""
Job API — asynchronous paper → manga generation with pollable, persisted state.
Endpoints: see the API table in README.md.
"""

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.jobs import get_job_manager, MIN_TEXT_CHARS, VALID_LANGUAGES, VALID_LAYOUTS, VALID_RANGES, VALID_THEMES

router = APIRouter()

MAX_PDF_BYTES = 60 * 1024 * 1024


def _flag(value: Optional[str]) -> Optional[bool]:
    if value is None or value == "":
        return None
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@router.post("", status_code=202)
async def create_job(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    title: str = Form(""),
    theme: str = Form("kumomo"),
    language: str = Form("en-US"),
    layout: Optional[str] = Form(None),
    qa: Optional[str] = Form(None),
    panel_range: str = Form("auto"),
    fast: Optional[str] = Form(None),
):
    if theme not in VALID_THEMES:
        raise HTTPException(status_code=422, detail=f"theme must be one of {sorted(VALID_THEMES)}")
    if language not in VALID_LANGUAGES:
        raise HTTPException(status_code=422, detail=f"language must be one of {sorted(VALID_LANGUAGES)}")
    if layout is not None and layout not in VALID_LAYOUTS:
        raise HTTPException(status_code=422, detail=f"layout must be one of {sorted(VALID_LAYOUTS)}")
    if panel_range not in VALID_RANGES:
        raise HTTPException(status_code=422, detail=f"panel_range must be one of {sorted(VALID_RANGES)}")

    pdf_bytes = None
    filename = ""
    if file is not None and file.filename:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        pdf_bytes = await file.read(MAX_PDF_BYTES + 1)
        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise HTTPException(status_code=413, detail="PDF larger than 60 MB")
        if not pdf_bytes.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="File is not a valid PDF")
        filename = file.filename
    if pdf_bytes is None and not (text and len(text.strip()) >= MIN_TEXT_CHARS):
        raise HTTPException(status_code=400, detail=f"Upload a PDF or send at least {MIN_TEXT_CHARS} characters of text")

    try:
        await get_job_manager().preflight()   # no key, a model not activated, no balance: said now, before anything is paid
        state = get_job_manager().create(pdf_bytes=pdf_bytes, text=text, filename=filename, title=title.strip(), theme=theme, language=language,
                                         layout=layout, qa=_flag(qa), panel_range=panel_range, fast=_flag(fast))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        if getattr(e, "account", False):
            raise HTTPException(status_code=400, detail=str(e))
        raise
    return {"job_id": state["job_id"], "status": state["status"]}


@router.get("")
async def list_jobs():
    return {"jobs": get_job_manager().list()}


@router.get("/{job_id}")
async def get_job(job_id: str):
    state = get_job_manager().get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return state


@router.post("/{job_id}/panels/{n}/regenerate", status_code=202)
async def regenerate_panel(job_id: str, n: int):
    try:
        result = await get_job_manager().regenerate_panel(job_id, n)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return result


@router.post("/{job_id}/relayout")
async def relayout_job(job_id: str):
    """Re-letter a finished v3 job's panels with the current lettering and board code (free: no image calls)."""
    try:
        result = await get_job_manager().relayout(job_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return result


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    state = await get_job_manager().cancel(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": state["status"]}
