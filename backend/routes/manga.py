"""
Legacy manga routes (kept for compatibility with older clients).

The synchronous v1 pipeline (4-panel Gemini batches, regex storyboards, global progress) was removed.
These endpoints now delegate to the v2 services:
- /upload-pdf   → v2 extractor (clean text, title, captions)
- /storyboard   → v2 script (brief → chaptered JSON → fit → fact-check → translate), synchronous
- /generate, /generate/pdf → create an asynchronous job and return its id (poll /api/jobs/{id})
- /progress     → progress of the most recent job
- /export       → stitch client-provided panels into one long image
"""

import base64
import io
import traceback
from typing import Optional

from fastapi import Depends, APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import get_config
from services.cast import build_cast
from services.engine_select import build_engines
from routes.jobs import MAX_PDF_BYTES
from services.jobs import get_job_manager, MIN_TEXT_CHARS
from services.paper_extractor import extract_paper, extract_text_only
from services.storyboarder_v2 import StoryboarderV2, panel_range_for


# the legacy paid routes (a storyboard or a whole manga per call; the web UI uses /api/jobs) share the switch of /api/generate
from routes.generation import require_generate_api  # noqa: E402

router = APIRouter()


class TextToMangaRequest(BaseModel):
    """文本转漫画请求（legacy shape）"""
    model_config = ConfigDict(extra="ignore")

    text: str
    title: str = ""
    character: str = "kumomo"
    language: str = "en-US"
    num_panels: Optional[int] = None  # ignored: the model chooses within the configured range


@router.get("/progress")
async def get_generation_progress():
    """Progress of the most recent job (legacy shape)."""
    jobs = get_job_manager().list(limit=1)
    if not jobs:
        return {"stage": "idle", "current_panel": 0, "total_panels": 0, "message": "", "progress_percent": 0}
    job = jobs[0]
    progress = job.get("progress") or {}
    return {
        "stage": job.get("status"),
        "current_panel": progress.get("done", 0),
        "total_panels": progress.get("total", 0),
        "message": job.get("job_id"),
        "progress_percent": progress.get("percent", 0),
    }


@router.post("/clear-cache")
async def clear_cache():
    """The v2 pipeline keeps no storyboard cache; kept for compatibility."""
    return {"cleared": 0, "message": "No cache in v2 (every job re-reads the paper)"}


async def _read_pdf(file: UploadFile) -> bytes:
    content = await file.read(MAX_PDF_BYTES + 1)
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF larger than 60 MB")
    return content


@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Parse a PDF and return its cleaned text (v2 extractor)."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    content = await _read_pdf(file)
    try:
        paper = extract_paper(content, filename=file.filename or "")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {e}")
    text = paper.body
    return {
        "filename": file.filename,
        "title": paper.title,
        "total_pages": paper.stats.get("pages", 0),
        "full_text": text,
        "text_preview": text[:500] + ("..." if len(text) > 500 else ""),
        "image_count": 0,
        "text_chunks": 1,
        "captions": paper.captions,
        "stats": paper.stats,
    }


@router.post("/storyboard", dependencies=[Depends(require_generate_api)])
async def generate_storyboard(request: TextToMangaRequest):
    """Write the manga script for a text (synchronous; no images)."""
    if len(request.text.strip()) < MIN_TEXT_CHARS:
        raise HTTPException(status_code=400, detail=f"Text too short ({len(request.text)} chars); need at least {MIN_TEXT_CHARS}")
    engines = None
    try:
        settings = get_config().manga_settings
        paper = extract_text_only(request.text, title=request.title)
        cast, _ = build_cast(request.character)
        engines = build_engines()
        writer = StoryboarderV2(engines.text, engines.fast_text, engines.check, engines.translate)
        storyboard = await writer.build(paper, cast, request.language, panel_range_for("auto", settings.min_panels, settings.max_panels))
        data = storyboard.to_dict()
        data["panel_count"] = len(storyboard.panels)
        data["character_theme"] = storyboard.theme
        data["usage"] = writer.usage
        return data
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if engines is not None:
            await engines.close()


def _job_response(state: dict) -> dict:
    job_id = state["job_id"]
    return {
        "job_id": job_id,
        "status": state["status"],
        "detail": f"Generation is asynchronous: poll GET /api/jobs/{job_id} (images under /output/{job_id}/)",
    }


@router.post("/generate", status_code=202, dependencies=[Depends(require_generate_api)])
async def generate_manga(request: TextToMangaRequest):
    """Start a manga job from text (legacy entry point; returns the job id instead of blocking)."""
    try:
        state = get_job_manager().create(pdf_bytes=None, text=request.text, title=request.title, theme=request.character, language=request.language)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _job_response(state)


@router.post("/generate/pdf", status_code=202, dependencies=[Depends(require_generate_api)])
async def generate_manga_from_pdf(
    file: UploadFile = File(...),
    character: str = Form(default="kumomo"),
    language: str = Form(default="en-US"),
):
    """Start a manga job from a PDF (legacy entry point; returns the job id instead of blocking)."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    content = await _read_pdf(file)
    try:
        state = get_job_manager().create(pdf_bytes=content, text=None, filename=file.filename or "", theme=character, language=language)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _job_response(state)


@router.post("/export")
async def export_manga(
    panels_base64: list[str],
    title: str = "manga",
    layout: str = "vertical",
):
    """Stitch client-provided panel images into one long image."""
    try:
        from PIL import Image

        images = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in panels_base64]
        if not images:
            raise HTTPException(status_code=400, detail="No images provided")

        if layout == "vertical":
            width = max(img.width for img in images)
            resized = [img.resize((width, int(img.height * width / img.width))) for img in images]
            canvas = Image.new("RGB", (width, sum(img.height for img in resized)), "white")
            y = 0
            for img in resized:
                canvas.paste(img, (0, y))
                y += img.height
        else:
            total_width = sum(img.width for img in images)
            max_height = max(img.height for img in images)
            canvas = Image.new("RGB", (total_width, max_height), "white")
            x = 0
            for img in images:
                canvas.paste(img, (x, (max_height - img.height) // 2))
                x += img.width

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        safe = "".join(c for c in title if c.isalnum() or c in "_- ")[:60] or "manga"
        return Response(content=buffer.getvalue(), media_type="image/png",
                        headers={"Content-Disposition": f"attachment; filename={safe}.png"})
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
