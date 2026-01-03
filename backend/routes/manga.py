"""
Manga API Routes
漫画生成完整流程接口
"""

import base64
import traceback
from typing import Optional, Dict, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.pdf_parser import get_parser, ParsedDocument
from services.storyboarder import get_storyboarder, Storyboard, clear_storyboard_cache
from services.manga_generator import get_manga_generator, GeneratedManga
from services.progress import get_progress, set_stage, reset_progress


router = APIRouter()


# ==================== 进度查询 ====================

@router.get("/progress")
async def get_generation_progress():
    """Get current generation progress"""
    return get_progress().to_dict()


# ==================== 缓存管理 ====================

@router.post("/clear-cache")
async def clear_cache():
    """Clear the storyboard cache to force regeneration"""
    count = clear_storyboard_cache()
    return {"cleared": count, "message": f"Cleared {count} cached storyboards"}


# ==================== 请求/响应模型 ====================

class TextToMangaRequest(BaseModel):
    """文本转漫画请求"""
    text: str
    title: str = ""
    character: str = "chiikawa"
    language: str = "zh-CN"
    num_panels: Optional[int] = None  # 兼容旧前端，实际由 Gemini 自动决定数量

    class Config:
        extra = "ignore"  # 忽略未知字段


class StoryboardResponse(BaseModel):
    """分镜脚本响应"""
    title: str
    summary: str
    character_theme: str
    panel_count: int
    panels: list[dict]


class MangaResponse(BaseModel):
    """漫画生成响应"""
    title: str
    character_theme: str
    panel_count: int
    panels: list[dict]
    combined_image_base64: Optional[str] = None


class PipelineStatus(BaseModel):
    """流水线状态"""
    stage: str
    progress: float
    message: str


# ==================== PDF 上传与解析 ====================

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """上传并解析 PDF 文件"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        parser = get_parser()
        content = await file.read()

        import io
        pdf_stream = io.BytesIO(content)
        document = await parser.parse(pdf_stream)

        return {
            "filename": file.filename,
            "total_pages": document.total_pages,
            "full_text": document.full_text,  # 完整文本用于生成
            "text_preview": document.full_text[:500] + "..." if len(document.full_text) > 500 else document.full_text,
            "image_count": len(document.all_images),
            "text_chunks": len(document.get_text_chunks())
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")


# ==================== 分镜脚本生成 ====================

@router.post("/storyboard", response_model=StoryboardResponse)
async def generate_storyboard(request: TextToMangaRequest):
    """
    从文本生成分镜脚本

    AI 会根据论文内容自动决定需要多少个片段
    """
    import hashlib
    text_hash = hashlib.sha256(request.text.encode()).hexdigest()[:16]
    print(f"[API] ========== /storyboard REQUEST ==========")
    print(f"[API] character={request.character}, language={request.language}")
    print(f"[API] title={request.title}")
    print(f"[API] text_len={len(request.text)}, text_hash={text_hash}")
    print(f"[API] Text preview: {request.text[:300]}...")
    try:
        storyboarder = get_storyboarder(request.character)

        storyboard = await storyboarder.generate_storyboard(
            text=request.text,
            title=request.title,
            language=request.language
        )

        response = StoryboardResponse(
            title=storyboard.title,
            summary=storyboard.summary,
            character_theme=storyboard.character_theme,
            panel_count=len(storyboard.panels),
            panels=[
                {
                    "panel_number": p.panel_number,
                    "panel_type": p.panel_type.value,
                    "characters": p.characters,
                    "character_emotions": p.character_emotions,
                    "visual_description": p.visual_description,
                    "dialogue": p.dialogue,
                    "visual_metaphor": p.visual_metaphor,
                    "props": p.props,
                    "background": p.background,
                    "layout_hint": p.layout_hint
                }
                for p in storyboard.panels
            ]
        )
        print(f"[API] /storyboard returning: title={response.title}, panel_count={response.panel_count}")
        return response

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 完整流水线 ====================

@router.post("/generate", response_model=MangaResponse)
async def generate_manga(request: TextToMangaRequest):
    """
    完整的论文转漫画流水线

    1. 使用 AI 理解论文，生成分镜脚本
    2. 使用图像生成模型为每个分镜生成图像
    3. 合并为完整漫画
    """
    print(f"[API] /generate endpoint hit!")
    try:
        print(f"[API] /generate called: character={request.character}, text_len={len(request.text)}, title={request.title}")

        # Step 1: 生成分镜脚本（AI 自动决定片段数量）
        storyboarder = get_storyboarder(request.character)
        storyboard = await storyboarder.generate_storyboard(
            text=request.text,
            title=request.title,
            language=request.language
        )

        # Step 2: 生成漫画图像
        print(f"[API] /generate step 2: generating manga images...")
        generator = get_manga_generator()
        manga = await generator.generate_from_storyboard(storyboard)
        print(f"[API] /generate step 2 done: {len(manga.panels)} panels generated")

        # Step 3: 合并图像
        print(f"[API] /generate step 3: combining images...")
        try:
            combined_image = manga.get_combined_image(layout="vertical")
            print(f"[API] /generate step 3 done: combined image size = {len(combined_image)} bytes")
        except Exception as e:
            print(f"[API] /generate step 3 FAILED: {e}")
            traceback.print_exc()
            raise

        print(f"[API] /generate step 4: encoding to base64...")
        combined_base64 = base64.b64encode(combined_image).decode()
        print(f"[API] /generate step 4 done: base64 length = {len(combined_base64)}")

        print(f"[API] /generate returning manga: title={manga.title}, panels={len(manga.panels)}")
        return MangaResponse(
            title=manga.title,
            character_theme=manga.character_theme,
            panel_count=len(manga.panels),
            panels=[
                {
                    "panel_number": p.panel_number,
                    "characters": p.characters,
                    "dialogue": p.dialogue,
                    "image_base64": p.image_base64
                }
                for p in manga.panels
            ],
            combined_image_base64=combined_base64
        )

    except Exception as e:
        traceback.print_exc()
        # 检查是否有部分结果可以返回
        try:
            from pathlib import Path
            progress_dir = Path(__file__).parent.parent.parent / "output" / "progress"
            if progress_dir.exists():
                # 找到最新的部分结果
                partial_files = sorted(progress_dir.glob("*_partial_*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
                final_files = sorted(progress_dir.glob("*_final.png"), key=lambda x: x.stat().st_mtime, reverse=True)

                latest = None
                if final_files:
                    latest = final_files[0]
                elif partial_files:
                    latest = partial_files[0]

                if latest:
                    print(f"[API] Returning partial result: {latest}")
                    with open(latest, "rb") as f:
                        partial_data = f.read()
                    partial_base64 = base64.b64encode(partial_data).decode()
                    return MangaResponse(
                        title=f"[Partial] {request.title or 'Manga'}",
                        character_theme=request.character,
                        panel_count=0,
                        panels=[],
                        combined_image_base64=partial_base64
                    )
        except Exception as recovery_error:
            print(f"[API] Failed to recover partial result: {recovery_error}")

        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/pdf")
async def generate_manga_from_pdf(
    file: UploadFile = File(...),
    character: str = Form(default="chiikawa"),
    language: str = Form(default="zh-CN")
):
    """
    从 PDF 文件直接生成漫画

    流程:
    1. 解析 PDF 提取文本
    2. Gemini 3 Pro 理解论文内容，自动决定分镜数量
    3. Nano Banana Pro 生成每个分镜的图像
    4. 合并输出
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Step 1: 解析 PDF
        parser = get_parser()
        content = await file.read()

        import io
        pdf_stream = io.BytesIO(content)
        document = await parser.parse(pdf_stream)

        # 获取文本（如果太长则分块处理第一块）
        text_chunks = document.get_text_chunks(max_tokens=10000)
        text = text_chunks[0] if text_chunks else document.full_text[:15000]

        print(f"[API] PDF parsed: {len(text)} chars")

        # Step 2: 生成分镜（AI 自动决定片段数量）
        storyboarder = get_storyboarder(character)
        storyboard = await storyboarder.generate_storyboard(
            text=text,
            title=Path(file.filename).stem,
            language=language
        )

        print(f"[API] Storyboard generated: {len(storyboard.panels)} panels")

        # Step 3: 生成漫画
        generator = get_manga_generator()
        manga = await generator.generate_from_storyboard(storyboard)

        print(f"[API] Manga generated: {len(manga.panels)} panels")

        # 保存并返回
        output_path = manga.save(generator.output_dir)
        print(f"[API] Saved to: {output_path}")

        # 返回合并图像
        combined_image = manga.get_combined_image()

        # 处理文件名编码（HTTP header 必须是 ASCII）
        import urllib.parse
        safe_filename = urllib.parse.quote(output_path.name)

        return Response(
            content=combined_image,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}"
            }
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 导出 ====================

@router.post("/export")
async def export_manga(
    panels_base64: list[str],
    title: str = "manga",
    layout: str = "vertical"
):
    """导出漫画为长图"""
    try:
        from PIL import Image
        import io

        images = []
        for img_b64 in panels_base64:
            img_data = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_data))
            images.append(img)

        if not images:
            raise HTTPException(status_code=400, detail="No images provided")

        # 合并图像
        if layout == "vertical":
            max_width = max(img.width for img in images)
            total_height = sum(img.height for img in images)
            canvas = Image.new("RGB", (max_width, total_height), "white")

            y = 0
            for img in images:
                x = (max_width - img.width) // 2
                canvas.paste(img, (x, y))
                y += img.height
        else:
            # 横向布局
            total_width = sum(img.width for img in images)
            max_height = max(img.height for img in images)
            canvas = Image.new("RGB", (total_width, max_height), "white")

            x = 0
            for img in images:
                y = (max_height - img.height) // 2
                canvas.paste(img, (x, y))
                x += img.width

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG", quality=95)

        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename={title}.png"
            }
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
