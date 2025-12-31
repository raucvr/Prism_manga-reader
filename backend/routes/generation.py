"""
Generation API Routes
文本和图像生成接口
"""

import base64
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines import (
    get_client,
    get_text_engine,
    get_image_engine,
    Message,
    GenerationConfig,
    ImageGenerationConfig,
    ImageContent
)


router = APIRouter()


# ==================== 请求/响应模型 ====================

class TextGenerationRequest(BaseModel):
    """文本生成请求"""
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False


class TextGenerationResponse(BaseModel):
    """文本生成响应"""
    content: str
    model: str
    usage: dict


class ImageGenerationRequest(BaseModel):
    """图像生成请求"""
    prompt: str
    style: str = "full_color_manga"
    width: int = 1024
    height: int = 1536
    negative_prompt: str = ""
    reference_image_base64: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    """图像生成响应"""
    image_base64: str
    mime_type: str
    model: str
    revised_prompt: Optional[str] = None


# ==================== 文本生成 ====================

@router.post("/text", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """生成文本"""
    try:
        client = await get_client()

        config = GenerationConfig(
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        response = await client.generate_text(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            config=config
        )

        return TextGenerationResponse(
            content=response.content,
            model=response.model,
            usage=response.usage
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text/stream")
async def generate_text_stream(request: TextGenerationRequest):
    """流式生成文本"""
    try:
        engine = get_text_engine()

        messages = []
        if request.system_prompt:
            messages.append(Message.system(request.system_prompt))
        messages.append(Message.user(request.prompt))

        config = GenerationConfig(
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        async def stream_generator():
            async for chunk in engine.generate_text_stream(messages, config):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 图像生成 ====================

@router.post("/image", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    """生成图像（使用 Nano Banana Pro）"""
    try:
        client = await get_client()

        config = ImageGenerationConfig(
            width=request.width,
            height=request.height,
            style=request.style,
            negative_prompt=request.negative_prompt
        )

        # 处理参考图像
        reference_images = None
        if request.reference_image_base64:
            reference_images = [
                ImageContent.from_base64(request.reference_image_base64)
            ]

        response = await client.generate_image(
            prompt=request.prompt,
            config=config,
            reference_images=reference_images
        )

        if not response.images:
            raise HTTPException(status_code=500, detail="No image generated")

        image = response.images[0]

        return ImageGenerationResponse(
            image_base64=image.data,
            mime_type=image.mime_type,
            model=response.model,
            revised_prompt=response.revised_prompt
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 多模态对话 ====================

class MultimodalRequest(BaseModel):
    """多模态对话请求"""
    prompt: str
    system_prompt: Optional[str] = None
    images_base64: list[str] = []
    temperature: float = 0.7
    max_tokens: int = 4096


@router.post("/multimodal", response_model=TextGenerationResponse)
async def generate_multimodal(request: MultimodalRequest):
    """多模态对话（支持图像输入）"""
    try:
        client = await get_client()

        # 转换图像
        images = [
            ImageContent.from_base64(img)
            for img in request.images_base64
        ]

        config = GenerationConfig(
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        response = await client.generate_text(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            images=images,
            config=config
        )

        return TextGenerationResponse(
            content=response.content,
            model=response.model,
            usage=response.usage
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
