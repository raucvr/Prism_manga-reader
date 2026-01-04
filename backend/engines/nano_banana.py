"""
Nano Banana Pro Engine (Google Gemini Image Generation)
封装 Google Generative AI API，支持文本生成和图像生成
"""

import asyncio
import base64
import json
from typing import Optional, AsyncIterator

import httpx

from .base import (
    BaseEngine,
    Message,
    MessageRole,
    ImageContent,
    GenerationConfig,
    ImageGenerationConfig,
    TextResponse,
    ImageResponse,
)


class NanoBananaEngine(BaseEngine):
    """
    Google Gemini 引擎
    基于 Google Gemini API，支持文本和图像生成
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        model: str = "gemini-3-pro"
    ):
        super().__init__(api_key, base_url, model)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def name(self) -> str:
        return "Nano Banana Pro"

    @property
    def supports_image_generation(self) -> bool:
        return True

    @property
    def supports_vision(self) -> bool:
        return True

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=10.0),
                headers={"Content-Type": "application/json"}
            )
        return self._client

    def _build_contents(self, messages: list[Message]) -> list[dict]:
        """将消息转换为 Gemini API 格式"""
        contents = []

        for msg in messages:
            # Gemini 使用 "user" 和 "model" 作为角色
            role = "user" if msg.role in (MessageRole.USER, MessageRole.SYSTEM) else "model"

            parts = []

            # 添加文本内容
            if msg.content:
                parts.append({"text": msg.content})

            # 添加图像内容
            for img in msg.images:
                if img.is_base64:
                    parts.append({
                        "inline_data": {
                            "mime_type": img.mime_type,
                            "data": img.data
                        }
                    })
                elif img.is_url:
                    # Gemini 需要先下载图片转为 base64
                    # 这里简化处理，实际使用时需要异步下载
                    parts.append({
                        "file_data": {
                            "mime_type": img.mime_type,
                            "file_uri": img.data
                        }
                    })

            contents.append({"role": role, "parts": parts})

        return contents

    def _build_generation_config(self, config: Optional[GenerationConfig]) -> dict:
        """构建生成配置"""
        if config is None:
            config = GenerationConfig()

        return {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_tokens,
            "topP": config.top_p,
            "topK": config.top_k,
        }

    async def generate_text(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None
    ) -> TextResponse:
        """生成文本响应"""
        client = await self._get_client()

        url = f"{self.base_url}/models/{self.model}:generateContent"
        params = {"key": self.api_key}

        payload = {
            "contents": self._build_contents(messages),
            "generationConfig": self._build_generation_config(config)
        }

        response = await client.post(url, params=params, json=payload)
        response.raise_for_status()
        data = response.json()

        # 提取响应内容
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No response candidates returned")

        content = ""
        for part in candidates[0].get("content", {}).get("parts", []):
            if "text" in part:
                content += part["text"]

        usage = data.get("usageMetadata", {})

        return TextResponse(
            content=content,
            model=self.model,
            usage={
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0)
            },
            finish_reason=candidates[0].get("finishReason", "")
        )

    async def generate_text_stream(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """流式生成文本"""
        client = await self._get_client()

        url = f"{self.base_url}/models/{self.model}:streamGenerateContent"
        params = {"key": self.api_key, "alt": "sse"}

        payload = {
            "contents": self._build_contents(messages),
            "generationConfig": self._build_generation_config(config)
        }

        async with client.stream("POST", url, params=params, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        candidates = data.get("candidates", [])
                        for candidate in candidates:
                            for part in candidate.get("content", {}).get("parts", []):
                                if "text" in part:
                                    yield part["text"]
                    except json.JSONDecodeError:
                        continue

    async def generate_image(
        self,
        prompt: str,
        config: Optional[ImageGenerationConfig] = None,
        reference_images: list[ImageContent] = None
    ) -> ImageResponse:
        """
        使用 Nano Banana Pro 生成图像
        利用 Gemini 的原生图像生成能力
        """
        if config is None:
            config = ImageGenerationConfig()

        client = await self._get_client()

        # 使用当前配置的模型（应该是图像生成模型）
        url = f"{self.base_url}/models/{self.model}:generateContent"
        params = {"key": self.api_key}

        # 构建提示词，强调漫画风格
        style_prompt = self._build_manga_prompt(prompt, config)

        # 构建请求内容
        parts = [{"text": style_prompt}]

        # 如果有参考图像，添加到请求中
        if reference_images:
            for img in reference_images:
                if img.is_base64:
                    parts.append({
                        "inline_data": {
                            "mime_type": img.mime_type,
                            "data": img.data
                        }
                    })

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "temperature": config.temperature,  # 使用配置的温度，不要硬编码
            }
        }

        response = await client.post(url, params=params, json=payload)
        response.raise_for_status()
        data = response.json()

        # 提取生成的图像
        images = []
        revised_prompt = None

        candidates = data.get("candidates", [])
        for candidate in candidates:
            for part in candidate.get("content", {}).get("parts", []):
                if "inline_data" in part:
                    inline_data = part["inline_data"]
                    images.append(ImageContent(
                        data=inline_data.get("data", ""),
                        mime_type=inline_data.get("mime_type", "image/png"),
                        is_base64=True
                    ))
                elif "text" in part:
                    # 可能包含修改后的提示词说明
                    revised_prompt = part["text"]

        if not images:
            raise ValueError("No images generated")

        return ImageResponse(
            images=images,
            model=self.model,
            prompt=prompt,
            revised_prompt=revised_prompt
        )

    def _build_manga_prompt(self, prompt: str, config: ImageGenerationConfig) -> str:
        """构建漫画风格的提示词"""
        style_map = {
            "full_color_manga": "full color manga style, vibrant colors, clean lines",
            "black_white_manga": "black and white manga style, high contrast, screentone shading",
            "chiikawa": "Chiikawa style, cute characters, soft pastel colors, simple round shapes",
            "watercolor": "watercolor manga style, soft edges, flowing colors"
        }

        style_desc = style_map.get(config.style, style_map["full_color_manga"])

        # 构建完整提示词
        full_prompt = f"""Generate a manga panel illustration with the following requirements:

Style: {style_desc}
Aspect Ratio: {config.width}x{config.height}

Content: {prompt}

Important:
- Make it suitable for educational/explainer content
- Include clear visual storytelling
- If there are characters, make them expressive and cute
- Ensure text bubbles are readable if included"""

        if config.negative_prompt:
            full_prompt += f"\n\nAvoid: {config.negative_prompt}"

        return full_prompt

    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
