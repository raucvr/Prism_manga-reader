"""
OpenRouter Engine
封装 OpenRouter API，兼容 OpenAI 格式，支持多种 LLM 模型
支持通过 OpenRouter 调用 Nano Banana Pro (Gemini) 图像生成
"""

import json
from typing import Optional, AsyncIterator

import httpx

from .base import (
    BaseEngine,
    Message,
    ImageContent,
    GenerationConfig,
    ImageGenerationConfig,
    TextResponse,
    ImageResponse,
)


class OpenRouterEngine(BaseEngine):
    """
    OpenRouter 引擎
    使用 OpenAI 兼容的 API 格式访问多种 LLM
    支持通过 Gemini 生成图像
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "google/gemini-3-pro-preview"
    ):
        super().__init__(api_key, base_url, model)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def name(self) -> str:
        return "OpenRouter"

    @property
    def supports_image_generation(self) -> bool:
        # 支持任何包含 "image" 的 Gemini 模型
        return "gemini" in self.model.lower() and "image" in self.model.lower()

    @property
    def supports_vision(self) -> bool:
        vision_models = [
            "anthropic/claude-3",
            "openai/gpt-4-vision",
            "openai/gpt-4o",
            "google/gemini"
        ]
        return any(self.model.startswith(prefix) for prefix in vision_models)

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(180.0, connect=10.0),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://prism.ai.local",
                    "X-Title": "Prism"
                }
            )
        return self._client

    def _build_messages(self, messages: list[Message]) -> list[dict]:
        """将消息转换为 OpenAI 兼容格式"""
        result = []

        for msg in messages:
            role = msg.role.value

            if msg.images and self.supports_vision:
                content = []

                if msg.content:
                    content.append({"type": "text", "text": msg.content})

                for img in msg.images:
                    if img.is_url:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": img.data}
                        })
                    elif img.is_base64:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{img.mime_type};base64,{img.data}"
                            }
                        })

                result.append({"role": role, "content": content})
            else:
                result.append({"role": role, "content": msg.content})

        return result

    async def generate_text(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None
    ) -> TextResponse:
        """生成文本响应"""
        if config is None:
            config = GenerationConfig()

        client = await self._get_client()

        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": self._build_messages(messages),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        }

        if config.stop_sequences:
            payload["stop"] = config.stop_sequences

        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})

        return TextResponse(
            content=content,
            model=data.get("model", self.model),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            },
            finish_reason=choice.get("finish_reason", "")
        )

    async def generate_text_stream(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """流式生成文本"""
        if config is None:
            config = GenerationConfig()

        client = await self._get_client()

        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": self._build_messages(messages),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "stream": True
        }

        if config.stop_sequences:
            payload["stop"] = config.stop_sequences

        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def generate_image(
        self,
        prompt: str,
        config: Optional[ImageGenerationConfig] = None,
        reference_images: list[ImageContent] = None
    ) -> ImageResponse:
        """
        通过 OpenRouter 调用 Nano Banana Pro (Gemini) 生成图像
        """
        if config is None:
            config = ImageGenerationConfig()

        client = await self._get_client()
        url = f"{self.base_url}/chat/completions"

        # 构建消息
        content = []

        # 添加参考图像
        if reference_images:
            for img in reference_images:
                if img.is_url:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img.data}
                    })
                elif img.is_base64:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img.mime_type};base64,{img.data}"
                        }
                    })

        # 添加文本提示
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0.8,
            "max_tokens": 4096
        }

        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        # 从响应中提取图像
        images = []
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content_resp = message.get("content", "")

        # 检查 message.images 字段（OpenRouter Gemini 图像响应格式）
        message_images = message.get("images", [])
        for img_item in message_images:
            if isinstance(img_item, dict):
                img_type = img_item.get("type", "")
                if img_type == "image_url":
                    img_url = img_item.get("image_url", {}).get("url", "")
                    if img_url.startswith("data:image"):
                        # 解析 data URL: data:image/png;base64,xxxxx
                        parts = img_url.split(",", 1)
                        if len(parts) == 2:
                            mime_match = parts[0].split(";")[0].replace("data:", "")
                            images.append(ImageContent.from_base64(parts[1], mime_match))

        # 如果 content 是列表（备用格式）
        if not images and isinstance(content_resp, list):
            for item in content_resp:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        img_data = item.get("image", {})
                        if "data" in img_data:
                            images.append(ImageContent.from_base64(img_data["data"], "image/png"))
                        elif "url" in img_data:
                            images.append(ImageContent.from_url(img_data["url"]))

        text_content = content_resp if isinstance(content_resp, str) else ""

        return ImageResponse(
            images=images,
            model=data.get("model", self.model),
            prompt=prompt,
            revised_prompt=text_content if not images else None
        )

    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
