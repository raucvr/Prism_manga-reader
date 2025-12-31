"""
Engines Module
统一导出所有 AI 引擎，提供工厂函数创建引擎实例
"""

from typing import Optional, Union

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
from .nano_banana import NanoBananaEngine
from .openrouter import OpenRouterEngine

# 导入配置加载器
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config, ProviderConfig


__all__ = [
    # 基础类
    "BaseEngine",
    "Message",
    "MessageRole",
    "ImageContent",
    "GenerationConfig",
    "ImageGenerationConfig",
    "TextResponse",
    "ImageResponse",
    # 引擎实现
    "NanoBananaEngine",
    "OpenRouterEngine",
    # 工厂函数
    "create_engine",
    "get_default_engine",
    "get_image_engine",
    "get_text_engine",
    "reset_client",
]


def create_engine(
    provider_name: str,
    model: Optional[str] = None,
    provider_config: Optional[ProviderConfig] = None
) -> BaseEngine:
    """
    根据服务商名称创建引擎实例

    Args:
        provider_name: 服务商名称 (google_genai, openrouter, local_llm)
        model: 模型名称，不指定则使用服务商配置的第一个模型
        provider_config: 服务商配置，不指定则从配置文件读取

    Returns:
        BaseEngine 实例
    """
    if provider_config is None:
        config = get_config()
        provider_config = config.get_provider(provider_name)

    if provider_config is None:
        raise ValueError(f"Unknown provider: {provider_name}")

    if not provider_config.enabled:
        raise ValueError(f"Provider {provider_name} is not enabled")

    # 确定使用的模型
    if model is None:
        if provider_config.models:
            model = provider_config.models[0]
        else:
            raise ValueError(f"No models configured for provider {provider_name}")

    # 创建对应的引擎
    engine_map = {
        "google_genai": NanoBananaEngine,
        "openrouter": OpenRouterEngine,
    }

    engine_class = engine_map.get(provider_name)
    if engine_class is None:
        raise ValueError(f"No engine implementation for provider: {provider_name}")

    return engine_class(
        api_key=provider_config.api_key,
        base_url=provider_config.base_url,
        model=model
    )


def get_default_engine() -> BaseEngine:
    """
    获取默认引擎（Nano Banana Pro）

    Returns:
        配置的默认引擎实例
    """
    config = get_config()

    # 首先尝试 Google GenAI (Nano Banana Pro)
    google_config = config.get_provider("google_genai")
    if google_config and google_config.enabled:
        return create_engine("google_genai", provider_config=google_config)

    # 回退到第一个启用的服务商
    for provider in config.get_enabled_providers():
        return create_engine(provider.name, provider_config=provider)

    raise RuntimeError("No enabled providers found in configuration")


def get_image_engine() -> BaseEngine:
    """
    获取图像生成引擎 - 从配置读取模型

    Returns:
        图像生成引擎实例
    """
    config = get_config()
    image_model = config.image_model  # 从配置读取图像模型

    print(f"[Engines] Using image model: {image_model}")

    # 1. 首选直接调用 Google Gemini API
    google_config = config.get_provider("google_genai")
    if google_config and google_config.enabled:
        # 如果是 OpenRouter 格式的模型名，转换为 Google 格式
        model_name = image_model
        if "/" in model_name:
            model_name = model_name.split("/")[-1]  # google/gemini-3-pro -> gemini-3-pro
        return create_engine(
            "google_genai",
            model=model_name,
            provider_config=google_config
        )

    # 2. 通过 OpenRouter 调用
    openrouter_config = config.get_provider("openrouter")
    if openrouter_config and openrouter_config.enabled:
        return create_engine(
            "openrouter",
            model=image_model,
            provider_config=openrouter_config
        )

    raise RuntimeError("No image engine available. Enable google_genai or openrouter provider.")


def get_text_engine(prefer_fast: bool = False) -> BaseEngine:
    """
    获取文本生成引擎

    Args:
        prefer_fast: 是否优先选择快速模型

    Returns:
        文本生成引擎实例
    """
    config = get_config()

    if prefer_fast:
        # 优先使用 flash 模型
        google_config = config.get_provider("google_genai")
        if google_config and google_config.enabled:
            for model in google_config.models:
                if "flash" in model.lower():
                    return create_engine("google_genai", model=model, provider_config=google_config)

    # 使用 fallback 模型
    fallback = config.fallback_text_model
    if fallback and "/" in fallback:
        provider, model = fallback.split("/", 1)
        if provider == "openrouter":
            openrouter_config = config.get_provider("openrouter")
            if openrouter_config and openrouter_config.enabled:
                return create_engine("openrouter", model=model, provider_config=openrouter_config)

    # 回退到默认引擎
    return get_default_engine()


class ModelClient:
    """
    统一模型客户端
    封装多个引擎，提供简化的 API 调用接口
    """

    def __init__(self):
        self._engines: dict[str, BaseEngine] = {}
        self._default_engine: Optional[BaseEngine] = None
        self._image_engine: Optional[BaseEngine] = None

    async def initialize(self) -> None:
        """初始化客户端，加载默认引擎"""
        try:
            self._default_engine = get_default_engine()
            self._image_engine = get_image_engine()
        except RuntimeError as e:
            print(f"[ModelClient] Warning: {e}")

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        images: list[ImageContent] = None,
        config: Optional[GenerationConfig] = None,
        engine: Optional[BaseEngine] = None
    ) -> TextResponse:
        """
        生成文本

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            images: 附加图像
            config: 生成配置
            engine: 指定引擎，不指定则使用默认

        Returns:
            TextResponse
        """
        if engine is None:
            engine = self._default_engine
        if engine is None:
            engine = get_default_engine()

        messages = []
        if system_prompt:
            messages.append(Message.system(system_prompt))
        messages.append(Message.user(prompt, images or []))

        return await engine.generate_text(messages, config)

    async def generate_image(
        self,
        prompt: str,
        config: Optional[ImageGenerationConfig] = None,
        reference_images: list[ImageContent] = None
    ) -> ImageResponse:
        """
        生成图像

        Args:
            prompt: 图像描述
            config: 图像生成配置
            reference_images: 参考图像

        Returns:
            ImageResponse
        """
        if self._image_engine is None:
            self._image_engine = get_image_engine()

        return await self._image_engine.generate_image(prompt, config, reference_images)

    async def close(self) -> None:
        """关闭所有引擎连接"""
        for engine in self._engines.values():
            await engine.close()
        if self._default_engine:
            await self._default_engine.close()
        if self._image_engine and self._image_engine != self._default_engine:
            await self._image_engine.close()


# 全局客户端实例
_client: Optional[ModelClient] = None


async def get_client() -> ModelClient:
    """获取全局模型客户端"""
    global _client
    if _client is None:
        _client = ModelClient()
        await _client.initialize()
    return _client


async def reset_client() -> None:
    """重置全局客户端（配置重载时调用）"""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
