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
from .byteplus import BytePlusEngine, DEFAULT_TEXT_MODEL as BYTEPLUS_DEFAULT_TEXT_MODEL

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
    "BytePlusEngine",
    # 工厂函数
    "create_engine",
    "get_default_engine",
    "get_image_engine",
    "get_vision_engine",
    "get_text_engine",
    "split_model_spec",
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
        "byteplus": BytePlusEngine,
    }

    engine_class = engine_map.get(provider_name)
    if engine_class is None:
        raise ValueError(f"No engine implementation for provider: {provider_name}")

    return engine_class(
        api_key=provider_config.api_key,
        base_url=provider_config.base_url,
        model=model
    )


KNOWN_PROVIDERS = ("byteplus", "openrouter", "google_genai")


def split_model_spec(spec: str) -> tuple[Optional[str], str]:
    """``"byteplus/seed-2-0-pro"`` → ``("byteplus", "seed-2-0-pro")``; ``"google/gemini-x"`` → ``(None, "google/gemini-x")``.

    Only known provider names are treated as a prefix, so OpenRouter model ids such as
    ``google/gemini-3-pro-image-preview`` keep working unchanged.
    """
    if spec and "/" in spec:
        provider, model = spec.split("/", 1)
        if provider in KNOWN_PROVIDERS:
            return provider, model
    return None, spec


def _engine_for_spec(spec: str, *, image: bool = False) -> Optional[BaseEngine]:
    """Create an engine for a ``provider/model`` spec if that provider is enabled; else None."""
    provider, model = split_model_spec(spec)
    if provider is None:
        return None
    config = get_config()
    provider_config = config.get_provider(provider)
    if provider_config is None or not provider_config.enabled:
        return None
    if provider == "byteplus":
        # BytePlus engines serve text and images from one client: text model + Seedream image model
        if image:
            return BytePlusEngine(api_key=provider_config.api_key, base_url=provider_config.base_url or None, model=BYTEPLUS_DEFAULT_TEXT_MODEL, image_model=model)
        return BytePlusEngine(api_key=provider_config.api_key, base_url=provider_config.base_url or None, model=model)
    return create_engine(provider, model=model, provider_config=provider_config)


def get_default_engine() -> BaseEngine:
    """
    获取默认文本引擎：优先 core_engine.text_model / default_model（支持 provider/model），
    其次 Google GenAI，最后第一个启用的服务商。
    """
    config = get_config()

    for spec in (config.text_model, config.default_model):
        engine = _engine_for_spec(spec) if spec else None
        if engine is not None:
            return engine

    google_config = config.get_provider("google_genai")
    if google_config and google_config.enabled:
        return create_engine("google_genai", provider_config=google_config)

    for provider in config.get_enabled_providers():
        return create_engine(provider.name, provider_config=provider)

    raise RuntimeError("No enabled providers found in configuration")


def get_image_engine() -> BaseEngine:
    """
    获取图像生成引擎 - 从配置读取模型（支持 byteplus/… Seedream、openrouter/…、google_genai）
    """
    config = get_config()
    image_model = config.image_model

    print(f"[Engines] Using image model: {image_model}")

    engine = _engine_for_spec(image_model, image=True)
    if engine is not None:
        return engine

    # 1. 首选直接调用 Google Gemini API
    google_config = config.get_provider("google_genai")
    if google_config and google_config.enabled:
        model_name = image_model.split("/")[-1] if "/" in image_model else image_model
        return create_engine("google_genai", model=model_name, provider_config=google_config)

    # 2. 通过 OpenRouter 调用
    openrouter_config = config.get_provider("openrouter")
    if openrouter_config and openrouter_config.enabled:
        return create_engine("openrouter", model=image_model, provider_config=openrouter_config)

    raise RuntimeError("No image engine available. Enable byteplus, google_genai or openrouter provider.")


def get_vision_engine() -> BaseEngine:
    """获取视觉理解引擎（面板 QA 判定），默认与文本引擎相同"""
    config = get_config()
    engine = _engine_for_spec(config.vision_model) if config.vision_model else None
    return engine or get_default_engine()


def get_text_engine(prefer_fast: bool = False) -> BaseEngine:
    """
    获取文本生成引擎

    Args:
        prefer_fast: 是否优先选择快速模型（core_engine.fast_text_model）
    """
    config = get_config()

    if prefer_fast:
        engine = _engine_for_spec(config.fast_text_model) if config.fast_text_model else None
        if engine is not None:
            return engine
        google_config = config.get_provider("google_genai")
        if google_config and google_config.enabled:
            for model in google_config.models:
                if "flash" in model.lower():
                    return create_engine("google_genai", model=model, provider_config=google_config)

    fallback = config.fallback_text_model
    if fallback and "/" in fallback:
        engine = _engine_for_spec(fallback)
        if engine is not None:
            return engine
        provider, model = fallback.split("/", 1)
        if provider == "openrouter":
            openrouter_config = config.get_provider("openrouter")
            if openrouter_config and openrouter_config.enabled:
                return create_engine("openrouter", model=model, provider_config=openrouter_config)

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
        except Exception as e:  # noqa: BLE001 — missing keys raise ValueError
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


def _invalidate_client_on_config_change(_config) -> None:
    """Sync hook for the config watcher thread: drop the cached client so the next call rebuilds it."""
    global _client
    _client = None


get_config().on_change(_invalidate_client_on_config_change)
