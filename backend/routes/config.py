"""
Configuration API Routes
配置管理相关接口
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import get_config
from engines import reset_client
from services.manga_generator import reset_manga_generator


router = APIRouter()


class ProviderStatus(BaseModel):
    """服务商状态"""
    name: str
    enabled: bool
    has_api_key: bool
    models: list[str]


class ConfigResponse(BaseModel):
    """配置响应"""
    default_model: str
    fallback_text_model: str
    providers: list[ProviderStatus]
    manga_settings: dict
    output_settings: dict


@router.get("", response_model=ConfigResponse)
async def get_current_config():
    """获取当前配置（隐藏敏感信息）"""
    config = get_config()

    providers = []
    for name, provider in config.providers.items():
        providers.append(ProviderStatus(
            name=name,
            enabled=provider.enabled,
            has_api_key=bool(provider.api_key and len(provider.api_key) > 0),
            models=provider.models
        ))

    return ConfigResponse(
        default_model=config.default_model,
        fallback_text_model=config.fallback_text_model,
        providers=providers,
        manga_settings={
            "default_style": config.manga_settings.default_style,
            "aspect_ratio": config.manga_settings.aspect_ratio,
            "render_text_in_image": config.manga_settings.render_text_in_image,
            "panels_per_page": config.manga_settings.panels_per_page,
            "default_character": config.manga_settings.default_character
        },
        output_settings={
            "image_format": config.output_settings.image_format,
            "image_quality": config.output_settings.image_quality,
            "max_width": config.output_settings.max_width,
            "max_height": config.output_settings.max_height
        }
    )


@router.get("/providers")
async def list_providers():
    """列出所有可用的服务商"""
    config = get_config()
    return {
        name: {
            "enabled": provider.enabled,
            "models": provider.models
        }
        for name, provider in config.providers.items()
    }


@router.get("/models")
async def list_available_models():
    """列出所有可用模型"""
    config = get_config()

    models = []
    for provider in config.get_enabled_providers():
        for model in provider.models:
            models.append({
                "provider": provider.name,
                "model": model,
                "full_name": f"{provider.name}/{model}"
            })

    return {"models": models}


@router.post("/reload")
async def reload_config():
    """重新加载配置文件"""
    try:
        # 重置所有缓存的实例
        await reset_client()
        reset_manga_generator()
        # 重新加载配置
        config = get_config()
        config.reload()
        return {"status": "success", "message": "Configuration reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
