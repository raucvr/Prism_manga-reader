"""
API Routes Module
所有 API 路由的统一入口
"""

from fastapi import APIRouter

from .config import router as config_router
from .generation import router as generation_router
from .manga import router as manga_router

# 主 API 路由
api_router = APIRouter()

# 注册子路由
api_router.include_router(config_router, prefix="/config", tags=["配置"])
api_router.include_router(generation_router, prefix="/generate", tags=["生成"])
api_router.include_router(manga_router, prefix="/manga", tags=["漫画"])
