"""
Nano Banana Station - FastAPI Application
主应用入口
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config_loader import get_config, CONFIG_DIR
from engines import get_client


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("[Nano Banana Station] Starting up...")

    # 初始化配置
    config = get_config()
    config.start_watching()
    print(f"[Config] Loaded from {CONFIG_DIR}")
    print(f"[Config] Default model: {config.default_model}")
    print(f"[Config] Enabled providers: {[p.name for p in config.get_enabled_providers()]}")

    # 初始化模型客户端
    try:
        client = await get_client()
        print("[Engines] Model client initialized")
    except Exception as e:
        print(f"[Engines] Warning: {e}")

    yield

    # 关闭时
    print("[Nano Banana Station] Shutting down...")
    config.stop_watching()

    # 关闭模型客户端
    try:
        client = await get_client()
        await client.close()
    except Exception:
        pass


# 创建 FastAPI 应用
app = FastAPI(
    title="Nano Banana Station",
    description="将学术论文转换为可爱漫画的 AI 服务",
    version="0.1.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务（生成的图片等）
output_dir = PROJECT_ROOT / "output"
output_dir.mkdir(exist_ok=True)
app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")


# ==================== 路由注册 ====================

from routes import api_router
app.include_router(api_router, prefix="/api")


# ==================== 根路由 ====================

@app.get("/")
async def root():
    """根路由 - 健康检查"""
    return {
        "service": "Nano Banana Station",
        "status": "running",
        "version": "0.1.0"
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    config = get_config()
    return {
        "status": "healthy",
        "config_loaded": True,
        "providers": {
            name: provider.enabled
            for name, provider in config.providers.items()
        }
    }


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).parent)]
    )
