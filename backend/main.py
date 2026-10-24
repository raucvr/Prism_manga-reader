"""
Prism - FastAPI Application
主应用入口
"""

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config_loader import get_config, CONFIG_DIR, as_int
from engines import get_client
from routes import jobs as job_routes


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("[Prism] Starting up...")

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
    print("[Prism] Shutting down...")
    config.stop_watching()

    # 关闭模型客户端
    try:
        client = await get_client()
        await client.close()
    except Exception:
        pass


# 创建 FastAPI 应用
app = FastAPI(
    title="Prism",
    description="将学术论文转换为可爱漫画的 AI 服务（v2: Seedream 5.0 Pro + Seed 2.0，异步任务 /api/jobs）",
    version="0.2.0",
    lifespan=lifespan
)

# ==================== 上传大小限制 ====================

MULTIPART_SLACK = 64 * 1024  # multipart boundaries, part headers and the small form fields next to the PDF
UPLOAD_PATHS = {"/api/jobs", "/api/manga/upload-pdf", "/api/manga/generate/pdf"}


class UploadSizeLimit:
    """Answer 413 for an oversized PDF upload (POST /api/jobs and the legacy /api/manga PDF routes) before its body is read.

    Starlette parses the whole multipart body (spooling the file to disk) before create_job runs, so the route's
    exact check alone would refuse a huge upload only after receiving all of it.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"].rstrip("/") not in UPLOAD_PATHS:
            await self.app(scope, receive, send)
            return
        limit = job_routes.MAX_PDF_BYTES + MULTIPART_SLACK
        detail = f"PDF larger than {job_routes.MAX_PDF_BYTES / (1024 * 1024):g} MB"
        try:
            declared = int(dict(scope["headers"]).get(b"content-length", b""))
        except ValueError:
            declared = None  # chunked upload: count the bytes instead
        if declared is not None and declared > limit:
            await JSONResponse({"detail": detail}, status_code=413)(scope, receive, send)
            return

        received = 0

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:  # FastAPI re-raises an HTTPException from body parsing as this 413
                    raise HTTPException(status_code=413, detail=detail)
            return message

        await self.app(scope, counting_receive, send)


# ==================== 跨站请求防护 ====================

STATE_CHANGING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def app_ports(environ=os.environ) -> tuple[int, int]:
    """(frontend, backend) ports: start.sh and start.ps1 export the ones they use; the defaults are Prism's usual ones."""
    return as_int(environ.get("FRONTEND_PORT"), 3000), as_int(environ.get("BACKEND_PORT"), 8000)


def local_origins(*ports: int) -> list[str]:
    return [f"http://{host}:{port}" for port in ports for host in ("localhost", "127.0.0.1")]


FRONTEND_PORT, BACKEND_PORT = app_ports()
FRONTEND_ORIGINS = local_origins(FRONTEND_PORT)
APP_ORIGINS = frozenset(local_origins(FRONTEND_PORT, BACKEND_PORT))  # the backend's own pages (/docs) count too


class SameOriginWrites:
    """Answer 403 to a state-changing request (POST, PUT, PATCH, DELETE) that a page on another site sent.

    Any web page can make the browser POST a plain form (urlencoded or multipart, so no CORS preflight) to
    localhost:3000/api/jobs or :8000, and CORSMiddleware only hides the response: the paid job would still start.
    Browsers always send Origin with such a request, and current ones Sec-Fetch-Site as well; curl, scripts and the
    tests send neither header and pass. The UI's own requests reach the backend through the Next.js proxy with its
    Origin (localhost or 127.0.0.1 on the frontend port) unchanged.
    """

    def __init__(self, app, origins=APP_ORIGINS):
        self.app = app
        self.origins = frozenset(origins)
        self.detail = ("Refused: this request came from another site. Prism accepts changes only from its own pages ("
                       + ", ".join(sorted(self.origins)) + ").")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in STATE_CHANGING_METHODS:
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        origin = headers.get(b"origin", b"").decode("latin-1").strip().lower()
        site = headers.get(b"sec-fetch-site", b"").decode("latin-1").strip().lower()
        if (origin and origin not in self.origins) or site == "cross-site":  # "null" (sandboxed frame, file://) too
            print(f"[Security] Refused {scope['method']} {scope['path']} from another site "
                  f"(Origin: {origin[:100] or '-'}, Sec-Fetch-Site: {site[:20] or '-'})")
            await JSONResponse({"detail": self.detail}, status_code=403)(scope, receive, send)
            return
        await self.app(scope, receive, send)


LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "[::1]", "testserver"})   # testserver: FastAPI's TestClient


def host_name(value: str) -> str:
    """The name part of a Host header: "localhost:3000" -> "localhost", "[::1]:8000" -> "[::1]"."""
    value = value.split(",")[0].strip().lower()
    if value.startswith("["):
        return value.split("]", 1)[0] + "]"
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


class LocalHostOnly:
    """Answer 403 to a request whose Host, or X-Forwarded-Host (the Next.js proxy's copy of the browser's Host), is not
    this machine. With DNS rebinding a web page points a name of its own at 127.0.0.1 and reads the jobs, the uploaded
    papers and the prompts through either port; its requests carry that name as their Host. Reads included: the write
    guard above only stops spending."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        foreign = [value.decode("latin-1") for value in (headers.get(b"host", b""), headers.get(b"x-forwarded-host", b""))
                   if value and host_name(value.decode("latin-1")) not in LOCAL_HOSTNAMES]
        if foreign:
            print(f"[Security] Refused {scope.get('method', 'WEBSOCKET')} {scope['path']} for the host {foreign[0][:100]}")
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await JSONResponse({"detail": "Refused: Prism answers only to localhost and 127.0.0.1."}, status_code=403)(
                    scope, receive, send)
            return
        await self.app(scope, receive, send)


# Added before CORS, so CORS stays the outer layer and the 413 carries its headers
app.add_middleware(UploadSizeLimit)
# Outside the upload limit, so a cross-site upload is refused before any of its body is read
app.add_middleware(SameOriginWrites, origins=APP_ORIGINS)
app.add_middleware(LocalHostOnly)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
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
        "service": "Prism",
        "status": "running",
        "version": "0.2.0"
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
    # Loopback only (anything on the network could otherwise start paid jobs), and no reload: it kills running jobs
    uvicorn.run("main:app", host="127.0.0.1", port=BACKEND_PORT)
