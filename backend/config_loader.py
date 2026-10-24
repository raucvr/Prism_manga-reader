"""
Configuration Loader Module
从 config/api_config.yaml 读取配置，支持环境变量替换和热更新
"""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
API_CONFIG_PATH = CONFIG_DIR / "api_config.yaml"


def expand_env_vars(value: Any) -> Any:
    """递归替换配置中的环境变量 ${VAR_NAME}"""
    if isinstance(value, str):
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        for var_name in matches:
            env_value = os.environ.get(var_name, "")
            value = value.replace(f"${{{var_name}}}", env_value)
        return value
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value


def as_int(value: Any, default: int) -> int:
    """Tolerant int parsing for YAML values ('', None, '5', 5.0, '${UNSET}', .inf)."""
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return default
        return int(float(value))
    except (TypeError, ValueError, OverflowError):  # OverflowError: .inf, 1e400
        return default


def as_bool(value: Any, default: bool) -> bool:
    """Tolerant bool parsing: true/false/yes/no/on/off/1/0 (strings included)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in ("true", "yes", "on", "1"):
        return True
    if text in ("false", "no", "off", "0"):
        return False
    return default


@dataclass
class ProviderConfig:
    """API 服务商配置"""
    name: str
    enabled: bool = False
    api_key: str = ""
    base_url: str = ""
    models: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "ProviderConfig":
        return cls(
            name=name,
            enabled=as_bool(data.get("enabled"), False),
            api_key=str(data.get("api_key") or ""),
            base_url=str(data.get("base_url") or ""),
            models=data.get("models", []) if isinstance(data.get("models"), list) else [data.get("model", "")]
        )


@dataclass
class MangaSettings:
    """漫画生成设置"""
    default_style: str = "full_color_manga"
    aspect_ratio: str = "2:3"
    render_text_in_image: bool = True
    panels_per_page: int = 1
    default_character: str = "kumomo"
    # v2 pipeline (Seedream): one image per panel, bounded parallelism, vision QA
    layout: str = "single"           # single | strip
    panel_width: int = 1024
    panel_height: int = 1536
    concurrency: int = 5
    qa_enabled: bool = True
    qa_blind_check: bool = True      # second, script-free transcription must agree with the judge
    max_attempts: int = 3
    min_panels: int = 16
    max_panels: int = 32
    style_anchor: bool = False
    fast: bool = False
    # v3 (default): text-free art, typeset text and code-drawn exhibits; v2: the image model letters the text
    pipeline: str = "v3"
    # v3: letter the same art in English and Chinese (the job's language first); the second language costs text calls only
    bilingual: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "MangaSettings":
        return cls(
            default_style=data.get("default_style", "full_color_manga"),
            aspect_ratio=data.get("aspect_ratio", "2:3"),
            render_text_in_image=as_bool(data.get("render_text_in_image"), True),
            panels_per_page=as_int(data.get("panels_per_page"), 1),
            default_character=str(data.get("default_theme") or data.get("default_character") or "kumomo"),
            layout=str(data.get("layout") or "single"),
            panel_width=as_int(data.get("panel_width"), 1024),
            panel_height=as_int(data.get("panel_height"), 1536),
            concurrency=max(1, as_int(data.get("concurrency"), 5)),
            qa_enabled=as_bool(data.get("qa_enabled"), True),
            qa_blind_check=as_bool(data.get("qa_blind_check"), True),
            max_attempts=max(1, as_int(data.get("max_attempts"), 3)),
            min_panels=as_int(data.get("min_panels"), 16),
            max_panels=as_int(data.get("max_panels"), 32),
            style_anchor=as_bool(data.get("style_anchor"), False),
            fast=as_bool(data.get("fast"), False),
            pipeline="v2" if str(data.get("pipeline") or "v3").strip().lower() == "v2" else "v3",
            bilingual=as_bool(data.get("bilingual"), True),
        )


@dataclass
class OutputSettings:
    """输出设置"""
    image_format: str = "png"
    image_quality: int = 95
    max_width: int = 1024
    max_height: int = 1536

    @classmethod
    def from_dict(cls, data: dict) -> "OutputSettings":
        return cls(
            image_format=str(data.get("image_format") or "png"),
            image_quality=as_int(data.get("image_quality"), 95),
            max_width=as_int(data.get("max_width"), 1024),
            max_height=as_int(data.get("max_height"), 1536)
        )


class ConfigChangeHandler(FileSystemEventHandler):
    """配置文件变更监听器"""
    def __init__(self, callback):
        self.callback = callback

    def on_modified(self, event):
        if event.src_path.endswith("api_config.yaml"):
            self._reload()

    def on_created(self, event):  # an atomic save renamed in from another directory arrives as a creation only
        self.on_modified(event)

    def on_moved(self, event):
        # an atomic save (temp file renamed over api_config.yaml) arrives as a move only, with no modified event
        if event.dest_path.endswith("api_config.yaml"):
            self._reload()

    def _reload(self):
        try:
            self.callback()
        except Exception as e:  # noqa: BLE001 — an exception here would end watchdog's observer thread for good
            print(f"[ConfigManager] Config reload failed: {e}")


class ConfigManager:
    """配置管理器 - 单例模式"""
    _instance: Optional["ConfigManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._config: dict = {}
        self._providers: dict[str, ProviderConfig] = {}
        self._manga_settings: Optional[MangaSettings] = None
        self._output_settings: Optional[OutputSettings] = None
        self._observer: Optional[Observer] = None
        self._callbacks: list = []
        self._loaded = False  # True after the first successful load: later failures keep that config
        self._source: Optional[Path] = None  # the file the active config was read from
        self.reload()

    def reload(self) -> bool:
        """重新加载配置文件（缺失时回退到 api_config.yaml.example，并给出警告而不是崩溃）

        Returns False when the file cannot be used; the first load raises instead, later loads keep the previous config.
        """
        try:
            path = API_CONFIG_PATH
            if not path.exists():
                if self._source == path:  # deleted, or renamed away mid-save: never swap in the template's placeholder key
                    raise FileNotFoundError(f"{path.name} is missing")
                example = CONFIG_DIR / "api_config.yaml.example"
                print(f"[ConfigManager] WARNING: {path} not found; falling back to {example.name}. "
                      f"Copy it to api_config.yaml and set your API key.")
                path = example
            with open(path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
            if raw_config is None:  # empty or comments only
                raw_config = {}
            if not isinstance(raw_config, dict):
                raise ValueError(f"{path.name} must contain a YAML mapping, not {type(raw_config).__name__}")
            if not raw_config and self._loaded:  # an editor truncates the file before writing it
                raise ValueError(f"{path.name} is empty (saved halfway?)")

            # 先全部解析到局部变量，成功后一次性替换（热更新失败时保留旧配置）
            config = expand_env_vars(raw_config)
            providers = {name: ProviderConfig.from_dict(name, data or {}) for name, data in (config.get("providers") or {}).items()}
            manga_settings = MangaSettings.from_dict(config.get("manga_settings") or {})
            output_settings = OutputSettings.from_dict(config.get("output") or {})

            self._config, self._providers = config, providers
            self._manga_settings, self._output_settings = manga_settings, output_settings
            self._loaded, self._source = True, path

            # 触发回调
            for callback in self._callbacks:
                try:
                    callback(self)
                except Exception as cb_error:  # noqa: BLE001
                    print(f"[ConfigManager] on_change callback failed: {cb_error}")
            return True

        except Exception as e:
            print(f"[ConfigManager] Failed to load config: {e}")
            if not self._loaded:  # first load: nothing to fall back to
                raise
            print("[ConfigManager] Keeping the previous configuration")
            return False

    def start_watching(self) -> None:
        """启动配置文件热更新监听"""
        if self._observer is not None:
            return

        self._observer = Observer()
        handler = ConfigChangeHandler(self.reload)
        self._observer.schedule(handler, str(CONFIG_DIR), recursive=False)
        self._observer.start()
        print(f"[ConfigManager] Watching config changes at {CONFIG_DIR}")

    def stop_watching(self) -> None:
        """停止配置文件监听"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

    def on_change(self, callback) -> None:
        """注册配置变更回调"""
        self._callbacks.append(callback)

    # ==================== 属性访问 ====================

    def _core(self) -> dict:
        core = self._config.get("core_engine")
        return core if isinstance(core, dict) else {}  # `core_engine:` with every line commented out is null

    @property
    def default_model(self) -> str:
        """获取默认模型"""
        return self._core().get("default_model", "nano-banana-pro")

    @property
    def fallback_text_model(self) -> str:
        """获取备用文本模型"""
        return self._core().get("fallback_text_model", "")

    @property
    def image_model(self) -> str:
        """获取图像生成模型"""
        return self._core().get("image_model", "google/gemini-3-pro-image-preview")

    @property
    def text_model(self) -> str:
        """分析 / 分镜文本模型（provider/model）"""
        return self._core().get("text_model", "")

    @property
    def fast_text_model(self) -> str:
        """翻译 / 改写用的快速文本模型（provider/model）"""
        return self._core().get("fast_text_model", "")

    @property
    def vision_model(self) -> str:
        """面板 QA 视觉模型（provider/model）"""
        return self._core().get("vision_model", "")

    @property
    def providers(self) -> dict[str, ProviderConfig]:
        """获取所有服务商配置"""
        return self._providers

    @property
    def manga_settings(self) -> MangaSettings:
        """获取漫画设置"""
        return self._manga_settings

    @property
    def output_settings(self) -> OutputSettings:
        """获取输出设置"""
        return self._output_settings

    @property
    def generate_api_enabled(self) -> bool:
        """是否开放 /api/generate/*（直接调用付费模型的原始文本 / 图像接口，界面不使用；默认关闭）"""
        server = self._config.get("server")
        return as_bool(server.get("enable_generate_api"), False) if isinstance(server, dict) else False

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """获取指定服务商配置"""
        return self._providers.get(name)

    def get_enabled_providers(self) -> list[ProviderConfig]:
        """获取所有已启用的服务商"""
        return [p for p in self._providers.values() if p.enabled]

    def get_raw_config(self) -> dict:
        """获取原始配置字典"""
        return self._config


# 全局配置实例
config = ConfigManager()


def get_config() -> ConfigManager:
    """获取配置管理器实例"""
    return config
