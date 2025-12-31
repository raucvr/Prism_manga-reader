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
            enabled=data.get("enabled", False),
            api_key=data.get("api_key", ""),
            base_url=data.get("base_url", ""),
            models=data.get("models", []) if isinstance(data.get("models"), list) else [data.get("model", "")]
        )


@dataclass
class MangaSettings:
    """漫画生成设置"""
    default_style: str = "full_color_manga"
    aspect_ratio: str = "2:3"
    render_text_in_image: bool = True
    panels_per_page: int = 4
    default_character: str = "chiikawa"

    @classmethod
    def from_dict(cls, data: dict) -> "MangaSettings":
        return cls(
            default_style=data.get("default_style", "full_color_manga"),
            aspect_ratio=data.get("aspect_ratio", "2:3"),
            render_text_in_image=data.get("render_text_in_image", True),
            panels_per_page=data.get("panels_per_page", 4),
            default_character=data.get("default_character", "chiikawa")
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
            image_format=data.get("image_format", "png"),
            image_quality=data.get("image_quality", 95),
            max_width=data.get("max_width", 1024),
            max_height=data.get("max_height", 1536)
        )


class ConfigChangeHandler(FileSystemEventHandler):
    """配置文件变更监听器"""
    def __init__(self, callback):
        self.callback = callback

    def on_modified(self, event):
        if event.src_path.endswith("api_config.yaml"):
            self.callback()


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
        self.reload()

    def reload(self) -> None:
        """重新加载配置文件"""
        try:
            with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)

            # 展开环境变量
            self._config = expand_env_vars(raw_config)

            # 解析 providers
            self._providers = {}
            for name, data in self._config.get("providers", {}).items():
                self._providers[name] = ProviderConfig.from_dict(name, data)

            # 解析 manga_settings
            self._manga_settings = MangaSettings.from_dict(
                self._config.get("manga_settings", {})
            )

            # 解析 output settings
            self._output_settings = OutputSettings.from_dict(
                self._config.get("output", {})
            )

            # 触发回调
            for callback in self._callbacks:
                callback(self)

        except Exception as e:
            print(f"[ConfigManager] Failed to load config: {e}")
            raise

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

    @property
    def default_model(self) -> str:
        """获取默认模型"""
        return self._config.get("core_engine", {}).get("default_model", "nano-banana-pro")

    @property
    def fallback_text_model(self) -> str:
        """获取备用文本模型"""
        return self._config.get("core_engine", {}).get("fallback_text_model", "")

    @property
    def image_model(self) -> str:
        """获取图像生成模型"""
        return self._config.get("core_engine", {}).get("image_model", "google/gemini-3-pro-image-preview")

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
