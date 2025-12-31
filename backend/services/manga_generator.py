"""
Manga Generator Service - 简化版
利用 Gemini 3 Pro Image 内置的 Chiikawa 知识

核心设计：
1. Gemini 内置 Chiikawa 知识，不需要复杂的角色 prompt
2. 每次只生成一个 panel，确保文字渲染准确
3. Gemini 直接在图片中渲染对白气泡
"""

import asyncio
import base64
import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines import get_client, ImageGenerationConfig, ImageContent
from config_loader import get_config
from services.storyboarder import Storyboard, Panel, PanelType, CharacterLibrary
from services.progress import set_stage, set_panel_progress, reset_progress


@dataclass
class GeneratedPanel:
    """生成的漫画格"""
    panel_number: int
    image_base64: str
    mime_type: str = "image/png"
    dialogue: Dict[str, str] = field(default_factory=dict)
    characters: List[str] = field(default_factory=list)
    width: int = 0
    height: int = 0


@dataclass
class GeneratedManga:
    """生成的完整漫画"""
    title: str
    panels: List[GeneratedPanel]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    character_theme: str = ""
    language: str = "zh-CN"

    def get_combined_image(self, layout: str = "vertical", render_dialogues: bool = False) -> bytes:
        """
        合并所有面板为一张长图

        Args:
            layout: 布局方式 ("vertical" 竖向, "grid" 网格)
            render_dialogues: 是否额外渲染对白（Gemini 已直接渲染，通常不需要）
        """
        if not self.panels:
            return b""

        images = []
        for panel in self.panels:
            img_data = base64.b64decode(panel.image_base64)
            img = Image.open(io.BytesIO(img_data))
            images.append(img)

        if layout == "vertical":
            return self._combine_vertical(images)
        else:
            return self._combine_grid(images)

    def _combine_vertical(self, images: List[Image.Image]) -> bytes:
        """垂直拼接图像"""
        max_width = max(img.width for img in images)
        gap = 20
        total_height = sum(img.height for img in images) + gap * (len(images) - 1)

        canvas = Image.new("RGB", (max_width, total_height), "white")

        y_offset = 0
        for img in images:
            x_offset = (max_width - img.width) // 2
            canvas.paste(img, (x_offset, y_offset))
            y_offset += img.height + gap

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG", quality=95)
        return buffer.getvalue()

    def _combine_grid(self, images: List[Image.Image], cols: int = 2) -> bytes:
        """网格拼接图像"""
        if not images:
            return b""

        rows = (len(images) + cols - 1) // cols
        cell_width = max(img.width for img in images)
        cell_height = max(img.height for img in images)
        gap = 10

        canvas_width = cell_width * cols + gap * (cols - 1)
        canvas_height = cell_height * rows + gap * (rows - 1)

        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = col * (cell_width + gap) + (cell_width - img.width) // 2
            y = row * (cell_height + gap) + (cell_height - img.height) // 2
            canvas.paste(img, (x, y))

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG", quality=95)
        return buffer.getvalue()

    def save(self, output_dir: Path) -> Path:
        """保存漫画到文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in self.title if c.isalnum() or c in "_ -")[:30]
        filename = f"{safe_title}_{timestamp}.png"

        combined = self.get_combined_image()
        output_path = output_dir / filename
        with open(output_path, "wb") as f:
            f.write(combined)

        return output_path


class MangaGenerator:
    """
    漫画生成器 - 简化版

    核心策略：
    1. 利用 Gemini 内置的 Chiikawa 知识
    2. 每次只生成一个 panel，确保文字准确
    3. Gemini 直接渲染对白气泡和文字
    """

    def __init__(self):
        self.config = get_config()
        self.output_dir = Path(__file__).parent.parent.parent / "output"
        self.output_dir.mkdir(exist_ok=True)
        self.char_lib = CharacterLibrary()

    async def generate_from_storyboard(
        self,
        storyboard: Storyboard,
        save_progress: bool = True
    ) -> GeneratedManga:
        """
        根据分镜脚本生成漫画

        每个 panel 单独生成，避免文字乱码
        """
        print(f"[MangaGenerator] Starting: '{storyboard.title}' with {len(storyboard.panels)} panels")

        # Report progress
        set_stage("generating", f"Starting manga generation")
        set_panel_progress(0, len(storyboard.panels))

        # 创建进度保存目录
        progress_dir = self.output_dir / "progress"
        progress_dir.mkdir(exist_ok=True)

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in storyboard.title if c.isalnum() or c in "_ -")[:20]

        generated_panels = []
        failed_count = 0

        # 逐个生成（关键：每次只生成一个 panel）
        for i, panel in enumerate(storyboard.panels):
            try:
                print(f"[MangaGenerator] Generating panel {panel.panel_number}/{len(storyboard.panels)}...")

                result = await self._generate_panel(panel, storyboard.language)
                generated_panels.append(result)

                # Update progress
                set_panel_progress(len(generated_panels), len(storyboard.panels))

                # 保存单个 panel
                if save_progress and result.image_base64:
                    panel_path = progress_dir / f"{safe_title}_{session_id}_panel{panel.panel_number:03d}.png"
                    try:
                        img_data = base64.b64decode(result.image_base64)
                        with open(panel_path, "wb") as f:
                            f.write(img_data)
                    except Exception as e:
                        print(f"[MangaGenerator] Failed to save: {e}")

                # 每 5 个 panel 保存一次合并图
                if save_progress and len(generated_panels) % 5 == 0:
                    await self._save_partial_manga(
                        storyboard, generated_panels, progress_dir, safe_title, session_id
                    )

            except Exception as e:
                failed_count += 1
                print(f"[MangaGenerator] Panel {panel.panel_number} failed: {e}")
                # 创建占位符
                generated_panels.append(self._create_placeholder_panel(panel, 1024, 1228))

        # 保存最终结果
        if save_progress and generated_panels:
            await self._save_partial_manga(
                storyboard, generated_panels, progress_dir, safe_title, session_id, is_final=True
            )

        print(f"[MangaGenerator] Completed: {len(generated_panels)} panels, {failed_count} failed")

        # Mark as completed
        set_stage("completed", f"Generated {len(generated_panels)} panels")

        return GeneratedManga(
            title=storyboard.title,
            panels=generated_panels,
            character_theme=storyboard.character_theme,
            language=storyboard.language
        )

    async def _generate_panel(self, panel: Panel, language: str, max_retries: int = 3) -> GeneratedPanel:
        """
        生成单个漫画格

        使用简化的 prompt，让 Gemini 使用内置的 Chiikawa 知识
        包含重试逻辑以处理网络错误
        """
        client = await get_client()
        output_settings = self.config.output_settings

        # 简化的 prompt - 利用 Gemini 内置知识
        prompt = self._build_simple_prompt(panel, language)

        width, height = self._get_panel_dimensions(
            panel.layout_hint,
            output_settings.max_width,
            output_settings.max_height
        )

        negative_prompt = getattr(self.config.manga_settings, 'negative_prompt', None)
        if not negative_prompt:
            negative_prompt = "photorealistic, 3d render, anime style, complex shading, multiple panels"

        config = ImageGenerationConfig(
            width=width,
            height=height,
            style=self.config.manga_settings.default_style,
            negative_prompt=negative_prompt
        )

        # 加载参考图片（可选，Gemini 已有内置知识）
        reference_images = self._load_reference_images(panel)
        if reference_images:
            print(f"[MangaGenerator] Using {len(reference_images)} reference images")

        # 带重试的生成逻辑
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await client.generate_image(prompt, config, reference_images=reference_images)

                if response.images:
                    image = response.images[0]
                    print(f"[MangaGenerator] Panel {panel.panel_number} generated")
                    return GeneratedPanel(
                        panel_number=panel.panel_number,
                        image_base64=image.data,
                        mime_type=image.mime_type,
                        dialogue=panel.dialogue,
                        characters=panel.characters,
                        width=width,
                        height=height
                    )

            except Exception as e:
                last_error = e
                retry_msg = f" (attempt {attempt + 1}/{max_retries})" if attempt < max_retries - 1 else ""
                print(f"[MangaGenerator] Generation failed{retry_msg}: {e}")

                # 如果还有重试机会，等待后重试
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避: 1s, 2s, 4s
                    print(f"[MangaGenerator] Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)

        # 所有重试都失败了
        print(f"[MangaGenerator] All {max_retries} attempts failed for panel {panel.panel_number}")
        return self._create_placeholder_panel(panel, width, height)

    def _build_simple_prompt(self, panel: Panel, language: str) -> str:
        """
        构建简化的图像生成 prompt

        关键：让 Gemini 使用它内置的 Chiikawa 知识
        """
        lang_map = {"zh-CN": "中文", "en-US": "English", "ja-JP": "日本語"}
        lang_name = lang_map.get(language, "中文")

        # 构建对白部分
        dialogue_text = ""
        if panel.dialogue:
            dialogue_lines = []
            for char, text in panel.dialogue.items():
                dialogue_lines.append(f"- {char}: \"{text}\"")
            dialogue_text = "\n".join(dialogue_lines)

        # 构建角色列表
        chars = ", ".join(panel.characters) if panel.characters else "Chiikawa and Hachiware"

        prompt = f"""生成一幅 Chiikawa (ちいかわ) 风格的漫画单格。

## 场景
{panel.visual_description}

## 角色
{chars}

## 对白（请在图中用气泡渲染，语言：{lang_name}）
{dialogue_text if dialogue_text else "无对白"}

## 风格要求
- Nagano/Chiikawa 官方画风
- 粗黑线条、柔和粉彩色
- 简单圆润的角色造型
- 清晰易读的对白气泡

生成单幅漫画图像，包含角色和对白气泡。"""

        return prompt

    def _load_reference_images(self, panel: Panel) -> List[ImageContent]:
        """加载参考图片（可选）"""
        reference_images = []

        image_paths = self.char_lib.get_all_reference_images_for_panel(
            panel.characters,
            panel.character_emotions
        )

        for img_path in image_paths[:4]:  # 限制数量
            try:
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()

                ext = Path(img_path).suffix.lower()
                mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
                mime_type = mime_map.get(ext, "image/png")

                reference_images.append(ImageContent(
                    data=img_data,
                    mime_type=mime_type,
                    is_base64=True
                ))
            except Exception as e:
                print(f"[MangaGenerator] Failed to load ref image: {e}")

        return reference_images

    def _get_panel_dimensions(self, layout_hint: str, max_width: int, max_height: int) -> tuple:
        """根据布局提示确定面板尺寸"""
        if layout_hint == "wide":
            return max_width, max_height // 2
        elif layout_hint == "tall":
            return max_width // 2, max_height
        else:
            return max_width, int(max_width * 1.2)

    def _create_placeholder_panel(self, panel: Panel, width: int, height: int) -> GeneratedPanel:
        """创建占位符面板"""
        img = Image.new("RGB", (width, height), "#f5f5f5")
        draw = ImageDraw.Draw(img)
        draw.rectangle([5, 5, width-5, height-5], outline="#cccccc", width=2)

        text = f"Panel {panel.panel_number}"
        if panel.characters:
            text += f"\n{', '.join(panel.characters)}"

        draw.text((width//2, height//2), text, fill="#999999", anchor="mm")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return GeneratedPanel(
            panel_number=panel.panel_number,
            image_base64=img_base64,
            dialogue=panel.dialogue,
            characters=panel.characters,
            width=width,
            height=height
        )

    async def _save_partial_manga(
        self,
        storyboard: Storyboard,
        panels: List[GeneratedPanel],
        progress_dir: Path,
        safe_title: str,
        session_id: str,
        is_final: bool = False
    ):
        """保存部分生成的漫画"""
        try:
            partial_manga = GeneratedManga(
                title=storyboard.title,
                panels=panels,
                character_theme=storyboard.character_theme,
                language=storyboard.language
            )

            suffix = "final" if is_final else f"partial_{len(panels)}"
            filename = f"{safe_title}_{session_id}_{suffix}.png"
            output_path = progress_dir / filename

            combined = partial_manga.get_combined_image()
            with open(output_path, "wb") as f:
                f.write(combined)

            print(f"[MangaGenerator] Saved {'final' if is_final else 'partial'}: {output_path.name}")

            # 保存元数据
            meta_path = progress_dir / f"{safe_title}_{session_id}_meta.json"
            meta = {
                "title": storyboard.title,
                "total_panels": len(storyboard.panels),
                "generated_panels": len(panels),
                "session_id": session_id,
                "is_complete": is_final and len(panels) == len(storyboard.panels)
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[MangaGenerator] Failed to save partial: {e}")


# 全局实例
_generator: Optional[MangaGenerator] = None


def get_manga_generator() -> MangaGenerator:
    """获取漫画生成器实例"""
    global _generator
    if _generator is None:
        _generator = MangaGenerator()
    return _generator


def reset_manga_generator() -> None:
    """重置漫画生成器"""
    global _generator
    _generator = None
