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
        # 安全的文件名：保留中文字符
        safe_title = "".join(c for c in self.title if c.isalnum() or c in "_ -" or '\u4e00' <= c <= '\u9fff')[:30]
        if not safe_title:
            safe_title = "manga"
        filename = f"{safe_title}_{timestamp}.png"

        combined = self.get_combined_image()
        output_path = output_dir / filename
        with open(output_path, "wb") as f:
            f.write(combined)

        return output_path


class MangaGenerator:
    """
    漫画生成器 - 批量版

    核心策略：
    1. 利用 Gemini 内置的 Chiikawa 知识
    2. 每次生成4个 panel（2x2网格），提高效率
    3. Gemini 直接渲染对白气泡和文字
    """

    def __init__(self):
        self.config = get_config()
        self.output_dir = Path(__file__).parent.parent.parent / "output"
        self.output_dir.mkdir(exist_ok=True)
        self.char_lib = CharacterLibrary()
        self.panels_per_batch = 4  # 每次生成4个panel

    async def generate_from_storyboard(
        self,
        storyboard: Storyboard,
        save_progress: bool = True
    ) -> GeneratedManga:
        """
        根据分镜脚本生成漫画

        每次生成4个panel（2x2网格），提高效率
        """
        print(f"[MangaGenerator] Starting: '{storyboard.title}' with {len(storyboard.panels)} panels (theme: {storyboard.character_theme})")

        # 存储当前主题用于生成
        self.current_theme = storyboard.character_theme

        # Report progress
        set_stage("generating", f"Starting manga generation")
        set_panel_progress(0, len(storyboard.panels))

        # 创建以 PDF 标题命名的子文件夹
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 安全的文件夹名：保留中文字符，移除特殊字符
        safe_title = "".join(c for c in storyboard.title if c.isalnum() or c in "_ -" or '\u4e00' <= c <= '\u9fff')[:50]
        if not safe_title:
            safe_title = "manga"

        # 每次生成创建独立的子文件夹
        manga_folder = self.output_dir / f"{safe_title}_{session_id}"
        manga_folder.mkdir(parents=True, exist_ok=True)
        progress_dir = manga_folder  # 所有文件保存在这个文件夹中

        print(f"[MangaGenerator] Output folder: {manga_folder}")

        generated_panels = []
        failed_count = 0

        # 动态批量生成
        total_panels = len(storyboard.panels)
        is_cjk = storyboard.language in ["zh-CN", "ja-JP"]

        # 使用动态批次大小
        panel_index = 0
        batch_num = 0

        while panel_index < total_panels:
            # 计算这批的最佳大小
            batch_size = self._calculate_optimal_batch_size(
                storyboard.panels[panel_index:panel_index + 4],
                is_cjk
            )
            batch_end = min(panel_index + batch_size, total_panels)
            batch_panels = storyboard.panels[panel_index:batch_end]
            batch_num += 1

            try:
                print(f"[MangaGenerator] Generating batch {batch_num}: panels {panel_index+1}-{batch_end}/{total_panels} (batch_size={len(batch_panels)})...")

                result = await self._generate_panel_batch(batch_panels, storyboard.language)
                generated_panels.append(result)

                # Update progress
                set_panel_progress(batch_end, total_panels)

                # 保存批次图像
                if save_progress and result.image_base64:
                    panel_path = progress_dir / f"{safe_title}_{session_id}_batch{batch_num:03d}.png"
                    try:
                        img_data = base64.b64decode(result.image_base64)
                        with open(panel_path, "wb") as f:
                            f.write(img_data)
                    except Exception as e:
                        print(f"[MangaGenerator] Failed to save: {e}")

                # 每 3 个批次保存一次合并图
                if save_progress and len(generated_panels) % 3 == 0:
                    await self._save_partial_manga(
                        storyboard, generated_panels, progress_dir, safe_title, session_id
                    )

            except Exception as e:
                failed_count += 1
                print(f"[MangaGenerator] Batch {batch_num} failed: {e}")
                # 创建占位符
                width, height = self._get_batch_dimensions(len(batch_panels))
                generated_panels.append(self._create_placeholder_batch(batch_panels, width, height))

            # 更新索引到下一批
            panel_index = batch_end

        # 保存最终结果
        if save_progress and generated_panels:
            await self._save_partial_manga(
                storyboard, generated_panels, progress_dir, safe_title, session_id, is_final=True
            )

        print(f"[MangaGenerator] Completed: {batch_num} batches ({total_panels} panels), {failed_count} failed")

        # Mark as completed
        set_stage("completed", f"Generated {total_panels} panels in {len(generated_panels)} batches")

        return GeneratedManga(
            title=storyboard.title,
            panels=generated_panels,
            character_theme=storyboard.character_theme,
            language=storyboard.language
        )

    async def _generate_panel(self, panel: Panel, language: str, max_retries: int = 3) -> GeneratedPanel:
        """
        生成单个漫画格

        使用简化的 prompt，让 Gemini 使用内置知识
        包含重试逻辑以处理网络错误
        """
        client = await get_client()
        output_settings = self.config.output_settings

        # 获取当前主题
        theme = getattr(self, 'current_theme', 'chiikawa')

        # 简化的 prompt - 利用 Gemini 内置知识
        prompt = self._build_simple_prompt(panel, language, theme)

        width, height = self._get_panel_dimensions(
            panel.layout_hint,
            output_settings.max_width,
            output_settings.max_height
        )

        negative_prompt = getattr(self.config.manga_settings, 'negative_prompt', None)
        if not negative_prompt:
            negative_prompt = "photorealistic, 3d render, anime style, complex shading, multiple panels"
        # 添加角色一致性相关的负面提示
        negative_prompt += ", inconsistent characters, characters not matching reference images, different colors from reference, wrong character proportions"

        config = ImageGenerationConfig(
            width=width,
            height=height,
            style=self.config.manga_settings.default_style,
            negative_prompt=negative_prompt,
            temperature=0.3  # 低温度确保角色一致性
        )

        # 加载参考图片
        # chibikawa 主题必须加载所有原创角色的参考图片以保持一致性
        if theme == "chibikawa":
            reference_images = self._load_all_chibikawa_references()
        else:
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

    async def _generate_panel_batch(self, panels: List[Panel], language: str, max_retries: int = 3) -> GeneratedPanel:
        """
        批量生成多个漫画格

        根据面板数量使用不同布局：
        - 1 panel: 单张图 1024x1024
        - 2 panels: 横向排列 2048x1024
        - 3-4 panels: 2x2网格 2048x2048
        """
        client = await get_client()
        output_settings = self.config.output_settings

        # 获取当前主题
        theme = getattr(self, 'current_theme', 'chiikawa')

        # 根据面板数量确定尺寸
        width, height = self._get_batch_dimensions(len(panels))

        # 构建批量 prompt
        prompt = self._build_batch_prompt(panels, language, theme)

        negative_prompt = getattr(self.config.manga_settings, 'negative_prompt', None)
        if not negative_prompt:
            negative_prompt = "photorealistic, 3d render, anime style, complex shading, blurry, messy lines"
        # 添加角色一致性相关的负面提示
        negative_prompt += ", inconsistent characters, characters not matching reference images, changing character designs between panels, different colors from reference, wrong character proportions"

        config = ImageGenerationConfig(
            width=width,
            height=height,
            style=self.config.manga_settings.default_style,
            negative_prompt=negative_prompt,
            temperature=0.3  # 低温度确保角色一致性
        )

        # 加载参考图片
        reference_images = []
        # chibikawa 主题必须加载所有原创角色的参考图片以保持一致性
        if theme == "chibikawa":
            reference_images = self._load_all_chibikawa_references()
        else:
            for panel in panels[:1]:  # 只用第一个panel的参考图
                reference_images.extend(self._load_reference_images(panel))
        if reference_images:
            print(f"[MangaGenerator] Using {len(reference_images)} reference images")

        # 带重试的生成逻辑
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await client.generate_image(prompt, config, reference_images=reference_images)

                if response.images:
                    image = response.images[0]
                    print(f"[MangaGenerator] Batch generated ({len(panels)} panels)")
                    return GeneratedPanel(
                        panel_number=panels[0].panel_number,
                        image_base64=image.data,
                        mime_type=image.mime_type,
                        dialogue={},  # 批量模式不单独存储对白
                        characters=[],
                        width=width,
                        height=height
                    )

            except Exception as e:
                last_error = e
                retry_msg = f" (attempt {attempt + 1}/{max_retries})" if attempt < max_retries - 1 else ""
                print(f"[MangaGenerator] Batch generation failed{retry_msg}: {e}")

                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[MangaGenerator] Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)

        print(f"[MangaGenerator] All {max_retries} attempts failed for batch")
        return self._create_placeholder_batch(panels, width, height)

    def _calculate_optimal_batch_size(self, panels: List[Panel], is_cjk: bool) -> int:
        """
        根据面板文本长度动态计算最佳批次大小

        长对话时减少每批面板数量，以保证文字清晰可读
        """
        if not panels:
            return 1

        # 计算所有面板中最长的单个对白和总对白长度
        max_single_dialogue = 0
        total_dialogue_len = 0

        for panel in panels:
            if panel.dialogue:
                for char, text in panel.dialogue.items():
                    max_single_dialogue = max(max_single_dialogue, len(text))
                    total_dialogue_len += len(text)

        print(f"[MangaGenerator] Dialogue analysis: max_single={max_single_dialogue}, total={total_dialogue_len}, is_cjk={is_cjk}")

        # CJK 语言的阈值（更严格，确保文字清晰）
        if is_cjk:
            # 如果单个对白超过 150 字，只生成 1 个面板
            if max_single_dialogue > 150:
                print(f"[MangaGenerator] Long CJK dialogue ({max_single_dialogue} chars), using 1 panel")
                return 1
            # 如果单个对白超过 80 字，只生成 2 个面板
            elif max_single_dialogue > 80:
                print(f"[MangaGenerator] Medium CJK dialogue ({max_single_dialogue} chars), using 2 panels")
                return min(2, len(panels))
            # 否则生成 4 个面板
            else:
                return min(4, len(panels))
        else:
            # 英文：也根据对话长度调整
            # 如果单个对白超过 300 字符，只生成 1 个面板
            if max_single_dialogue > 300:
                print(f"[MangaGenerator] Long English dialogue ({max_single_dialogue} chars), using 1 panel")
                return 1
            # 如果单个对白超过 150 字符，只生成 2 个面板
            elif max_single_dialogue > 150:
                print(f"[MangaGenerator] Medium English dialogue ({max_single_dialogue} chars), using 2 panels")
                return min(2, len(panels))
            # 否则生成 4 个面板
            else:
                return min(4, len(panels))

    def _get_batch_dimensions(self, batch_size: int) -> tuple:
        """
        根据批次大小返回图像尺寸

        - 1 panel: 1024x1024
        - 2 panels: 2048x1024 (横向排列)
        - 3-4 panels: 2048x2048 (2x2网格)
        """
        if batch_size == 1:
            return (1024, 1024)
        elif batch_size == 2:
            return (2048, 1024)
        else:
            return (2048, 2048)

    def _build_batch_prompt(self, panels: List[Panel], language: str, theme: str = "chiikawa") -> str:
        """
        构建批量图像生成 prompt

        根据面板数量使用不同布局：
        - 1 panel: 单张图
        - 2 panels: 横向排列 (1x2)
        - 3-4 panels: 2x2网格
        """
        lang_map = {"zh-CN": "中文", "en-US": "English", "ja-JP": "日本語"}
        lang_name = lang_map.get(language, "中文")
        is_cjk = language in ["zh-CN", "ja-JP"]
        num_panels = len(panels)

        # 根据面板数量选择布局和位置标签
        if num_panels == 1:
            layout_desc = "single manga panel"
            positions = [""]
        elif num_panels == 2:
            layout_desc = "1x2 horizontal manga layout"
            positions = ["Left", "Right"]
        else:
            layout_desc = "2x2 manga grid"
            positions = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]

        # 构建面板描述 - 不截断文字
        panel_lines = []
        for i, panel in enumerate(panels):
            pos = positions[i] if i < len(positions) else f"Panel {i+1}"
            chars = ", ".join(panel.characters) if panel.characters else ""

            # 完整对白，不截断
            dialogues = []
            if panel.dialogue:
                for char, text in panel.dialogue.items():
                    dialogues.append(f'{char}: "{text}"')
            dialogue_str = " | ".join(dialogues) if dialogues else ""

            if pos:
                panel_lines.append(f"{pos}: {chars} - {dialogue_str}")
            else:
                panel_lines.append(f"Characters: {chars}\nDialogue: {dialogue_str}")

        panels_text = "\n".join(panel_lines)

        # 简洁的风格描述
        if theme == "ghibli":
            style = "Ghibli (Spirited Away style, Haku in HUMAN form only)"
        elif theme == "chibikawa":
            style = "Chibikawa (ORIGINAL cute characters - draw EXACTLY as shown in reference images)"
        else:
            style = "Chiikawa (Nagano style)"

        # 根据面板数量和语言调整文字说明
        if num_panels == 1:
            if is_cjk:
                text_note = f"Text: Render {lang_name} dialogue in speech bubbles. Use LARGE, CLEAR font. Make text easily readable."
            else:
                text_note = f"Text: Clear speech bubbles in {lang_name}. Render text CLEARLY."
        else:
            if is_cjk:
                text_note = f"Text: {lang_name} dialogue in speech bubbles. Use LARGE, CLEAR font for readability."
            else:
                text_note = f"Text: Clear speech bubbles in {lang_name}. Render text LEGIBLY."

        # Chibikawa 原创角色描述 - 明确标注每张图片对应哪个角色（顺序必须和加载顺序一致）
        chibikawa_char_desc = ""
        if theme == "chibikawa":
            # 图片加载顺序: kumo.jpeg, nezu.jpeg, papi.jpeg (按 CHIBIKAWA_IMAGES 字典顺序)
            chibikawa_char_desc = """⚠️ REFERENCE IMAGES PROVIDED - STRICT MAPPING ⚠️

I am attaching 3 reference images in this exact order:
• Image 1: kumo (the curious student)
• Image 2: nezu (the skeptic)
• Image 3: papi (the mentor/professor)

You MUST draw each character EXACTLY as shown in their corresponding reference image.
"""

        # 根据布局生成不同的 prompt - 把角色描述放在最前面
        if num_panels == 1:
            if theme == "chibikawa":
                prompt = f"""{chibikawa_char_desc}
---
TASK: Create a {layout_desc} in {style} style.

Audience: Nobel Prize scholars who LOVE cute characters.
Balance: Academic rigor + cute charm.

Panel Content:
{panels_text}

Background: Simple classroom/lab.
{text_note}

⚠️ REMINDER: Character appearance MUST match the reference images exactly. Do not improvise."""
            else:
                prompt = f"""{layout_desc}, {style}.

Audience: Nobel Prize scholars who LOVE {style.split()[0]} characters.
Balance: Academic rigor + cute charm.

{panels_text}

Background: Simple classroom/lab.
{text_note}"""
        else:
            if theme == "chibikawa":
                prompt = f"""{chibikawa_char_desc}
---
TASK: Create a {layout_desc} in {style} style.

Audience: Nobel Prize scholars who LOVE cute characters.
Balance: Academic rigor + cute charm.

Background: Simple classroom/lab (CONSISTENT across all panels!)
{text_note}

Panel Layout:
{panels_text}

Generate {layout_desc} with black borders between panels.

⚠️ REMINDER: Character appearance MUST match the reference images exactly in EVERY panel. Do not improvise."""
            else:
                prompt = f"""{layout_desc}, {style}.

Audience: Nobel Prize scholars who LOVE {style.split()[0]} characters.
Balance: Academic rigor + cute charm.

Background: Simple classroom/lab (CONSISTENT across all panels!)
{text_note}

Panels:
{panels_text}

Generate {layout_desc} with black borders between panels."""

        return prompt

    def _create_placeholder_batch(self, panels: List[Panel], width: int, height: int) -> GeneratedPanel:
        """创建批量占位符，支持不同布局"""
        img = Image.new("RGB", (width, height), "#f5f5f5")
        draw = ImageDraw.Draw(img)
        num_panels = len(panels)

        draw.rectangle([5, 5, width-5, height-5], outline="#cccccc", width=2)

        if num_panels == 1:
            # 单张图
            positions = [(width // 2, height // 2)]
        elif num_panels == 2:
            # 横向排列
            mid_x = width // 2
            draw.line([(mid_x, 0), (mid_x, height)], fill="#cccccc", width=2)
            positions = [(mid_x // 2, height // 2), (mid_x + mid_x // 2, height // 2)]
        else:
            # 2x2网格
            mid_x, mid_y = width // 2, height // 2
            draw.line([(mid_x, 0), (mid_x, height)], fill="#cccccc", width=2)
            draw.line([(0, mid_y), (width, mid_y)], fill="#cccccc", width=2)
            positions = [(mid_x//2, mid_y//2), (mid_x + mid_x//2, mid_y//2),
                         (mid_x//2, mid_y + mid_y//2), (mid_x + mid_x//2, mid_y + mid_y//2)]

        for i, panel in enumerate(panels):
            if i < len(positions):
                x, y = positions[i]
                text = f"Panel {panel.panel_number}"
                draw.text((x, y), text, fill="#999999", anchor="mm")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return GeneratedPanel(
            panel_number=panels[0].panel_number if panels else 0,
            image_base64=img_base64,
            dialogue={},
            characters=[],
            width=width,
            height=height
        )

    def _build_simple_prompt(self, panel: Panel, language: str, theme: str = "chiikawa") -> str:
        """
        构建简化的图像生成 prompt（单个panel，备用）
        简洁版 - 现代模型足够聪明
        """
        lang_map = {"zh-CN": "中文", "en-US": "English", "ja-JP": "日本語"}
        lang_name = lang_map.get(language, "中文")
        is_cjk = language in ["zh-CN", "ja-JP"]

        # 对白 - 不截断，完整输出
        dialogues = []
        if panel.dialogue:
            for char, text in panel.dialogue.items():
                dialogues.append(f'{char}: "{text}"')
        dialogue_str = " | ".join(dialogues) if dialogues else "(no dialogue)"

        chars = ", ".join(panel.characters) if panel.characters else ""
        if theme == "ghibli":
            style = "Ghibli (Haku in HUMAN form)"
        elif theme == "chibikawa":
            style = "Chibikawa (ORIGINAL characters - match reference images EXACTLY)"
        else:
            style = "Chiikawa"

        # CJK 语言强调文字清晰
        if is_cjk:
            text_note = f"Render {lang_name} text LARGE and CLEAR in speech bubbles."
        else:
            text_note = "Render text CLEARLY in speech bubbles."

        # Chibikawa 原创角色描述 - 明确标注图片顺序
        chibikawa_char_desc = ""
        if theme == "chibikawa":
            # 图片加载顺序: kumo.jpeg, nezu.jpeg, papi.jpeg
            chibikawa_char_desc = """⚠️ REFERENCE IMAGES - STRICT MAPPING ⚠️

Image 1: kumo | Image 2: nezu | Image 3: papi
Draw each character EXACTLY as shown in their reference image.
"""

        if theme == "chibikawa":
            prompt = f"""{chibikawa_char_desc}
---
TASK: Single manga panel in {style} style.

Characters: {chars}
Dialogue ({lang_name}): {dialogue_str}
Background: Simple classroom/lab.

{text_note}

⚠️ REMINDER: Character designs MUST match reference images exactly."""
        else:
            prompt = f"""Single manga panel, {style} style.

Characters: {chars}
Dialogue ({lang_name}): {dialogue_str}
Background: Simple classroom/lab.

{text_note}"""

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

    def _load_all_chibikawa_references(self) -> List[ImageContent]:
        """
        加载所有 chibikawa 原创角色的参考图片
        这是确保角色一致性的关键 - 每次图像生成都必须包含这些参考图
        """
        reference_images = []

        image_paths = self.char_lib.get_all_chibikawa_reference_images()
        print(f"[MangaGenerator] Loading {len(image_paths)} chibikawa reference images")

        for img_path in image_paths:
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
                print(f"[MangaGenerator] Loaded reference: {Path(img_path).name}")
            except Exception as e:
                print(f"[MangaGenerator] Failed to load chibikawa ref image {img_path}: {e}")

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
