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

from engines import get_client, ImageGenerationConfig, ImageContent, GenerationConfig
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
        # 动态构建角色名映射 - 从 character_images 目录加载
        self.kumomo_char_map = {}
        for char_name in self.char_lib.get_kumomo_character_names():
            self.kumomo_char_map[char_name] = char_name

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

                # 保存批次图像（每个批次单独备份）
                if save_progress and result.image_base64:
                    panel_path = progress_dir / f"{safe_title}_{session_id}_batch{batch_num:03d}.png"
                    try:
                        img_data = base64.b64decode(result.image_base64)
                        with open(panel_path, "wb") as f:
                            f.write(img_data)
                    except Exception as e:
                        print(f"[MangaGenerator] Failed to save: {e}")

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
            await self._save_final_manga(
                storyboard, generated_panels, progress_dir, safe_title, session_id
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

    async def _generate_panel_batch(self, panels: List[Panel], language: str, max_retries: int = 5) -> GeneratedPanel:
        """
        批量生成多个漫画格（带验证和强化纠错）

        流程：
        1. 生成图像
        2. 验证角色/剧情是否符合剧本（kumomo 主题必须验证）
        3. 如果不通过，带详细反馈重新生成
        4. 最多尝试 max_retries 次，每次都验证
        """
        client = await get_client()

        # 获取当前主题
        theme = getattr(self, 'current_theme', 'chiikawa')

        # 根据面板数量确定尺寸
        width, height = self._get_batch_dimensions(len(panels))

        # 收集此批次所有出现的角色 (用于动态加载参考图)
        batch_characters = set()
        if theme == "kumomo":
            for p in panels:
                for c in p.characters:
                    norm_c = c.lower().strip()
                    # 尝试映射到标准名称 kumo/nezu/papi
                    for key, val in self.kumomo_char_map.items():
                        if key in norm_c:
                            batch_characters.add(val)
                            break
            print(f"[MangaGenerator] Active characters in batch: {batch_characters}")

        # 构建基础 prompt (传入 batch_characters)
        base_prompt = self._build_batch_prompt(panels, language, theme, batch_characters)

        negative_prompt = getattr(self.config.manga_settings, 'negative_prompt', None)
        if not negative_prompt:
            negative_prompt = "photorealistic, 3d render, anime style, complex shading, blurry, messy lines"
        # 强化负面提示 - 明确禁止 Chiikawa 角色特征
        negative_prompt += ", cat ears, rabbit ears, chiikawa, hachiware, usagi, inconsistent characters"

        config = ImageGenerationConfig(
            width=width,
            height=height,
            style=self.config.manga_settings.default_style,
            negative_prompt=negative_prompt,
            temperature=0.2  # 低温度确保角色一致性
        )

        # 加载参考图片 (动态: 只加载批次需要的角色)
        reference_images = []
        if theme == "kumomo":
            chars_to_load = list(batch_characters) if batch_characters else self.char_lib.get_kumomo_character_names()
            reference_images = self._load_specific_kumomo_references(chars_to_load)
        else:
            for panel in panels[:1]:
                reference_images.extend(self._load_reference_images(panel))
        if reference_images:
            print(f"[MangaGenerator] Using {len(reference_images)} reference images")

        # 生成 + 验证循环（强化纠错）
        validation_feedback = ""
        last_valid_image = None  # 保存最后一次通过验证或生成的图像

        for attempt in range(max_retries):
            try:
                # 更新进度状态，让前端知道正在做什么
                if theme == "kumomo" and attempt > 0:
                    set_stage("generating", f"Validating characters (retry {attempt}/{max_retries})")

                # 构建 prompt（如果有验证反馈则加入更强的纠错指令）
                if validation_feedback:
                    prompt = f"""RETRY: {validation_feedback}
{base_prompt}"""
                    print(f"[MangaGenerator] Attempt {attempt+1}/{max_retries}: Regenerating with feedback: {validation_feedback}")
                else:
                    prompt = base_prompt
                    print(f"[MangaGenerator] Attempt {attempt+1}/{max_retries}: Generating...")

                # 生成图像
                response = await client.generate_image(prompt, config, reference_images=reference_images)

                if response.images:
                    image = response.images[0]
                    print(f"[MangaGenerator] Batch generated ({len(panels)} panels)")

                    # kumomo 主题: 每次都验证，确保角色正确
                    if theme == "kumomo":
                        set_stage("generating", f"Validating character consistency...")
                        generated_img = ImageContent(
                            data=image.data,
                            mime_type=image.mime_type,
                            is_base64=True
                        )
                        is_valid, feedback = await self._validate_generated_image(
                            generated_img, panels, reference_images, theme
                        )

                        if is_valid:
                            print(f"[MangaGenerator] ✓ Validation PASSED on attempt {attempt+1}")
                            return GeneratedPanel(
                                panel_number=panels[0].panel_number,
                                image_base64=image.data,
                                mime_type=image.mime_type,
                                dialogue={},
                                characters=[],
                                width=width,
                                height=height
                            )
                        else:
                            print(f"[MangaGenerator] ✗ Validation FAILED on attempt {attempt+1}: {feedback}")
                            validation_feedback = feedback
                            last_valid_image = image  # 保存这次生成的图像以防万一

                            # 如果是最后一次尝试，返回最后生成的图像（虽然未通过验证）
                            if attempt == max_retries - 1:
                                print(f"[MangaGenerator] ⚠ All {max_retries} attempts failed validation, returning last generated image")
                                return GeneratedPanel(
                                    panel_number=panels[0].panel_number,
                                    image_base64=image.data,
                                    mime_type=image.mime_type,
                                    dialogue={},
                                    characters=[],
                                    width=width,
                                    height=height
                                )
                            continue  # 重新生成
                    else:
                        # 非 kumomo 主题直接返回
                        return GeneratedPanel(
                            panel_number=panels[0].panel_number,
                            image_base64=image.data,
                            mime_type=image.mime_type,
                            dialogue={},
                            characters=[],
                            width=width,
                            height=height
                        )

            except Exception as e:
                print(f"[MangaGenerator] Attempt {attempt+1}/{max_retries} failed with error: {e}")

                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 8)  # 最多等待 8 秒
                    print(f"[MangaGenerator] Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)

        # 所有尝试都失败了
        print(f"[MangaGenerator] All {max_retries} attempts failed for batch")

        # 如果有之前生成的图像（未通过验证），仍然返回它
        if last_valid_image:
            print(f"[MangaGenerator] Returning last generated image despite validation failure")
            return GeneratedPanel(
                panel_number=panels[0].panel_number,
                image_base64=last_valid_image.data,
                mime_type=last_valid_image.mime_type,
                dialogue={},
                characters=[],
                width=width,
                height=height
            )

        return self._create_placeholder_batch(panels, width, height)

    def _calculate_optimal_batch_size(self, panels: List[Panel], is_cjk: bool) -> int:
        """
        固定返回 4 个面板的批次大小

        对话长度在分镜生成阶段控制（prompt限制单句50字符以内）
        """
        if not panels:
            return 1

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

    async def _validate_generated_image(
        self,
        generated_image: ImageContent,
        panels: List[Panel],
        reference_images: List[ImageContent],
        theme: str
    ) -> tuple[bool, str]:
        """
        验证生成的漫画是否符合要求（思维链 CoT 版）

        核心策略：先让模型描述看到的内容，再判断是否匹配
        这样可以避免模型直接说"PASS"而没有真正检查

        Returns:
            (is_valid, feedback) - 是否通过验证，以及反馈信息
        """
        # 只对 kumomo 主题进行验证（原创角色需要严格一致）
        if theme != "kumomo":
            return True, ""

        client = await get_client()

        # 动态生成验证提示词
        char_names = self.char_lib.get_kumomo_character_names()
        num_chars = len(char_names)

        # 构建角色映射说明
        image_mapping = "\n".join([f"Image {i+1} = {name}'s design" for i, name in enumerate(char_names)])
        last_img_num = num_chars + 1

        validate_prompt = f"""Look at the reference character designs (Image 1-{num_chars}) and the generated manga (Image {last_img_num}).

{image_mapping}
Image {last_img_num} = generated manga

Question: Do the characters in Image {last_img_num} look EXACTLY like their reference designs?

IMPORTANT: The character must have the SAME appearance as the reference image.
A different animal is NOT the same character.

Answer only: PASS or FAIL"""

        # 发送参考图 + 生成的漫画图进行验证
        all_images = reference_images + [generated_image]

        config = GenerationConfig(
            temperature=0.0,  # 最低温度 - 确定性输出
            top_p=0.1,        # 极低采样 - 只选最可能的结果
            top_k=1,          # 只选概率最高的token
            max_tokens=500
        )

        try:
            response = await client.generate_text(
                prompt=validate_prompt,
                images=all_images,
                config=config
            )

            result = response.content.strip().upper()
            print(f"[MangaGenerator] Validation result: {result}")

            # 简单判断：只有明确 PASS 才通过，其他都失败
            if "PASS" in result and "FAIL" not in result:
                return True, ""
            else:
                return False, "Characters don't match reference"

        except Exception as e:
            print(f"[MangaGenerator] Validation error: {e}")
            # 验证出错时也默认失败，宁可重试
            return False, f"Validation error: {str(e)[:50]}"

    def _build_batch_prompt(self, panels: List[Panel], language: str, theme: str = "chiikawa", batch_characters: set = None) -> str:
        """
        构建批量图像生成 prompt - 使用详细的剧本信息

        根据面板数量使用不同布局：
        - 1 panel: 单张图
        - 2 panels: 横向排列 (1x2)
        - 3-4 panels: 2x2网格

        batch_characters: kumomo 主题时，当前批次出现的角色集合
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

        # 构建详细的面板描述
        panel_lines = []
        for i, panel in enumerate(panels):
            pos = positions[i] if i < len(positions) else f"Panel {i+1}"

            # 标题
            title = f" ({panel.panel_title})" if getattr(panel, 'panel_title', '') else ""

            # 角色
            chars = ", ".join(panel.characters) if panel.characters else ""

            # 详细画面描述（最重要）
            visual = getattr(panel, 'visual_description', '') or ""

            # 对白
            dialogues = []
            if panel.dialogue:
                for char, text in panel.dialogue.items():
                    dialogues.append(f'{char}: "{text}"')
            dialogue_str = " | ".join(dialogues) if dialogues else ""

            # 旁白/解释
            narration = getattr(panel, 'narration', '') or ""

            # 背景
            bg = getattr(panel, 'background', 'simple classroom') or "simple classroom"

            # 组合成详细描述
            if pos:
                desc = f"[{pos}{title}]\n"
            else:
                desc = ""
            desc += f"Characters: {chars}\n"
            if visual:
                desc += f"Visual: {visual}\n"
            if dialogue_str:
                desc += f"Dialogue: {dialogue_str}\n"
            if narration:
                desc += f"Narration box: {narration}\n"
            desc += f"Background: {bg}"

            panel_lines.append(desc)

        panels_text = "\n---\n".join(panel_lines)

        # 风格描述
        if theme == "ghibli":
            style = "Ghibli"
        elif theme == "kumomo":
            style = "Kumomo"
        else:
            style = "Chiikawa"

        # 文字说明
        if is_cjk:
            text_note = f"Render {lang_name} text LARGE and CLEAR in speech bubbles and narration boxes."
        else:
            text_note = f"Render text CLEARLY in speech bubbles and narration boxes."

        # Kumomo 原创角色参考 - 开头和结尾都提醒
        char_ref_start = ""
        char_ref_end = ""
        if theme == "kumomo":
            # 动态生成角色参考说明
            char_names = self.char_lib.get_kumomo_character_names()
            char_lines = [f"- Image {i+1} = {name} (draw {name} EXACTLY like this)" for i, name in enumerate(char_names)]
            char_ref_start = "CHARACTER DESIGNS (attached images above):\n" + "\n".join(char_lines) + "\n\n"
            char_ref_end = "\n\nREMINDER: Draw characters EXACTLY like the reference images above. Do NOT create different animals."

        prompt = f"""{char_ref_start}{layout_desc}, {style} style. {text_note}

{panels_text}{char_ref_end}"""

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

    def _load_all_kumomo_references(self) -> List[ImageContent]:
        """加载所有原创角色参考图"""
        return self._load_specific_kumomo_references(self.char_lib.get_kumomo_character_names())

    def _load_specific_kumomo_references(self, required_chars: List[str]) -> List[ImageContent]:
        """
        动态加载指定的 Kumomo 原创角色参考图片

        只加载当前批次需要的角色，减少干扰
        """
        reference_images = []

        all_paths = self.char_lib.get_all_kumomo_reference_images()
        print(f"[MangaGenerator] Filtering references for: {required_chars}")

        for img_path in all_paths:
            path_obj = Path(img_path)
            filename = path_obj.name.lower()

            # 检查此图片是否属于所需角色
            is_needed = False
            for rc in required_chars:
                if rc in filename:
                    is_needed = True
                    break

            if not is_needed:
                continue

            try:
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()

                ext = path_obj.suffix.lower()
                mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
                mime_type = mime_map.get(ext, "image/png")

                reference_images.append(ImageContent(
                    data=img_data,
                    mime_type=mime_type,
                    is_base64=True
                ))
                print(f"[MangaGenerator] Loaded reference: {filename}")
            except Exception as e:
                print(f"[MangaGenerator] Failed to load kumomo ref image {img_path}: {e}")

        return reference_images

    async def _save_final_manga(
        self,
        storyboard: Storyboard,
        panels: List[GeneratedPanel],
        progress_dir: Path,
        safe_title: str,
        session_id: str
    ):
        """保存最终生成的漫画和分镜脚本"""
        import json

        try:
            final_manga = GeneratedManga(
                title=storyboard.title,
                panels=panels,
                character_theme=storyboard.character_theme,
                language=storyboard.language
            )

            # 保存漫画图片
            filename = f"{safe_title}_{session_id}_final.png"
            output_path = progress_dir / filename

            combined = final_manga.get_combined_image()
            with open(output_path, "wb") as f:
                f.write(combined)

            print(f"[MangaGenerator] Saved final: {output_path.name}")

            # 保存分镜脚本 JSON（用于验证图像生成质量）
            json_filename = f"{safe_title}_{session_id}_storyboard.json"
            json_path = progress_dir / json_filename

            storyboard_data = storyboard.to_dict()
            storyboard_data["session_id"] = session_id
            storyboard_data["generated_at"] = datetime.now().isoformat()

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(storyboard_data, f, ensure_ascii=False, indent=2)

            print(f"[MangaGenerator] Saved storyboard: {json_filename}")

        except Exception as e:
            print(f"[MangaGenerator] Failed to save final: {e}")


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
