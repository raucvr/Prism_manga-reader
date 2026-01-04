"""
Storyboarder Service - 简化版
利用 Gemini 3 Pro Image 内置的 Chiikawa 知识

核心设计：
1. Gemini 内置 Chiikawa 知识，不需要复杂的角色 prompt
2. 使用同一个模型完成分镜生成和图像生成
3. 每个 panel 单独生成，避免长内容导致文字乱码
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines import get_client, GenerationConfig, ImageContent
from config_loader import get_config


class PanelType(str, Enum):
    """面板类型 - 灵活接受各种类型"""
    TITLE = "title"
    INTRODUCTION = "intro"
    EXPLANATION = "explain"
    EXAMPLE = "example"
    DIAGRAM = "diagram"
    REACTION = "reaction"
    CONCLUSION = "conclusion"
    # 扩展类型 - 支持 Gemini 生成的各种类型
    ACTION = "action"
    TRANSITION = "transition"
    METAPHOR = "metaphor"
    CONCEPT = "concept"
    METHODOLOGY = "methodology"
    DISCOVERY = "discovery"
    SUMMARY = "summary"
    HUMOR = "humor"
    OTHER = "other"  # 通用类型

    @classmethod
    def from_string(cls, value: str) -> "PanelType":
        """灵活解析 panel_type，未知类型映射到 OTHER"""
        value = value.lower().strip()
        # 直接匹配
        for member in cls:
            if member.value == value:
                return member
        # 部分匹配
        if "intro" in value or "title" in value:
            return cls.INTRODUCTION
        if "explain" in value or "concept" in value or "detail" in value:
            return cls.EXPLANATION
        if "example" in value or "analogy" in value:
            return cls.EXAMPLE
        if "react" in value or "emotion" in value:
            return cls.REACTION
        if "conclu" in value or "summary" in value or "ending" in value:
            return cls.CONCLUSION
        if "action" in value or "moment" in value:
            return cls.ACTION
        if "transition" in value or "shift" in value:
            return cls.TRANSITION
        if "metaphor" in value:
            return cls.METAPHOR
        if "method" in value:
            return cls.METHODOLOGY
        if "discov" in value or "reveal" in value:
            return cls.DISCOVERY
        if "humor" in value or "gag" in value or "chaos" in value:
            return cls.HUMOR
        # 默认返回 OTHER
        return cls.OTHER


@dataclass
class Panel:
    """单个漫画格"""
    panel_number: int
    panel_type: PanelType
    visual_description: str       # 详细画面内容（给 Gemini 生成图像用）
    characters: List[str]         # 出场角色
    character_emotions: Dict[str, str]  # 角色表情
    dialogue: Dict[str, str]      # 对白（气泡内的文字）
    narration: str = ""           # 旁白/原理解释（画面外的说明文字）
    panel_title: str = ""         # 面板标题（如：以前的难题）
    visual_metaphor: str = ""     # 视觉隐喻
    props: List[str] = field(default_factory=list)
    background: str = ""
    layout_hint: str = "normal"


@dataclass
class Storyboard:
    """完整分镜脚本"""
    title: str
    summary: str
    character_theme: str
    panels: List[Panel]
    source_document: str = ""
    language: str = "zh-CN"
    is_fallback: bool = False  # 是否为 fallback storyboard（不应被缓存）

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "summary": self.summary,
            "character_theme": self.character_theme,
            "language": self.language,
            "panel_count": len(self.panels),
            "panels": [
                {
                    "panel_number": p.panel_number,
                    "panel_type": p.panel_type.value,
                    "visual_description": p.visual_description,
                    "characters": p.characters,
                    "character_emotions": p.character_emotions,
                    "dialogue": p.dialogue,
                    "narration": p.narration,
                    "visual_metaphor": p.visual_metaphor,
                    "props": p.props,
                    "background": p.background,
                    "layout_hint": p.layout_hint,
                }
                for p in self.panels
            ]
        }


class Storyboarder:
    """
    分镜生成器 - 简化版

    利用 Gemini 3 Pro Image 的能力：
    1. 内置 Chiikawa 知识，不需要复杂的角色描述
    2. 能理解论文内容并转化为漫画分镜
    """

    def __init__(self, character_theme: str = "chiikawa"):
        self.character_theme = character_theme
        self.config = get_config()
        self.char_lib = CharacterLibrary()  # 动态加载原创角色

    async def generate_storyboard(
        self,
        text: str,
        title: str = "",
        language: str = "zh-CN"
    ) -> Storyboard:
        """
        从论文生成分镜脚本 - 三步流程

        Step 1: 生成英文技术解读
        Step 2: 用英文生成高质量漫画分镜
        Step 3: 翻译对白到目标语言（如果不是英文）
        """
        import hashlib

        # 验证输入
        if not text or len(text) < 100:
            raise ValueError(f"Input text too short ({len(text)} chars). PDF may not have been parsed correctly.")

        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        cache_key = f"v{CACHE_VERSION}_{text_hash}_{language}_{self.character_theme}"
        print(f"[Storyboarder] Input: {len(text)} chars, hash={text_hash}, title={title}")

        # 检查缓存
        if cache_key in _storyboard_cache:
            print(f"[Storyboarder] Using cached storyboard for {cache_key}")
            return _storyboard_cache[cache_key]

        print(f"[Storyboarder] Cache miss, generating new storyboard...")

        client = await get_client()

        # ========== Step 1: 生成英文技术解读 ==========
        print(f"[Storyboarder] Step 1: Generating technical analysis ({len(text)} chars)...")

        analysis_prompt = self._build_analysis_prompt(text, title)

        config = GenerationConfig(
            temperature=0.3,  # 低温度确保准确性
            max_tokens=32000  # 足够生成完整技术分析
        )

        analysis_response = await client.generate_text(
            prompt=analysis_prompt,
            config=config
        )

        technical_analysis = analysis_response.content
        print(f"[Storyboarder] Analysis: {len(technical_analysis)} chars")

        # ========== Step 2: 用英文生成高质量漫画分镜 ==========
        print(f"[Storyboarder] Step 2: Generating manga storyboard (in English for quality)...")

        # 始终用英文生成分镜，确保最高质量
        storyboard_prompt = self._build_storyboard_prompt(technical_analysis, title, "en-US")

        # 加载角色参考图片（用于原创角色如 kumomo）
        reference_images = self._load_character_reference_images()
        if reference_images:
            print(f"[Storyboarder] Including {len(reference_images)} character reference images")

        config = GenerationConfig(
            temperature=0.7,
            max_tokens=32000  # 足够生成 100+ 个分镜
        )

        response = await client.generate_text(
            prompt=storyboard_prompt,
            images=reference_images if reference_images else None,
            config=config
        )

        # 解析响应（暂时设为英文）
        storyboard = self._parse_response(response.content, title, text, "en-US")

        print(f"[Storyboarder] Generated {len(storyboard.panels)} panels")

        # ========== Step 3: 翻译对白到目标语言 ==========
        if language != "en-US":
            print(f"[Storyboarder] Step 3: Translating dialogues to {language}...")
            storyboard = await self._translate_storyboard(storyboard, language, client, technical_analysis)
            print(f"[Storyboarder] Translation completed")

        storyboard.language = language

        # ========== Step 4: 强制执行对话长度限制 ==========
        storyboard = self._enforce_dialogue_limits(storyboard, language)

        # 只缓存成功的结果（>= 10 panels 且非 fallback）
        is_fallback = getattr(storyboard, 'is_fallback', False)
        min_panels_to_cache = 10

        if len(storyboard.panels) >= min_panels_to_cache and not is_fallback:
            _storyboard_cache[cache_key] = storyboard
            print(f"[Storyboarder] Cached storyboard as {cache_key} ({len(storyboard.panels)} panels)")
        else:
            print(f"[Storyboarder] NOT caching: {len(storyboard.panels)} panels (min={min_panels_to_cache}), fallback={is_fallback}")

        return storyboard

    async def _translate_storyboard(
        self,
        storyboard: Storyboard,
        target_language: str,
        client,
        technical_analysis: str = ""
    ) -> Storyboard:
        """
        翻译分镜到目标语言（对白 + 旁白）
        包含技术分析作为上下文，确保专业术语翻译准确
        """
        lang_map = {"zh-CN": "Simplified Chinese", "ja-JP": "Japanese"}
        target_lang_name = lang_map.get(target_language, target_language)

        # 收集所有需要翻译的文本
        # 格式: panel_number|type|key|text
        # type: dialogue, narration, title
        all_texts = []

        # 标题
        if storyboard.title:
            all_texts.append(f"0|title|title|{storyboard.title}")

        for panel in storyboard.panels:
            # 对白
            for char, text in panel.dialogue.items():
                if text:
                    all_texts.append(f"{panel.panel_number}|dialogue|{char}|{text}")
            # 旁白
            if panel.narration:
                all_texts.append(f"{panel.panel_number}|narration|narration|{panel.narration}")

        if not all_texts:
            return storyboard

        # 批量翻译
        texts_to_translate = "\n".join(all_texts)

        # 包含技术分析摘要作为上下文
        context_summary = technical_analysis[:8000] if technical_analysis else ""

        prompt = f"""You are translating a manga storyboard about a scientific paper to {target_lang_name}.

# CONTEXT (Technical Analysis of the Paper)
{context_summary}

# TRANSLATION RULES
- This is a manga explaining an ACADEMIC PAPER - preserve ALL technical terms accurately
- Translate naturally in conversational {target_lang_name}
- Keep exact numbers, formulas, method names (e.g., "Amber ff14SB", "hazard ratio 0.73", "AdS/CFT")
- DO NOT add explanations or notes
- Output ONLY the translations in the same format

# FORMAT
Input: panel_number|type|key|text
Output: panel_number|type|key|translated_text

Types:
- dialogue: Character speech bubbles
- narration: Explanation text boxes
- title: Main title

# TEXTS TO TRANSLATE
{texts_to_translate}

# OUTPUT (translations only, same format):"""

        config = GenerationConfig(
            temperature=0.3,
            max_tokens=32000  # 足够翻译所有内容
        )

        response = await client.generate_text(
            prompt=prompt,
            config=config
        )

        print(f"[Storyboarder] Translation response length: {len(response.content)} chars")

        # 解析翻译结果
        translated_dialogues = {}  # {panel_num: {char: text}}
        translated_narrations = {}  # {panel_num: text}
        translated_title = None

        for line in response.content.strip().split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|", 3)
            if len(parts) >= 4:
                try:
                    panel_num = int(parts[0])
                    text_type = parts[1].strip().lower()
                    key = parts[2].strip().lower()
                    translated = parts[3].strip()

                    if text_type == "dialogue":
                        if panel_num not in translated_dialogues:
                            translated_dialogues[panel_num] = {}
                        translated_dialogues[panel_num][key] = translated
                    elif text_type == "narration":
                        translated_narrations[panel_num] = translated
                    elif text_type == "title" and panel_num == 0:
                        translated_title = translated
                except ValueError:
                    continue

        print(f"[Storyboarder] Parsed: {len(translated_dialogues)} dialogue panels, {len(translated_narrations)} narrations")

        # 应用翻译
        dialogue_count = 0
        narration_count = 0

        # 翻译标题
        if translated_title:
            storyboard.title = translated_title
            print(f"[Storyboarder] Translated title: {translated_title[:50]}...")

        for panel in storyboard.panels:
            # 应用对白翻译
            if panel.panel_number in translated_dialogues:
                dialogue_lower_map = {k.lower(): k for k in panel.dialogue.keys()}
                for char, translated in translated_dialogues[panel.panel_number].items():
                    if char in dialogue_lower_map:
                        original_key = dialogue_lower_map[char]
                        panel.dialogue[original_key] = translated
                        dialogue_count += 1

            # 应用旁白翻译
            if panel.panel_number in translated_narrations:
                panel.narration = translated_narrations[panel.panel_number]
                narration_count += 1

        print(f"[Storyboarder] Applied {dialogue_count} dialogues, {narration_count} narrations")

        return storyboard

    def _enforce_dialogue_limits(self, storyboard: Storyboard, language: str) -> Storyboard:
        """
        强制执行对话和旁白长度限制

        限制（基于气泡空间）:
        - 中文/日文对话: 40 字符
        - 英文对话: 100 字符
        - 中文/日文旁白: 60 字符
        - 英文旁白: 150 字符
        """
        is_cjk = language in ["zh-CN", "ja-JP"]

        # 设置限制
        dialogue_limit = 40 if is_cjk else 100
        narration_limit = 60 if is_cjk else 150

        truncated_count = 0

        for panel in storyboard.panels:
            # 截断对话
            for char, text in list(panel.dialogue.items()):
                if len(text) > dialogue_limit:
                    # 智能截断：在标点或空格处截断
                    truncated = text[:dialogue_limit]
                    # 尝试在最后一个标点处截断
                    for punct in ['。', '！', '？', '，', '.', '!', '?', ',', ' ']:
                        last_punct = truncated.rfind(punct)
                        if last_punct > dialogue_limit * 0.6:  # 至少保留 60% 内容
                            truncated = truncated[:last_punct + 1]
                            break
                    else:
                        truncated = truncated + "..."

                    panel.dialogue[char] = truncated
                    truncated_count += 1

            # 截断旁白
            narration = getattr(panel, 'narration', '') or ''
            if len(narration) > narration_limit:
                truncated = narration[:narration_limit]
                for punct in ['。', '！', '？', '.', '!', '?', ' ']:
                    last_punct = truncated.rfind(punct)
                    if last_punct > narration_limit * 0.6:
                        truncated = truncated[:last_punct + 1]
                        break
                else:
                    truncated = truncated + "..."
                panel.narration = truncated
                truncated_count += 1

        if truncated_count > 0:
            print(f"[Storyboarder] Truncated {truncated_count} dialogues/narrations to fit limits")

        return storyboard

    def _load_character_reference_images(self) -> List[ImageContent]:
        """
        加载角色参考图片（用于原创角色）

        目前支持 kumomo 主题的原创角色
        """
        import base64

        images = []
        image_base_path = Path(__file__).parent.parent.parent / "config" / "character_images"

        if self.character_theme == "kumomo":
            # 动态加载原创角色参考图
            char_images = self.char_lib.kumomo_images_ordered

            for char_name, filename in char_images:
                img_path = image_base_path / filename
                if img_path.exists():
                    try:
                        with open(img_path, "rb") as f:
                            img_data = f.read()
                        img_base64 = base64.b64encode(img_data).decode()

                        # 确定 MIME 类型
                        suffix = img_path.suffix.lower()
                        mime_type = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"

                        images.append(ImageContent(
                            data=img_base64,
                            mime_type=mime_type
                        ))
                        print(f"[Storyboarder] Loaded reference image for {char_name}")
                    except Exception as e:
                        print(f"[Storyboarder] Failed to load {char_name} image: {e}")

        return images

    def _build_analysis_prompt(self, text: str, title: str) -> str:
        """
        构建技术解读 prompt - 简洁版，让模型自由发挥
        """
        print(f"[Storyboarder] Analysis prompt: {len(text)} chars")

        return f"""Analyze this academic paper. Extract ALL key information:
- Research question and novelty
- Methodology (exact steps, algorithms, parameters)
- Results (exact numbers, statistics, comparisons)
- Conclusions and limitations

Copy exact values from the paper. Output in English.

# Paper
{text[:80000]}"""

    def _build_storyboard_prompt(self, text: str, title: str, language: str) -> str:
        """
        构建分镜生成 prompt（始终用英文生成，后续翻译）

        使用详细的剧本格式，包含完整角色设定
        """
        # 根据主题选择不同的角色和详细设定
        if self.character_theme == "ghibli":
            style_name = "Studio Ghibli"
            characters_desc = """
• haku - The wise mentor/professor. A calm, elegant young man in traditional clothing. Speaks thoughtfully and explains complex concepts clearly. Often gestures gracefully when teaching.
• chihiro - The curious student. A young girl with a ponytail, eager to learn. Asks good questions, sometimes confused but always determined. Shows emotions openly.
• calcifer - The skeptic/critic. A small fire spirit who adds humor. Questions assumptions, points out flaws, makes witty remarks. Floats around and changes colors with mood."""
            example_chars = "haku, chihiro"
        elif self.character_theme == "kumomo":
            style_name = "Kumomo (original cute characters)"
            # 动态生成角色描述，包含角色分工
            role_desc = {"mentor": "professor/mentor", "student": "curious student", "skeptic": "skeptic/critic"}
            chars_with_roles = self.char_lib.get_kumomo_characters_with_roles()
            char_lines = [f"• {name} - {role_desc.get(role, role)} (appearance defined by reference image)" for name, role in chars_with_roles]
            characters_desc = "\n" + "\n".join(char_lines)
            # 使用前两个角色作为示例
            char_names = [name for name, _ in chars_with_roles]
            example_chars = f"{char_names[0]}, {char_names[1]}" if len(char_names) >= 2 else char_names[0] if char_names else "character1, character2"
        else:  # chiikawa
            style_name = "Chiikawa"
            characters_desc = """
• hachiware - The wise mentor/professor. A cat-like creature with a split-colored face. Calm, knowledgeable, explains concepts patiently. Often holds papers or points at diagrams.
• chiikawa - The curious student. A small, round white creature. Innocent, easily confused but hardworking. Shows emotions with "Wa!" or "Eh?". Sweats when nervous.
• usagi - The skeptic/critic. A rabbit with intense eyes. Energetic, questions everything with "Ura!!", sometimes chaotic but insightful. Jumps around when excited."""
            example_chars = "hachiware, chiikawa"

        example_format = f"""===
Panel 1: The Old Problem
Characters: {example_chars}
Scene: Research lab with messy papers on desk
Visual: The student stares at a pile of items (representing the research problem). The mentor holds a measuring device, explaining the traditional approach. The student looks confused with sweat drops.
Dialogue:
- {example_chars.split(", ")[0]}: "Previously, we used method X to solve this problem..."
- {example_chars.split(", ")[1]}: "But that seems incomplete!"
Narration: Traditional methods have limitations that this paper addresses.
===
Panel 2: The New Idea
Characters: {example_chars}
Scene: Same lab, mentor now excited
Visual: The mentor excitedly points at a new diagram. The student's eyes widen with understanding. Props show the key innovation.
Dialogue:
- {example_chars.split(", ")[0]}: "Wait! The secret is in THIS approach!"
- {example_chars.split(", ")[1]}: "Oh! That makes sense!"
Narration: The paper's novel contribution explained simply.
==="""

        return f"""Create a {style_name}-style manga explaining this paper. Generate 20-60 detailed panels.

CHARACTER PROFILES (use these consistently throughout):
{characters_desc}

Source material:
{text[:100000]}

Format each panel with ALL these fields:
{example_format}

Requirements:
- Visual: Describe character poses, expressions, props, actions in detail
- Dialogue: Keep each line SHORT (max 40 characters for Chinese, 100 for English). Split long explanations across multiple panels.
- Narration: Scientific explanation as text box (max 60 characters for Chinese, 150 for English)
- Keep characters IN CHARACTER throughout (mentor teaches, student asks, skeptic questions)
- Use exact numbers and terms from the paper"""

    def _fix_json(self, json_str: str) -> str:
        """尝试修复常见的 JSON 格式错误"""
        fixed = json_str

        # 移除尾部多余的逗号 (在 ] 或 } 之前)
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)

        # 修复缺少逗号的情况 (}{ 或 }[)
        fixed = re.sub(r'}\s*{', '},{', fixed)
        fixed = re.sub(r'}\s*\[', '},[', fixed)
        fixed = re.sub(r'\]\s*{', '],{', fixed)

        # 修复未闭合的字符串 (在 } 或 ] 之前添加引号)
        # 这个比较复杂，暂时跳过

        # 确保 JSON 以 } 结尾
        fixed = fixed.strip()
        if not fixed.endswith('}'):
            # 尝试找到最后一个完整的 panel 并截断
            last_panel_end = fixed.rfind('}')
            if last_panel_end > 0:
                fixed = fixed[:last_panel_end + 1]
                # 确保 panels 数组正确关闭
                if '"panels"' in fixed and fixed.count('[') > fixed.count(']'):
                    fixed += ']}'

        return fixed

    def _extract_panels_from_broken_json(self, json_str: str) -> List[dict]:
        """从格式错误的 JSON 中逐个提取 panel 对象"""
        panels = []

        # 使用正则匹配每个 panel 对象
        # 匹配 {"panel_number": ... } 格式的对象
        panel_pattern = r'\{\s*"panel_number"\s*:\s*\d+[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'

        matches = re.findall(panel_pattern, json_str, re.DOTALL)

        for match in matches:
            try:
                # 清理并尝试解析单个 panel
                cleaned = match.strip()
                panel = json.loads(cleaned)
                if "panel_number" in panel:
                    panels.append(panel)
            except json.JSONDecodeError:
                # 尝试修复单个 panel
                try:
                    fixed = self._fix_json(match)
                    panel = json.loads(fixed)
                    if "panel_number" in panel:
                        panels.append(panel)
                except:
                    continue

        # 按 panel_number 排序
        panels.sort(key=lambda x: x.get("panel_number", 0))

        return panels

    def _parse_response(
        self,
        response: str,
        title: str,
        source_text: str,
        language: str
    ) -> Storyboard:
        """解析 AI 响应 - 支持自然语言格式"""

        panels = []

        # 尝试解析自然语言格式 (=== 分隔)
        if "===" in response:
            panels = self._parse_natural_language_format(response)
            print(f"[Storyboarder] Parsed {len(panels)} panels from natural language format")

        # 回退: 尝试 JSON 格式
        if not panels:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response

            try:
                data = json.loads(json_str)
                for p in data.get("panels", []):
                    panel = self._dict_to_panel(p, len(panels) + 1)
                    if panel:
                        panels.append(panel)
            except json.JSONDecodeError:
                # 尝试修复 JSON
                try:
                    fixed_json = self._fix_json(json_str)
                    data = json.loads(fixed_json)
                    for p in data.get("panels", []):
                        panel = self._dict_to_panel(p, len(panels) + 1)
                        if panel:
                            panels.append(panel)
                except:
                    pass

        if not panels:
            print(f"[Storyboarder] All parse attempts failed, using fallback")
            return self._create_fallback_storyboard(title, source_text, language)

        # 最终排序确保故事顺序正确
        panels.sort(key=lambda p: p.panel_number)
        print(f"[Storyboarder] Final panel order: {[p.panel_number for p in panels[:10]]}...")

        return Storyboard(
            title=title or "学习笔记",
            summary="",
            character_theme=self.character_theme,
            panels=panels,
            source_document=source_text[:1000],
            language=language
        )

    def _parse_natural_language_format(self, response: str) -> List[Panel]:
        """Parse rich natural language format storyboard (=== delimited)"""
        panels = []

        # Split by ===
        sections = re.split(r'={3,}', response)

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Parse panel number and title (Panel 1: The Old Problem)
            panel_header_match = re.search(r'(?:Panel|分镜)\s*(\d+)(?:\s*[:：]\s*(.+?))?(?:\n|$)', section, re.IGNORECASE)
            panel_number = int(panel_header_match.group(1)) if panel_header_match else len(panels) + 1
            panel_title = panel_header_match.group(2).strip() if panel_header_match and panel_header_match.group(2) else ""

            # Parse characters
            chars_match = re.search(r'(?:Characters|角色):\s*(.+?)(?:\n|$)', section, re.IGNORECASE)
            characters = []
            if chars_match:
                chars_str = chars_match.group(1)
                characters = [c.strip().lower() for c in re.split(r'[,，、]', chars_str)]

            # Parse scene (简短场景描述)
            scene_match = re.search(r'(?:Scene|场景):\s*(.+?)(?:\n|$)', section, re.IGNORECASE)
            background = scene_match.group(1).strip() if scene_match else "simple classroom"

            # Parse visual description (详细画面内容)
            visual_match = re.search(
                r'(?:Visual|画面|画面内容):\s*(.+?)(?=\n(?:Dialogue|对白|Narration|旁白):|\n===|$)',
                section, re.DOTALL | re.IGNORECASE
            )
            visual_description = visual_match.group(1).strip() if visual_match else ""

            # Parse dialogue
            dialogue = {}
            dialogue_section = re.search(
                r'(?:Dialogue|对白):\s*(.+?)(?=\n(?:Narration|旁白):|\n===|$)',
                section, re.DOTALL | re.IGNORECASE
            )
            if dialogue_section:
                dialogue_text = dialogue_section.group(1)
                # Match - character: "dialogue" format
                # Only match double quotes to avoid cutting at apostrophes (don't, aren't, etc.)
                dialogue_matches = re.findall(
                    r'-\s*(\w+):\s*"([^"]+)"',
                    dialogue_text
                )
                # Also try curly quotes if no matches
                if not dialogue_matches:
                    dialogue_matches = re.findall(
                        r'-\s*(\w+):\s*"([^"]+)"',
                        dialogue_text
                    )
                for char, text in dialogue_matches:
                    dialogue[char.lower()] = text

            # Parse narration (旁白/原理解释)
            narration_match = re.search(
                r'(?:Narration|旁白|原理解释):\s*(.+?)(?=\n===|$)',
                section, re.DOTALL | re.IGNORECASE
            )
            narration = narration_match.group(1).strip() if narration_match else ""

            if visual_description or dialogue or narration:
                panel = Panel(
                    panel_number=panel_number,
                    panel_type=PanelType.EXPLANATION,
                    visual_description=visual_description,
                    characters=characters or self._get_default_characters(),
                    character_emotions={},
                    dialogue=dialogue,
                    narration=narration,
                    panel_title=panel_title,
                    background=background
                )
                panels.append(panel)

        # 按 panel_number 排序
        panels.sort(key=lambda p: p.panel_number)
        return panels

    def _dict_to_panel(self, p: dict, default_num: int) -> Optional[Panel]:
        """将字典转换为 Panel 对象"""
        try:
            panel_type_str = p.get("panel_type", "explain")
            panel_type = PanelType.from_string(panel_type_str)

            return Panel(
                panel_number=p.get("panel_number", default_num),
                panel_type=panel_type,
                visual_description=p.get("visual_description", p.get("visual", "")),
                characters=p.get("characters", self._get_default_characters()),
                character_emotions=p.get("character_emotions", {}),
                dialogue=p.get("dialogue", {}),
                narration=p.get("narration", ""),
                panel_title=p.get("panel_title", p.get("title", "")),
                visual_metaphor=p.get("visual_metaphor", ""),
                props=p.get("props", []),
                background=p.get("background", p.get("scene", "simple classroom")),
                layout_hint=p.get("layout_hint", "normal")
            )
        except Exception as e:
            print(f"[Storyboarder] Panel parse error: {e}")
            return None

    def _get_default_characters(self) -> list:
        """根据主题返回默认角色列表"""
        if self.character_theme == "ghibli":
            return ["haku", "chihiro"]
        elif self.character_theme == "kumomo":
            return self.char_lib.get_kumomo_character_names()
        return ["chiikawa", "hachiware"]

    def _create_fallback_storyboard(
        self,
        title: str,
        source_text: str,
        language: str
    ) -> Storyboard:
        """创建备用分镜（解析失败时）- 根据主题使用对应角色"""
        # 根据主题选择角色和对白
        if self.character_theme == "ghibli":
            panels = [
                Panel(
                    panel_number=1,
                    panel_type=PanelType.TITLE,
                    visual_description="Haku holding a scroll, Chihiro looking curious, in a magical library",
                    characters=["haku", "chihiro"],
                    character_emotions={"haku": "explaining", "chihiro": "curious"},
                    dialogue={"haku": "今天来学习一个有趣的话题！", "chihiro": "是什么呢？"},
                    background="magical library with floating books"
                ),
                Panel(
                    panel_number=2,
                    panel_type=PanelType.EXPLANATION,
                    visual_description="Haku explaining with a glowing diagram, Chihiro and Calcifer listening",
                    characters=["haku", "chihiro", "calcifer"],
                    character_emotions={"haku": "explaining", "chihiro": "thinking", "calcifer": "default"},
                    dialogue={"haku": "让我来解释一下～", "calcifer": "这个很有意思..."},
                    background="classroom with warm lighting"
                ),
                Panel(
                    panel_number=3,
                    panel_type=PanelType.CONCLUSION,
                    visual_description="All three characters happy together",
                    characters=["haku", "chihiro", "calcifer"],
                    character_emotions={"haku": "happy", "chihiro": "happy", "calcifer": "excited"},
                    dialogue={"chihiro": "原来如此！", "haku": "做得好！", "calcifer": "哼，我早就知道了！"},
                    background="sunny classroom"
                )
            ]
        elif self.character_theme == "kumomo":
            # 动态获取原创角色名
            char_names = self.char_lib.get_kumomo_character_names()
            c1 = char_names[0] if len(char_names) > 0 else "character1"
            c2 = char_names[1] if len(char_names) > 1 else "character2"
            c3 = char_names[2] if len(char_names) > 2 else c1  # 如果只有2个角色，重复使用第一个
            all_chars = char_names[:3] if len(char_names) >= 3 else char_names

            panels = [
                Panel(
                    panel_number=1,
                    panel_type=PanelType.TITLE,
                    visual_description=f"{c1} holding a book, {c2} looking curious",
                    characters=[c1, c2],
                    character_emotions={c1: "explaining", c2: "curious"},
                    dialogue={c1: "今天来学习一个有趣的话题！", c2: "是什么呢？"},
                    background="simple study room"
                ),
                Panel(
                    panel_number=2,
                    panel_type=PanelType.EXPLANATION,
                    visual_description=f"{c1} at whiteboard explaining, others listening",
                    characters=all_chars,
                    character_emotions={c: "thinking" for c in all_chars},
                    dialogue={c1: "让我来解释一下～"},
                    background="classroom"
                ),
                Panel(
                    panel_number=3,
                    panel_type=PanelType.CONCLUSION,
                    visual_description=f"All characters celebrating",
                    characters=all_chars,
                    character_emotions={c: "happy" for c in all_chars},
                    dialogue={c2: "原来如此！", c1: "做得好！"},
                    background="festive background with sparkles"
                )
            ]
        else:  # chiikawa (default)
            panels = [
                Panel(
                    panel_number=1,
                    panel_type=PanelType.TITLE,
                    visual_description="Hachiware holding a book, Chiikawa looking curious",
                    characters=["hachiware", "chiikawa"],
                    character_emotions={"hachiware": "explaining", "chiikawa": "confused"},
                    dialogue={"hachiware": "今天来学习一个有趣的话题！", "chiikawa": "诶...？"},
                    background="simple study room"
                ),
                Panel(
                    panel_number=2,
                    panel_type=PanelType.EXPLANATION,
                    visual_description="Hachiware at whiteboard, Chiikawa listening",
                    characters=["hachiware", "chiikawa"],
                    character_emotions={"hachiware": "explaining", "chiikawa": "thinking"},
                    dialogue={"hachiware": "让我来解释一下～", "chiikawa": "嗯嗯..."},
                    background="classroom"
                ),
                Panel(
                    panel_number=3,
                    panel_type=PanelType.CONCLUSION,
                    visual_description="All three characters celebrating together",
                    characters=["chiikawa", "hachiware", "usagi"],
                    character_emotions={"chiikawa": "happy", "hachiware": "happy", "usagi": "yelling"},
                    dialogue={"chiikawa": "原来如此！", "hachiware": "做得好！", "usagi": "呀哈！"},
                    background="festive background with sparkles"
                )
            ]

        return Storyboard(
            title=title or "学习笔记",
            summary="让我们一起来学习吧！",
            character_theme=self.character_theme,
            panels=panels,
            source_document=source_text[:500],
            language=language,
            is_fallback=True  # 标记为 fallback，不应被缓存
        )


# 保留 CharacterLibrary 用于参考图片加载（如果需要）
class CharacterLibrary:
    """角色库 - 动态从 character_images 目录加载原创角色"""

    def __init__(self):
        self.image_base_path = Path(__file__).parent.parent.parent / "config" / "character_images"
        self._load_kumomo_characters()
        self._load_config()

    def _load_kumomo_characters(self):
        """
        从 character_images 目录动态加载原创角色

        文件命名格式: "数字. 角色名.扩展名"
        例如: "1. papi.jpeg", "2. kumo.png"

        角色分工按数字顺序:
        - 1 = 教授/导师 (mentor)
        - 2 = 学生 (student)
        - 3 = 质疑者 (skeptic)
        """
        import re

        self.kumomo_characters = {}  # name -> name
        self.kumomo_images_ordered = []  # [(name, filename), ...]
        self.kumomo_images = {}  # name -> filename
        self.kumomo_roles = {}  # name -> role

        # 角色分工映射
        role_map = {1: "mentor", 2: "student", 3: "skeptic"}

        if not self.image_base_path.exists():
            print(f"[CharacterLibrary] Warning: {self.image_base_path} does not exist")
            return

        # 扫描目录中的图片文件
        image_extensions = {'.jpeg', '.jpg', '.png', '.webp'}
        image_files = [
            f for f in self.image_base_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        # 解析文件名并排序
        parsed_chars = []
        for img_file in image_files:
            filename = img_file.name
            stem = img_file.stem

            # 尝试解析 "数字. 角色名" 格式
            match = re.match(r'^(\d+)\.\s*(.+)$', stem)
            if match:
                order = int(match.group(1))
                char_name = match.group(2).strip().lower()
            else:
                # 没有数字前缀，按文件名排序
                order = 999
                char_name = stem.lower()

            role = role_map.get(order, "character")
            parsed_chars.append((order, char_name, filename, role))

        # 按数字排序
        parsed_chars.sort(key=lambda x: x[0])

        for order, char_name, filename, role in parsed_chars:
            self.kumomo_characters[char_name] = char_name
            self.kumomo_images_ordered.append((char_name, filename))
            self.kumomo_images[char_name] = filename
            self.kumomo_roles[char_name] = role

        char_info = [(name, self.kumomo_roles[name]) for name, _ in self.kumomo_images_ordered]
        print(f"[CharacterLibrary] Loaded {len(self.kumomo_characters)} characters: {char_info}")

    def _load_config(self):
        """加载角色配置"""
        import yaml
        config_path = Path(__file__).parent.parent.parent / "config" / "characters.yaml"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self.characters = data.get("characters", {})
        except Exception as e:
            print(f"[CharacterLibrary] Warning: {e}")
            self.characters = {}

    def get_reference_images(self, char_name: str, emotion: str = None) -> List[str]:
        """获取角色参考图片路径"""
        image_paths = []
        char_name_lower = char_name.lower()

        # 检查是否是 kumomo 原创角色
        normalized_name = self.kumomo_characters.get(char_name_lower)
        if normalized_name:
            # 使用 kumomo 原创角色的参考图片
            img_filename = self.kumomo_images.get(normalized_name)
            if img_filename:
                full_path = self.image_base_path / img_filename
                if full_path.exists():
                    image_paths.append(str(full_path))
                    return image_paths

        # 回退到 YAML 配置
        char = self.characters.get(char_name, {})
        ref_images = char.get("reference_images", {})

        # 主要参考图
        for img_path in ref_images.get("main", []):
            full_path = self.image_base_path / img_path
            if full_path.exists():
                image_paths.append(str(full_path))

        # 表情参考图
        if emotion:
            expr_path = ref_images.get("expressions", {}).get(emotion)
            if expr_path:
                full_path = self.image_base_path / expr_path
                if full_path.exists():
                    image_paths.append(str(full_path))

        return image_paths

    def get_all_kumomo_reference_images(self) -> List[str]:
        """
        获取所有原创角色的参考图片（按文件名排序）
        """
        image_paths = []
        for char_name, img_filename in self.kumomo_images_ordered:
            full_path = self.image_base_path / img_filename
            if full_path.exists():
                image_paths.append(str(full_path))
        return image_paths

    def get_kumomo_character_names(self) -> List[str]:
        """获取所有原创角色名称（按顺序）"""
        return [name for name, _ in self.kumomo_images_ordered]

    def get_kumomo_character_role(self, char_name: str) -> str:
        """获取角色的分工 (mentor/student/skeptic)"""
        return self.kumomo_roles.get(char_name.lower(), "character")

    def get_kumomo_characters_with_roles(self) -> List[tuple]:
        """获取所有角色及其分工 [(name, role), ...]"""
        return [(name, self.kumomo_roles.get(name, "character")) for name, _ in self.kumomo_images_ordered]

    def get_all_reference_images_for_panel(
        self,
        characters: List[str],
        emotions: Dict[str, str]
    ) -> List[str]:
        """获取 panel 中所有角色的参考图片"""
        all_images = []
        seen = set()

        for char_name in characters:
            emotion = emotions.get(char_name)
            for img in self.get_reference_images(char_name, emotion):
                if img not in seen:
                    all_images.append(img)
                    seen.add(img)

        return all_images

    def has_reference_images(self, char_name: str) -> bool:
        """检查角色是否有参考图片"""
        return len(self.get_reference_images(char_name)) > 0


# 全局实例
_storyboarder: Optional[Storyboarder] = None

# Cache version - increment this to invalidate all cached storyboards
# v2: Added translation context fix for zh-CN
# v3: Removed 25-char truncation limit for CJK
# v4: Added kumomo theme with original characters (Pip, Kumomo, Pippin)
# v5: Fixed cache key collision - now uses full text hash instead of first 5000 chars
# v6: Added character reference images to storyboard generation for original characters
# v7: Fixed character mapping: kumo=云朵, nezu=刺猬, papi=小狗
# v8: Fixed default characters, fallback storyboard, and validation logic for kumomo theme
# v9: Smart caching - only cache successful results (>=10 panels, not fallback)
# v10: Stronger reference image prompts - explicitly tell model "IMAGE 1/2/3 are input images"
# v11: Simplified prompts - minimal text descriptions, let model focus on reference image files
# v12: Enhanced validation with CoT (Chain-of-Thought), dynamic ref loading, character mapping
# v13: Ultra-simple prompts: "[papi looks like this: papi.jpeg]" - let model focus on images
# v14: Fix temperature hardcoded to 1.0 in API, explicit image order in prompt
# v15: Dynamic character loading from character_images directory
# v16: Character roles from filename prefix (1.=mentor, 2.=student, 3.=skeptic)
# v17: Enforce dialogue length limits (40 chars CJK, 100 chars EN) to fit speech bubbles
# v18: Fix dialogue parsing - apostrophes in contractions (don't, aren't) were causing truncation
# v19: Translate ALL text fields (dialogue + narration + title), not just dialogue
CACHE_VERSION = 19

# Simple storyboard cache (text hash -> storyboard)
_storyboard_cache: Dict[str, Storyboard] = {}


def clear_storyboard_cache() -> int:
    """Clear the storyboard cache. Returns the number of entries cleared."""
    global _storyboard_cache
    count = len(_storyboard_cache)
    _storyboard_cache = {}
    print(f"[Storyboarder] Cleared {count} cached storyboards")
    return count


def get_storyboarder(character_theme: str = None) -> Storyboarder:
    """获取分镜生成器实例"""
    global _storyboarder

    if character_theme is None:
        config = get_config()
        character_theme = config.manga_settings.default_character

    if _storyboarder is None or _storyboarder.character_theme != character_theme:
        _storyboarder = Storyboarder(character_theme)

    return _storyboarder


def reset_storyboarder() -> None:
    """重置分镜生成器"""
    global _storyboarder
    _storyboarder = None
