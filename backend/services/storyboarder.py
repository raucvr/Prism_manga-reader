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

from engines import get_client, GenerationConfig
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
    visual_description: str       # 场景描述（给 Gemini 生成图像用）
    characters: List[str]         # 出场角色
    character_emotions: Dict[str, str]  # 角色表情
    dialogue: Dict[str, str]      # 对白
    visual_metaphor: str = ""     # 视觉隐喻
    props: List[str] = field(default_factory=list)
    background: str = ""
    layout_hint: str = "normal"
    # 简化后不再需要复杂的 full_image_prompt
    # Gemini 内置 Chiikawa 知识，只需要简单描述


@dataclass
class Storyboard:
    """完整分镜脚本"""
    title: str
    summary: str
    character_theme: str
    panels: List[Panel]
    source_document: str = ""
    language: str = "zh-CN"

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
                    "visual_metaphor": p.visual_metaphor,
                    "props": p.props,
                    "background": p.background,
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

    async def generate_storyboard(
        self,
        text: str,
        title: str = "",
        language: str = "zh-CN"
    ) -> Storyboard:
        """
        从论文生成分镜脚本

        使用 Gemini 3 Pro Image 分析论文，生成分镜描述
        模型内置 Chiikawa 知识，会自动使用合适的角色和风格
        """
        import hashlib
        global _storyboard_cache

        # Check cache first
        cache_key = hashlib.md5(f"{text[:5000]}:{title}:{language}:{self.character_theme}".encode()).hexdigest()
        if cache_key in _storyboard_cache:
            cached = _storyboard_cache[cache_key]
            print(f"[Storyboarder] Using cached storyboard ({len(cached.panels)} panels)")
            return cached

        client = await get_client()

        print(f"[Storyboarder] Analyzing paper ({len(text)} chars)...")

        # 简化的 prompt - 利用 Gemini 内置知识
        prompt = self._build_storyboard_prompt(text, title, language)

        config = GenerationConfig(
            temperature=0.7,
            max_tokens=16000  # 足够生成 100 个分镜的 JSON
        )

        response = await client.generate_text(
            prompt=prompt,
            config=config
        )

        # 解析响应
        storyboard = self._parse_response(response.content, title, text, language)

        print(f"[Storyboarder] Generated {len(storyboard.panels)} panels")

        # Cache the result
        _storyboard_cache[cache_key] = storyboard

        return storyboard

    def _build_storyboard_prompt(self, text: str, title: str, language: str) -> str:
        """
        构建分镜生成 prompt

        使用简单的自然语言格式，避免复杂 JSON 解析问题
        """
        lang_map = {"zh-CN": "中文", "en-US": "English", "ja-JP": "日本語"}
        lang_name = lang_map.get(language, "中文")

        dialogue_lang = "Chinese" if language == "zh-CN" else ("Japanese" if language == "ja-JP" else "English")

        return f"""Convert this academic paper into a Chiikawa-style educational manga storyboard.

# Paper Content
{text[:100000]}

# Requirements

1. **Characters**: Hachiware (teacher), Chiikawa (student), Usagi (occasional comic relief)

2. **Panel count**: Generate 20-80 panels based on paper complexity, covering ALL important content

3. **Order**: Follow the paper's logical sequence strictly, do NOT skip any middle sections

4. **Language**: Dialogue in {dialogue_lang}, scene descriptions in English

# Output Format

Separate each panel with ===, format as follows:

===
Panel 1
Characters: hachiware, chiikawa
Scene: Hachiware holding a book, Chiikawa looking curious, in a cozy study room
Dialogue:
- hachiware: "Let's learn something interesting today!"
- chiikawa: "Huh...? What is it?"
===
Panel 2
Characters: hachiware, chiikawa
Scene: Hachiware pointing at a whiteboard with diagrams, Chiikawa listening attentively
Dialogue:
- hachiware: "This paper is about..."
- chiikawa: "Hmm..."
===

Output all panels in this format. Ensure complete coverage of the paper content."""

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

        return Storyboard(
            title=title or "学习笔记",
            summary="",
            character_theme=self.character_theme,
            panels=panels,
            source_document=source_text[:1000],
            language=language
        )

    def _parse_natural_language_format(self, response: str) -> List[Panel]:
        """Parse natural language format storyboard (=== delimited)"""
        panels = []

        # Split by ===
        sections = re.split(r'={3,}', response)

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Parse panel number (支持 Panel 1 或 分镜 1)
            panel_num_match = re.search(r'(?:Panel|分镜)\s*(\d+)', section, re.IGNORECASE)
            panel_number = int(panel_num_match.group(1)) if panel_num_match else len(panels) + 1

            # Parse characters (支持 Characters: 或 角色:)
            chars_match = re.search(r'(?:Characters|角色):\s*(.+)', section, re.IGNORECASE)
            characters = []
            if chars_match:
                chars_str = chars_match.group(1)
                characters = [c.strip().lower() for c in re.split(r'[,，、]', chars_str)]

            # Parse scene description (支持 Scene: 或 场景:)
            scene_match = re.search(
                r'(?:Scene|场景):\s*(.+?)(?=\n(?:Dialogue|对白):|\n---|$)',
                section, re.DOTALL | re.IGNORECASE
            )
            visual_description = scene_match.group(1).strip() if scene_match else ""

            # Parse dialogue (支持 Dialogue: 或 对白:)
            dialogue = {}
            dialogue_section = re.search(
                r'(?:Dialogue|对白):\s*(.+?)(?=\n===|$)',
                section, re.DOTALL | re.IGNORECASE
            )
            if dialogue_section:
                dialogue_text = dialogue_section.group(1)
                # Match - character: "dialogue" format (支持中英文引号)
                dialogue_matches = re.findall(
                    r'-\s*(\w+):\s*["\"\'](.+?)["\"\']',
                    dialogue_text
                )
                for char, text in dialogue_matches:
                    dialogue[char.lower()] = text

            if visual_description or dialogue:
                panel = Panel(
                    panel_number=panel_number,
                    panel_type=PanelType.EXPLANATION,
                    visual_description=visual_description,
                    characters=characters or ["chiikawa", "hachiware"],
                    character_emotions={},
                    dialogue=dialogue,
                    background="pastel background"
                )
                panels.append(panel)

        return panels

    def _dict_to_panel(self, p: dict, default_num: int) -> Optional[Panel]:
        """将字典转换为 Panel 对象"""
        try:
            panel_type_str = p.get("panel_type", "explain")
            panel_type = PanelType.from_string(panel_type_str)

            return Panel(
                panel_number=p.get("panel_number", default_num),
                panel_type=panel_type,
                visual_description=p.get("visual_description", ""),
                characters=p.get("characters", ["chiikawa", "hachiware"]),
                character_emotions=p.get("character_emotions", {}),
                dialogue=p.get("dialogue", {}),
                visual_metaphor=p.get("visual_metaphor", ""),
                props=p.get("props", []),
                background=p.get("background", "simple pastel background"),
                layout_hint=p.get("layout_hint", "normal")
            )
        except Exception as e:
            print(f"[Storyboarder] Panel parse error: {e}")
            return None

    def _create_fallback_storyboard(
        self,
        title: str,
        source_text: str,
        language: str
    ) -> Storyboard:
        """创建备用分镜（解析失败时）"""
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
            language=language
        )


# 保留 CharacterLibrary 用于参考图片加载（如果需要）
class CharacterLibrary:
    """角色库 - 主要用于加载参考图片"""

    def __init__(self):
        self.image_base_path = Path(__file__).parent.parent.parent / "config" / "character_images"
        self._load_config()

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
        char = self.characters.get(char_name, {})
        ref_images = char.get("reference_images", {})
        image_paths = []

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

# Simple storyboard cache (text hash -> storyboard)
_storyboard_cache: Dict[str, Storyboard] = {}


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
