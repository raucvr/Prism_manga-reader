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
        从论文生成分镜脚本 - 三步流程

        Step 1: 生成英文技术解读
        Step 2: 用英文生成高质量漫画分镜
        Step 3: 翻译对白到目标语言（如果不是英文）
        """
        import hashlib
        global _storyboard_cache

        # Use FULL text hash to avoid cache collision between papers with similar beginnings
        # Previously used text[:5000] which caused wrong storyboards to be returned
        full_text_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_key = hashlib.md5(f"v{CACHE_VERSION}:{full_text_hash}:{title}:{language}:{self.character_theme}".encode()).hexdigest()

        print(f"[Storyboarder] ========== NEW GENERATION REQUEST ==========")
        print(f"[Storyboarder] Cache version: {CACHE_VERSION}")
        print(f"[Storyboarder] Input text length: {len(text)} chars")
        print(f"[Storyboarder] Full text hash: {full_text_hash}")
        print(f"[Storyboarder] Cache key: {cache_key}")
        print(f"[Storyboarder] Title: {title}")
        print(f"[Storyboarder] Language: {language}, Theme: {self.character_theme}")
        print(f"[Storyboarder] Text preview (first 500 chars):")
        print(f"[Storyboarder] >>> {text[:500]} <<<")
        print(f"[Storyboarder] Current cache size: {len(_storyboard_cache)} entries")
        print(f"[Storyboarder] Cached keys: {list(_storyboard_cache.keys())[:3]}...")

        if cache_key in _storyboard_cache:
            cached = _storyboard_cache[cache_key]
            print(f"[Storyboarder] ⚠️ CACHE HIT - Using cached storyboard ({len(cached.panels)} panels)")
            print(f"[Storyboarder] ⚠️ If content is wrong, restart backend to clear cache!")
            return cached

        print(f"[Storyboarder] ✓ Cache miss - generating fresh storyboard")

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
        print(f"[Storyboarder] Technical analysis generated ({len(technical_analysis)} chars)")
        print(f"[Storyboarder] Analysis preview (first 500 chars):")
        print(f"[Storyboarder] >>> {technical_analysis[:500]} <<<")

        # ========== Step 2: 用英文生成高质量漫画分镜 ==========
        print(f"[Storyboarder] Step 2: Generating manga storyboard (in English for quality)...")

        # 始终用英文生成分镜，确保最高质量
        storyboard_prompt = self._build_storyboard_prompt(technical_analysis, title, "en-US")

        config = GenerationConfig(
            temperature=0.7,
            max_tokens=32000  # 足够生成 100+ 个分镜
        )

        response = await client.generate_text(
            prompt=storyboard_prompt,
            config=config
        )

        # 解析响应（暂时设为英文）
        storyboard = self._parse_response(response.content, title, text, "en-US")

        print(f"[Storyboarder] Generated {len(storyboard.panels)} panels in English")
        # Log first 3 panels for verification
        for i, panel in enumerate(storyboard.panels[:3]):
            print(f"[Storyboarder] Panel {panel.panel_number} dialogue: {panel.dialogue}")

        # ========== Step 3: 翻译对白到目标语言 ==========
        if language != "en-US":
            print(f"[Storyboarder] Step 3: Translating dialogues to {language}...")
            storyboard = await self._translate_storyboard(storyboard, language, client, technical_analysis)
            print(f"[Storyboarder] Translation completed")

        storyboard.language = language

        # Cache the result
        _storyboard_cache[cache_key] = storyboard

        return storyboard

    async def _translate_storyboard(
        self,
        storyboard: Storyboard,
        target_language: str,
        client,
        technical_analysis: str = ""
    ) -> Storyboard:
        """
        翻译分镜对白到目标语言
        包含技术分析作为上下文，确保专业术语翻译准确
        """
        lang_map = {"zh-CN": "Simplified Chinese", "ja-JP": "Japanese"}
        target_lang_name = lang_map.get(target_language, target_language)

        # 收集所有对白
        all_dialogues = []
        for panel in storyboard.panels:
            for char, text in panel.dialogue.items():
                all_dialogues.append(f"{panel.panel_number}|{char}|{text}")

        if not all_dialogues:
            return storyboard

        # 批量翻译
        dialogues_text = "\n".join(all_dialogues)

        # 包含技术分析摘要作为上下文
        context_summary = technical_analysis[:8000] if technical_analysis else ""

        prompt = f"""You are translating manga dialogues about a scientific paper to {target_lang_name}.

# CONTEXT (Technical Analysis of the Paper)
{context_summary}

# TRANSLATION RULES
- This is a manga explaining an ACADEMIC PAPER - preserve ALL technical terms accurately
- Translate naturally in conversational {target_lang_name}
- Keep exact numbers, formulas, method names (e.g., "Amber ff14SB", "hazard ratio 0.73")
- DO NOT add explanations or notes
- Output ONLY the translations in the same format

# FORMAT
Input: panel_number|character|dialogue
Output: panel_number|character|translated_dialogue

# DIALOGUES TO TRANSLATE
{dialogues_text}

# OUTPUT (translations only):"""

        config = GenerationConfig(
            temperature=0.3,
            max_tokens=32000  # 足够翻译所有对话
        )

        response = await client.generate_text(
            prompt=prompt,
            config=config
        )

        print(f"[Storyboarder] Translation response length: {len(response.content)} chars")

        # 解析翻译结果
        translated_map = {}
        for line in response.content.strip().split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|", 2)
            if len(parts) >= 3:
                try:
                    panel_num = int(parts[0])
                    char = parts[1].strip().lower()
                    translated = parts[2].strip()
                    if panel_num not in translated_map:
                        translated_map[panel_num] = {}
                    translated_map[panel_num][char] = translated
                except ValueError:
                    continue

        print(f"[Storyboarder] Parsed {len(translated_map)} panels from translation")

        # 应用翻译 - 使用大小写不敏感匹配
        applied_count = 0
        for panel in storyboard.panels:
            if panel.panel_number in translated_map:
                # 创建小写 key 的映射
                dialogue_lower_map = {k.lower(): k for k in panel.dialogue.keys()}
                for char, translated in translated_map[panel.panel_number].items():
                    # 使用小写匹配
                    if char in dialogue_lower_map:
                        original_key = dialogue_lower_map[char]
                        panel.dialogue[original_key] = translated
                        applied_count += 1

        print(f"[Storyboarder] Applied {applied_count} translations")

        return storyboard

    def _build_analysis_prompt(self, text: str, title: str) -> str:
        """
        构建技术解读 prompt - 生成详细的英文技术分析
        """
        # Log the actual content being sent to AI
        print(f"[Storyboarder] Building analysis prompt with {len(text)} chars of paper content")
        print(f"[Storyboarder] Paper title: {title}")
        print(f"[Storyboarder] First 500 chars of paper: {text[:500]}...")

        return f"""You are a senior academic reviewer. Analyze this paper and produce a COMPREHENSIVE technical breakdown in English.

⚠️ CRITICAL: You MUST analyze ONLY the content provided below. Do NOT invent, assume, or hallucinate any information not present in the paper text.

# Paper Content (This is the ONLY source of truth - analyze ONLY what you see here)
{text[:80000]}

# Required Analysis Structure

## 1. Core Innovation & Research Question
- What is the main problem being addressed?
- What is novel about this approach?
- How does it differ from existing methods?

## 2. Methodology Deep Dive
- Detailed step-by-step technical workflow
- Key algorithms, equations, and formulas (COPY EXACTLY as written in paper)
- Data sources, sample sizes, and experimental setup
- Parameter choices and their justifications

## 3. Quantitative Results
- ALL numerical results with EXACT values (e.g., "r = 0.82", not "high correlation")
- Statistical metrics (p-values, confidence intervals, effect sizes, correlation coefficients)
- Comparison with baselines/prior work
- Performance across different conditions/datasets

## 4. Technical Implementation Details
- Software/hardware used (EXACT names: "Amber ff14SB", "TIP3P", etc.)
- Force fields, water models, simulation parameters
- Hyperparameters and their EXACT values
- Training procedures (if applicable)
- Computational requirements

## 5. Mathematical Formulas
- Write out ALL key equations EXACTLY as they appear in the paper
- Include variable definitions
- Do NOT paraphrase or simplify formulas

## 6. Key Insights & Interpretations
- What do the results mean mechanistically?
- Surprising or counterintuitive findings
- Key design decisions explained (e.g., "LDE excludes ligand information because...")

## 7. Limitations & Future Directions
- Acknowledged limitations
- Assumptions made
- Open questions for future research

## 8. Critical Evaluation
- Strengths of the methodology
- Potential weaknesses or concerns
- Alternative approaches that could be considered

⚠️ CRITICAL INSTRUCTIONS:
- COPY all numbers, equations, and technical terms EXACTLY from the paper
- Do NOT substitute terms (e.g., if paper says "MDS", don't say "UMAP")
- Do NOT round or approximate values
- Do NOT simplify equations
- Include EXACT software/tool names as written
- This analysis will be used to create an academic manga - any error will be immediately noticed by expert readers

⚠️ ANTI-HALLUCINATION RULES:
- If information is NOT in the paper, write "Not mentioned in paper" - do NOT invent details
- If you cannot find specific numbers, write "Specific value not provided" - do NOT make up numbers
- If methodology details are unclear, write "Details not specified" - do NOT assume
- NEVER fabricate authors, affiliations, or citations not in the provided text
- Your analysis MUST be traceable back to specific text in the paper above

Output the complete technical analysis in English."""

    def _build_storyboard_prompt(self, text: str, title: str, language: str) -> str:
        """
        构建分镜生成 prompt（始终用英文生成，后续翻译）

        使用简单的自然语言格式，避免复杂 JSON 解析问题
        支持不同主题：chiikawa, ghibli
        """
        # 根据主题选择不同的角色和风格
        if self.character_theme == "ghibli":
            style_name = "Studio Ghibli"
            characters_desc = "Haku (wise mentor/senior researcher), Chihiro (curious student), Calcifer (witty fire spirit who adds humor)"
            example_dialogue = """===
Panel 1
Characters: haku, chihiro
Scene: Haku and Chihiro in a magical library with floating books, soft watercolor lighting
Dialogue:
- haku: "This study uses a randomized controlled trial with n=2,847 participants..."
- chihiro: "What was the stratification criteria?"
===
Panel 2
Characters: haku, chihiro, calcifer
Scene: All three examining a glowing diagram projected in the air
Dialogue:
- haku: "The primary endpoint showed a hazard ratio of 0.73 (95% CI: 0.61-0.87)..."
- calcifer: "Hmph! But what about the selection bias in the control group?"
==="""
        elif self.character_theme == "chibikawa":
            # 原创角色主题 - 使用参考图片保证角色一致性，不用文字描述外形
            style_name = "Chibikawa (original cute characters)"
            characters_desc = """papi (the wise mentor/professor),
kumo (the curious student),
nezu (the skeptic who asks tough questions)"""
            example_dialogue = """===
Panel 1
Characters: papi, kumo
Scene: papi holding the paper, kumo looking curious, in a research lab
Dialogue:
- papi: "This study uses a randomized controlled trial with n=2,847 participants..."
- kumo: "What was the stratification criteria?"
===
Panel 2
Characters: papi, kumo, nezu
Scene: All three examining a complex diagram on a whiteboard
Dialogue:
- papi: "The primary endpoint showed a hazard ratio of 0.73 (95% CI: 0.61-0.87)..."
- nezu: "But what about the selection bias in the control group?"
==="""
        else:  # chiikawa (default)
            style_name = "Chiikawa"
            characters_desc = "Hachiware (senior researcher/professor), Chiikawa (curious PhD student), Usagi (the skeptic who asks tough questions)"
            example_dialogue = """===
Panel 1
Characters: hachiware, chiikawa
Scene: Hachiware holding the paper, pointing at a specific figure, in a research lab
Dialogue:
- hachiware: "This study uses a randomized controlled trial with n=2,847 participants..."
- chiikawa: "What was the stratification criteria?"
===
Panel 2
Characters: hachiware, chiikawa, usagi
Scene: All three examining a complex diagram on a whiteboard
Dialogue:
- hachiware: "The primary endpoint showed a hazard ratio of 0.73 (95% CI: 0.61-0.87)..."
- usagi: "But what about the selection bias in the control group?"
==="""

        return f"""Create a {style_name}-style manga explaining this paper.

# Technical Analysis (This is your ONLY source of truth)
{text[:100000]}

# Key Points
- Readers: Nobel Prize-level scholars who LOVE {style_name} style
- Balance: Academic rigor + {style_name} charm
- ⚠️ CRITICAL: ONLY use facts from the Technical Analysis above - NO hallucination
- ⚠️ If the analysis says "Not mentioned" or "Not specified", do NOT invent that information
- Copy exact numbers, formulas, method names (e.g., "Amber ff14SB" not "CHARMM36m")
- Dialogue can be as LONG as needed to explain concepts fully - NEVER truncate or cut off mid-sentence
- If a concept requires detailed explanation, use the full dialogue without shortening
- Every fact in your manga MUST come from the Technical Analysis above

# Characters
{characters_desc}

# Format
- 40-100 panels, dialogue in English
- Background: simple classroom/lab (keep consistent!)
- Use === to separate panels:

{example_dialogue}

Generate all panels."""

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
                    characters=characters or self._get_default_characters(),
                    character_emotions={},
                    dialogue=dialogue,
                    background="simple classroom"
                )
                panels.append(panel)

        # 按 panel_number 排序，确保故事顺序正确
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
                visual_description=p.get("visual_description", ""),
                characters=p.get("characters", self._get_default_characters()),
                character_emotions=p.get("character_emotions", {}),
                dialogue=p.get("dialogue", {}),
                visual_metaphor=p.get("visual_metaphor", ""),
                props=p.get("props", []),
                background=p.get("background", "simple classroom"),
                layout_hint=p.get("layout_hint", "normal")
            )
        except Exception as e:
            print(f"[Storyboarder] Panel parse error: {e}")
            return None

    def _get_default_characters(self) -> list:
        """根据主题返回默认角色列表"""
        if self.character_theme == "ghibli":
            return ["haku", "chihiro"]
        elif self.character_theme == "chibikawa":
            return ["pip", "kumomo"]
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
        elif self.character_theme == "chibikawa":
            # 原创角色主题
            panels = [
                Panel(
                    panel_number=1,
                    panel_type=PanelType.TITLE,
                    visual_description="Pip (orange puppy with floppy ears) holding a book, Kumomo (blue cloud creature with leaf sprout) looking curious",
                    characters=["pip", "kumomo"],
                    character_emotions={"pip": "explaining", "kumomo": "curious"},
                    dialogue={"pip": "今天来学习一个有趣的话题！", "kumomo": "是什么呢？"},
                    background="simple study room"
                ),
                Panel(
                    panel_number=2,
                    panel_type=PanelType.EXPLANATION,
                    visual_description="Pip at whiteboard explaining, Kumomo and Pippin (hedgehog with striped tail) listening",
                    characters=["pip", "kumomo", "pippin"],
                    character_emotions={"pip": "explaining", "kumomo": "thinking", "pippin": "skeptical"},
                    dialogue={"pip": "让我来解释一下～", "pippin": "这个靠谱吗..."},
                    background="classroom"
                ),
                Panel(
                    panel_number=3,
                    panel_type=PanelType.CONCLUSION,
                    visual_description="All three original characters celebrating together - Pip, Kumomo, and Pippin",
                    characters=["pip", "kumomo", "pippin"],
                    character_emotions={"pip": "happy", "kumomo": "happy", "pippin": "excited"},
                    dialogue={"kumomo": "原来如此！", "pip": "做得好！", "pippin": "还不错嘛！"},
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
            language=language
        )


# 保留 CharacterLibrary 用于参考图片加载（如果需要）
class CharacterLibrary:
    """角色库 - 主要用于加载参考图片"""

    # Chibikawa 原创角色 - 直接使用 kumo, nezu, papi 作为名字
    CHIBIKAWA_CHARACTERS = {
        "kumo": "kumo",   # 好奇的学生
        "nezu": "nezu",   # 怀疑论者
        "papi": "papi",   # 导师教授
    }

    # Chibikawa 角色对应的参考图片文件
    CHIBIKAWA_IMAGES = {
        "kumo": "kumo.jpeg",
        "nezu": "nezu.jpeg",
        "papi": "papi.jpeg",
    }

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
        image_paths = []
        char_name_lower = char_name.lower()

        # 检查是否是 chibikawa 原创角色
        normalized_name = self.CHIBIKAWA_CHARACTERS.get(char_name_lower)
        if normalized_name:
            # 使用 chibikawa 原创角色的参考图片
            img_filename = self.CHIBIKAWA_IMAGES.get(normalized_name)
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

    def get_all_chibikawa_reference_images(self) -> List[str]:
        """获取所有 chibikawa 原创角色的参考图片"""
        image_paths = []
        for char_name, img_filename in self.CHIBIKAWA_IMAGES.items():
            full_path = self.image_base_path / img_filename
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

# Cache version - increment this to invalidate all cached storyboards
# v2: Added translation context fix for zh-CN
# v3: Removed 25-char truncation limit for CJK
# v4: Added chibikawa theme with original characters (Pip, Kumomo, Pippin)
# v5: Fixed cache key collision - now uses full text hash instead of first 5000 chars
CACHE_VERSION = 5

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
