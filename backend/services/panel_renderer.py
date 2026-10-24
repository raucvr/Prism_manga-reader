"""
Panel renderer v2 — one Seedream image per panel (or per two-panel strip).

Prompt template (≈180–260 words, under Seedream's 600-word guidance), derived from the experiments:
  style + reference mapping → layout line → explicit character scope → scene / action / emotions
  → quoted bubbles with speakers (+ tail rule) → narration → ≤2 labels → text rules → "no other text" rule

v3 (``text_free``): build_art_prompt draws the picture with NO text at all — bubbles, narration and board exhibits are
typeset by code afterwards — and reserves room for them: a calm top band (scene) or a large blank whiteboard (board).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from engines.base import BaseEngine, ImageContent, ImageGenerationConfig, ImageResponse
from services.storyboarder_v2 import SCENE_CHART_RE, Cast, PanelV2

ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


def _ordinal(i: int) -> str:
    return ORDINALS[i] if i < len(ORDINALS) else f"#{i + 1}"

TEXT_RULES = {
    "en-US": "Render every quoted text exactly as written, in English, in a large, clean, legible hand-lettered comic font.",
    "zh-CN": "Render every quoted text exactly as written, in Simplified Chinese characters, large and legible, in a clean rounded comic font.",
    "ja-JP": "Render every quoted text exactly as written, in Japanese, large and legible, in a clean rounded comic font.",
}
BUBBLE_RULES = "Each speech bubble has a tail pointing at its speaker's mouth; characters without a bubble are silent. Nothing is cut off; no extra words."
NO_STRAY_TEXT = ("Apart from the quoted texts above, the image contains NO other letters, words, numbers or symbols: any diagram, poster, board, "
                 "screen, book or sign is wordless (plain boxes, arrows, shapes, dots or scribble lines instead of writing).")
SHEET_CAPTIONS = " Never copy the captions on the reference sheets (e.g. \"Soft Spikes\")."


@dataclass
class RenderResult:
    image: ImageContent
    prompt: str
    usage: dict
    size: str
    seconds: float


def _reference_block(cast: Cast, ref_names: list[str], force_ordinals: bool = False) -> str:
    looks = {m.name: m.look for m in cast.members}
    if not ref_names:
        return ""
    if len(ref_names) == 1 and not force_ordinals:
        n = ref_names[0]
        return f" The reference image is {n}: {looks.get(n, n)}. Draw {n} exactly as in the reference sheet: same species, colors, face and proportions."
    parts = [f"The {_ordinal(i)} reference image is {n}: {looks.get(n, n)}." for i, n in enumerate(ref_names)]
    return " " + " ".join(parts) + " Draw every character exactly as in its reference sheet: same species, colors, face and proportions."


def _looks_block(cast: Cast, present: list[str], ref_names: list[str]) -> str:
    """Looks of the characters in the picture that have no reference sheet (Seedream would otherwise get a bare name)."""
    looks = [f"{m.name}: {m.look}" for m in cast.members if m.name in present and m.name not in ref_names]
    return f" Characters without a reference sheet look like this: {'; '.join(looks)}." if looks else ""


def _scope_line(cast: Cast, present: list[str], noun: str = "panel") -> str:
    others = [m.name for m in cast.members if m.name not in present]
    line = f"Characters in this {noun}: {', '.join(present)} only."
    if others:
        line += f" {', '.join(others)} do NOT appear."
    return line + (" No other people, children, animals or creatures appear anywhere in the picture; apart from the listed characters, "
                   "every room, office, court, street, vehicle and building in it is empty of people.")


# a scene that mentions written text makes Seedream letter gibberish ("a sentence with one word blacked out")
TEXTY_RE = re.compile(r"\b(?:sentences?|text|words?|letters?|writing|written|captions?|headlines?|paragraphs?|notes?|numbers?|digits?"
                      r"|sums?|equations?|formulas?|arithmetic|calculations?)\b|\d", re.I)
TEXTY_HINT = ("Apart from the quoted texts, draw any sentence, word, number or formula in this scene as blank cards, blocks or bars "
              "with no letters or digits.")


def _panel_block(panel: PanelV2, label: Optional[str] = None, per_panel_scope: bool = False) -> str:
    lines = []
    if label:
        lines.append(label)
    if per_panel_scope:
        lines.append(f"Characters in this panel: {', '.join(panel.characters)} only; nobody else.")
    if panel.scene:
        lines.append(f"Scene: {panel.scene}")
    action = panel.visual or ""
    if panel.emotions:
        action += " Emotions: " + ", ".join(f"{k} is {v}" for k, v in panel.emotions.items()) + "."
    lines.append(f"Action: {action}")
    if TEXTY_RE.search(f"{panel.scene} {panel.visual}"):
        lines.append(TEXTY_HINT)
    if panel.dialogue:
        lines.append("Speech bubbles:")
        for b in panel.dialogue:
            lines.append(f'- {b.speaker} says: "{b.text}"')
        speakers = sorted({b.speaker for b in panel.dialogue})
        silent = [c for c in panel.characters if c not in speakers]
        if silent:
            lines.append(f"Speakers: {', '.join(speakers)}. {', '.join(silent)} {'is' if len(silent) == 1 else 'are'} silent.")
    else:
        lines.append("No speech bubbles in this panel.")
    if panel.narration:
        lines.append(f'Narration box along the top edge: "{panel.narration}"')
    if panel.labels:
        lines.append("Labels drawn on props: " + ", ".join(f'"{x}"' for x in panel.labels[:2]))
    return "\n".join(lines)


def _feedback_block(feedback: Optional[str]) -> str:
    return f"\n\nThe previous attempt had these problems — fix them: {feedback}" if feedback else ""


def build_single_prompt(panel: PanelV2, cast: Cast, language: str, ref_names: list[str], feedback: Optional[str] = None, anchor: bool = False) -> str:
    style = cast.style + _reference_block(cast, ref_names, force_ordinals=anchor) + _looks_block(cast, panel.characters, ref_names)
    if anchor:
        style += (f" The {_ordinal(len(ref_names))} reference image is the previous panel of this same manga: match its palette, line weight and "
                  "lighting, but do not copy its composition, poses or any of its text.")
    return (
        f"{style}\nA single manga panel, portrait, with a thin black border and a small white margin. {_scope_line(cast, panel.characters)}\n\n"
        f"{_panel_block(panel)}\n\n{TEXT_RULES.get(language, TEXT_RULES['en-US'])} {BUBBLE_RULES} {NO_STRAY_TEXT}"
        f"{SHEET_CAPTIONS if ref_names else ''}{_feedback_block(feedback)}"
    )


def build_strip_prompt(panels: list[PanelV2], cast: Cast, language: str, ref_names: list[str], feedback: Optional[str] = None) -> str:
    present = []
    for p in panels:
        for c in p.characters:
            if c not in present:
                present.append(c)
    blocks = "\n\n".join(_panel_block(p, label=f"PANEL {i + 1} ({pos}):", per_panel_scope=True) for i, (p, pos) in enumerate(zip(panels, ["left", "right"])))
    return (
        f"{cast.style}{_reference_block(cast, ref_names)}{_looks_block(cast, present, ref_names)}\nA manga strip with exactly two equal panels side by side, separated by a white gutter, each with a thin black "
        f"border. Each panel is its own scene as described. {_scope_line(cast, present, noun='strip')}\n\n{blocks}\n\n"
        f"{TEXT_RULES.get(language, TEXT_RULES['en-US'])} {BUBBLE_RULES} {NO_STRAY_TEXT}{SHEET_CAPTIONS if ref_names else ''}"
        f"{_feedback_block(feedback)}"
    )


# ----------------------------------------------------------------------------- v3: text-free art (the text is typeset later)

WORDLESS = ("WORDLESS IMAGE: no letters, words, numbers, digits, symbols, speech bubbles, captions or signatures anywhere; "
            "screens, books, papers, clipboards, cards, boards, signs and clock faces are blank or show only simple shapes.")
ART_SHEET_CAPTIONS = " Never copy the captions on the reference sheets."
ART_TEXTY_HINT = "Draw anything in this scene that would carry writing (a sentence, word, number or formula) as blank cards, blocks or bars."
SCENE_COMPOSITION = ("Composition: the top 28% of the panel is calm, plain background (wall or sky) with nothing in it, reserved for text "
                     "added later; no whiteboard, blackboard or screen anywhere in the picture; all heads are below that band")
BOARD_COMPOSITION = ("Composition: the top 22% of the panel is plain empty wall reserved for text. Below it, a LARGE, COMPLETELY BLANK white "
                     "whiteboard spans almost the full width of the panel and about 48% of its height, seen straight on, evenly lit, with a "
                     "thin dark frame. Nothing stands in front of it or overlaps it, and nothing is written or drawn on it.")
ART_MAX_WORDS = 450
# a scene that mentions writing gets the blank-cards hint ("nezu scribbles on his clipboard" came back with glyphs on it)
ART_TEXTY_RE = re.compile(TEXTY_RE.pattern + r"|\b(?:writes?|scribbl\w*|jot(?:s|ted|ting)?)\b", re.I)
# quoted words invite the image model to letter them: dialogue never enters the art prompt, quoted bits of a scene are dropped
# (a straight single quote only opens after a non-word character, so "kumo's" and "papis'" stay)
QUOTED_RE = re.compile(r"\"[^\"\n]*\"|“[^”\n]*”|‘[^’\n]*’|「[^」\n]*」|『[^』\n]*』|«[^»\n]*»|(?<![\w'])'[^'\n]+'(?!\w)")
# the exhibit is drawn by code, so the picture needs a blank board, never a chart the image model would make up
CHART_RE = re.compile(r"\b(?:(?:bar|line|pie|flow|results?|comparison|data|scores?|summary|recap)\s+)?"
                      r"(?:charts?|graphs?|plots?|diagrams?|flowcharts?|infographics?|histograms?|pipelines?|slides?)\b", re.I)
QUALIFIED_RE = re.compile(r"\b(?:comparison|results?|data|scores?|summary|recap|numbers?|values?)\s+(?:tables?|lists?|bars?)\b", re.I)
KIND_WORDS = {"bar": r"bars?", "table": r"tables?", "list": r"lists?|checklists?", "flow": r"arrows?", "links": r"arrows?"}
# "the kitchen table" is furniture even when the exhibit is a table
FURNITURE_CUES = frozenset("sit sits sat sitting seated around on under beneath kitchen dining coffee wooden picnic work side "
                           "round lab breakfast folding cluttered".split())
# a clause describing what is on the board ("the board where the Transformer bar towers over the rest") invites a chart
BOARD_CONTENT_RE = re.compile(r"\b((?:white)?board)\s+(?:where|that shows|which shows|that lists|which lists|showing|shows|displaying"
                              r"|listing|comparing|full of|covered (?:with|in)|filled with)\b.*?(?=[,;.!?]|\s+(?:while|as|but|and then)\b|$)",
                              re.I)
_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"
# a whole number token: 28.4, 1,000, 4.9%, 3:00, 1/2, 10x, 2.3 · 10^19, 2.3×10¹⁹, 1e19, 10¹⁹ (both ends of "4-5");
# never a piece of a name (P100, GPT-2, ConvS2S, 3D, 50th)
NUMBER_RE = re.compile(rf"(?<![\w.])(?<![A-Za-z][\-−])[+\-−±~≈]?\d+(?:[.,:/]\d+)*(?:\s*[·×x*]\s*10(?:\^?[+\-−]?\d+|[{_SUPERSCRIPTS}]+))?"
                       rf"(?:[eE][+\-−]?\d+)?(?:\s*%|[{_SUPERSCRIPTS}]+|[x×])?(?!\w)")
OFF_PANEL = "something just out of view"
COUNT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")


def _tidy(text: str) -> str:
    return re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s+", " ", text)).strip()


def _unquoted(text: Optional[str]) -> str:
    return _tidy(re.sub(r"[\"“”「」『』«»]", "", QUOTED_RE.sub(" ", text or "")))   # stray marks of an unbalanced quote go too


def _no_numbers(text: str) -> str:
    """Digits invite the image model to draw them: a small count becomes a word ("4 ovens" -> "four ovens"), any other
    number (a result, a percentage, a year) is dropped; names with digits in them (P100, GPT-2) stay."""
    def word(m: re.Match) -> str:
        s = m.group(0)
        return COUNT_WORDS[int(s)] if re.fullmatch(r"[0-9]+", s) and int(s) < len(COUNT_WORDS) else " "   # not isdigit: "10¹⁹" is
    return _tidy(NUMBER_RE.sub(word, text))


def _no_chart(text: str) -> str:
    """A chart the writer left in a scene ("Kumo looks at the bar chart") would come back as a fake one, and a scene has
    no board to put it on: the characters look at something out of view instead."""
    return re.sub(rf"\b(?:(?:the|a|an|this|that|these|those|their|its)\s+)?(?:(?:rising|growing|shrinking|falling|tall|short|big|small)\s+)?"
                  rf"(?:{SCENE_CHART_RE.pattern})", OFF_PANEL, text, flags=re.I)


def _clip_words(text: str, limit: Optional[int]) -> str:
    words = text.split()
    if limit is None or len(words) <= limit:
        return text
    return " ".join(words[:limit]).rstrip(",;:-") + "." if limit > 0 else ""


def _board_text(text: str, kind: str) -> str:
    """A board panel's setting or action with every chart word turned into "board", what is on the board left out and
    its numbers dropped."""
    text = QUALIFIED_RE.sub("board", CHART_RE.sub("board", text))
    if kind in KIND_WORDS:
        def board(m: re.Match) -> str:
            before = re.findall(r"[a-z]+", text[max(0, m.start() - 40):m.start()].lower())[-3:]
            return m.group(0) if kind == "table" and FURNITURE_CUES & set(before) else "board"
        text = re.sub(rf"\b(?:{KIND_WORDS[kind]})\b", board, text, flags=re.I)
    text = re.sub(r"\b((?:white)?board)(?:['’]s)?(?:\s+board)+\b", r"\1", _no_numbers(text), flags=re.I)
    return _tidy(BOARD_CONTENT_RE.sub(r"\1", text))


def stage_order(panel: PanelV2) -> list[str]:
    """Left to right: the speakers in the order they speak (bubbles are read left to right), then the silent characters."""
    speakers = [b.speaker for b in panel.dialogue if b.speaker in panel.characters]
    return list(dict.fromkeys([*speakers, *panel.characters]))


def _placement(names: list[str]) -> str:
    if len(names) == 1:
        return f"{names[0]} in the center"
    if len(names) == 2:
        return f"{names[0]} on the left and {names[1]} on the right"
    if len(names) == 3:
        return f"{names[0]} on the left, {names[1]} in the middle and {names[2]} on the right"
    return f"from left to right {', '.join(names[:-1])} and {names[-1]}" if names else ""


def build_art_prompt(panel: PanelV2, cast: Cast, ref_names: list[str], feedback: Optional[str] = None, anchor: bool = False) -> str:
    """v3: the picture of one panel with NO text in it — dialogue, narration, labels and the board's exhibit are typeset
    by code afterwards, so none of them is ever sent. The top stays calm for that text, the first speaker stands on the
    left so the bubbles read in order, and a board panel gets a large blank whiteboard. At most ART_MAX_WORDS words."""
    style = cast.style + _reference_block(cast, ref_names, force_ordinals=anchor) + _looks_block(cast, panel.characters, ref_names)
    if anchor:
        style += (f" The {_ordinal(len(ref_names))} reference image is the previous panel of this same manga: match its palette, "
                  "line weight and lighting, but do not copy its composition or poses.")
    head = f"{style}\nA single manga panel, portrait, with a thin black border and a small white margin. {_scope_line(cast, panel.characters)}"
    # as lettering: a "board" panel without an exhibit is lettered as a scene, so it is drawn as one
    board = panel.kind == "board" and isinstance(panel.board, dict) and bool(panel.board)
    setting, action = _unquoted(panel.scene), _unquoted(panel.visual)
    if board:
        kind = str(panel.board.get("kind") or "").lower()
        setting, action = _board_text(setting, kind), _board_text(action, kind)
    else:
        setting, action = _no_numbers(_no_chart(setting)), _no_numbers(_no_chart(action))
    emotions = (_no_numbers(_unquoted(" Emotions: " + ", ".join(f"{k} is {v}" for k, v in panel.emotions.items()) + "."))
                if panel.emotions else "")
    order = stage_order(panel)
    where = _placement(order)
    if board:
        composition = BOARD_COMPOSITION + (f" At the bottom, below the board, {where} {'is' if len(order) == 1 else 'are'} shown from "
                                           "the chest up." if where else "")
    else:
        composition = SCENE_COMPOSITION + (f"; {where}." if where else ".")
    texty = not board and ART_TEXTY_RE.search(f"{panel.scene} {panel.visual}")
    rules = WORDLESS + (ART_SHEET_CAPTIONS if ref_names else "")
    fix = _unquoted(feedback)

    def assemble(setting_words: Optional[int], action_words: Optional[int], fix_words: Optional[int]) -> str:
        lines = [f"Setting: {_clip_words(setting, setting_words)}"] if setting else []
        act = _tidy(f"{_clip_words(action, action_words)} {emotions}")
        if act:
            lines.append(f"Action (draw nothing on the board): {act}" if board else f"Action: {act}")
        if texty:
            lines.append(ART_TEXTY_HINT)
        lines.append(composition)
        return f"{head}\n\n" + "\n".join(lines) + f"\n\n{rules}{_feedback_block(_clip_words(fix, fix_words))}"

    # only an over-long prompt is clipped: the scene and the feedback give way, the rules never do
    for caps in ((None, None, None), (45, 90, 80), (30, 60, 60), (20, 40, 40), (15, 25, 25), (15, 25, 0)):
        prompt = assemble(*caps)
        if len(prompt.split()) <= ART_MAX_WORDS:
            break
    return prompt


def select_reference_names(panels: list[PanelV2], sheets: dict[str, ImageContent], cast: Cast) -> list[str]:
    """Sheets of the characters that appear (in cast order); all sheets when none of them has one."""
    present = {c for p in panels for c in p.characters}
    names = [m.name for m in cast.members if m.name in present and m.name in sheets]
    if not names:
        names = [m.name for m in cast.members if m.name in sheets]
    return names


class PanelRenderer:
    def __init__(self, image_engine: BaseEngine, cast: Cast, sheets: dict[str, ImageContent], language: str,
                 panel_size: tuple[int, int] = (1024, 1536), strip_size: tuple[int, int] = (2240, 1120),
                 fast: bool = False, output_format: str = "jpeg", text_free: bool = False):
        self.engine = image_engine
        self.cast = cast
        self.sheets = sheets
        self.language = language
        self.panel_size = panel_size
        self.strip_size = strip_size
        self.fast = fast
        self.output_format = output_format
        self.text_free = text_free   # v3: wordless art (build_art_prompt); the text is typeset afterwards

    def _config(self, size: tuple[int, int]) -> ImageGenerationConfig:
        cfg = ImageGenerationConfig(width=size[0], height=size[1], style=self.cast.theme)
        cfg.extra = {"fast": self.fast, "output_format": self.output_format, "watermark": False}
        return cfg

    async def _generate(self, prompt: str, refs: list[ImageContent], size: tuple[int, int]) -> RenderResult:
        import time
        t0 = time.time()
        response: ImageResponse = await self.engine.generate_image(prompt, self._config(size), reference_images=refs)
        return RenderResult(image=response.images[0], prompt=prompt, usage=getattr(response, "usage", {}) or {},
                            size=getattr(response, "size", None) or f"{size[0]}x{size[1]}", seconds=round(time.time() - t0, 1))

    async def render_panel(self, panel: PanelV2, feedback: Optional[str] = None, anchor: Optional[ImageContent] = None) -> RenderResult:
        names = select_reference_names([panel], self.sheets, self.cast)
        refs = [self.sheets[n] for n in names]
        use_anchor = anchor is not None and len(refs) < 10
        if use_anchor:
            refs = refs + [anchor]
        if self.text_free:
            prompt = build_art_prompt(panel, self.cast, names, feedback=feedback, anchor=use_anchor)
        else:
            prompt = build_single_prompt(panel, self.cast, self.language, names, feedback=feedback, anchor=use_anchor)
        return await self._generate(prompt, refs, self.panel_size)

    async def render_strip(self, panels: list[PanelV2], feedback: Optional[str] = None) -> RenderResult:
        if self.text_free:   # v3 letters every panel on its own art; a lettered strip prompt here would put text in the art
            raise ValueError("text-free (v3) art is drawn one panel at a time; use render_panel")
        names = select_reference_names(panels, self.sheets, self.cast)
        refs = [self.sheets[n] for n in names]
        prompt = build_strip_prompt(panels, self.cast, self.language, names, feedback=feedback)
        return await self._generate(prompt, refs, self.strip_size)
