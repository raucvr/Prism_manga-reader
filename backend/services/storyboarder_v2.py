"""
Storyboarder v2 — paper → brief → chaptered manga script (JSON), with budget fitting and translation.

Design rules (from the owner's objective + Seedream experiments):
- one idea per panel, ≤2 bubbles × ≤14 words, narration ≤20 words (zh/ja: ≤22 / ≤32 characters)
- all facts come from the paper; exact numbers; no invented results
- no underscores / LaTeX in on-image text (the image model drops them) — plain words or √ ² ×
- text is never truncated: over-budget lines are rewritten by the model in a "fit" pass

v3 (build_v3): the reader must come away knowing the abstract and having seen its key numbers. The brief splits the
abstract into claims with verified quotes; the story runs hook → problem → idea → how → proof → limits → payoff → recap
on one running example, with scene panels and board panels (exhibits drawn by code from the paper's exact values); a
deterministic lint (services.boards / services.evidence, budgets, claim coverage) drives up to two fix rounds; the
fact-check covers board strings; services.abstract_quiz gates the visible text before translation.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from decimal import Decimal, InvalidOperation
from typing import Optional

from engines.base import BaseEngine, GenerationConfig, Message
from services.paper_extractor import PaperText


# ----------------------------------------------------------------------------- data

def _as_list(value) -> list:
    """Model replies sometimes send one item where a list belongs ("labels": "Attention"); never iterate a str or dict."""
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _claim_ids(value) -> list[int]:
    """Claim ids of a reply: 3, "3", "C3", "#3", "claim 3", 3.0 and {"id": 3} all mean claim 3."""
    out = []
    for x in _as_list(value):
        if isinstance(x, dict):
            x = x.get("id", x.get("claim"))
        if isinstance(x, bool):
            continue
        if isinstance(x, float) and x.is_integer():
            x = int(x)
        m = re.fullmatch(r"\s*(?:claims?\s*|[C#]\s*)?(\d+)\s*", str(x), re.I)
        if m and int(m.group(1)) not in out:
            out.append(int(m.group(1)))
    return out


def _text(value) -> str:
    """A text field of a reply: a string (or number) stripped; a list or object is no text (str() would draw "['...']")."""
    return str(value).strip() if isinstance(value, (str, int, float)) and not isinstance(value, bool) else ""


def _labels(value) -> list[str]:
    """Label texts of a reply: a string is one label (never split into characters); non-text items are dropped."""
    return [str(x).strip() for x in _as_list(value) if isinstance(x, (str, int, float)) and str(x).strip()]


@dataclass
class Bubble:
    speaker: str
    text: str


@dataclass
class PanelV2:
    n: int
    chapter: str
    title: str
    characters: list[str]
    scene: str
    visual: str
    dialogue: list[Bubble] = field(default_factory=list)
    narration: str = ""
    labels: list[str] = field(default_factory=list)
    emotions: dict[str, str] = field(default_factory=dict)
    covers: list[str] = field(default_factory=list)   # ids of the brief's key points this panel explains
    detail: str = ""     # the precise facts behind the panel, shown as text below the picture (never drawn)
    # v3: "scene" (the cast acting) or "board" (an exhibit the program draws on a blank whiteboard in the scene)
    kind: str = "scene"
    board: Optional[dict] = None     # exhibit spec (services/boards.py), only for kind "board"
    beat: str = ""                   # story beat: hook|problem|idea|how|proof|limits|payoff|recap
    claims: list[int] = field(default_factory=list)   # indexes of StoryboardV2.abstract_claims this panel conveys

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dialogue"] = [asdict(b) for b in self.dialogue]
        return d

    @classmethod
    def from_dict(cls, d: dict, default_n: int) -> "PanelV2":
        def text(key: str) -> str:
            value = d.get(key)
            return str(value).strip() if value is not None else ""

        dialogue = []
        for b in _as_list(d.get("dialogue")):
            if isinstance(b, dict) and str(b.get("text") or "").strip():   # "  " is no line
                dialogue.append(Bubble(speaker=str(b.get("speaker") or "").strip().lower(), text=str(b["text"]).strip()))
        characters = d.get("characters")
        if isinstance(characters, str):
            characters = characters.split(",")
        elif isinstance(characters, dict):  # {"papi": "curious", ...}
            characters = list(characters)
        emotions = d.get("emotions") if isinstance(d.get("emotions"), dict) else {}
        try:
            n = int(d.get("n", default_n) or default_n)
        except (TypeError, ValueError):
            n = default_n
        return cls(
            n=n,
            chapter=text("chapter").lower() or "main",
            title=text("title"),
            characters=[str(c).strip().lower() for c in _as_list(characters) if c is not None and str(c).strip()],
            scene=text("scene"),
            visual=text("visual"),
            dialogue=dialogue,
            narration=text("narration"),
            labels=_labels(d.get("on_image_labels") or d.get("labels"))[:2],
            emotions={str(k).lower(): str(v).strip() for k, v in emotions.items() if v is not None and str(v).strip()},
            covers=list(dict.fromkeys(c.strip() for x in _as_list(d.get("covers")) if x is not None for c in str(x).split(",") if c.strip())),
            detail=text("detail"),
            kind="board" if text("kind").lower() == "board" or text("type").lower() == "board" else "scene",
            board=d.get("board") if isinstance(d.get("board"), dict) else None,
            beat=text("beat").lower(),
            claims=_claim_ids(d.get("claims", d.get("abstract_claims"))),
        )


@dataclass
class Chapter:
    id: str
    title: str
    panel_numbers: list[int] = field(default_factory=list)


@dataclass
class StoryboardV2:
    title: str
    summary: str
    theme: str
    language: str
    chapters: list[Chapter]
    panels: list[PanelV2]
    brief: dict = field(default_factory=dict)
    card: dict = field(default_factory=dict)   # the whole paper in a few lines, shown as text in the reader
    # v3
    version: int = 2
    abstract_claims: list[dict] = field(default_factory=list)   # [{"id": 0, "text": "...", "quote": "..."}] from the paper's abstract
    running_example: str = ""
    quiz: dict = field(default_factory=dict)                    # last abstract-quiz result on the visible text

    def to_dict(self) -> dict:
        d = {
            "title": self.title,
            "summary": self.summary,
            "theme": self.theme,
            "language": self.language,
            "chapters": [asdict(c) for c in self.chapters],
            "panels": [p.to_dict() for p in self.panels],
            "brief": self.brief,
            "card": self.card,
        }
        if self.version >= 3:
            d.update({"version": self.version, "abstract_claims": self.abstract_claims, "running_example": self.running_example, "quiz": self.quiz})
        return d


@dataclass
class CastMember:
    name: str
    role: str          # mentor | student | skeptic
    look: str          # short physical description used in prompts
    voice: str         # how they talk


@dataclass
class Cast:
    theme: str
    style: str                 # art-style sentence for the image prompt
    members: list[CastMember]

    def names(self) -> list[str]:
        return [m.name for m in self.members]

    def describe(self) -> str:
        role_desc = {"mentor": "the professor/mentor", "student": "the student", "skeptic": "the critic"}
        return "\n".join(f"- {m.name}: {role_desc.get(m.role, m.role)}. Looks: {m.look}. Voice: {m.voice}" for m in self.members)


LANG_NAMES = {"en-US": "English", "zh-CN": "Simplified Chinese", "ja-JP": "Japanese"}
BUDGETS = {
    # bubble words (en) / characters (cjk), narration, max bubbles per panel
    "en-US": {"bubble": 14, "narration": 20, "label": 3, "unit": "words"},
    "zh-CN": {"bubble": 22, "narration": 32, "label": 6, "unit": "characters"},
    "ja-JP": {"bubble": 24, "narration": 34, "label": 8, "unit": "characters"},
}
PANEL_RANGES = {"short": (12, 18), "auto": None, "long": (28, 40)}
CHAPTER_PLAN = [
    ("hook", "Why should anyone care?"),
    ("problem", "The old problem"),
    ("idea", "The key idea"),
    ("how", "How it works"),
    ("evidence", "Does it work? The evidence"),
    ("payoff", "Back to the example"),
    ("limits", "Limits and open questions"),
    ("takeaway", "Takeaway"),
]


# ----------------------------------------------------------------------------- helpers

PEOPLE_RE = re.compile(r"\b(?:people|persons?|man|men|woman|women|boys?|girls?|child(?:ren)?|kids?|students?|pupils?|librarians?|teachers?"
                       r"|workers?|crowds?|audiences?|spectators?|guides?|hikers?|scientists?|researchers?|engineers?|doctors?|players?"
                       r"|judges?|referees?|customers?|visitors?|staff|humans?|figures?|clerks?|passengers?|tourists?|chefs?|bakers?|athletes?)\b", re.I)
NO_PEOPLE_RE = re.compile(r"\b(?:no|without|not any|no other|nobody|no one)\b(?:\s+\w+){0,3}?\s+(?:people|persons?|humans?|figures?|crowds?"
                          r"|audiences?|spectators?|staff)\b", re.I)


COPIES_RE = re.compile(r"\b(?:\d+|two|three|four|five|six|seven|eight|nine|ten|twelve|many|several|dozens? of)\s+(?:\w+\s+){0,3}?"
                       r"(?:papis?|kumos?|nezus?|dogs?|hedgehogs?|clouds?|copies|clones|versions|twins)\b"
                       r"|\b(?:clones?|copies|mini(?:ature)?|tiny|little|small)\s+(?:versions?\s+of\s+)?(?:papis?|kumos?|nezus?)\b", re.I)


def _mentions_people(text: str) -> bool:
    """A scene that names people to draw ("eight small librarians") or copies of a cast member ("eight identical small
    papis"), not one that says there are none."""
    text = NO_PEOPLE_RE.sub(" ", text)
    return bool(PEOPLE_RE.search(text) or COPIES_RE.search(text))


DRAWN_CLAIM_RE = re.compile(r"\d|\b(?:charts?|bars?|graphs?|plots?|stacks?|compare[sd]?|comparison|taller|shorter|bigger|smaller)\b", re.I)
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]")
LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*")


def _code_like(token: str) -> bool:
    """Numbers, acronyms and CamelCase names (28.4, BLEU, GPT-4, ResNet-152) that a translation keeps in Latin script."""
    return any(c.isdigit() for c in token) or sum(c.isupper() for c in token) >= 2


def _latin_keep(brief: dict) -> frozenset:
    """Lower-cased words a translation may keep in Latin script: the brief's capitalised words (Transformer) and glossary terms."""
    text = json.dumps({k: v for k, v in brief.items() if k != "factcheck"}, ensure_ascii=False)
    terms = " ".join(str(g.get("term") or "") if isinstance(g, dict) else str(g) for g in _as_list(brief.get("glossary")))
    return frozenset(w.lower() for w in re.findall(r"\b[A-Z][A-Za-z]{3,}", text) + re.findall(r"[A-Za-z]{4,}", terms))


def _has_script(text: str, language: str, keep: frozenset = frozenset()) -> bool:
    """True when a translated line contains the target script, or only what the translation prompt says to keep exact:
    numbers, acronyms and CamelCase names ("28.4 BLEU！", "ResNet-152？") and the words in ``keep`` ("Transformer！")."""
    if language not in ("zh-CN", "ja-JP"):
        return True
    if CJK_RE.search(text):
        return True
    rest = LATIN_TOKEN_RE.sub(lambda m: " " if _code_like(m.group()) else m.group(), text)
    return all(w.lower() in keep for w in re.findall(r"[A-Za-z]{4,}", rest))


def text_len(text: str, language: str) -> int:
    """English: words. Chinese/Japanese: display width (full-width = 1, ASCII letters/digits = ½), so
    "Transformer纯用注意力" counts 12, not 17."""
    if BUDGETS.get(language, BUDGETS["en-US"])["unit"] == "words":
        return len(text.split())
    compact = re.sub(r"\s+", "", text)
    return int(math.ceil(sum(0.5 if ord(ch) < 128 else 1.0 for ch in compact)))


def _span_end(text: str, start: int, open_ch: str, close_ch: str) -> Optional[int]:
    """End index of the open_ch...close_ch span starting at ``start`` (string-aware); None when it never closes."""
    depth, in_str, esc = 0, False, False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return j + 1
    return None


def _balanced_objects(text: str):
    """Yield every top-level {...} substring (string-aware), in order."""
    i = text.find("{")
    while i != -1:
        end = _span_end(text, i, "{", "}")
        if end is None:
            return
        yield text[i:end]
        i = text.find("{", end)


def _top_level_lists(text: str) -> list[str]:
    """Every [...] that is not inside an object (e.g. a bare list between prose), in order. None at all when an object
    never closes, so a truncated or malformed reply still goes to the repair round."""
    found, k = [], 0
    while True:
        i, j = text.find("[", k), text.find("{", k)
        if j != -1 and (i == -1 or j < i):
            k = _span_end(text, j, "{", "}")
            if k is None:
                return []
            continue
        if i == -1:
            return found
        end = _span_end(text, i, "[", "]")
        if end is None:  # a stray "[" in prose
            k = i + 1
            continue
        found.append(text[i:end])
        k = end


LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
# Seed cannot write "[CLS]": it emits this placeholder instead (same string in every job), which then reached the reader
LEAKED_TOKEN_RE = re.compile(r"\s?<\[(BOS|[A-Z]+)_never_used_[0-9a-f]+\]>")


def scrub_surrogates(value):
    """Copy of a JSON-like value with every lone UTF-16 surrogate replaced by U+FFFD (json.loads turns a model's half
    emoji escape "\\ud83d" into one, and UTF-8 cannot encode it), the model's leaked "[CLS]" placeholder restored, and
    NaN / Infinity (which json.loads accepts) made null, so storyboard.json stays valid JSON for the browser."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        value = LEAKED_TOKEN_RE.sub(lambda m: " [CLS]" if m.group(1) == "BOS" else "", value)
        return LONE_SURROGATE_RE.sub("\ufffd", value)
    if isinstance(value, dict):
        return {scrub_surrogates(k): scrub_surrogates(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_surrogates(v) for v in value]
    return value


def _loads(candidate: str):
    """json.loads (retried once without trailing commas), without lone surrogates; raw control characters inside strings
    (a newline in a bubble) are accepted: two quiz gates were lost to "Invalid control character" alone."""
    try:
        return scrub_surrogates(json.loads(candidate, strict=False))
    except json.JSONDecodeError:
        return scrub_surrogates(json.loads(re.sub(r",\s*([}\]])", r"\1", candidate), strict=False))


def extract_json(text: str, key: Optional[str] = None) -> dict:
    """Pull a JSON object out of a model reply; tolerate code fences, prose, trailing commas and bare lists.

    With ``key`` the first object containing that key wins, then one holding it one level down (e.g. {"result": {...}}),
    then a JSON list is wrapped as {key: [...]}: the whole reply, a fenced list, or a list of objects between prose.
    A malformed or cut-off payload raises (never a fallback to a list item or a "[]" from the prose), so the caller's
    repair round runs.
    """
    fenced = re.findall(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.S | re.I)
    stripped = text.strip()
    candidates = fenced + ([stripped] if stripped.startswith("[") else []) + list(_balanced_objects(text))
    parsed: list = []
    broken = False  # a fenced block or {...} that does not parse: the payload is there but malformed
    last_error: Exception = ValueError("no JSON object in model reply")
    for candidate in candidates:
        try:
            parsed.append(_loads(candidate))
        except json.JSONDecodeError as e:
            last_error, broken = e, broken or candidate != stripped
    if key:
        for data in parsed:
            if isinstance(data, dict) and key in data:
                return data
        for data in parsed:
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, dict) and key in v:
                        return v
        for data in parsed:
            if isinstance(data, list):
                return {key: data}
        lists = []
        for candidate in [] if broken else _top_level_lists(text):  # a broken payload goes to the repair round
            try:
                lists.append(_loads(candidate))
            except json.JSONDecodeError:
                continue
        no_object = not any(isinstance(d, dict) for d in parsed)  # an empty list never replaces a parsed object
        for data in [x for x in lists if any(isinstance(i, dict) for i in x)] + [x for x in lists if not x and no_object]:
            return {key: data}  # "[1]" in prose is a citation, not the payload
        first = text.find("{")
        if first != -1 and text[:first].rstrip().endswith("["):  # a list of objects that does not parse (cut off?)
            raise ValueError("model reply is a JSON list that does not parse")  # never return one of its items
    if not parsed:
        raise last_error
    for data in parsed:
        if isinstance(data, dict):
            return data
    raise ValueError("model reply contained no JSON object")


PAPER_PREFIX = "You work on one research paper for an educational manga. The paper comes first; your task follows it.\n\n"


def _messages(system: str, user: str, paper_text: str = "") -> list[Message]:
    """Paper first: a prompt that ends with the paper text ("PAPER TEXT:\n<body>") sends the paper as the first message,
    identical in every call of a job, so the provider serves it from its prefix cache (cached input costs a fifth on
    Seed and a thirtieth on DeepSeek); the role and the task follow it. Other prompts go as they are."""
    if paper_text:
        for body in (paper_text, paper_text.strip()):
            marker = "\n\nPAPER TEXT:\n" + body
            if body and user.endswith(marker):
                head = [Message.system(PAPER_PREFIX + "PAPER TEXT:\n" + paper_text.strip())]
                return head + ([Message.system(system)] if system else []) + [Message.user(user[:-len(marker)])]
    return [Message.system(system), Message.user(user)] if system else [Message.user(user)]


async def _ask_json(engine: BaseEngine, system: str, user: str, *, thinking: bool, temperature: float, max_tokens: int = 16000,
                    schema: Optional[dict] = None, key: Optional[str] = None, paper: str = "", cache_key: str = "") -> tuple[dict, dict]:
    """One JSON call (with one repair round); ``paper`` (the paper text the prompt ends with) goes first for the prefix
    cache, ``cache_key`` keeps a job's calls on one cache."""
    config = GenerationConfig(temperature=temperature, max_tokens=max_tokens)
    config.thinking = thinking
    config.prompt_cache_key = cache_key or None
    if schema is not None:
        config.response_format = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema, "strict": True}}
    messages = _messages(system, user, paper)
    response = await engine.generate_text(messages, config)
    usage = dict(response.usage or {})
    if not (response.content or "").strip():   # nothing to repair: a "repaired" empty reply would read as "no problems found"
        raise ValueError(f"the model sent an empty reply (finish_reason={response.finish_reason or 'unknown'})")
    try:
        data = extract_json(response.content, key=key)
    except (ValueError, json.JSONDecodeError):
        # one repair round without schema constraints
        repair = GenerationConfig(temperature=0.0, max_tokens=max_tokens)
        repair.thinking = False
        fixed = await engine.generate_text(
            [Message.user("Return the following as strictly valid JSON (same content, no commentary):\n\n" + scrub_surrogates(response.content[:60000]))], repair)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[k] = int(usage.get(k, 0) or 0) + int((fixed.usage or {}).get(k, 0) or 0)
        try:
            data = extract_json(fixed.content, key=key)
        except (ValueError, json.JSONDecodeError) as e:   # say what came back: the log alone could not tell (Right On Q's quiz)
            raise ValueError(f"no usable JSON after a repair round: {e} (finish_reason {response.finish_reason or 'unknown'}; "
                             f"the reply began {scrub_surrogates(response.content[:60])!r})") from e
    return data, usage


async def _ask_check(engine: BaseEngine, user: str, **kwargs) -> tuple[dict, dict]:
    """A check part (``_ask_json``), asked once more when its reply was empty or no repair could parse it: checks run in
    parallel parts, and one lost part leaves its panels unchecked (a DeepSeek part of Right On Q's fact-check once came
    back empty with finish_reason "stop"). Account and content-filter errors are not ValueErrors and pass through."""
    try:
        return await _ask_json(engine, "", user, **kwargs)
    except ValueError as e:   # json.JSONDecodeError is a ValueError
        print(f"[StoryboarderV2] a check reply could not be used ({e}); asking once more")
        return await _ask_json(engine, "", user, **kwargs)


# ----------------------------------------------------------------------------- prompts

SYSTEM_WRITER = (
    "You are a science communicator and manga scriptwriter. You write accurate, warm, lightly funny scripts that teach "
    "real research papers to smart readers who have ADHD: short lines, one idea per panel, concrete visual metaphors, "
    "frequent payoffs, zero filler, zero hallucination."
)


def brief_prompt(paper: PaperText) -> str:
    caps = "\n".join(paper.captions[:20])
    return f"""Read the paper and write a compact, faithful BRIEF for a manga adaptation.

Rules: every statement must come from the paper; copy exact numbers, dataset names, baselines and units; if something is unclear say "unclear" instead of guessing.

Output JSON only:
{{
 "title": "paper title",
 "field": "one phrase",
 "problem": "2-3 sentences: what was hard/slow/wrong before, in plain language",
 "key_idea": "ONE sentence a smart teenager could repeat",
 "contributions": ["each contribution the authors claim, one sentence each"],
 "mechanisms": [{{"name": "...", "plain": "2 sentences, no jargon", "metaphor": "a concrete, drawable everyday image for it", "exact": "the precise technical statement with symbols spelled in words"}}],
 "setup": "1-3 sentences: the data, baselines, metrics and model sizes of the experiments",
 "evidence": [{{"claim": "...", "numbers": "exact figures with units, dataset, split, baseline, single model or ensemble", "why_it_matters": "..."}}],
 "limitations": ["only limitations the authors state"],
 "takeaway": "1-2 sentences: the paper's own conclusion",
 "glossary": [{{"term": "...", "plain": "..."}}],
 "fun_facts": ["optional surprising details worth a joke"],
 "points": [{{"id": "P1", "kind": "problem|contribution|mechanism|setup|result|analysis|limitation", "priority": "core|important|detail", "text": "one precise statement with the exact numbers and setting"}}]
}}
"mechanisms": every component a reader needs to understand how the method works (usually 3-8). "evidence": every main result and key ablation or analysis (usually 4-8): go through every results table and figure, including the appendix, and keep each number with its exact setting. "limitations": only what the authors themselves say; never add your own.
"points": 15-30 points that together cover the whole paper: its problem, every contribution, every mechanism, the experimental setup, every main result and key analysis or ablation, and the limitations the authors state. priority "core" = a reader cannot understand the paper without it (usually 8-14 points); "important" = a curious reader should get it; "detail" = everything else worth knowing.

FIGURE/TABLE CAPTIONS (for orientation):
{caps}

PAPER TEXT:
{paper.body}"""


def _brief_items(brief: dict) -> list[dict]:
    """Every checkable statement of the brief with an id that says where it lives."""
    items = [{"id": key, "text": brief[key]} for key in ("problem", "key_idea", "setup", "takeaway")
             if isinstance(brief.get(key), str) and brief[key].strip()]
    for key in ("contributions", "limitations"):
        items += [{"id": f"{key}.{i}", "text": t} for i, t in enumerate(_as_list(brief.get(key))) if isinstance(t, str) and t.strip()]
    for i, m in enumerate(_as_list(brief.get("mechanisms"))):
        if isinstance(m, dict):
            items += [{"id": f"mechanisms.{i}.{f}", "text": m[f]} for f in ("plain", "exact") if isinstance(m.get(f), str) and m[f].strip()]
    for i, e in enumerate(_as_list(brief.get("evidence"))):
        if isinstance(e, dict):
            items += [{"id": f"evidence.{i}.{f}", "text": e[f]} for f in ("claim", "numbers") if isinstance(e.get(f), str) and e[f].strip()]
    items += [{"id": f"points.{pt['id']}", "text": str(pt.get("text") or "")} for pt in _points(brief) if str(pt.get("text") or "").strip()]
    items += [{"id": f"abstract_claims.{i}", "text": c["text"]} for i, c in enumerate(_as_list(brief.get("abstract_claims")))
              if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip()]
    return items


def brief_check_prompt(brief: dict, paper: PaperText, items: Optional[list[dict]] = None) -> str:
    return f"""You verify the BRIEF of a research paper before it is turned into a manga; every later step trusts it.
For EVERY item below, find the sentence, table or figure of the PAPER that supports it. An item passes only if the paper supports it with the same numbers, the same setting (model size, single model or ensemble, dataset, split, which variant) and the same strength and scope. Flag items that are unsupported, overstated (all, first, only, best, always, completely), misattributed or wrong, and limitations the authors do not state.
For each flagged item give "corrected": the item rewritten so the paper supports it (same role, roughly the same length), or "DELETE" when nothing in the paper supports it.
Output JSON only: {{"problems": [{{"id": "...", "issue": "...", "quote": "the paper's own words", "corrected": "..."}}]}} — an empty list when everything is supported.

ITEMS:
{json.dumps(_brief_items(brief) if items is None else items, ensure_ascii=False)}

PAPER TEXT:
{paper.body}"""


def storyboard_prompt(brief: dict, cast: Cast, language: str, panel_range: tuple[int, int]) -> str:
    lo, hi = panel_range
    budget = BUDGETS["en-US"]
    chapters = "\n".join(f"- {cid}: {title}" for cid, title in CHAPTER_PLAN)
    return f"""Write the manga script for this paper brief. Write it in English (it is translated later).

CAST (use these lowercase names exactly; keep everyone in role)
{cast.describe()}

CHAPTERS (use these chapter ids in order; every chapter gets at least one panel; "how" gets one panel PER mechanism)
{chapters}

HARD RULES
- Between {lo} and {hi} panels total; choose the count by how much the paper actually contains. Number panels 1..N in reading order.
- ONE idea per panel. Max 2 speech bubbles per panel, each ≤ {budget['bubble']} words. Optional narration box ≤ {budget['narration']} words that states the precise technical fact (numbers go here).
- Every number, name and claim must come from the brief. Never invent results. Use the exact figures.
- No absolute claims (all, every, always, never, nobody) unless the paper itself makes them. No claims about the paper's later impact, adoption or today's practice ("changed everything", "now standard", "to this day"): the takeaway restates the paper's own conclusion.
- Keep the paper's qualifiers (deep, many, most, on this dataset) and tie every number to its exact setting: model size, single model or ensemble, dataset and split. A trend is never turned into a hard threshold.
- Describe each method exactly as the paper does: a simplification may drop detail but must not change what happens. Limitations are only those the paper states; never invent a weakness. Never claim a model beats humans unless the paper does, and credit a gain to everything the paper credits it to.
- "visual": concrete and drawable — poses, expressions, props and the metaphor. Mention only characters listed in "characters". Only the characters listed in "characters" appear. Metaphors use objects, places, diagrams and the cast only — never extra people, children, crowds or other creatures (no guides, hikers, students, librarians). Never ask for written words, numbers or formulas in the picture ("a sentence on the board", "a line of text with one word blacked out", "sums on a chalkboard"): show words as a row of blank cards or blocks, one of them covered or highlighted, and amounts as bars or piles; numbers belong in the narration or labels. Never draw copies or miniature versions of a cast member: show counts with simple objects (eight colored cards, not eight little papis).
- "on_image_labels": ≤ 2 short labels (≤ 3 words) drawn on props/boards; everything else in the picture is wordless.
- On-image text (bubbles, narration, labels) must not contain underscores, LaTeX, backslashes or markdown. Write math in words or with plain symbols (√, ², ×, ÷, ≈). Avoid quotation marks inside bubbles.
- Every panel carries at least one fact from the brief, the hook and the takeaway included (the hook states the problem, the takeaway the main result); no filler panels and no panel that repeats an earlier one. The narration adds a fact the bubbles do not say; it never restates a bubble. The first time a metric or technical term appears (BLEU, F1, a dataset), explain it in plain words using the glossary, in the narration or the detail.
- COVERAGE: each panel lists in "covers" the ids of the brief points it explains. Every "core" point must be covered by at least one panel, and "important" points as far as the panel budget allows; choose the panel count so that every core point fits (one main point per panel; a closely related second point may go in the narration or the detail).
- "detail": 1-3 sentences (≤ 60 words) for readers who want more: the points this panel covers, restated plainly with their exact numbers and settings, and the mechanism in technical terms. Add nothing the points do not say: no judgments such as first, only, best, optimal, all, order of magnitude, unless a point says it. It is shown as text below the picture and never drawn.
- If a visual compares amounts (bars, stacks, piles), the larger or better one in the paper must look larger; never draw a comparison that contradicts the narration.
- Emotions: give each character in the panel one emotion word.
- Pacing for ADHD readers: open with a hook that answers "why should I care", give a payoff every 2-3 panels, let the student ask what the reader is wondering, let the critic voice the limitation, end with a memorable takeaway.

OUTPUT: JSON only, exactly this shape:
{{"title": "short manga title", "summary": "one sentence", "panels": [{{"n": 1, "chapter": "hook", "title": "...", "characters": ["papi", "kumo"], "scene": "...", "visual": "...", "dialogue": [{{"speaker": "papi", "text": "..."}}], "narration": "...", "on_image_labels": ["..."], "emotions": {{"papi": "curious"}}, "covers": ["P1"], "detail": "..."}}]}}

BRIEF:
{json.dumps(brief, ensure_ascii=False)}"""


def fit_prompt(items: list[dict], language: str) -> str:
    budget = BUDGETS.get(language, BUDGETS["en-US"])
    return f"""Rewrite each text so it fits the limit while keeping the exact meaning, numbers and tone ({LANG_NAMES.get(language, language)}).
Limits: speech bubble ≤ {budget['bubble']} {budget['unit']}, narration ≤ {budget['narration']} {budget['unit']}, label on a prop ≤ {budget['label']} {budget['unit']}. No underscores, LaTeX or markdown.
A "visual" item is a scene for the picture that mentions people or copies of a cast member: rewrite it as ONE English sentence (scene and action together) that shows the same idea with objects, places, diagrams and the cast only (each cast member once; counts as simple objects such as cards), and states that nobody else is there.
Return JSON only: {{"items": [{{"id": "...", "text": "rewritten"}}]}}

ITEMS:
{json.dumps(items, ensure_ascii=False)}"""


FACTCHECK_CHUNK = 30   # lines per fact-check call: a long script is checked in parallel parts (a thinking checker takes minutes)


def _factcheck_items(sb: StoryboardV2, paper: PaperText, language: str, panels: Optional[set] = None, card: bool = True) -> list[dict]:
    items = []
    for p in sb.panels:
        if panels is not None and p.n not in panels:
            continue
        for i, b in enumerate(p.dialogue):
            items.append({"id": f"{p.n}.d{i}", "speaker": b.speaker, "text": b.text})
        if p.narration:
            items.append({"id": f"{p.n}.n", "text": p.narration})
        if p.detail:
            items.append({"id": f"{p.n}.x", "text": p.detail})
        if p.visual and DRAWN_CLAIM_RE.search(p.visual):   # numbers and comparisons in a scene description end up drawn
            items.append({"id": f"{p.n}.v", "text": p.visual})
        if p.board:
            items += _board_check_items(p, whole=language == "en-US", paper_text=paper.body)
    for key, value in (sb.card.items() if card else ()):
        for i, text in enumerate([value] if isinstance(value, str) else value):
            items.append({"id": f"card.{key}.{i}", "text": text})
    return items


def factcheck_parts(sb: StoryboardV2, paper: PaperText, language: str = "en-US", size: int = FACTCHECK_CHUNK) -> list[tuple[set, bool]]:
    """The script cut into whole panels of about ``size`` fact-check lines each: (panel numbers, with the card)."""
    parts, current, count = [], set(), 0
    for p in sb.panels:
        n = len(_factcheck_items(sb, paper, language, {p.n}, card=False))
        if current and count + n > size:
            parts.append(current)
            current, count = set(), 0
        current.add(p.n)
        count += n
    if current or not parts:
        parts.append(current)
    return [(panels, i == 0) for i, panels in enumerate(parts)]


def factcheck_prompt(sb: StoryboardV2, paper: PaperText, language: str = "en-US", panels: Optional[set] = None, card: bool = True) -> str:
    items = _factcheck_items(sb, paper, language, panels, card)
    budget = BUDGETS.get(language, BUDGETS["en-US"]) if sb.version < 3 else _v3_budget(language)["scene"]
    unit = "words" if language == "en-US" else "characters"
    boards = _board_check_rules(language) if any(p.board for p in sb.panels) else ""
    target = ("" if language == "en-US" else
              f"\nThe lines are in {LANG_NAMES[language]}: judge them in that language and write every corrected line in natural "
              f"{LANG_NAMES[language]} with full-width punctuation, keeping technical terms, names and numbers exact.")
    return f"""You are a meticulous fact-checker for an educational manga about a research paper.
Compare EVERY line below against the PAPER TEXT. A line passes only if a sentence, table or figure of the paper supports it with the same strength and scope; simplified wording is fine when a reader of the paper would agree with it. Flag a line when it:
- states a number, name, comparison, mechanism or claim that the paper does not support;
- attributes a figure to the wrong model or setting (single model vs ensemble, validation vs test, wrong dataset or model size; e.g. a Transformer-big cost presented as a baseline cost);
- distorts the mechanism (e.g. "divide by the number of queries" when the paper divides by the square root of the key dimension; "only the last layer is trained" when all parameters are fine-tuned);
- overgeneralises: an absolute claim (all, every, always, completely, the whole field) the paper does not make, a result on some tasks stated for all tasks, a gradual trend stated as a hard threshold, or a qualifier of the paper's claim dropped (deep, many, most, on this dataset);
- claims that the method produced a specific output on the story's example ("it translated the sentence correctly", "it got the answer right", "it found the link perfectly") when the paper does not show that output for that example; what a figure of the paper shows may be described as the figure shows it, with the paper's hedges;
- claims impact, adoption or present-day status the paper does not state ("changed everything", "now a standard building block", "to this day", "every phone uses this"; also 至今, 直到现在, 现在已经, 今でも, 現在では): an open question is one "the paper leaves open", not one unsolved today;
- says a model beats humans when the paper does not, or compares against only the weaker of several human baselines;
- credits one ingredient with all or most of a gain that the paper attributes to several (e.g. bidirectionality alone when the paper credits it together with the pre-training tasks);
- states a limitation or weakness the paper does not report, or the opposite of what it reports.
Casual tone, simplification and metaphors are fine; do not flag style. When a number is wrong, the corrected line states the paper's figure (with its setting) instead of dropping the number. Ids ending in .v are scene descriptions that get drawn: flag any number, comparison or claim in them that the paper does not support (a chart must not show the wrong side as bigger).{boards}{target}
For each problem give a corrected line that keeps the same meaning intent and roughly the same length (bubbles ≤ {budget['bubble']} {unit}, narration ≤ {budget['narration']} {unit}; ids ending in .x are a panel's detail and ids starting with card. are the paper card, both reader text of up to 60 words), no underscores or LaTeX.
Output JSON only: {{"problems": [{{"id": "4.n", "issue": "...", "corrected": "..."}}]}} — use an empty list when everything is supported.

LINES:
{json.dumps(items, ensure_ascii=False)}

PAPER TEXT:
{paper.body}"""


def _translation_panel(p: PanelV2) -> dict:
    d = {"n": p.n, "title": p.title, "dialogue": [{"speaker": b.speaker, "text": b.text} for b in p.dialogue], "narration": p.narration,
         "labels": list(p.labels), "detail": p.detail}
    if p.board:   # v3: the strings drawn on the board, keyed by their path
        d["board"] = {str(path): text for path, text in _board_strings(p.board)}
    return d


# the cast's props and running jokes, translated the same way in every manga (9 of 14 Chinese scripts made the critic's
# berry tart an egg tart, 蛋挞, which the art never shows)
FIXED_TERMS = {"zh-CN": {"berry tart": "浆果挞", "tart": "浆果挞", "berries": "浆果", "professor": "教授", "PhD student": "博士生",
                         "critic": "评论家", "skeptic": "怀疑派"}}


def _fixed_terms(language: str) -> str:
    terms = FIXED_TERMS.get(language)
    return ("\n- Fixed terms (use exactly these): " + "; ".join(f"{en} = {zh}" for en, zh in terms.items())) if terms else ""


def translate_prompt(sb: StoryboardV2, language: str) -> str:
    budget = BUDGETS[language]
    glossary = [g for g in (sb.brief.get("glossary") or [])][:30]
    names = ", ".join(sorted({str(n).strip().capitalize() for p in sb.panels for n in [*p.characters, *(b.speaker for b in p.dialogue)]
                              if str(n).strip()})) or "the cast's names"
    payload = {
        "title": sb.title,
        "summary": sb.summary,
        "chapters": [{"id": c.id, "title": c.title} for c in sb.chapters],
        "panels": [_translation_panel(p) for p in sb.panels],
        "card": sb.card,
    }
    if sb.version >= 3:
        scene, board = _v3_budget(language)["scene"], _v3_budget(language)["board"]
        limits = (f"each bubble ≤ {scene['bubble']} characters and narration ≤ {scene['narration']} characters on a scene panel; on a "
                  f"panel with a board, each bubble ≤ {board['bubble']} and narration ≤ {board['narration']}; panel title ≤ 12 characters; "
                  f"board strings about as short as the English")
        fields = "labels, detail, every string value of a panel's \"board\" (keep its keys) and every string in card"
    else:
        limits = (f"each bubble ≤ {budget['bubble']} characters, narration ≤ {budget['narration']} characters, panel title ≤ 12 characters, "
                  f"each label ≤ 6 characters")
        fields = "labels, detail and every string in card"
    return f"""Translate this manga script into natural, conversational {LANG_NAMES[language]} for readers who are new to the topic.
Rules:
- Translate ONLY the string values of title, summary, chapter titles, panel titles, dialogue[].text, narration, {fields}. Keep every key, every "n" and every "speaker" unchanged.
- Keep every qualifier and hedge (deep, many, most, some, about, up to, on this dataset) and the claim's exact strength: never strengthen or generalise a statement.
- Keep model names, datasets, metrics and numbers exact (e.g. BLEU, WMT 2014, 28.4, Transformer, GSM8K). Widely used English acronyms (RNN, CNN, GPU, Q/K/V) stay in English; labels that are pure acronyms or numbers stay unchanged. A general term gets its {LANG_NAMES[language]} name, with the English abbreviation in parentheses the first time ("大语言模型（LLM）", "思维链（CoT）").
- Write the way a {LANG_NAMES[language]} science comic for teenagers talks: short, lively sentences in natural word order. Never copy the English sentence structure, appositives ("Kumo, a PhD student, ...") or idioms word for word.
- The cast keep their names in Latin letters ({names}): never transliterate them (their name tabs in the pictures are Latin); roles may be translated ("Kumo，博士生").{_fixed_terms(language)}
- Length limits: {limits} — rephrase to fit while keeping the meaning.
- No underscores, LaTeX or markdown in any text. Use full-width punctuation appropriate for {LANG_NAMES[language]}.
Glossary (term → plain meaning): {json.dumps(glossary, ensure_ascii=False)}
Output JSON only with the same structure.

{json.dumps(payload, ensure_ascii=False)}"""


# a translation that is stronger than its English line (7 of 38 Chinese fact-check findings were added in translation:
# "extremely hard" became 根本没法, "solved" 完美解决), or drops the English line's hedge
ZH_STRONG_RE = re.compile(r"完美|完全|根本|彻底|所有|全部|一定|绝对|总是|永远|毫无|全都|完胜|碾压|任何")
EN_STRONG_RE = re.compile(r"\b(?:perfect(?:ly)?|complete(?:ly)?|entire(?:ly)?|total(?:ly)?|fully|all|every(?:thing|one)?|always|never|"
                          r"absolute(?:ly)?|definitely|whole|any|nothing|none|no)\b", re.I)
HEDGES = [   # (English hedge, the Chinese words that keep it)
    (re.compile(r"\bmost(?:ly)?\b", re.I), re.compile(r"大多|多数|大部分|大都")),
    (re.compile(r"\b(?:about|around|roughly|approximately)\s+\d", re.I), re.compile(r"约|大约|左右|大概|差不多|近")),
    (re.compile(r"\b(?:nearly|almost)\b", re.I), re.compile(r"几乎|将近|接近|差不多|近")),
    (re.compile(r"\bup to\b", re.I), re.compile(r"最高|最多|多达|高达|上限")),
    (re.compile(r"\b(?:over|more than)\s+\d", re.I), re.compile(r"超过|以上|多于|逾|多")),
    (re.compile(r"\bsingle[- ]model\b", re.I), re.compile(r"单模型|单个模型|单一模型|单独")),
    (re.compile(r"\bensembles?\b", re.I), re.compile(r"集成|组合")),
    (re.compile(r"\b(?:may|might)\b", re.I), re.compile(r"可能|或许|也许")),
    (re.compile(r"\b(?:seems?|apparently)\b", re.I), re.compile(r"似乎|好像|看来|看起来")),
]
DOUBLED_RE = re.compile(r"(先先|的的|了了|是是|在在|和和|也也|都都)")


def translation_problems(english: str, translated: str, language: str, names: frozenset = frozenset()) -> list[str]:
    """What a translated line does to its English line that the reader would be misled by: a stronger claim, a lost
    hedge, a transliterated cast name, a doubled character, a prop under another name. Only zh-CN has rules."""
    if language != "zh-CN" or not english or not translated:
        return []
    out = []
    strong = ZH_STRONG_RE.findall(translated)
    if strong and not EN_STRONG_RE.search(english):
        out.append(f"stronger than the English ('{strong[0]}'): keep the English line's strength")
    for en_hedge, zh_hedge in HEDGES:
        m = en_hedge.search(english)
        if m and not zh_hedge.search(translated):
            out.append(f"drops the English hedge '{m.group(0).strip()}'")
    for name in names:
        if re.search(rf"\b{re.escape(name)}\b", english, re.I) and not re.search(rf"{re.escape(name)}", translated, re.I):
            out.append(f"the name {name.capitalize()} must stay in Latin letters")
    if "蛋挞" in translated:
        out.append("the tart is a berry tart: 浆果挞, never 蛋挞")
    if "×" in english and "×" not in translated and re.search(r"(?<![A-Za-z])[xX](?![A-Za-z])", translated):
        out.append("write × as ×, not X")
    doubled = DOUBLED_RE.search(translated)
    if doubled:
        out.append(f"a doubled character ('{doubled.group(0)}')")
    return out


def translation_fix_prompt(items: list[dict], language: str) -> str:
    return f"""Correct these {LANG_NAMES[language]} lines of a manga. Each comes with the English line it translates and what is wrong with the translation. Fix only that: keep the English line's meaning and strength, every number, name and technical term exactly, and about the same length; natural {LANG_NAMES[language]} with full-width punctuation; no underscores, LaTeX or markdown.
Output JSON only: {{"items": [{{"id": "...", "text": "corrected line"}}]}}

ITEMS:
{json.dumps(items, ensure_ascii=False)}"""


def backport_prompt(items: list[dict]) -> str:
    return f"""A fact-check of the Chinese translation of a manga script against the PAPER found the problems below. Each item gives the English line the Chinese line was translated from. The English line may have the same problem (an overgeneralisation, a dropped qualifier, a claim or an output the paper does not support), or the translation may have added it.
For each item: if the English line has the problem, return the corrected English line (same intent, about as short, the paper's numbers and qualifiers, no underscores or LaTeX); if the English line is fine as it is, return "KEEP".
Output JSON only: {{"items": [{{"id": "...", "text": "corrected English line or KEEP"}}]}}

ITEMS:
{json.dumps(items, ensure_ascii=False)}

PAPER TEXT:
"""


def _line_at(sb: StoryboardV2, pid: str) -> Optional[str]:
    """The text a fact-check id names ("5.d0" bubble, "5.n" narration, "5.b.<path>" board string), or None."""
    n_str, _, part = str(pid).partition(".")
    panel = next((p for p in sb.panels if str(p.n) == n_str), None)
    if panel is None:
        return None
    if part == "n":
        return panel.narration or None
    if part.startswith("d") and part[1:].isdigit():
        i = int(part[1:])
        return panel.dialogue[i].text if i < len(panel.dialogue) else None
    if part.startswith("b.") and panel.board:
        return dict(_board_strings(panel.board)).get(part[2:])
    return None


def _set_line_at(sb: StoryboardV2, pid: str, text: str) -> bool:
    n_str, _, part = str(pid).partition(".")
    panel = next((p for p in sb.panels if str(p.n) == n_str), None)
    if panel is None:
        return False
    if part == "n" and panel.narration:
        panel.narration = text
        return True
    if part.startswith("d") and part[1:].isdigit() and int(part[1:]) < len(panel.dialogue):
        panel.dialogue[int(part[1:])].text = text
        return True
    if part.startswith("b.") and panel.board:
        new = _set_board_strings(panel.board, {part[2:]: text})
        if new is not panel.board:
            panel.board = new
            return True
    return False


CHAPTER_ALIASES = {
    "intro": "hook", "introduction": "hook", "opening": "hook", "why": "hook", "motivation": "hook",
    "background": "problem", "challenge": "problem", "old problem": "problem",
    "key idea": "idea", "key_idea": "idea", "insight": "idea", "solution": "idea",
    "method": "how", "methods": "how", "mechanism": "how", "mechanisms": "how", "how it works": "how", "approach": "how",
    "results": "evidence", "result": "evidence", "experiments": "evidence", "proof": "evidence",
    "limitations": "limits", "limitation": "limits", "caveats": "limits", "critique": "limits",
    "conclusion": "takeaway", "conclusions": "takeaway", "summary": "takeaway", "ending": "takeaway", "outro": "takeaway",
}


def _normalize_chapter(chapter: str, previous: str) -> str:
    ids = [cid for cid, _ in CHAPTER_PLAN]
    c = (chapter or "").strip().lower()
    if c in ids:
        return c
    if c in CHAPTER_ALIASES:
        return CHAPTER_ALIASES[c]
    for cid in ids:
        if cid in c:
            return cid
    return previous


NARRATOR_RE = re.compile(r"\b(?:narrat(?:or|ion)|caption|voice[- ]?over|text box|title card)\b")


def _match_cast(name: str, valid) -> Optional[str]:
    """'Professor Papi', 'papi (smiling)' → 'papi'; the earliest whole-word name wins ('kumo (to papi)' → 'kumo'),
    independent of set order. None when no cast member is named."""
    n = (name or "").strip().lower()
    if n in valid:
        return n
    hits = [(m.start(), -len(member), member) for member in valid for m in [re.search(rf"\b{re.escape(member)}\b", n)] if m]
    return min(hits)[2] if hits else None


def _is_narrator(speaker: str, valid) -> bool:
    """'Narrator (V.O.)', 'The narrator', 'Caption box' → True; a speaker naming a cast member is that member's speech."""
    return _match_cast(speaker, valid) is None and bool(NARRATOR_RE.search((speaker or "").lower()))


# ----------------------------------------------------------------------------- pipeline steps

def _plain(text: str) -> str:
    """On-image text without markdown or TeX marks: Seedream letters them literally ("*only*")."""
    return re.sub(r"\*+", "", str(text or "").replace("_", " ").replace("\\", "")).strip()


def _normalize_panels(panels: list[PanelV2], cast: Cast, max_bubbles: Optional[int] = 2) -> None:
    """Number panels 1..N and clean them: narrator lines become narration (never dropped) before speech is capped at
    ``max_bubbles`` (None: no cap), names map onto the cast, emotions keep only characters in the panel, unknown chapters
    inherit the previous."""
    for i, p in enumerate(panels, start=1):
        p.n = i
    valid = set(cast.names())
    default_speaker = cast.names()[0]
    previous_chapter = CHAPTER_PLAN[0][0]
    for p in panels:
        speech = []
        for b in p.dialogue:
            if not _is_narrator(b.speaker, valid):
                speech.append(b)
            elif b.text not in p.narration:
                p.narration = f"{p.narration} {b.text}".strip()
        for b in speech:
            b.text = _plain(b.text)
        speech = [b for b in speech if b.text]   # a line of only markup ("**") would be an empty bubble
        p.dialogue = speech[:max_bubbles] if max_bubbles else speech
        p.narration, p.labels = _plain(p.narration), [_plain(lab) for lab in p.labels]
        p.characters = list(dict.fromkeys(m for c in p.characters for m in [_match_cast(c, valid)] if m))
        for b in p.dialogue:
            matched = _match_cast(b.speaker, valid)
            b.speaker = matched or (p.characters[0] if p.characters else default_speaker)
        if not p.characters:
            p.characters = [b.speaker for b in p.dialogue][:1] or [default_speaker]
        for b in p.dialogue:
            if b.speaker not in p.characters:
                p.characters.append(b.speaker)
        p.emotions = {m: v for k, v in p.emotions.items() for m in [_match_cast(k, p.characters)] if m}
        p.chapter = _normalize_chapter(p.chapter, previous_chapter)
        previous_chapter = p.chapter


def _chapters(panels: list[PanelV2]) -> list[Chapter]:
    return [Chapter(id=cid, title=ctitle, panel_numbers=nums) for cid, ctitle in CHAPTER_PLAN
            for nums in [[p.n for p in panels if p.chapter == cid]] if nums]


def _points(brief: dict) -> list[dict]:
    """The brief's key points with stripped ids and lower-case priorities, each id once."""
    out, seen = [], set()
    for pt in _as_list(brief.get("points")):
        pid = str(pt.get("id") or "").strip() if isinstance(pt, dict) else ""
        if pid and pid not in seen:
            seen.add(pid)
            out.append({**pt, "id": pid, "priority": str(pt.get("priority") or "").strip().lower()})
    return out


def uncovered(sb: "StoryboardV2", priority: str = "core") -> list[dict]:
    """The brief's points of this priority that no panel lists in its covers."""
    covered = {c for p in sb.panels for c in p.covers}
    return [pt for pt in _points(sb.brief) if pt["priority"] == priority and pt["id"] not in covered]


def coverage(sb: "StoryboardV2") -> dict:
    """{"core": [covered, total], "important": [covered, total]} for the reader and the job report. A v3 script reports
    the abstract claims its panels convey as its core points (and under "claims")."""
    if sb.version >= 3 and sb.abstract_claims:
        ids = {c.get("id") for c in sb.abstract_claims}
        done = len(ids & {c for p in sb.panels for c in p.claims})
        return {"core": [done, len(ids)], "important": [0, 0], "claims": [done, len(ids)]}
    out = {}
    for priority in ("core", "important"):
        total = [pt for pt in _points(sb.brief) if pt["priority"] == priority]
        out[priority] = [len(total) - len(uncovered(sb, priority)), len(total)]
    return out


UNCLEAR = {"unclear", "unknown", "n/a", "none", "not stated"}


def _card_text(value) -> str:
    """A brief value as reader text: strings as they are, lists of strings joined, anything else (or "unclear") dropped."""
    if isinstance(value, list):
        value = "; ".join(t for t in map(_card_text, value) if t)
    text = value.strip() if isinstance(value, str) else ""
    return "" if text.lower().strip(".") in UNCLEAR else text


def paper_card(brief: dict) -> dict:
    """The whole paper in a few lines (shown as text in the reader, translated and fact-checked with the script)."""
    def items(key: str, limit: int) -> list[str]:
        return [t for t in map(_card_text, _as_list(brief.get(key))) if t][:limit]

    how, results = [], []
    for m in _as_list(brief.get("mechanisms")):
        if isinstance(m, dict):
            name, plain = _card_text(m.get("name")), _card_text(m.get("plain"))
            how.append(f"{name}: {plain}" if name and plain else name or plain)
    for e in _as_list(brief.get("evidence")):
        if isinstance(e, dict):
            claim, numbers = _card_text(e.get("claim")), _card_text(e.get("numbers"))
            results.append(f"{claim} ({numbers})" if claim and numbers else claim)
    card = {"problem": _card_text(brief.get("problem")), "key_idea": _card_text(brief.get("key_idea")),
            "contributions": items("contributions", 6), "how": [h for h in how if h][:8], "results": [r for r in results if r][:8],
            "limitations": items("limitations", 5), "takeaway": _card_text(brief.get("takeaway"))}
    return {k: v for k, v in card.items() if v}


def _insert_position(panels: list[PanelV2], new: PanelV2, after: int) -> int:
    """Index in ``panels`` to insert ``new`` at: inside the run of its own chapter (clamped to it), or right after the
    panel it follows when its chapter is unknown (then it takes that panel's chapter). Never after the takeaway unless
    the new panel is part of it."""
    ids = [cid for cid, _ in CHAPTER_PLAN]
    if new.chapter not in ids:
        k = min(max(after, 0), len(panels))
        takeaway = [i for i, p in enumerate(panels) if p.chapter == "takeaway"]
        if takeaway and k > takeaway[0]:   # the manga keeps ending on its takeaway
            k = takeaway[0]
        new.chapter = panels[k - 1].chapter if k else panels[0].chapter
        return k
    run = [i for i, p in enumerate(panels) if p.chapter == new.chapter]
    if run:
        return min(max(after, run[0]), run[-1] + 1)
    earlier = [i for i, p in enumerate(panels) if p.chapter in ids and ids.index(p.chapter) < ids.index(new.chapter)]
    return earlier[-1] + 1 if earlier else 0


def cover_prompt(sb: "StoryboardV2", cast: Cast, missing: list[dict], max_panels: int) -> str:
    budget = BUDGETS["en-US"]
    script = [{"n": p.n, "chapter": p.chapter, "title": p.title, "covers": p.covers,
               "dialogue": [asdict(b) for b in p.dialogue], "narration": p.narration, "detail": p.detail} for p in sb.panels]
    return f"""The manga script below does not cover these CORE points of the paper:
{json.dumps(missing, ensure_ascii=False)}

Fix that without touching the existing panels:
- Write new panels that explain them (one point per panel, or two closely related points), each with "after": the number of the existing panel it follows, so the story still reads in order. The script may grow to at most {max_panels} panels in total ({max(0, max_panels - len(sb.panels))} more).
- If there is no room, or a point is a small detail of an existing panel, add a sentence to that panel's detail instead: {{"n": 7, "add_detail": "...", "covers": ["P9"]}}. If an existing panel already explains a point but does not list it, just tag it: {{"n": 7, "covers": ["P9"]}}.
- New panels follow the script's rules: cast {", ".join(cast.names())} only (lowercase names), max 2 speech bubbles of ≤ {budget['bubble']} words, optional narration ≤ {budget['narration']} words, ≤ 2 short on-image labels, a concrete drawable "visual" with no extra people and no written words, "covers" with the point ids, and a "detail" of 1-3 sentences. Every statement comes from the points; keep their exact numbers and qualifiers.
Output JSON only: {{"new_panels": [{{"after": 7, "chapter": "how", "title": "...", "characters": ["papi"], "scene": "...", "visual": "...", "dialogue": [{{"speaker": "papi", "text": "..."}}], "narration": "...", "on_image_labels": [], "emotions": {{"papi": "curious"}}, "covers": ["P9"], "detail": "..."}}], "details": []}}

SCRIPT:
{json.dumps(script, ensure_ascii=False)}"""


def _translated_card(card: dict, reply: dict, language: str, keep: frozenset) -> dict:
    """Take each translated card field that has the same shape and is in the target script; keep the English otherwise."""
    out = dict(card)
    for key, value in card.items():
        new = reply.get(key)
        if isinstance(value, str) and isinstance(new, str) and new.strip() and _has_script(new, language, keep):
            out[key] = new.strip()
        elif isinstance(value, list) and isinstance(new, list) and len(new) == len(value) \
                and all(isinstance(x, str) and x.strip() and _has_script(x, language, keep) for x in new):
            out[key] = [x.strip() for x in new]
    return out


# ----------------------------------------------------------------------------- v3: story layer
# The reader test: someone who never saw the paper reads only what is visible in the panels (bubbles, narration, boards)
# and must come away knowing everything the abstract says, having seen its key numbers as evidence.

V3_PANELS = (16, 24)           # story length; an explicit longer range (the "long" option) is kept
QUIZ_PASS = 0.85               # share of abstract questions a test reader must answer (abstract_quiz.PASS_SCORE wins)
QUIZ_EXTRA_PANELS = 4          # how far past its range the quiz revision may grow the script
BEATS = ("hook", "claims", "problem", "idea", "how", "proof", "payoff", "limits", "recap")
BEAT_CHAPTER = {"hook": "hook", "claims": "hook", "problem": "problem", "idea": "idea", "how": "how", "proof": "evidence",
                "payoff": "payoff", "limits": "limits", "recap": "takeaway"}
# panels that list the abstract's claims instead of teaching them: the quiz reads around them, and a claim counts as
# conveyed (with its numbers) only on the other panels
LIST_BEATS = ("claims", "recap")
BEAT_ALIASES = {
    "stakes": "hook", "intro": "hook", "opening": "hook",
    "why_old_fails": "problem", "old_way": "problem", "old_way_fails": "problem", "background": "problem",
    "key_idea": "idea", "insight": "idea", "solution": "idea",
    "mechanism": "how", "method": "how", "how_it_works": "how", "demo": "how",
    "evidence": "proof", "result": "proof", "results": "proof", "challenge": "proof", "reaction": "proof", "perspective": "proof",
    "limitation": "limits", "limitations": "limits", "caveat": "limits", "caveats": "limits",
    "resolution": "payoff", "ending": "payoff", "conclusion": "payoff", "takeaway": "payoff",
    "summary": "recap", "review": "recap",
    "claim_board": "claims", "claims_board": "claims", "claim": "claims", "teaser": "claims", "promise": "claims", "bet": "claims",
    "predict": "proof", "prediction": "proof", "guess": "proof", "reveal": "proof",
}
# v3 text is typeset by code, so budgets follow comic practice rather than what the image model can letter;
# Chinese gets about 1.8 characters per English word
BUDGETS_V3 = {
    "en-US": {"unit": "words", "scene": {"bubbles": 3, "bubble": 16, "narration": 20, "total": 44},
              "board": {"bubbles": 2, "bubble": 12, "narration": 14, "total": 28}},
    "zh-CN": {"unit": "characters", "scene": {"bubbles": 3, "bubble": 29, "narration": 36, "total": 80},
              "board": {"bubbles": 2, "bubble": 22, "narration": 25, "total": 50}},
}

SYSTEM_STORY = (
    "You are an award-winning manga writer and science communicator. You turn research papers into short, gripping manga "
    "stories that teach the real content, with the paper's real numbers, to smart readers who have ADHD. You never invent facts."
)


def _v3_budget(language: str) -> dict:
    return BUDGETS_V3.get(language) or BUDGETS_V3["en-US" if language == "en-US" else "zh-CN"]


def story_range(panel_range: tuple[int, int]) -> tuple[int, int]:
    """Panel range of a v3 story: 16-24 panels, inside the caller's range where they overlap; an explicitly longer
    range is kept as it is."""
    lo, hi = panel_range
    if lo > V3_PANELS[1]:
        return lo, max(lo, hi)
    lo = min(max(lo, V3_PANELS[0]), V3_PANELS[1])
    return lo, max(lo, min(hi, V3_PANELS[1]))


_WARNED: set[str] = set()


def _optional(module: str):
    """services.boards / evidence / abstract_quiz, imported on use: this module (and the v2 pipeline) still loads without
    them, and a missing or broken one only disables its checks, with one warning per process."""
    try:
        return importlib.import_module(f"services.{module}")
    except Exception as e:  # ImportError, or a module that fails while loading
        if module not in _WARNED:
            _WARNED.add(module)
            print(f"[StoryboarderV2] WARNING: services.{module} unavailable ({type(e).__name__}: {e}); its v3 checks are skipped")
        return None


def _reraise_account(e: BaseException) -> None:
    """A best-effort step never hides an account problem (a refused key, a model not activated, no balance: engines
    mark those errors ``account``): the job must stop and say so, not quietly skip its checks."""
    if getattr(e, "account", False):
        raise e


def _ledger(paper_text: str):
    """The tables read from the PDF layout (services.ledger), or None when the paper text has none or the module is missing."""
    mod = _optional("ledger")
    found = mod.ledger_of(paper_text) if mod is not None else None
    return found if found else None


def _tables_section(paper_text: str) -> str:
    mod = _optional("ledger")
    return mod.section(paper_text) if mod is not None else ""


# ----------------------------------------------------------------------------- v3: abstract and its claims

ABSTRACT_LINE_RE = re.compile(r"^\s*(?:abstract|a b s t r a c t)\s*[.:—–-]?\s*$", re.I)
ABSTRACT_RUN_IN_RE = re.compile(r"^\s*(?:Abstract|ABSTRACT)\s*[.:—–-]\s*(\S.*)$|^\s*ABSTRACT\s+(\S.*)$")
FIRST_SECTION_RE = re.compile(r"^\s*(?:(?:\d{1,2}|[IVX]{1,5})\.?\s+)?(?:introduction|background|related work|motivation|preliminaries)\s*$", re.I)
NUMBERED_HEADING_RE = re.compile(r"^\s*(?:\d{1,2}|[IVX]{1,5})\.?\s+[A-Z][A-Za-z][A-Za-z ,:&\-]{0,60}$")
ABSTRACT_END_RE = re.compile(r"^\s*(?:[∗*†‡§¶]|(?:Figure|Fig\.|Table)\s*\d+[.:]|Keywords\b|Key words\b|Index Terms\b|CCS Concepts\b"
                             r"|Example sentence shown in a figure\b)", re.I)
VENUE_RE = re.compile(r"Conference on|Proceedings of|Workshop on|Symposium on|Under review|Preprint\b|Published as|Copyright\b|©")


def find_abstract(body: str, max_lines: int = 400) -> str:
    """The paper's abstract: from its "Abstract" heading to the first section heading ("1 Introduction"), cut at the first
    footnote (∗ † ‡), venue line ("31st Conference on ..."), caption, or heading-like line of a figure that interrupts it
    ("Standard Prompting"). "" when the paper has no abstract heading."""
    lines = body.splitlines()[:max_lines]
    start, first = None, ""
    for i, line in enumerate(lines):
        if ABSTRACT_LINE_RE.match(line):
            start = i + 1
            break
    if start is None:
        for i, line in enumerate(lines):
            m = ABSTRACT_RUN_IN_RE.match(line)
            if m:
                start, first = i + 1, (m.group(1) or m.group(2)).strip()
                break
    if start is None:
        return ""
    kept = [first] if first else []
    for line in lines[start:start + 80]:
        text = line.strip()
        if not text or re.fullmatch(r"\d{1,2}\.?|[IVX]{1,4}\.", text):   # a page number, or a section number on its own line
            continue
        if ABSTRACT_END_RE.match(text) or VENUE_RE.search(text) or FIRST_SECTION_RE.match(text):
            break
        if kept and NUMBERED_HEADING_RE.match(text) and re.search(r"[.!?]$", kept[-1]):
            break   # "2 Background" after a finished sentence (a wrapped "8 GPUs" never follows one)
        if kept and len(text.split()) <= 4 and text[:1].isupper() and not re.search(r"[.!?,;:)]$", text) \
                and re.search(r"[.!?]", " ".join(kept)):
            break   # a heading inside a figure placed in the abstract's column
        kept.append(text)
    return re.sub(r"\s+", " ", " ".join(kept)).strip()[:4000]


ABBREVIATION_RE = re.compile(r"\b(?:e\.g|i\.e|et al|etc|vs|cf|Fig|Figs|Eq|Eqs|Sec|Tab|approx|resp|No)\.$", re.I)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'(\[])", str(text or "").strip())
    out: list[str] = []
    for part in parts:
        if out and ABBREVIATION_RE.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        elif part.strip():
            out.append(part.strip())
    return out


DASHES_RE = re.compile(r"[\u2010-\u2015\u2212]")
STOPWORDS = frozenset("a an the of and or to in on for with by from at as is are was were be been that this these those it its "
                      "we our their which while than into over can also both such".split())


def _norm_text(text: str) -> str:
    s = DASHES_RE.sub("-", unicodedata.normalize("NFKC", str(text or "")))
    s = re.sub(r"-\s*\n\s*", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _squash(text: str) -> str:
    """Letters, digits and CJK only: survives spacing, punctuation and the extractor's lost hyphens ("Englishto-German")."""
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", _norm_text(text))


def quote_in(quote: str, source: str) -> bool:
    """True when the quote occurs in the source, ignoring case, spacing and punctuation; "..." may join fragments that
    occur in that order. Fragments shorter than 10 letters prove nothing."""
    frags = [_squash(f) for f in re.split(r"\s*(?:\.\.\.|…)\s*", str(quote or ""))]
    frags = [f for f in frags if f]
    if not frags or any(len(f) < 10 for f in frags):
        return False
    src, pos = _squash(source), 0
    for f in frags:
        j = src.find(f, pos)
        if j < 0:
            return False
        pos = j + len(f)
    return True


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+|\d+(?:\.\d+)?", _norm_text(text)) if t[0].isdigit() or (len(t) > 2 and t not in STOPWORDS)}


def _best_sentence(texts: list[str], sentences: list[str], threshold: float = 0.5) -> Optional[str]:
    """The sentence sharing the largest share of words with any of ``texts`` (a near-verbatim quote, the plain claim)."""
    scored = [(len(want & _content_tokens(s)) / len(want), s) for want in map(_content_tokens, texts) if want for s in sentences]
    score, best = max(scored, default=(0.0, None))
    return best if score >= threshold else None


def _sentence_covered(sentence: str, quotes: list[str]) -> bool:
    """At least 60% of the sentence's words, and all of its numbers, are in claim quotes taken from it."""
    words = _content_tokens(sentence)
    if not words:
        return True
    got: set[str] = set()
    for q in quotes:
        qt = _content_tokens(q)
        if qt and len(qt & words) >= 0.5 * len(qt):
            got |= qt & words
    return len(got) >= 0.6 * len(words) and {w for w in words if w[0].isdigit()} <= got


def verify_claims(raw, abstract: str, body: str) -> list[dict]:
    """The brief's abstract claims, checked deterministically: each quote must occur in the abstract (in the paper when
    there is no abstract). A claim whose quote is not found is re-anchored on the abstract sentence its words match best
    (at least half of them) or dropped; an abstract sentence that no claim covers becomes a claim of its own, so together the claims
    cover the whole abstract. Ordered as in the abstract, ids 0..N-1."""
    source = abstract or body
    sentences = split_sentences(abstract) if abstract else []
    claims, seen = [], set()
    for item in _as_list(raw):
        if isinstance(item, dict):
            text, quote = str(item.get("text") or item.get("claim") or "").strip(), str(item.get("quote") or "").strip()
        elif isinstance(item, str):
            text = quote = item.strip()
        else:
            continue
        if not quote_in(quote, source):
            anchor = _best_sentence([quote, text], sentences)
            if anchor is None:
                print(f"[StoryboarderV2] dropped an abstract claim whose quote is not in the abstract: {quote[:80]!r}")
                continue
            quote = anchor
        if _squash(quote) in seen:
            continue
        seen.add(_squash(quote))
        claims.append({"text": _plain(text or quote), "quote": quote})
    for s in sentences:
        # the abstract says it, so the reader must get it (a fragment of under four words is extraction debris)
        if len(s.split()) >= 4 and not _sentence_covered(s, [c["quote"] for c in claims]):
            claims.append({"text": s, "quote": s})
    src = _squash(source)

    def position(claim: dict) -> int:
        first = next((f for f in (_squash(x) for x in re.split(r"\.\.\.|…", claim["quote"])) if f), "")
        j = src.find(first) if first else -1
        return j if j >= 0 else len(src)

    claims.sort(key=position)
    return [{"id": i, **c} for i, c in enumerate(claims)]


def paper_examples(raw, body: str, limit: int = 5) -> list[dict]:
    """Example inputs the paper itself shows (a figure's sentence, a sample question), kept only when found in the paper."""
    out: list[dict] = []
    for item in _as_list(raw):
        text = str((item.get("text") or item.get("example") or "") if isinstance(item, dict) else (item or "")).strip()
        text = re.sub(r"^Example sentence shown in a figure:\s*", "", text, flags=re.I)
        source = str(item.get("source") or "").strip() if isinstance(item, dict) else ""
        if text and len(text) <= 400 and quote_in(text, body) and all(_squash(text) != _squash(e["text"]) for e in out):
            out.append({"text": text, "source": source})
    return out[:limit]


# ----------------------------------------------------------------------------- v3: prompts

def brief_prompt_v3(paper: PaperText, abstract: str) -> str:
    caps = "\n".join(paper.captions[:20])
    abstract_block = abstract or ("(no abstract heading was found: take the claims from the paper's own summary of its problem, "
                                  "contributions and results in the introduction and conclusion, quoting those sentences)")
    return f"""Read the paper and its ABSTRACT and write a compact, faithful BRIEF for a manga adaptation. The manga is judged by one test: a reader who never saw the paper must come away able to explain everything the abstract says, with its numbers.

Rules: every statement must come from the paper; copy exact numbers, dataset names, baselines and units; if something is unclear say "unclear" instead of guessing.

Output JSON only:
{{
 "title": "paper title",
 "field": "one phrase",
 "abstract_claims": [{{"id": 0, "text": "the claim in plain words, one line", "quote": "the exact words of the abstract it comes from"}}],
 "problem": "2-3 sentences: what was hard/slow/wrong before, in plain language",
 "key_idea": "ONE sentence a smart teenager could repeat",
 "contributions": ["each contribution the authors claim, one sentence each"],
 "mechanisms": [{{"name": "...", "plain": "2 sentences, no jargon", "metaphor": "a concrete, drawable everyday image for it", "exact": "the precise technical statement with symbols spelled in words"}}],
 "setup": "1-3 sentences: the data, baselines, metrics and model sizes of the experiments",
 "evidence": [{{"claim": "...", "numbers": "exact figures with units, dataset, split, baseline, single model or ensemble", "source": "where the paper reports it (Table 2, Section 6.1)", "higher_is_better": true, "why_it_matters": "..."}}],
 "limitations": ["only limitations the authors state"],
 "takeaway": "1-2 sentences: the paper's own conclusion",
 "examples": [{{"text": "an example input exactly as the paper shows it", "source": "Figure 3"}}],
 "glossary": [{{"term": "...", "plain": "what it is in plain words; for a metric, what it measures and whether higher or lower is better"}}],
 "fun_facts": ["optional surprising details worth a joke"]
}}
"abstract_claims": split the ABSTRACT into atomic claims (usually 6-15) that together cover EVERY sentence of it, in order: the problem, what is new, how it works, every result with its number and setting, and the conclusion. "text" states the claim in plain words a smart teenager understands, keeping its numbers and qualifiers exactly (single model or ensemble, which dataset, "up to", "over"); "quote" copies the words of the abstract it comes from character for character (a sentence or a clause of at least five words). Ids are 0, 1, 2, ... in abstract order.
"mechanisms": every component a reader needs to understand how the method works (usually 3-8).
"evidence": every result the abstract mentions, then the key ablations and analyses: go through every results table and figure, including the appendix, and keep each number with its exact setting and where the paper reports it.
"limitations": only what the authors themselves say; never add your own.
"examples": example inputs the paper itself shows that a story could follow from start to end: a sentence from a figure, a sample question, an input and its output (lines starting with "Example sentence shown in a figure:" are such examples). Copy them exactly; [] when the paper shows none.
"glossary": every technical term, metric and dataset that the abstract or the results use.

ABSTRACT:
{abstract_block}

FIGURE/TABLE CAPTIONS (for orientation):
{caps}

PAPER TEXT:
{paper.body}"""


def _roles(cast: Cast) -> dict[str, str]:
    """Cast names by role (mentor, student, skeptic); missing roles fall back to the members in order."""
    names = cast.names() or ["papi", "kumo", "nezu"]
    by_role: dict[str, str] = {}
    for m in cast.members:
        by_role.setdefault(m.role, m.name)
    return {role: by_role.get(role) or names[min(i, len(names) - 1)] for i, role in enumerate(("mentor", "student", "skeptic"))}


def _v3_rules(cast: Cast) -> str:
    """Panel kinds, board shapes, budgets and data rules shared by the story, fix and quiz-revision prompts."""
    en = BUDGETS_V3["en-US"]
    scene, board = en["scene"], en["board"]
    return f"""PANEL KINDS
- "scene": the cast acting in the running example. An illustrator who cannot write draws the picture, so "visual" describes only what is drawn (setting, poses, expressions, objects, the metaphor), never words, numbers, charts or writing; screens, books and boards in a scene are blank. Props never carry writing: no clipboards, notebooks, pens writing, newspapers, posters, signs, plaques, stone tablets, scrolls, calendars, clocks, word cards or screens with text; use blank cards, red yarn, building blocks, berries (the skeptic's own) or gestures instead. No props shaped like a person or creature (robots, dolls, statues, toy animals): they are drawn as extra characters. All words go in "dialogue" and "narration". Only the characters listed in "characters" appear: cast {", ".join(cast.names())} (lowercase names), never other people, crowds, creatures or copies of a cast member.
- "board": a large blank whiteboard in the scene carries an exhibit that a program draws exactly from "board", so it may hold exact text and numbers. "visual" describes only the setting and the cast's poses and reactions next to the board. Use a board for every result (with the paper's numbers), for a mechanism that needs a diagram, and for the recap. Board shapes (JSON):
  bar: {{"kind": "bar", "title": "...", "subtitle": "what the metric measures; higher is better", "unit": "BLEU", "higher_is_better": true, "bars": [{{"label": "short display name", "value": "@T2.r7.c1", "baseline": true}}, {{"label": "...", "value": "@T2.r9.c1", "highlight": true}}], "source": "Table 2"}}  (3-6 bars from ONE column of one table, so of the same metric, dataset, split and training setup: the paper's headline result is the ONE bar with "highlight": true, the strongest earlier system the paper compares with is marked "baseline": true, and other systems from the same column (other variants of the paper's model included) give context; every label names the system as the paper does ("OpenAI GPT", "ConvS2S Ensemble"), never "prior best" or "SOTA"; the program draws the gap between the highlighted bar and the baseline, so mark the baseline only when the paper makes that comparison. A "value" cites a cell of the TABLES section (see DATA RULES); a number that no table holds is written out, with "quote": a verbatim snippet of the paper that contains it, and "paper_label": the row or condition name exactly as in the paper)
  table: {{"kind": "table", "title": "...", "subtitle": "...", "columns": ["...", "..."], "rows": [{{"cells": ["Transformer (big)", "@T2.r9.c1"]}}], "highlight_rows": [0], "source": "Table 3"}}  (at most 4 columns and 6 rows; a cell may cite a table cell; a row with numbers that no table holds needs a "quote": a verbatim table row or sentence of the paper containing every number of the row)
  flow: {{"kind": "flow", "title": "...", "subtitle": "...", "steps": ["...", "..."], "highlight": [1], "note": "..."}}  (2-7 steps: a pipeline or an architecture)
  compare: {{"kind": "compare", "title": "...", "subtitle": "...", "left": {{"label": "old way", "items": ["..."]}}, "right": {{"label": "new way", "items": ["..."]}}}}  (at most 4 items a side; left = old, right = new)
  links: {{"kind": "links", "title": "...", "subtitle": "...", "tokens": ["The", "animal", "..."], "from": 1, "weights": [0.1, 0.6], "note": "..."}}  (at most 14 tokens of the running example and one weight per token: how much the token at index "from" attends to each; the board is always tagged as an illustration, so never call its weights exact, measured or reproduced from a figure)
  list: {{"kind": "list", "title": "...", "subtitle": "...", "items": ["..."], "open": false}}  (2-6 lines; "open": true draws empty check boxes: the claims board before the story has checked it; on the recap a line gets a tick only when a chart or table of the story showed its numbers, the program decides, and the other lines get a plain bullet)
  Board text is short: titles at most 10 words, bar labels 5, table columns 4 and cells 6, flow steps 8, compare items 10, list lines 14, link tokens one word each.

TEXT BUDGETS (a program checks them)
- scene panel: at most {scene['bubbles']} speech bubbles of at most {scene['bubble']} words each; narration at most {scene['narration']} words.
- scene panel: at most {scene['total']} words of bubbles and narration in total; board panel: at most {board['total']}. Big type keeps the manga readable on a phone, so say one thing per panel and let the next panel say the next.
- board panel: 1-2 cast members reacting to the board, 1-{board['bubbles']} speech bubbles of at most {board['bubble']} words each; narration at most {board['narration']} words.
- No underscores, LaTeX, markdown or emoji in any text; write math in words or plain symbols (√, ², ×, ≈).

DATA RULES (a program checks every number and rejects the panel otherwise)
- When the paper's tables could be read, the PAPER TEXT ends with a section "TABLES READ FROM THE PDF LAYOUT": each table has a "columns" line (c1 = BLEU › EN-DE, ...) and row lines ("[T2.r9]   Transformer (big)   c1: 28.4   c2: 41.8"). A number from that section is CITED, never typed: write @<table>.r<row>.c<column> ("@T2.r9.c1") as a bar's "value", as a table cell, or inside a bubble or the narration ("BERT reaches @T1.r5.c2 on MultiNLI"). The program writes the exact number, its row and its table.
- Every other number on a board, in a bubble or in the narration is copied character for character from the paper ("28.4", "2.3 · 10^19", "3.5 days"). Never compute, round, convert or estimate a number: no differences, ratios, percentages or "x times faster" unless the paper states that exact figure.
- A bar value or table row that does not cite a cell comes with a "quote": a verbatim snippet of the PAPER TEXT that contains it.
- A label says exactly which row or setting a value belongs to: model and size, single model or ensemble, dataset and split. Never call a row "prior best", "previous best" or "state of the art" unless the paper says so for that comparison; name the model instead.
- One board compares one metric on one dataset, split and training setup: the bars of a chart come from ONE column of one table; never mix metrics, units, datasets or setups (e.g. a semi-supervised result next to supervised-only ones) in one chart: use two boards, or a table with the setup as a column. Compare single models with single models, unless the paper's own claim includes ensembles (then say "ensembles included" in the subtitle). The program checks that the strongest earlier system of the column is on the chart.

FIDELITY
- Every fact, number, name and setting comes from the paper; keep the paper's qualifiers (single model, on this dataset, up to, over, about).
- Never overgeneralize: keep each claim exactly as broad as the paper's (if the paper says recurrent models cannot be parallelized within a training example, do not say it of every older model).
- No claims about later impact, adoption or today's practice. Limitations only as the authors state them. Never say a model beats humans unless the paper does.
- No verdict words: never "prove", "proven", "perfect(ly)", "case closed" or "always works"; the paper showed results on its tests ("held up on these tests", "on WMT 2014", "in this example"). Keep the paper's own hedges ("apparently", "seems", "about", "over").
- General knowledge the paper does not state (e.g. that recurrent networks "forget" early words) is not presented as the paper's claim; when the story needs it, a character says it is background, in one short line."""


def number_lines(claims: list[dict], paper_text: str, per_number: int = 3, width: int = 170) -> str:
    """For each key number of the abstract claims, the table cells that hold it (services.ledger: their ids, rows and
    columns, to cite on a board) and the paper lines that contain it (verbatim, trimmed around the number): quotes the
    writer can copy character for character, and the table rows a comparison comes from."""
    text = str(paper_text or "")
    ledger = _ledger(text)
    tables_at = text.find("\n" + _optional("ledger").LEDGER_HEAD) if ledger else -1
    lines = [ln.strip() for ln in (text[:tables_at] if tables_at >= 0 else text).splitlines() if ln.strip()]
    out, seen = [], set()
    for claim in claims:
        for written, key in _key_numbers(claim):
            if key in seen or len(re.sub(r"\D", "", written)) < 2:
                continue
            seen.add(key)
            pattern = re.compile(rf"(?<![\d.]){re.escape(written)}(?![\d])")
            cells = [c.describe() for c in ledger.cells_with(key)[:per_number]] if ledger else []
            hits = []
            for ln in lines:
                m = pattern.search(ln)
                if m:
                    start = max(0, m.start() - width // 2)
                    hits.append(("…" if start else "") + ln[start:start + width] + ("…" if start + width < len(ln) else ""))
                if len(hits) >= per_number - (1 if cells else 0):
                    break
            if cells or hits:
                out.append(f"{written}: " + " | ".join(cells + [f'"{h}"' for h in hits]))
    return "\n".join(out)


def _examples_block(brief: dict) -> str:
    examples = [e for e in _as_list(brief.get("examples")) if isinstance(e, dict) and e.get("text")]
    if not examples:
        return "The paper shows no example input that a story could follow, so use a small everyday example (see the rules)."
    return "Example inputs the paper itself shows (verified in the paper):\n" + "\n".join(
        f"- {e['text']}" + (f" ({e['source']})" if e.get("source") else "") for e in examples)


def story_prompt(brief: dict, claims: list[dict], cast: Cast, panel_range: tuple[int, int], paper: PaperText) -> str:
    lo, hi = panel_range
    r = _roles(cast)
    mentor, student, skeptic = r["mentor"], r["student"], r["skeptic"]
    facts = {k: v for k, v in brief.items() if k not in ("abstract_claims", "abstract", "examples", "brief_check", "factcheck")}
    return f"""Turn this research paper into a SHORT MANGA STORY (in English; it is translated later).

THE READER TEST (this is how the manga is judged): a reader who never saw the paper reads ONLY what is visible in the panels: speech bubbles, narration boxes and the boards drawn in the scenes. Afterwards they must be able to explain everything the paper's ABSTRACT says (the problem, what is new, how it works at the level of the abstract, every result with its number and setting, and the conclusion), and they must have SEEN the key numbers as evidence. Anything that is not visible in a panel does not exist for the reader.

ABSTRACT CLAIMS (every claim is conveyed by the visible text or a board of at least one panel, which lists the claim's id in "claims"):
{json.dumps(claims, ensure_ascii=False)}

CAST
{cast.describe()}
{student} is the reader's stand-in: curious, gets stuck, voices the natural misconception and asks what the reader would ask. {mentor} knows the paper and explains everything on the running example. {skeptic} is a sharp reviewer: doubts every claim, demands proof, names the limitations the authors state, and is won over only by data.

RUNNING EXAMPLE
{_examples_block(brief)}
- Use ONE running example from the first panel to the last: the problem is felt in it, the idea is shown on it, every mechanism is demonstrated on it, and the payoff solves it.
- Prefer an example the paper itself shows: copy it exactly and set "from_paper": true. Only when none fits, use a small everyday situation that mirrors the paper's task, set "from_paper": false, and let a character say once that it is an illustration; its made-up details are never presented as results, and it brings no numbers of its own beyond small counts.

STORY BEATS ("beat" of each panel, in this order; the panels of one beat form a chapter)
1. hook (1-2 panels): {student} has a concrete goal in the running example and runs into the problem the paper addresses; end on a question the reader wants answered.
2. claims (exactly 1 board panel, right after the hook): a "list" board with "open": true, titled like "What the paper claims", with the abstract's headline claims in 3-5 short lines, each with its key number. {skeptic} doubts them and bets something small from her own world (for example her berry tart) that they will not hold up; {student} takes the bet: together they will look at the evidence for each line. The board's subtitle says what the lines are (e.g. "The paper's own summary"), never how the board is drawn.
3. problem (1-2): why the existing approach fails, shown concretely on the example; say exactly what the paper says is wrong with it, and no more.
4. idea (1-2): the key idea in one memorable sentence, said by {student} after a nudge from {mentor}, shown on the example.
5. how (one panel per mechanism the abstract relies on, at most 3): each demonstrated on the running example; use a board (flow, links, compare) when a diagram explains it better than a scene. A metaphor gets at most one panel and is followed by the real mechanism.
6. proof (for EVERY result in the abstract): {skeptic} challenges ("Compared to what?"). For the one or two most important results, first a scene where {student} and {skeptic} each guess in words (no numbers) how it will turn out. Then a board panel answers with the paper's exact numbers against the strongest earlier system and the field (see the bar shape), and a reaction line says what the gap means in the paper's own terms and its scope: which test, single model or ensemble, which training data. One comparison per board; the most telling ablation may get its own board if there is room.
7. payoff (1-2): back to the running example, solved with the idea and SHOWN on a board: the worked steps, or the words that give the answer away, or the paper's own figure as a links board, hedged the way the paper hedges it; never claim the method produced an output for the example (a correct translation, a right answer) unless the paper shows that output. {student} reaches the goal.
8. limits (1-2): the limits or open questions the authors themselves state, each with the reason the paper gives, voiced by {skeptic}; if the paper states none, what it leaves open, in its own terms. Then the bet is settled in one line: {skeptic} pays up because the results the story showed held on these tests (name one), and keeps the limits as her fair point; never "every claim was verified".
9. recap (the last panel, a board panel): the claims board again ("open": false) with each line's key number, titled like "What the paper showed". It lists what the paper reports; it never says that every claim was verified or checked.
Each chapter opens with a question in a bubble or the narration (e.g. "Why was the old way so slow?"), and its last panel makes the reader want the next one.
The first panel where each cast member appears names them with their role in a bubble or the narration ("Kumo, a PhD student", "Professor Papi", "Nezu, the critic"), so the reader knows who is who.

TEACHING RULES
- Every panel teaches one concrete fact; jokes and emotions ride on top of facts, never instead of them. No filler panels and no panel that repeats an earlier one.
- The claims board and the recap only list the claims: every claim must ALSO be taught on a story panel (problem, idea, how, proof, limits or payoff), which lists the claim's id in "claims". A reader who skips the two lists must still learn the whole abstract.
- Explain every technical term, metric and dataset at its first appearance, in a bubble, the narration or the board's subtitle: what it is in plain words and, for a metric, what it measures and which direction is better (e.g. "BLEU: how closely a machine translation matches human translations; higher is better"). This includes words that appear on boards: "ensemble" (several models combined) the first time an ensemble appears, model variants (base vs big: a smaller and a larger version), tasks (what "constituency parsing" means), scores (F1, accuracy) and benchmark names. Board labels are plain words ("Transformer, 4 layers"), never codes like "4l". Use the glossary.
- Every comparative claim of the abstract (better, faster, cheaper, a fraction of the cost, beats humans) is shown with the paper's numbers on a board when the paper reports them (the rival's number included, e.g. the human error rate), and a claim made for several settings (e.g. both large and limited training data) shows each setting.
- The narration is read before the bubbles: it never refers to "the above".
- Narration states facts in plain words; bubbles carry the story and the explanations. The narration never repeats a bubble.
- {lo}-{hi} panels in total; choose the count by how much the abstract contains. Number panels 1..N in reading order.

{_v3_rules(cast)}

OUTPUT: JSON only:
{{"title": "short manga title", "summary": "one sentence", "running_example": {{"text": "one sentence: the situation the story follows", "from_paper": true, "source": "where the paper shows it (e.g. Figure 3), or empty"}},
 "panels": [{{"n": 1, "kind": "scene", "beat": "hook", "title": "short panel title", "characters": ["{student}"], "scene": "the setting", "visual": "what is drawn", "dialogue": [{{"speaker": "{student}", "text": "..."}}], "narration": "...", "emotions": {{"{student}": "frustrated"}}, "board": null, "claims": [0]}}]}}

BRIEF (checked against the paper):
{json.dumps(facts, ensure_ascii=False)}

WHERE THE ABSTRACT'S NUMBERS APPEAR IN THE PAPER (cite the @cells; copy quotes from these lines exactly; table rows name the systems and their rivals):
{number_lines(claims, paper.body) or "(no numbers)"}

PAPER TEXT:
{paper.body}"""


def _panel_json(p: PanelV2) -> dict:
    return {"n": p.n, "kind": p.kind, "beat": p.beat, "title": p.title, "characters": list(p.characters), "scene": p.scene,
            "visual": p.visual, "dialogue": [asdict(b) for b in p.dialogue], "narration": p.narration,
            "emotions": dict(p.emotions), "board": p.board, "claims": list(p.claims)}


def _panel_outline(p: PanelV2) -> dict:
    """A panel as context for a rewrite: its text, and only the kind and title of its board."""
    board = {"kind": p.board.get("kind"), "title": p.board.get("title")} if isinstance(p.board, dict) else None
    return {"n": p.n, "kind": p.kind, "beat": p.beat, "characters": list(p.characters), "dialogue": [asdict(b) for b in p.dialogue],
            "narration": p.narration, "board": board, "claims": list(p.claims)}


def fix_prompt(sb: StoryboardV2, cast: Cast, problems: dict[int, list[str]], paper_text: str = "", tables: str = "") -> str:
    """``paper_text`` when a problem needs the paper (evidence, claims, fact-check findings); otherwise ``tables``, the
    ledger section alone, so a rewrite can still cite table cells."""
    flagged = [{"panel": _panel_json(p), "problems": problems[p.n]} for p in sb.panels if p.n in problems]
    lines = number_lines(sb.abstract_claims, paper_text) if paper_text else ""
    paper = ((f"\n\nWHERE THE ABSTRACT'S NUMBERS APPEAR IN THE PAPER (cite the @cells; copy quotes from these lines exactly):\n{lines}" if lines else "")
             + f"\n\nPAPER TEXT:\n{paper_text}") if paper_text else (f"\n\n{tables}" if tables else "")
    return f"""Fix the flagged panels of this manga script. A checking program found the problems listed with each panel.
Return a replacement for EVERY flagged panel, with the same "n", that solves all of its problems while keeping its beat, its role in the story and the claims it conveys. Change nothing that is not needed.
- "evidence" problems: a number is not supported by the paper, or a chart's bars do not come from one table column. Cite the table cell that holds it ("value": "@T2.r9.c1"), or copy the value character for character from the paper and give that bar or table row a "quote" (a verbatim snippet of the PAPER TEXT containing its numbers); if nothing in the paper supports a number, remove it (drop that bar or row, or make the point without the number). Never compute, round or convert.
- "board" problems: make the board valid for its kind (see the shapes below), or turn the panel into a scene panel ("kind": "scene", "board": null).
- Budget problems: shorten the text, or move a definition into the board's subtitle; never drop a number or a claim.
- A claim no panel conveys, or a claim's missing number: state it, with its key number, in this panel's visible text or board, and add its id to "claims".

{_v3_rules(cast)}

Output JSON only: {{"panels": [{{"n": 3, "kind": "scene", "beat": "...", "title": "...", "characters": ["..."], "scene": "...", "visual": "...", "dialogue": [{{"speaker": "...", "text": "..."}}], "narration": "...", "emotions": {{}}, "board": null, "claims": [0]}}]}}

ABSTRACT CLAIMS:
{json.dumps(sb.abstract_claims, ensure_ascii=False)}

SCRIPT (context only; return only the flagged panels):
{json.dumps([_panel_outline(p) for p in sb.panels], ensure_ascii=False)}

FLAGGED PANELS:
{json.dumps(flagged, ensure_ascii=False)}{paper}"""


def fit_prompt_v3(items: list[dict], language: str) -> str:
    unit = _v3_budget(language)["unit"]
    return f"""Rewrite each text so it fits its limit ("max", in {unit}) while keeping the exact meaning, every number and the tone ({LANG_NAMES.get(language, language)}). No underscores, LaTeX or markdown.
A "visual" item is a scene for the picture that mentions people or copies of a cast member: rewrite it as ONE English sentence (scene and action together) that shows the same idea with objects, places, diagrams and the cast only (each cast member once; counts as simple objects such as cards), and states that nobody else is there.
Return JSON only: {{"items": [{{"id": "...", "text": "rewritten"}}]}}

ITEMS:
{json.dumps(items, ensure_ascii=False)}"""


def _cited(board: dict) -> dict:
    """The board as the fact-check sees it: a bar bound to a table cell says where its value comes from (table, column,
    row), so a chart titled for one dataset but drawn from another column stands out."""
    out = copy.deepcopy(board)
    for bar in out.get("bars") or []:
        if isinstance(bar, dict) and bar.get("cell"):
            bar.pop("quote", None)   # the row line repeats every column; "from" names the one that is drawn
            bar["from"] = bar.pop("cell")
    return out


def _board_check_items(p: PanelV2, whole: bool, paper_text: str = "") -> list[dict]:
    """Fact-check items of a board: the whole board (".b", its exact spec) and each string on it (".b.<path>")."""
    board = p.board
    ledger = _ledger(paper_text) if whole and paper_text else None
    if ledger and isinstance(board, dict) and any(isinstance(b, dict) and b.get("cell") for b in board.get("bars") or []):
        board = _cited(board)
        for bar in board["bars"]:
            c = ledger.cell(bar.get("from")) if isinstance(bar, dict) and bar.get("from") else None
            if c is not None:
                bar["from"] = f'{c.table_label}, column "{c.column}", row "{c.label}"'
    items = [{"id": f"{p.n}.b", "text": json.dumps(board, ensure_ascii=False)}] if whole else []
    return items + [{"id": f"{p.n}.b.{path}", "text": text} for path, text in _board_strings(p.board)]


def _board_check_rules(language: str) -> str:
    strings = ('Ids "N.b.<path>" are single strings on a board that a program draws exactly (a title, label, step, item or cell): '
               'check them like any line; "corrected" is the corrected string, about as short.')
    if language != "en-US":
        return "\n" + strings
    return ("\n" + 'Ids "N.b" are whole boards (charts, tables, diagrams, recap lists) given as their exact spec. ' + strings + """ Also flag:
- a chart or table that mixes different metrics, units, datasets, splits or training setups in one comparison (e.g. BLEU next to training cost, English-German next to English-French scores, or a semi-supervised result next to supervised-only ones);
- a label that presents an ensemble result as a single model (or a single model as an ensemble), or drops the model size or setting its value belongs to;
- a row or label called "prior best", "previous best", "previous state of the art" or similar when the paper does not say so for that comparison;
- a value attached to the wrong row, model or setting (compare the label with its quote and the paper's table);
- a bar chart whose drawn comparison is not one the paper makes: the program draws the gap between the highlighted bar and the bar marked "baseline", so flag a pair from different setups (semi-supervised vs supervised-only, dev vs test), a single model against an ensemble when the paper claims only a single-model result, or a baseline that is not the strongest earlier system the paper compares with.
For an "N.b" problem, "corrected" says in one sentence how the board must change.""")


# ----------------------------------------------------------------------------- v3: boards, numbers, lint (deterministic)

NUMBER_RE = re.compile(r"(?<![A-Za-z\d.])\d+(?:[.,]\d+)*")
# a paper shows results on its tests; the manga never closes the case for it
VERDICT_RE = re.compile(r"\b(?:prove[sdn]?|proving|proof that|case closed|perfect(?:ly)?|flawless(?:ly)?|always works|never fails)\b", re.I)
# props an image model fills with scribbled glyphs (the prototype's clipboard came back reading "好我")
WRITING_PROP_RE = re.compile(r"\b(?:clipboards?|notebooks?|notepads?|newspapers?|posters?|signs?|signboards?|calendars?|clocks?|"
                             r"wall clocks?|pens? (?:writing|scribbling|hovering)|scribbl\w*|jott\w*|menus?|labels?|chalkboards?|"
                             r"plaques?|tablets?|stone slabs?|scrolls?|banners?|certificates?|tickets?|receipts?|maps?|"
                             r"word cards?|cards? (?:with|showing|bearing) (?:words?|text|letters|the sentence)|"
                             r"(?:sentence|words?|letters) (?:written|laid out|spelled)\w*)\b", re.I)
# props shaped like a person or a creature: the image model draws them as one more character, and the art check rightly
# reports an extra figure (BERT's "toy robot" was redrawn three times)
FIGURE_PROP_RE = re.compile(r"\b(?:robots?|androids?|dolls?|puppets?|figurines?|statues?|mannequins?|snowm[ae]n|scarecrows?|"
                            r"teddy bears?|plush\w*|stuffed (?:animal|toy)s?|toy (?:animal|dog|cat|bear|bird|person|people)s?)\b", re.I)
# a chart in a scene: the illustrator draws a fake one, bars without labels or values that a reader takes for the evidence
# (Driver Misconduct's "Kumo looks at the bar chart" came back as two unreadable charts); charts belong on board panels
SCENE_CHART_RE = re.compile(r"\b(?:(?:bar|line|pie)\s+)?(?:charts?|graphs?|diagrams?|flowcharts?|infographics?|histograms?)\b|"
                            r"\b(?:scatter|line|box)\s+plots?\b|\bbar heights?\b|\b(?:rising|growing|shrinking|falling|tall|short)\s+bars\b|"
                            r"\bbars\s+(?:rise|rising|grow|growing|shrink|shrinking|climb\w*)\b|\baxis\b|"
                            r"\b(?:\w+\s+){0,3}bars?\s+on\s+(?:the|a)\s+(?:white)?board\b", re.I)   # "the fake remark bar on the board"
QUOTED_SPAN_RE = re.compile(r"[\"“”‘’'][^\"“”‘’']{3,}[\"“”‘’']")   # quoted words (the paper's own example) are not verdicts
# a board's words that describe the drawing instead of the paper ("Empty checkboxes: we will verify each" leaked from
# the board spec onto two Chinese and one English claims board)
DRAWING_NOTE_RE = re.compile(r"\b(?:check\s?box(?:es)?|empty box(?:es)?|tick box(?:es)?|we will (?:verify|check)|to be (?:verified|checked)|"
                             r"drawn by|the program|board spec)\b|复选框|勾选框|将逐一验证", re.I)
# words that point at a drawn exhibit: on a scene panel there is none ("Steps to reach the same accuracy" narrating a
# chart that was dropped)
CHART_WORDS_RE = re.compile(r"\b(?:this|these)\s+(?:chart|graph|plot|bars|table|diagram)\b|"
                            r"\b(?:the|this)\s+(?:chart|graph|plot)\s+(?:shows|says|proves)\b", re.I)
COMPARATIVE_RE = re.compile(r"\b(?:better|best|beats?|beating|outperform\w*|surpass\w*|exceed\w*|improv\w*|faster|cheaper|fewer|"
                            r"less|lower|higher|more than|over|fraction|state[- ]of[- ]the[- ]art|record|gain\w*|absolute)\b", re.I)


def _comparison_shown(panels: list[PanelV2]) -> bool:
    """A panel among these shows a comparison as a drawing: a bar chart with two or more bars, or a table."""
    for p in panels:
        b = p.board if p.kind == "board" and isinstance(p.board, dict) else None
        if b and (b.get("kind") == "table" or (b.get("kind") == "bar" and len(b.get("bars") or []) >= 2)):
            return True
    return False


# the story checks the evidence the panels show; it never certifies the whole paper ("every claim was verified" was a
# template line that readers caught on 9 of 13 reports while some claims had no evidence on any panel)
ALL_VERIFIED_RE = re.compile(r"\b(?:every|all(?: the)?|each)\s+(?:paper\s+)?(?:claims?|lines?|results?|points?|promises?)\s+"
                             r"(?:(?:was|were|is|are|got|have been|has been)\s+)?(?:verified|checked|confirmed|proven|held|delivered)\b|"
                             r"\bverified\s+(?:paper\s+)?claims\b|\bdelivered exactly what it (?:claimed|promised)\b|"
                             r"\bclaims?\s+(?:all\s+)?(?:verified|confirmed)\b", re.I)
# small numbers a line may write as words ("eight GPUs", "first place", 八块 GPU): they still show the figure
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "first": 1, "second": 2, "third": 3}
NUMBER_WORD_RE = re.compile(r"\b(?:" + "|".join(NUMBER_WORDS) + r")\b", re.I)
CJK_NUMERALS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
YEAR_RE = re.compile(r"(?:19|20)\d\d")
HIDDEN_BOARD_KEYS = frozenset({"kind", "quote", "paper_label", "highlight", "highlight_rows", "illustrative", "from", "weights", "higher_is_better",
                               "cell", "cell_ids", "unbound"})


def _numbers(text: str) -> list[str]:
    """The number tokens of a line, sorted: a rewrite or translation must keep them all."""
    return sorted(re.findall(r"\d+(?:[.,]\d+)*", str(text or "")))


def _number_tokens(text: str) -> list[tuple[str, str]]:
    """(as written, canonical value) of each number in a text, read the way services.boards reads board values:
    "2.3 · 10^19", "2.3 × 10¹⁹" and "2.3e19" are one number, "2.0" equals "2", and digits inside names (GPT-3, CIFAR-10,
    v1.1) are no numbers. Plain decimals when services.boards is unavailable."""
    text = str(text or "")
    boards = _optional("boards")
    if boards is not None:
        try:
            return [(n.text, n.key) for n in boards.find_numbers(text)]
        except Exception as e:  # fall back to plain decimals
            print(f"[StoryboarderV2] number scan failed: {type(e).__name__}: {e}")
    out = []
    for tok in NUMBER_RE.findall(re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)):
        try:
            out.append((tok, format(Decimal(tok.replace(",", ".")).normalize(), "f")))
        except InvalidOperation:
            continue
    return out


def _number_keys(text: str) -> set[str]:
    """Canonical values of the numbers a reader sees in a text, small numbers written as words or Chinese numerals
    included ("eight GPUs", "first place", "两个" show 8, 1 and 2)."""
    text = str(text or "")
    words = [NUMBER_WORDS[m.group().lower()] for m in NUMBER_WORD_RE.finditer(text)]
    return {key for _, key in _number_tokens(text)} | {str(v) for v in words + [CJK_NUMERALS[c] for c in text if c in CJK_NUMERALS]}


def _key_numbers(claim: dict) -> list[tuple[str, str]]:
    """The numbers a reader must see for a claim, as (written, canonical value): those in its abstract quote, except
    years (WMT 2014)."""
    out: dict[str, str] = {}
    for written, key in _number_tokens(str(claim.get("quote") or claim.get("text") or "")):
        if not YEAR_RE.fullmatch(written):
            out.setdefault(key, written)
    return [(written, key) for key, written in out.items()]


def _figures_kept(source: str, new: str) -> bool:
    """A translated line still shows every number of the English line (compared as values: "2.0" may become "2", a lone
    "8" may become 八); numbers it adds are fine."""
    have = _number_keys(new)
    return all(key in have for _, key in _number_tokens(source))


def _normalized_board(spec):
    boards = _optional("boards")
    if boards is None or not isinstance(spec, dict):
        return spec
    try:
        norm = boards.normalize_board(spec)
    except Exception as e:  # a malformed spec stays as it is, so lint reports it
        print(f"[StoryboarderV2] board normalization failed: {type(e).__name__}: {e}")
        norm = None
    return norm if isinstance(norm, dict) else spec


def board_issues(spec, paper_text: str) -> list[str]:
    """What is wrong with a board spec: its shape (services.boards) and its numbers (services.evidence)."""
    if not isinstance(spec, dict) or not spec:
        return ["board: missing"]
    boards = _optional("boards")
    if boards is None:
        return []
    try:
        norm = boards.normalize_board(spec)
        if not isinstance(norm, dict):
            return [f"board: {x}" for x in (boards.board_problems(spec) or ["not a valid board (unknown kind or missing fields)"])]
        problems = [f"board: {x}" for x in boards.board_problems(norm)]
    except Exception as e:
        return [f"board: not a valid board ({type(e).__name__}: {e})"]
    evidence = _optional("evidence")
    if evidence is not None:
        try:
            problems += [_with_hints(f"evidence: {x}", paper_text) for x in evidence.verify_board(norm, paper_text)]
        except Exception as e:  # a checker bug is no reason to drop the board; the fact-check still reads it
            print(f"[StoryboarderV2] WARNING: board evidence check failed: {type(e).__name__}: {e}")
    return problems


ROW_LABEL_RE = re.compile(r'^evidence: (?:bar|row \d+|table row \d+ \() ?"([^"]+)"\)?(?: cell \d+)?: the value ([^ ]+)')


def _with_hints(problem: str, paper_text: str, limit: int = 3) -> str:
    """An unsupported bar or row gets the paper lines that name it together with numbers, table rows and values close
    to the written one first: the fix round can copy the exact value and quote from there instead of a figure, whose
    extracted labels and values sit on different lines (CoT's Figure 2 "57" is Table 1's "56.9")."""
    m = ROW_LABEL_RE.match(problem)
    boards = _optional("boards")
    if not m or boards is None:
        return problem
    want = {w for w in _content_tokens(m.group(1)) if not w[0].isdigit()}
    target = boards.parse_value(m.group(2))
    if not want:
        return problem
    scored = []
    for ln in str(paper_text or "").splitlines():
        ln = ln.strip()
        if len(ln) < 8:
            continue
        overlap = len(want & _content_tokens(ln))
        values = [n for n in boards.find_numbers(ln) if not n.short]
        exact = target is not None and any(abs(float(n.key) - target) < 1e-9 for n in values)
        named = overlap >= max(1, math.ceil(0.5 * len(want)))
        if not values or not (named or exact and len(values) >= 2):   # a table row with the value, whatever its row header
            continue
        close = target is not None and any(abs(float(n.key) - target) <= max(1.0, 0.03 * abs(target)) for n in values)
        scored.append((named and close, exact, len(values) >= 2, overlap, -len(scored), ln[:160]))
    hits = [x[-1] for x in sorted(scored, reverse=True)[:limit]]
    return problem + (" Paper lines that name it: " + " | ".join(f'"{h}"' for h in hits) if hits else "")


def board_text(spec) -> str:
    """Everything a reader sees on a board (titles, labels, values, steps, cells, source), not the quotes behind it."""
    out: list[str] = []

    def walk(value, key=""):
        if key in HIDDEN_BOARD_KEYS:
            return
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and key == "value":
            out.append(str(value))
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, str(k))
        elif isinstance(value, list):
            for v in value:
                walk(v, key if key in ("cells", "items", "steps", "tokens", "columns") else "")

    walk(spec if isinstance(spec, dict) else {})
    return " ".join(out)


def visible_text(p: PanelV2) -> str:
    """The reader's view of a panel: bubbles, narration and board."""
    return " ".join([b.text for b in p.dialogue] + [p.narration, board_text(p.board) if p.board else ""]).strip()


def _board_strings(spec) -> list[tuple]:
    """(path, text) of each translatable board string, from services.boards ([] without it)."""
    boards = _optional("boards")
    if boards is None or not isinstance(spec, dict):
        return []
    try:
        return [(path, str(text)) for path, text in boards.board_strings(spec) if isinstance(text, str) and text.strip()]
    except Exception as e:
        print(f"[StoryboarderV2] board strings unavailable: {type(e).__name__}: {e}")
        return []


def _set_board_strings(spec, updates: dict[str, str]):
    """The board with some strings replaced (``updates`` keyed by str(path) as the prompts show it), or ``spec`` itself
    when nothing changed: services.boards refuses replacements that change a number."""
    before = {str(path): (path, text) for path, text in _board_strings(spec)}
    real = {before[k][0]: v for k, v in updates.items() if k in before}
    boards = _optional("boards")
    if boards is None or not real:
        return spec
    try:
        new = boards.set_board_strings(spec, real)
    except Exception as e:
        print(f"[StoryboarderV2] board strings not applied: {type(e).__name__}: {e}")
        return spec
    after = {str(path): text for path, text in _board_strings(new)} if isinstance(new, dict) else {}
    return new if any(after.get(k) != text for k, (_, text) in before.items()) else spec


LOWER_BETTER_RE = re.compile(r"\b(?:lower|smaller|less|fewer)\s+is\s+better\b|越(?:低|小|少)越好", re.I)
TEX_MARK_RE = re.compile(r"\$|\*\*")


def _numbers_supported(text: str, paper_text: str) -> bool:
    """No number in the text that the paper lacks (lone digits are fine); True without services.evidence."""
    evidence = _optional("evidence")
    if evidence is None or not str(text or "").strip():
        return True
    try:
        return not evidence.verify_text_numbers(str(text), paper_text)
    except Exception as e:  # a checker bug is no reason to drop text
        print(f"[StoryboarderV2] number check failed: {type(e).__name__}: {e}")
        return True


def _strip_marks(value, key: str = ""):
    """A board without TeX / markdown marks in what is drawn; quotes and paper labels stay verbatim (they are matched
    against the paper)."""
    if key in ("quote", "paper_label"):
        return value
    if isinstance(value, str):
        return re.sub(r"\s+", " ", TEX_MARK_RE.sub("", value)).strip()
    if isinstance(value, dict):
        return {k: _strip_marks(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_marks(v) for v in value]
    return value


def _bars_as_table(board: dict) -> dict:
    """Bars that mix units or notations as a table: the values and their quotes stay, and no bar length suggests that
    they compare."""
    table = {"kind": "table", "title": board.get("title", ""), "columns": ["", board.get("unit", "")],
             "rows": [{"cells": [b.get("label", ""), b["value"]], **({"quote": b["quote"]} if b.get("quote") else {})}
                      for b in board["bars"]],
             "highlight_rows": [i for i, b in enumerate(board["bars"]) if b.get("highlight")]}
    return table | {k: board[k] for k in ("subtitle", "note", "source") if board.get(k)}


LEDGER_ROW_RE = re.compile(r"^\[[A-Z]{1,3}\d{1,3}[a-z]?\.r\d+\]")


def _snap_to_paper(bar: dict, paper_text: str, boards) -> Optional[dict]:
    """A bar whose value the paper does not print as written (read off a figure: CoT's "57" and "55") snapped to the paper
    line that names it with the nearest value within max(1, 2%): its exact number and that line as the quote. None when
    no such line exists or two lines are equally near (no guessing between rows)."""
    target = boards.parse_value(bar.get("value"))
    names = [{w for w in _content_tokens(str(bar.get(k) or "")) if not w[0].isdigit()} for k in ("paper_label", "label")]
    names = [n for n in names if n]   # the paper's row name first, the display label second: either may name the line
    if target is None or not names:
        return None
    best: list[tuple[float, str, str]] = []
    for raw in str(paper_text or "").splitlines():
        ln = raw.strip()
        if LEDGER_ROW_RE.match(ln):
            continue   # a ledger row holds every column of its table row: services.ledger binds near values within one column
        words = _content_tokens(ln)
        if len(ln) < 6 or not any(len(n & words) >= max(1, math.ceil(0.5 * len(n))) for n in names):
            continue
        values = [n for n in boards.find_numbers(ln) if not n.short and n.exponent is None and not YEAR_RE.fullmatch(n.text)]
        # a near value only from a table row (columns, several numbers): a prose sentence that mentions the system can hold
        # any number ("GPT-3 on SST-2 ... (54.3%)" is no reading of the prior best's 55)
        row = len(values) >= 3 or (len(values) >= 2 and len(re.findall(r"\S {2,}(?=\S)", ln)) >= 1)
        for n in values:
            gap = abs(float(n.key) - target)
            if gap == 0 or (row and gap <= max(1.0, 0.02 * abs(target))):
                best.append((gap, n.text, ln[:200]))
    if not best:
        return None
    best.sort(key=lambda t: t[0])
    if len(best) > 1 and best[1][0] == best[0][0] and best[1][1] != best[0][1]:
        return None
    _gap, value, line = best[0]
    print(f"[StoryboarderV2] bar \"{bar.get('label')}\": {bar.get('value')} snapped to the paper's {value} ({line[:80]})")
    return {**bar, "value": value, "quote": line}


def _salvage_board(spec, paper_text: str) -> Optional[dict]:
    """Last resort for a board the fix rounds left invalid, so that its evidence is not lost with it: the normalized board
    without what the paper does not support (bars, rows, steps and items holding a number the paper lacks; a title,
    subtitle, unit, note, source, column or side label holding one is cleared), "lower is better" taken from its own
    words, bars that mix units or notations shown as a table, and $ / ** marks removed. Counts and lengths outside the
    limits stay: the renderer copes with them. None when nothing usable is left or a number is still unsupported."""
    boards, evidence = _optional("boards"), _optional("evidence")
    if boards is None or not isinstance(spec, dict):
        return None

    def ok(text) -> bool:
        return _numbers_supported(str(text or ""), paper_text)

    def alone(entry: dict) -> bool:   # one bar or table row checked on its own: its quote, its label, its paper line
        return evidence is None or not evidence.verify_board(entry, paper_text)

    try:
        board = boards.normalize_board(spec)
        if not isinstance(board, dict):
            return None
        board = _strip_marks(copy.deepcopy(board))
        kind = board["kind"]
        if kind == "bar":
            said = " ".join([board["title"], board.get("subtitle", ""), board.get("unit", "")])
            if board.get("higher_is_better", True) and LOWER_BETTER_RE.search(said):
                board["higher_is_better"] = False   # the board's own words win: the delta badge must not call worse better
            board["bars"] = [b if alone({"kind": "bar", "title": "", "bars": [b]}) else _snap_to_paper(b, paper_text, boards)
                             for b in board["bars"]]
            board["bars"] = [b for b in board["bars"] if b and alone({"kind": "bar", "title": "", "bars": [b]})]
            if len(board["bars"]) < 2:   # a one-bar chart compares nothing: the panel is better told as a scene
                return None
            if any(x.startswith("the bars mix") for x in boards.board_problems(board)):
                board, kind = _bars_as_table(board), "table"
        if kind == "table":
            rows = board["rows"]
            kept = [i for i, r in enumerate(rows) if alone({"kind": "table", "title": "", "columns": [], "rows": [r]})]
            board["rows"] = [rows[i] for i in kept]
            board["highlight_rows"] = [kept.index(i) for i in board.get("highlight_rows", []) if i in kept]
            board["columns"] = [c if ok(c) else "" for c in board["columns"]]
            if not board["rows"]:
                return None
        elif kind == "flow":
            kept = [i for i, step in enumerate(board["steps"]) if ok(step)]
            board["highlight"] = [kept.index(i) for i in board.get("highlight", []) if i in kept]
            board["steps"] = [board["steps"][i] for i in kept]
            if len(board["steps"]) < 2:
                return None
        elif kind == "list":
            board["items"] = [x for x in board["items"] if ok(x)]
            if not board["items"]:
                return None
        elif kind == "compare":
            for side in (board["left"], board["right"]):
                side["items"] = [x for x in side["items"] if ok(x)]
                side["label"] = side["label"] if ok(side["label"]) else ""
            if not (board["left"]["items"] and board["right"]["items"]):
                return None
        for key in ("subtitle", "unit", "note", "source"):
            if key in board and not ok(board[key]):
                del board[key]
        if not ok(board.get("title")):
            board["title"] = ""
        final = boards.normalize_board(board)
        if not isinstance(final, dict) or (evidence is not None and evidence.verify_board(final, paper_text)):
            return None
        return final
    except Exception as e:  # a salvage bug must not fail the job: the panel is then drawn as a scene
        print(f"[StoryboarderV2] board salvage failed: {type(e).__name__}: {e}")
        return None


def _illustrative_example(sb: StoryboardV2) -> str:
    """The running example when it is made up (its own numbers may appear in the text although the paper lacks them)."""
    example = sb.brief.get("running_example") if isinstance(sb.brief.get("running_example"), dict) else {}
    return "" if example.get("from_paper", True) else sb.running_example


def _recap_panel(sb: StoryboardV2) -> PanelV2:
    recaps = [p for p in sb.panels if p.beat == "recap"]
    boards = [p for p in recaps if p.kind == "board"]
    return (boards or recaps or sb.panels)[-1]


GENERIC_LABEL_RE = re.compile(r"\b(?:prior|previous|earlier|old|best|top|sota|state[\s-]of[\s-]the[\s-]art|baseline|rival|"
                              r"published|reported|leaderboard|#\s*\d+|no\.\s*\d+)\b", re.I)
GENERIC_LABEL_WORDS = frozenset("prior previous earlier old best top sota state art baseline rival published reported leaderboard "
                                "single ensemble model models system systems result results ours".split())
EXACT_CLAIM_RE = re.compile(r"\b(?:exact|exactly|measured|reproduced|actual|real)\b", re.I)


def _label_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(w) > 1 and w not in GENERIC_LABEL_WORDS
            and w not in STOPWORDS}


def _higher_is_better(spec: dict) -> bool:
    higher = str(spec.get("higher_is_better", True)).strip().lower() not in ("false", "0", "no")
    if spec.get("higher_is_better") is None and re.search(r"lower is better|越低越好|error|cost|time", " ".join(
            str(spec.get(k) or "") for k in ("title", "subtitle", "unit")), re.I):
        higher = False
    return higher


def _ledger_bar_issues(spec: dict, bars: list[dict], lit: list[dict], ledger) -> Optional[list[str]]:
    """Checks of a bar chart whose bars are bound to ledger cells: one column of one table, and the strongest earlier
    system of that column (services.ledger.strongest_rival) on the chart. None when the ledger cannot judge the chart
    (no bound highlight), so the plain rival check runs instead."""
    cells = [(b, c) for b in bars if b.get("cell") and (c := ledger.cell(b["cell"])) is not None]
    if not cells:
        return None
    columns = sorted({f'{c.table_label} column c{c.col} ("{c.column}")' for _, c in cells})
    if len(columns) > 1:
        return [f"evidence: the bars come from different table columns ({'; '.join(columns)}): one chart shows one column "
                "of one table (one metric, dataset and setting); use two charts, or a table board"]
    # a bar that cites nothing, with a value that is not in the cited column, comes from elsewhere (BERT's chart put the
    # leaderboard GLUE scores 72.8 and 80.5 next to Table 1's average 74.0: two different metrics)
    table, col = cells[0][1].table, cells[0][1].col
    held = {n.key for c in ledger.column(table, col) for n in _optional("boards").find_numbers(c.value)} if _optional("boards") else set()
    strays = [str(b.get("label") or b.get("value")) for b in bars if not b.get("cell")
              and not ({n.key for n in _optional("boards").find_numbers(str(b.get("value") or ""))} <= held)]
    if strays:
        return [f'evidence: bars {", ".join(strays)} are not in the column the other bars cite ({columns[0]}): a chart shows the '
                "values of one table column (cite them), or only values the text reports together for one comparison"]
    me = next((c for b, c in cells if b in lit), None) if len(lit) == 1 else None
    if me is None:
        return None
    context = " ".join(str(spec.get(k) or "") for k in ("title", "subtitle", "note"))
    rival = ledger.strongest_rival(me.id, _higher_is_better(spec), context)
    if rival is None or any(c.id == rival.id for _, c in cells):
        return []
    return [f'evidence: the strongest other system in this column ({rival.table_label}, "{rival.column}") is "{rival.label}" '
            f'with {rival.primary} (@{rival.id}), and it is not on the chart: add it ("value": "@{rival.id}") so the reader sees '
            'the whole field. Mark as "baseline" the system the claim of this chart compares with (the strongest earlier '
            'system when the claim is "better than before"); a system that is not a baseline stays a plain context bar']


def _story_board_issues(spec, ledger=None) -> list[str]:
    """Board rules the shape check does not know: who a bar is, what a highlight means, what a links board may claim,
    and (with the tables read from the PDF) one column per chart and its strongest earlier system on it."""
    if not isinstance(spec, dict):
        return []
    kind = str(spec.get("kind") or "").lower()
    problems: list[str] = []
    if kind in ("bar", "bars", "bar_chart", "leaderboard", "chart"):
        bars = [b for b in _as_list(spec.get("bars")) if isinstance(b, dict)]
        lit = [b for b in bars if str(b.get("highlight")).lower() in ("true", "1", "yes")]
        if len(lit) > 1:
            problems.append(f"{len(lit)} highlighted bars: highlight only the paper's headline result; show its other variants "
                            "as plain context bars (a chart compares one result with one baseline)")
        for b in bars:
            label, paper_label = str(b.get("label") or ""), str(b.get("paper_label") or "")
            if GENERIC_LABEL_RE.search(label) and not (_label_words(label) & _label_words(paper_label) if paper_label
                                                        else _label_words(label)):
                problems.append(f'bar "{label}" names no system: label it with the system the paper reports'
                                + (f' ("{paper_label}")' if paper_label else "") + '; "prior best" hides what the result is compared with')
        settings = _mixed_settings(bars)
        if settings:
            problems.append(f"the chart mixes {settings}: one setup per chart (the program draws the gap between the "
                            "highlighted bar and the baseline, so both must come from the same setup); use two charts or a table")
        base = [b for b in bars if str(b.get("baseline")).lower() in ("true", "1", "yes") and b not in lit]
        judged = _ledger_bar_issues(spec, bars, lit, ledger) if ledger else None
        if judged is not None:   # the ledger knows the whole column: its strongest system must be on the chart
            problems += judged
        elif len(lit) == 1 and base:
            stronger = _stronger_rivals(bars, lit[0], base[0], spec)
            if stronger:
                problems.append(f'the baseline "{base[0].get("label")}" is not the strongest earlier system on the chart '
                                f'({", ".join(stronger)} scores better): mark the strongest one the paper compares with as the baseline')
        if len(bars) == 2 and any(str(b.get("baseline")).lower() in ("true", "1", "yes") for b in bars):
            problems.append("only 2 bars: show the field too: add 1-3 other systems from the same table column (same metric, "
                            "dataset and setting), each with its quote, when the paper lists them")
    if kind in ("links", "attention", "word_links"):
        said = " ".join(str(spec.get(k) or "") for k in ("title", "subtitle", "note"))
        if EXACT_CLAIM_RE.search(said):
            problems.append("links board claims exact or measured weights: its weights are an illustration; say \"illustration\" "
                            "or describe what the paper's figure shows in words")
    return problems


# setup words in bar labels: two different ones of a group on one chart are two setups (a semi-supervised result next to a
# WSJ-only one, a dev score next to a test score)
SETTING_GROUPS = {
    "training data": {"wsj": "wsj only", "semi": "semi-supervised", "semisupervised": "semi-supervised", "supervised": "supervised",
                      "limited": "limited data", "large": "large data"},
    "splits": {"dev": "dev", "val": "validation", "validation": "validation", "test": "test"},
}


def _mixed_settings(bars: list[dict]) -> str:
    out = []
    for group, words in SETTING_GROUPS.items():
        if group == "training data":   # "large" is a model size too: only a "large data" phrase counts
            found = set()
            for b in bars:
                label = re.sub(r"\bsemi[\s-]+supervised", "semisupervised", str(b.get("label") or "").lower())
                found |= {words[w] for w in re.findall(r"[a-z]+", label) if w in words and w != "large" and w != "limited"}
                found |= {"large data" for _ in re.findall(r"\blarge(?:r)?\s+(?:data|training)", label)}
                found |= {"limited data" for _ in re.findall(r"\blimited\s+(?:data|training)", label)}
        else:
            found = {words[w] for b in bars for w in re.findall(r"[a-z]+", str(b.get("label") or "").lower()) if w in words}
        if len(found) > 1:
            out.append(f"{group} ({', '.join(sorted(found))})")
    return " and ".join(out)


def _stronger_rivals(bars: list[dict], best: dict, baseline: dict, spec: dict) -> list[str]:
    """Other systems (not variants of the highlighted one: no shared name word) that beat the marked baseline."""
    boards = _optional("boards")
    if boards is None:
        return []
    higher = _higher_is_better(spec)
    value = lambda b: boards.parse_value(b.get("value"))
    base_v, mine = value(baseline), _label_words(best.get("label", "")) | _label_words(best.get("paper_label", ""))
    if base_v is None:
        return []
    out = []
    for b in bars:
        if b is best or b is baseline or str(b.get("highlight")).lower() in ("true", "1", "yes"):
            continue
        v = value(b)
        variant = bool(mine & (_label_words(b.get("label", "")) | _label_words(b.get("paper_label", ""))))
        if v is not None and not variant and (v > base_v if higher else v < base_v):
            out.append(str(b.get("label") or b.get("value")))
    return out


GAP_RE = re.compile(r"(?:[+±]\s*|\bby\s+(?:over\s+|more than\s+|about\s+)?)?(\d+(?:\.\d+)?)\s*(?:-?\s*points?\b|pts?\b|%|percentage points?)",
                    re.I)


def _words_vs_chart(p: PanelV2) -> list[str]:
    """A gap the words state ("7.7 points") must be the gap the chart draws, or one of its values: otherwise the reader
    sees two different comparisons in one panel (a leaderboard score in the bubble, a table average on the board)."""
    boards = _optional("boards")
    spec = p.board if isinstance(p.board, dict) else None
    if boards is None or not spec:
        return []
    try:
        norm = boards.normalize_board(spec)
        if not norm or norm.get("kind") != "bar":
            return []
        shown = {round(v, 4) for v in (boards.parse_value(b["value"]) for b in norm["bars"]) if v is not None}
        badge = boards.delta_badge(norm)
        if badge and badge.get("kind") == "diff":
            gap = boards.parse_value(badge["text"].replace("\u2212", "-"))
            if gap is not None:
                shown.add(round(abs(gap), 4))
    except Exception:  # a checker bug never blocks the script
        return []
    words = " ".join([b.text for b in p.dialogue] + [p.narration])
    said = [m.group(1) for m in GAP_RE.finditer(words)]
    odd = [x for x in said if round(float(x), 4) not in shown]
    if not odd:
        return []
    return [f"the words say {', '.join(odd)} but the chart shows {', '.join(sorted(map(str, shown)))}: make the bubble and "
            "the chart show the same comparison (same table, same metric, same rival)"]


def _example_tokens(sb: StoryboardV2) -> set[str]:
    """Content words of the running example (its quoted part when the description quotes it)."""
    text = sb.running_example or ""
    quoted = re.findall(r"[\"“]([^\"”]{8,})[\"”]", text)
    words = _content_tokens(" ".join(quoted) if quoted else text)
    return {_stem(w) for w in words if not w[0].isdigit()} - {"example", "sentence", "training", "input", "mask"}


def _stem(word: str) -> str:
    """Crude singular: "balls" and "ball" (and "roger's" and "roger") count as the same word in the example check."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _teaching_panels(panels: list[PanelV2]) -> list[PanelV2]:
    """The panels that teach (not the claims board or the recap, which only list the claims)."""
    return [p for p in panels if p.beat not in LIST_BEATS] or list(panels)


def _home_panel(sb: StoryboardV2, claim: dict) -> PanelV2:
    """The teaching panel that should carry a claim no teaching panel conveys yet: the one whose words share the most with
    the claim; ties go to evidence panels when the claim has numbers, else to the idea and how panels; then the later."""
    teaching = _teaching_panels(sb.panels)
    want = _content_tokens(str(claim.get("text", "")) + " " + str(claim.get("quote", "")))
    preferred = ("proof",) if _key_numbers(claim) else ("idea", "how", "problem")
    def score(item):
        i, p = item
        return (len(want & _content_tokens(visible_text(p))), p.beat in preferred, p.kind == "board" and bool(_key_numbers(claim)), i)
    return max(enumerate(teaching), key=score)[1]


def lint_v3(sb: StoryboardV2, paper_text: str) -> dict[int, list[str]]:
    """Deterministic checks of an English v3 script: {panel n: [problems]} for the panels that have any.
    Per panel: kind vs board, the board's shape and evidence, text budgets, numbers missing from the paper.
    Story-wide: a claim no panel conveys, or a claim's key number shown on none of the panels that convey it; these go
    to the recap panel (the last panel when there is none), which restates every claim anyway."""
    out: dict[int, list[str]] = {}
    budget = BUDGETS_V3["en-US"]
    evidence = _optional("evidence")
    allowed = _illustrative_example(sb)
    ledger = _ledger(paper_text)
    cite = _optional("ledger") if ledger else None
    for p in sb.panels:
        problems: list[str] = []
        b = budget["board" if p.kind == "board" else "scene"]
        if p.kind == "board":
            if not isinstance(p.board, dict) or not p.board:
                problems.append("board panel without a board: give it a board (bar, table, flow, compare, links or list) or make it a scene panel")
            else:
                problems += board_issues(p.board, paper_text) + _story_board_issues(p.board, ledger) + _words_vs_chart(p)
                if p.board.get("unbound"):
                    problems.append(f"evidence: the board cites {', '.join(p.board['unbound'])}, which name no number cell of the "
                                    "TABLES section: cite an existing cell id, or copy the paper's number with a verbatim quote")
            if len(p.characters) > 2:
                problems.append(f"{len(p.characters)} cast members on a board panel: at most 2")
            if not p.dialogue:
                problems.append("board panel without a speech bubble: a cast member reacts to the board")
        elif p.board:
            problems.append('scene panel with a board: make it a board panel ("kind": "board") or remove the board')
        elif not p.dialogue and not p.narration:
            problems.append("no visible text: every panel teaches one fact in a bubble or the narration")
        props = sorted({m.group(0).lower() for m in WRITING_PROP_RE.finditer(f"{p.scene} {p.visual}")})
        if props:
            problems.append(f"the picture has props that carry writing ({', '.join(props)}): the illustrator draws them with "
                            "scribbles; use blank cards, red yarn, blocks, berries or gestures instead")
        charts = sorted({m.group(0).lower() for m in SCENE_CHART_RE.finditer(f"{p.scene} {p.visual}")}) if p.kind != "board" else []
        if charts:
            problems.append(f"the picture of a scene shows a chart ({', '.join(charts)}): the illustrator draws a fake one, bars "
                            "without labels or values; put the chart on a board panel (the program draws it) or show only the "
                            "characters' reaction")
        figures = sorted({m.group(0).lower() for m in FIGURE_PROP_RE.finditer(f"{p.scene} {p.visual}")})
        if figures:
            problems.append(f"the picture has props shaped like a person or creature ({', '.join(figures)}): they are drawn as "
                            "extra characters; use objects that are not figures (blocks, cards, a box, tools)")
        if len(p.dialogue) > b["bubbles"]:
            problems.append(f"{len(p.dialogue)} speech bubbles: at most {b['bubbles']} on a {p.kind} panel")
        texts = [(f"bubble {i} ({x.speaker})", x.text, b["bubble"]) for i, x in enumerate(p.dialogue)]
        if p.narration:
            texts.append(("narration", p.narration, b["narration"]))
        total = sum(text_len(text, "en-US") for _, text, _ in texts)
        if total > b["total"]:
            problems.append(f"{total} words of bubbles and narration in total: at most {b['total']} on a {p.kind} panel "
                            "(say less, or move a fact to a neighbouring panel)")
        verdicts = sorted({m.group(0).lower() for _, text, _ in texts for m in VERDICT_RE.finditer(QUOTED_SPAN_RE.sub(" ", text))})
        if verdicts:
            problems.append(f"verdict words {', '.join(verdicts)}: the paper shows results on its tests; write \"showed\", "
                            "\"on these tests\" or \"in this example\" instead")
        if isinstance(p.board, dict):
            note = next((m.group(0) for k in ("title", "subtitle", "note") for m in [DRAWING_NOTE_RE.search(str(p.board.get(k) or ""))] if m), "")
            if note:
                problems.append(f"the board's words describe the drawing (\"{note}\"): its title and subtitle say what the "
                                "lines are about (\"The paper's own summary\"), never how the board is drawn")
        elif p.kind != "board":
            pointed = next((m.group(0) for _, t, _ in texts for m in [CHART_WORDS_RE.search(t)] if m), "")
            if pointed:
                problems.append(f"\"{pointed}\": this is a scene panel with no chart or table; say the fact in words, or make "
                                "it a board panel that draws the exhibit")
        said = [t for _, t, _ in texts] + ([board_text(p.board)] if isinstance(p.board, dict) else [])
        certified = sorted({m.group(0).lower() for t in said for m in ALL_VERIFIED_RE.finditer(t)})
        if certified:
            problems.append(f"\"{certified[0]}\": the story showed evidence for some claims, it did not verify the paper; say "
                            "which results held on which tests (\"the BLEU results held on WMT 2014\"), never that every claim was verified")
        for where, text, limit in texts:
            words = text_len(text, "en-US")
            if words > limit:
                problems.append(f"{where} has {words} words: at most {limit} on a {p.kind} panel")
            if cite is not None and cite.unresolved(text, ledger):
                problems.append(f"evidence: {where} cites {', '.join(cite.unresolved(text, ledger))}, which name no number cell "
                                "of the TABLES section: cite an existing cell, or write the paper's number")
            if evidence is not None:
                try:
                    problems += [f"evidence: {where}: {x}" for x in evidence.verify_text_numbers(text, paper_text, allowed=allowed)]
                except Exception as e:
                    print(f"[StoryboarderV2] number check failed: {type(e).__name__}: {e}")
        if problems:
            out[p.n] = problems
    example = _example_tokens(sb)
    if example:   # the story follows one example: the reader sees it at the start and sees it solved at the end
        need = min(3, max(1, math.ceil(0.34 * len(example))))
        for beat in ("hook", "payoff"):
            panels = [p for p in sb.panels if p.beat == beat]
            if panels and not any(len(example & {_stem(w) for w in _content_tokens(visible_text(p))}) >= need for p in panels):
                out.setdefault(panels[0].n, []).append(
                    f"the {beat} does not show the running example ({sb.running_example[:140]}): quote it (or its key words) "
                    f"in a bubble, the narration or a links board, so the reader sees what the story is about"
                    + (" and sees it solved" if beat == "payoff" else ""))
    teaching = _teaching_panels(sb.panels)   # the claims board and the recap list claims; they never count as teaching them
    covered = {c for p in teaching for c in p.claims}
    for claim in sb.abstract_claims if sb.panels else []:
        cid = claim.get("id")
        if cid not in covered:
            home = _home_panel(sb, claim)
            out.setdefault(home.n, []).append(
                f"claim {cid} is not taught by any panel (the claims board and the recap do not count): state it, with its key "
                f"number, in this panel's visible text or board and add {cid} to \"claims\" ({claim.get('text', '')})")
            continue
        conveying = [p for p in teaching if cid in p.claims]
        shown = set().union(*(_number_keys(visible_text(p)) for p in conveying))
        missing = [written for written, key in _key_numbers(claim) if key not in shown]   # "2.0" shows 2, "eight" shows 8
        if missing:
            home = max(conveying, key=lambda p: (p.kind == "board", p.beat == "proof", p.n))
            out.setdefault(home.n, []).append(
                f"claim {cid} ({claim.get('text', '')}) needs its number {', '.join(missing)} visible on a panel that teaches "
                f"it: add it here, exactly as the paper writes it")
        elif _key_numbers(claim) and COMPARATIVE_RE.search(str(claim.get("text", ""))) and not _comparison_shown(conveying):
            home = max(conveying, key=lambda p: (p.beat == "proof", p.kind == "board", p.n))
            out.setdefault(home.n, []).append(
                f"claim {cid} ({claim.get('text', '')}) is a comparison, but no panel that teaches it draws one: show it on a bar "
                "chart (the paper's result, the strongest earlier system and the field, from one table column) or a table")
    return out


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def scrub_unsupported(sb: StoryboardV2, paper_text: str) -> list[str]:
    """Last line of defense after the fix rounds: a bubble or narration sentence whose number the paper still does not
    support is removed (an empty bubble goes). Returns what was removed (for the job report)."""
    evidence = _optional("evidence")
    if evidence is None:
        return []
    allowed = _illustrative_example(sb)
    removed = []
    cite = _optional("ledger")
    for p in sb.panels:
        def clean(text: str) -> str:
            keep = []
            for sentence in SENTENCE_SPLIT_RE.split(text or ""):
                try:
                    bad = evidence.verify_text_numbers(sentence, paper_text, allowed=allowed)
                except Exception:
                    bad = []
                if cite is not None and cite.REF_RE.search(sentence):   # a citation no fix round resolved: never drawn as "@T2.r9.c1"
                    bad = bad or ["unresolved table citation"]
                if bad:
                    removed.append(f"panel {p.n}: {sentence}")
                else:
                    keep.append(sentence)
            return " ".join(keep).strip()
        for b in p.dialogue:
            b.text = clean(b.text)
        p.dialogue = [b for b in p.dialogue if b.text]
        p.narration = clean(p.narration)
    if removed:
        print(f"[StoryboarderV2] removed {len(removed)} sentence(s) with numbers the paper does not support: {removed[:3]}")
    return removed


def _panel_items(value) -> list[dict]:
    """The panel objects of a reply: a list, one bare panel, or a dict keyed by panel number ({"1": {...}, "2": {...}})."""
    if isinstance(value, dict):
        if any(k in value for k in ("dialogue", "kind", "visual", "board", "narration")):
            return [value]
        out = []
        for key, item in value.items():
            if isinstance(item, dict):
                digits = re.sub(r"\D", "", str(key))
                out.append(item if "n" in item or not digits else {**item, "n": int(digits)})
        return out
    return [x for x in _as_list(value) if isinstance(x, dict)]


def _bubble_item(value):
    """A speech bubble as {"speaker", "text"}: replies sometimes send "kumo: text" or ["kumo", "text"] (a line without a
    speaker goes to the panel's first character)."""
    if isinstance(value, str):
        m = re.match(r"\s*([^:\n]{1,30}):\s*(\S.*)$", value, re.S)
        return {"speaker": m.group(1), "text": m.group(2)} if m and len(m.group(1).split()) <= 3 else {"speaker": "", "text": value}
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(x, str) for x in value):
        return {"speaker": value[0], "text": value[1]}
    return value


def _panel_from_reply(item: dict, default_n: int) -> PanelV2:
    """A v3 panel from a model reply: a board makes it a board panel unless the reply explicitly says "scene". A board sent
    as a JSON string and bubbles sent as strings or pairs are read too."""
    item = dict(item)
    if isinstance(item.get("board"), str) and item["board"].strip().startswith("{"):
        try:
            item["board"] = json.loads(item["board"])
        except json.JSONDecodeError:
            pass
    dialogue = item.get("dialogue")
    if isinstance(dialogue, dict) and "text" not in dialogue:   # {"kumo": "line", "papi": "line"}
        dialogue = [{"speaker": k, "text": v} for k, v in dialogue.items()]
    if dialogue is not None:
        item["dialogue"] = [_bubble_item(b) for b in _as_list(dialogue)]
    for key in ("narration", "title", "scene", "visual"):   # a list of sentences is one text, never "['...']"
        if isinstance(item.get(key), list):
            item[key] = " ".join(str(x).strip() for x in item[key] if isinstance(x, (str, int, float)) and str(x).strip())
    p = PanelV2.from_dict(item, default_n)
    declared = str(item.get("kind") or item.get("type") or "").strip().lower()
    if isinstance(item.get("board"), dict) and item["board"] and declared != "scene":
        p.kind = "board"
    return p


def _valid_boards(panels: list[PanelV2], paper_text: str, among: Optional[set[int]] = None) -> frozenset:
    """Numbers of the board panels (of ``among``) whose board passes every check."""
    return frozenset(p.n for p in panels if (among is None or p.n in among) and p.kind == "board" and isinstance(p.board, dict)
                     and p.board and not board_issues(p.board, paper_text))


def _normalize_beat(beat: str, previous: str) -> str:
    b = re.sub(r"[\s-]+", "_", (beat or "").strip().lower())
    if b in BEATS:
        return b
    if b in BEAT_ALIASES:
        return BEAT_ALIASES[b]
    return next((x for x in BEATS if x in b), previous)


def _normalize_v3_panels(panels: list[PanelV2], cast: Cast, claims: list[dict], ledger=None) -> None:
    """The v2 clean-up without its two-bubble cap (lint judges the budget; nothing is cut), plus beats, the chapters they
    map to, valid claim ids only, no drawn labels, ledger citations resolved (services.ledger: "@T2.r9.c1" becomes the
    cell's exact number), normalized boards, and at most the speakers on a crowded board panel."""
    _normalize_panels(panels, cast, max_bubbles=None)
    valid = {c.get("id") for c in claims}
    previous = BEATS[0]
    binder = _optional("ledger") if ledger else None
    for p in panels:
        p.beat = previous = _normalize_beat(p.beat, previous)
        p.chapter = BEAT_CHAPTER[p.beat]
        p.claims = [c for c in p.claims if c in valid]
        p.labels = []   # v3 art is wordless; every word is typeset or drawn on a board by code
        if binder is not None:
            for b in p.dialogue:
                b.text = binder.bind_text(b.text, ledger)
            p.narration = binder.bind_text(p.narration, ledger)
            if isinstance(p.board, dict):
                p.board = binder.bind_board(p.board, ledger)
        if p.board is not None:
            p.board = _normalized_board(p.board)
        if p.kind == "board" and len(p.characters) > 2:
            speakers = list(dict.fromkeys(b.speaker for b in p.dialogue))
            if len(speakers) <= 2:
                p.characters = (speakers + [c for c in p.characters if c not in speakers])[:2]
                p.emotions = {k: v for k, v in p.emotions.items() if k in p.characters}


# a recap title or subtitle that certifies every line ("Verified Results") over lines that only got a bullet
CERTIFYING_TITLE_RE = re.compile(r"\b(?:verified|validated|confirmed|proven|checked)\b", re.I)


def _honest_recap(sb: StoryboardV2, retitle: bool = True) -> None:
    """The recap ticks what the story showed: a line with numbers gets its tick only when the teaching panels show all of
    them (in a bubble, a caption, a board or a chart's badge); the others get a bullet ("checked" on the list board, which
    draws a legend for the two marks). Readers caught recaps that ticked every claim while some never appeared in the
    story, and a "Verified Results" title over the bullets; a rule that counted only numbers drawn on boards confused
    three readers, who had seen the bulleted numbers in a bubble. ``retitle`` (the English script only): replace a title
    or subtitle that certifies every line. Works on a script in any language (a job's relayout re-applies it)."""
    shown: set = set()
    boards = _optional("boards")
    for p in _teaching_panels(sb.panels):
        shown |= _number_keys(" ".join([p.narration or "", *(b.text for b in p.dialogue)]))
        if p.kind == "board" and isinstance(p.board, dict):
            shown |= _number_keys(board_text(p.board))
            badge = boards.delta_badge(p.board) if boards and p.board.get("kind") == "bar" else None
            if badge:   # the gap the chart draws ("+4.6") is shown too
                shown |= _number_keys(badge["text"])
    for p in sb.panels:
        b = p.board if p.beat == "recap" and isinstance(p.board, dict) else None
        if not b or b.get("kind") != "list" or b.get("open"):
            continue
        items = b.get("items") or []
        ticked = [i for i, item in enumerate(items) if _number_keys(str(item)) <= shown]   # no numbers: a point the story explained
        if len(ticked) < len(items):
            p.board = {**b, "checked": ticked}
            if retitle and CERTIFYING_TITLE_RE.search(str(b.get("title") or "")):
                p.board["title"] = "What the paper showed"
            if retitle and CERTIFYING_TITLE_RE.search(str(b.get("subtitle") or "")):   # "Verified results from this work"
                p.board.pop("subtitle")
        else:
            p.board = {k: v for k, v in b.items() if k != "checked"}


def _finish_v3(sb: StoryboardV2) -> None:
    """Panels 1..N and chapters from the beats (keeping chapter titles that were already translated)."""
    for i, p in enumerate(sb.panels, start=1):
        p.n = i
        p.chapter = BEAT_CHAPTER.get(p.beat, p.chapter)
        if isinstance(p.board, dict):
            p.board.pop("unbound", None)
    titles = {c.id: c.title for c in sb.chapters}
    sb.chapters = _chapters(sb.panels)
    for c in sb.chapters:
        c.title = titles.get(c.id, c.title)
    sb.version = 3


def _running_example(raw, brief: dict, panels: list[PanelV2], body: str) -> dict:
    """{"text", "from_paper", "source"}: from_paper only when one of the paper's verified examples (or the example itself)
    really shows up in the story."""
    if isinstance(raw, dict):
        text = str(raw.get("text") or raw.get("example") or "").strip()
        claimed = str(raw.get("from_paper", True)).strip().lower() not in ("false", "0", "no", "none", "")
        source = str(raw.get("source") or "").strip()
    else:
        text, claimed, source = str(raw or "").strip(), True, ""
    story = " ".join([text] + [visible_text(p) for p in panels])
    words = _content_tokens(story)
    used = [e for e in _as_list(brief.get("examples")) if isinstance(e, dict) and e.get("text")
            and len(_content_tokens(e["text"]) & words) >= 0.6 * max(1, len(_content_tokens(e["text"])))]
    from_paper = claimed and (bool(used) or quote_in(text, body))
    if from_paper and used and not source:
        source = str(used[0].get("source") or "")
    return {"text": text, "from_paper": bool(from_paper), "source": source if from_paper else ""}


def _board_translation(board: dict, translated: dict, language: str, keep: frozenset) -> dict[str, str]:
    """The translated board strings to apply: in the target script (or code-like, e.g. a model name) and with exactly
    the numbers of the English string."""
    out = {}
    for path, src in _board_strings(board):
        new = translated.get(str(path))
        if isinstance(new, str) and new.strip() and _has_script(new.strip(), language, keep) and _numbers(new) == _numbers(src):
            out[str(path)] = new.strip()
    return out


class StoryboarderV2:
    def __init__(self, text_engine: BaseEngine, fast_engine: Optional[BaseEngine] = None, check_engine: Optional[BaseEngine] = None,
                 translate_engine: Optional[BaseEngine] = None, cache_key: str = ""):
        """``text_engine`` writes (brief, story, fix rounds, quiz revisions) and grades the quiz; ``check_engine`` checks the
        work (brief check, fact-checks, the back-port of a translation's findings) and ``translate_engine`` translates;
        both default to the writer, and a different model family checks without the writer's blind spots (a blind test:
        DeepSeek V4 Pro found 56% of the errors two Seed passes had left, another Seed pass 10%). ``fast_engine`` fits
        lines and reads for the quiz."""
        self.text_engine = text_engine
        self.fast_engine = fast_engine or text_engine
        self.check_engine = check_engine or text_engine
        self.translate_engine = translate_engine or text_engine
        self.cache_key = cache_key   # one per job: its paper-first calls share the provider's prefix cache
        self.skipped: list[dict] = []   # checks that could not run (the job's quality record says so to the reader)
        self.usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    async def _warm(self, engine: BaseEngine, paper_text: str) -> None:
        """A 1-token call with the paper first, before parallel calls: they then read the paper from the provider's prefix
        cache (cached input costs a fifth on Seed, a thirtieth on DeepSeek) instead of each paying for it in full."""
        config = GenerationConfig(temperature=0.0, max_tokens=1)
        config.thinking = False
        config.prompt_cache_key = self.cache_key or None
        try:
            response = await engine.generate_text(_messages("", "Reply OK.\n\nPAPER TEXT:\n" + paper_text, paper_text), config)
            self._add_usage(response.usage or {})
        except Exception as e:  # noqa: BLE001  (a failed warm-up only costs the cache)
            _reraise_account(e)

    def _skip(self, stage: str, e: BaseException) -> None:
        """A best-effort check that could not run: kept for the job's quality record, so an unchecked script is never
        shown as a checked one."""
        self.skipped.append({"stage": stage, "error": f"{type(e).__name__}: {str(e)[:200]}"})

    def _add_usage(self, usage: dict) -> None:
        self.usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        self.usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)

    async def analyze(self, paper: PaperText) -> dict:
        brief, usage = await _ask_json(self.text_engine, SYSTEM_WRITER, brief_prompt(paper), thinking=True, temperature=0.2, max_tokens=16000, key="key_idea")
        self._add_usage(usage)
        if not brief.get("title") and paper.title:
            brief["title"] = paper.title
        return brief

    async def check_brief(self, brief: dict, paper: PaperText) -> int:
        """Correct (or drop) brief statements the paper does not support, before anything is written from them: errors in
        the brief otherwise reach the panels, the details and the card. A checker of its own reads the items in parallel
        parts (as the fact-check does). Returns the number of fixes (best effort)."""
        items = _brief_items(brief)
        size = FACTCHECK_CHUNK // 2 if self.check_engine is not self.text_engine else max(1, len(items))
        parts = [items[i:i + size] for i in range(0, len(items), size)] or [[]]
        if len(parts) > 1:
            await self._warm(self.check_engine, paper.body)
        results = await asyncio.gather(*(
            _ask_check(self.check_engine, brief_check_prompt(brief, paper, part), thinking=True, temperature=0.1,
                       max_tokens=12000, key="problems", paper=paper.body, cache_key=self.cache_key)
            for part in parts), return_exceptions=True)
        failed = [r for r in results if isinstance(r, BaseException)]
        for e in failed:
            _reraise_account(e)
        if len(failed) == len(results):   # the brief is still usable unchecked
            print(f"[StoryboarderV2] brief check skipped: {failed[0]}")
            self._skip("brief check", failed[0])
            return 0
        if failed:
            self._skip(f"brief check, {len(failed)} of {len(results)} parts", failed[0])
        data = {"problems": []}
        for result in results:
            if not isinstance(result, BaseException):
                part_data, usage = result
                self._add_usage(usage)
                data["problems"] += _as_list(part_data.get("problems"))
        fixes, drop = [], {}
        points = {pt["id"]: pt for pt in _as_list(brief.get("points")) if isinstance(pt, dict)
                  for pt in [{**pt, "id": str(pt.get("id") or "").strip()}]}
        for item in _as_list(data.get("problems")):
            if not isinstance(item, dict):
                continue
            pid, corrected = str(item.get("id") or "").strip(), _text(item.get("corrected"))
            if not pid or not corrected:
                continue
            delete = corrected.upper().strip(".") == "DELETE"
            parts = pid.split(".")
            if parts[0] in ("problem", "key_idea", "setup", "takeaway") and len(parts) == 1 and isinstance(brief.get(pid), str):
                fixes.append({"id": pid, "before": brief[pid], "after": "" if delete else corrected})
                brief[pid] = "" if delete else corrected
            elif parts[0] in ("contributions", "limitations") and len(parts) == 2 and parts[1].isdigit():
                brief[parts[0]] = items = _as_list(brief.get(parts[0]))   # the brief's own list: a correction must stick
                i = int(parts[1])
                if i < len(items):
                    fixes.append({"id": pid, "before": items[i], "after": "" if delete else corrected})
                    if delete:
                        drop.setdefault(parts[0], set()).add(i)
                    else:
                        items[i] = corrected
            elif parts[0] in ("mechanisms", "evidence") and len(parts) == 3 and parts[1].isdigit():
                items, i, f = _as_list(brief.get(parts[0])), int(parts[1]), parts[2]
                if i < len(items) and isinstance(items[i], dict) and f in items[i]:
                    fixes.append({"id": pid, "before": items[i][f], "after": "" if delete else corrected})
                    if delete and f in ("claim", "plain"):
                        drop.setdefault(parts[0], set()).add(i)
                    else:
                        items[i][f] = "" if delete else corrected
            elif parts[0] == "abstract_claims" and len(parts) == 2 and parts[1].isdigit():
                claims, i = _as_list(brief.get("abstract_claims")), int(parts[1])
                if i < len(claims) and isinstance(claims[i], dict):
                    # the abstract says it, so the claim stays: a "DELETE" falls back to the abstract's own words
                    after = str(claims[i].get("quote") or claims[i].get("text") or "") if delete else corrected
                    fixes.append({"id": pid, "before": claims[i].get("text"), "after": after})
                    claims[i]["text"] = after
            elif parts[0] == "points" and len(parts) >= 2:
                pt_id = ".".join(parts[1:])
                for raw in _as_list(brief.get("points")):
                    if isinstance(raw, dict) and str(raw.get("id") or "").strip() == pt_id:
                        fixes.append({"id": pid, "before": raw.get("text"), "after": "" if delete else corrected})
                        if delete:
                            drop.setdefault("points", set()).add(pt_id)
                        else:
                            raw["text"] = corrected
        for key, gone in drop.items():
            if key == "points":
                brief["points"] = [pt for pt in _as_list(brief.get("points")) if not (isinstance(pt, dict) and str(pt.get("id") or "").strip() in gone)]
            else:
                brief[key] = [x for i, x in enumerate(_as_list(brief.get(key))) if i not in gone]
        if fixes:
            print(f"[StoryboarderV2] brief check corrected {len(fixes)} statements: {[f['id'] for f in fixes]}")
        brief["brief_check"] = fixes
        return len(fixes)

    async def write(self, brief: dict, cast: Cast, panel_range: tuple[int, int], language: str) -> StoryboardV2:
        data, usage = await _ask_json(self.text_engine, SYSTEM_WRITER, storyboard_prompt(brief, cast, language, panel_range), thinking=True, temperature=0.6, max_tokens=24000, key="panels")
        self._add_usage(usage)
        panels = []
        for i, p in enumerate(data.get("panels") or [], start=1):
            try:
                panels.append(PanelV2.from_dict(p, i))
            except Exception as e:  # keep going on a malformed panel
                print(f"[StoryboarderV2] skipping malformed panel {i}: {e}")
        if not panels:
            raise ValueError("storyboard model returned no panels")
        numbers = [p.n for p in panels]
        if len(set(numbers)) == len(numbers):
            panels.sort(key=lambda p: p.n)          # trust the model's numbering when it is unique; else keep reply order
        _normalize_panels(panels, cast)
        return StoryboardV2(
            title=str(data.get("title") or brief.get("title") or "Paper manga"),
            summary=str(data.get("summary") or brief.get("key_idea") or ""),
            theme=cast.theme,
            language="en-US",
            chapters=_chapters(panels),
            panels=panels,
            brief=brief,
            card=paper_card(brief),
        )

    async def cover(self, sb: StoryboardV2, cast: Cast, panel_range: tuple[int, int]) -> int:
        """Give every core point of the brief a panel: missing ones get new panels (up to the panel budget, placed inside
        their own chapter) or a sentence in the detail of the closest panel. Returns the number of points it placed."""
        missing = uncovered(sb)
        if not missing:
            return 0
        try:
            data, usage = await _ask_json(self.text_engine, SYSTEM_WRITER, cover_prompt(sb, cast, missing, panel_range[1]),
                                          thinking=True, temperature=0.4, max_tokens=12000, key="new_panels")
        except Exception as e:  # coverage repair is best effort
            _reraise_account(e)
            print(f"[StoryboarderV2] coverage repair skipped: {e}")
            return 0
        self._add_usage(usage)
        missing_ids = {str(pt["id"]) for pt in missing}
        by_n = {p.n: p for p in sb.panels}
        new_items = [x for x in _as_list(data.get("new_panels")) if isinstance(x, dict)]
        details = [x for x in _as_list(data.get("details")) if isinstance(x, dict)] + [x for x in new_items if x.get("add_detail")]

        def to_detail(n, sentence: str, covers) -> None:   # an existing panel explains the point (or gets a sentence for it)
            try:
                panel = by_n.get(int(n))
            except (TypeError, ValueError):
                return
            if panel is None:
                return
            if sentence and sentence not in panel.detail:
                panel.detail = f"{panel.detail} {sentence}".strip()
            panel.covers += [c for c in PanelV2.from_dict({"covers": covers}, 0).covers if c not in panel.covers]

        for item in details:
            to_detail(item.get("n"), str(item.get("add_detail") or "").strip(), item.get("covers"))
        room = min(max(0, panel_range[1] - len(sb.panels)), len(missing_ids))
        titles = {p.title.strip().lower() for p in sb.panels if p.title.strip()}
        lines = {tuple(b.text for b in p.dialogue) for p in sb.panels if p.dialogue}
        inserts = []
        for i, item in enumerate(x for x in new_items if not x.get("add_detail")):
            try:
                new = PanelV2.from_dict(item, 0)
            except Exception as e:  # keep going on a malformed panel
                print(f"[StoryboarderV2] skipping malformed coverage panel: {e}")
                continue
            try:
                after = int(item.get("after"))
            except (TypeError, ValueError):
                after = len(sb.panels)
            repeated = new.title.strip().lower() in titles or tuple(b.text for b in new.dialogue) in lines
            useful = set(new.covers) & missing_ids and new.visual and (new.dialogue or new.narration) and not repeated
            if not useful:
                continue
            if len(inserts) >= room:   # no room left: its facts go into the detail of the panel it would follow
                to_detail(min(max(after, 1), len(sb.panels)), new.detail or new.narration, new.covers)
                continue
            inserts.append((_insert_position(sb.panels, new, after), i, new))
        if inserts:
            ordered = list(sb.panels)
            for pos, _i, new in sorted(inserts, key=lambda t: (t[0], t[1]), reverse=True):
                ordered.insert(pos, new)
            _normalize_panels(ordered, cast)
            sb.panels, sb.chapters = ordered, _chapters(ordered)
        placed = len(missing) - len(uncovered(sb))
        print(f"[StoryboarderV2] coverage repair placed {placed} of {len(missing)} missing core points ({len(inserts)} new panels)")
        return placed

    def violations(self, sb: StoryboardV2, language: str) -> list[dict]:
        if sb.version >= 3:
            return self._violations_v3(sb, language)
        budget = BUDGETS.get(language, BUDGETS["en-US"])
        items = []
        for p in sb.panels:
            if _mentions_people(f"{p.scene} {p.visual}"):   # Seedream draws whoever the scene mentions
                items.append({"id": f"{p.n}.v", "kind": "visual", "text": f"Scene: {p.scene}. Visual: {p.visual}"})
            for i, b in enumerate(p.dialogue):
                if text_len(b.text, language) > budget["bubble"] or "_" in b.text or "\\" in b.text:
                    items.append({"id": f"{p.n}.d{i}", "kind": "bubble", "text": b.text})
            if p.narration and (text_len(p.narration, language) > budget["narration"] or "_" in p.narration or "\\" in p.narration):
                items.append({"id": f"{p.n}.n", "kind": "narration", "text": p.narration})
            for i, lab in enumerate(p.labels):
                if text_len(lab, language) > budget["label"] or "_" in lab or "\\" in lab:
                    items.append({"id": f"{p.n}.l{i}", "kind": "label", "text": lab})
        return items

    def _violations_v3(self, sb: StoryboardV2, language: str) -> list[dict]:
        """Over-budget lines of a v3 script (limits by panel kind, given per item as "max") and scenes that mention people."""
        budget, items = _v3_budget(language), []
        for p in sb.panels:
            b = budget["board" if p.kind == "board" else "scene"]
            if _mentions_people(f"{p.scene} {p.visual}"):   # Seedream draws whoever the scene mentions
                items.append({"id": f"{p.n}.v", "kind": "visual", "text": f"Scene: {p.scene}. Visual: {p.visual}"})
            for i, x in enumerate(p.dialogue):
                if text_len(x.text, language) > b["bubble"] or "_" in x.text or "\\" in x.text:
                    items.append({"id": f"{p.n}.d{i}", "kind": "bubble", "text": x.text, "max": b["bubble"]})
            if p.narration and (text_len(p.narration, language) > b["narration"] or "_" in p.narration or "\\" in p.narration):
                items.append({"id": f"{p.n}.n", "kind": "narration", "text": p.narration, "max": b["narration"]})
        return items

    async def fit(self, sb: StoryboardV2, language: str, rounds: int = 2) -> int:
        """Rewrite over-budget lines with the fast model. Returns the number of lines rewritten. In a v3 script a rewrite
        that changes the line's numbers is refused (the longer exact line beats a short wrong one)."""
        rewritten = 0
        for _ in range(rounds):
            items = self.violations(sb, language)
            if not items:
                break
            prompt = fit_prompt_v3(items, language) if sb.version >= 3 else fit_prompt(items, language)
            data, usage = await _ask_json(self.fast_engine, "", prompt, thinking=False, temperature=0.2, max_tokens=8000, key="items")
            self._add_usage(usage)
            by_id = {str(x.get("id")): _text(x.get("text")) for x in _as_list(data.get("items")) if isinstance(x, dict) and _text(x.get("text"))}
            if sb.version >= 3:
                originals = {f"{p.n}.d{i}": b.text for p in sb.panels for i, b in enumerate(p.dialogue)} | {f"{p.n}.n": p.narration for p in sb.panels}
                by_id = {k: v for k, v in by_id.items() if k not in originals or _numbers(v) == _numbers(originals[k])}
            for p in sb.panels:
                for i, b in enumerate(p.dialogue):
                    new = by_id.get(f"{p.n}.d{i}")
                    if new:
                        b.text, rewritten = new, rewritten + 1
                new = by_id.get(f"{p.n}.n")
                if new:
                    p.narration, rewritten = new, rewritten + 1
                new = by_id.get(f"{p.n}.v")
                if new:
                    p.scene, p.visual, rewritten = "", new, rewritten + 1
                for i in range(len(p.labels)):
                    new = by_id.get(f"{p.n}.l{i}")
                    if new:
                        p.labels[i], rewritten = new, rewritten + 1
        # last resort: strip forbidden characters (never truncate)
        for p in sb.panels:
            for b in p.dialogue:
                b.text = _plain(b.text)
            p.narration = _plain(p.narration)
            p.labels = [_plain(lab) for lab in p.labels]
        return rewritten

    async def factcheck(self, sb: StoryboardV2, paper: PaperText, language: str = "en-US", board_flags: Optional[dict] = None) -> int:
        """Verify every bubble/narration (and board string) against the paper and apply corrections: once on the English
        script, and again after translation, which can drop a qualifier the first pass added. Returns #fixes.
        Whole-board findings ("N.b") cannot be fixed by replacing a string: they are collected in ``board_flags``
        ({n: [problem]}) for a fix round when the caller passes it."""
        # a checker of its own (DeepSeek thinks for minutes over a whole script) gets the script in parallel parts that
        # read the paper from the cache a warm-up filled; the writer's own model checks in one call, as before
        parts = factcheck_parts(sb, paper, language) if self.check_engine is not self.text_engine else [(None, True)]
        if len(parts) > 1:
            await self._warm(self.check_engine, paper.body)
        results = await asyncio.gather(*(
            _ask_check(self.check_engine, factcheck_prompt(sb, paper, language, panels, card), thinking=True, temperature=0.1,
                       max_tokens=12000, key="problems", paper=paper.body, cache_key=self.cache_key)
            for panels, card in parts), return_exceptions=True)
        failed = [r for r in results if isinstance(r, BaseException)]
        for e in failed:
            _reraise_account(e)
        stage = "fact-check" if language == "en-US" else f"fact-check ({language})"
        if len(failed) == len(results):   # fact-check is best effort
            print(f"[StoryboarderV2] fact-check skipped: {failed[0]}")
            self._skip(stage, failed[0])
            return 0
        if failed:
            print(f"[StoryboarderV2] {len(failed)} of {len(results)} fact-check parts failed: {failed[0]}")
            self._skip(f"{stage}, {len(failed)} of {len(results)} parts", failed[0])
        data = {"problems": []}
        for result in results:
            if not isinstance(result, BaseException):
                part, usage = result
                self._add_usage(usage)
                data["problems"] += _as_list(part.get("problems"))
        fixes = []
        board_updates: dict[int, dict[str, tuple[str, str]]] = {}
        for item in _as_list(data.get("problems")):   # a stray shape ("problems": 0) is no reason to fail the job
            if not isinstance(item, dict):
                continue
            pid, corrected = str(item.get("id", "")), _text(item.get("corrected"))
            if not corrected or "." not in pid or not _has_script(corrected, language, _latin_keep(sb.brief)):
                continue
            if pid.startswith("card."):
                _, key, index = (pid.split(".") + ["", ""])[:3]
                value = sb.card.get(key)
                if isinstance(value, str) and index in ("", "0"):
                    fixes.append({"id": pid, "before": value, "after": corrected, "issue": item.get("issue", "")})
                    sb.card[key] = corrected
                elif isinstance(value, list) and index.isdigit() and int(index) < len(value):
                    fixes.append({"id": pid, "before": value[int(index)], "after": corrected, "issue": item.get("issue", "")})
                    value[int(index)] = corrected
                continue
            n_str, part = pid.split(".", 1)
            try:
                n = int(n_str)
            except ValueError:
                continue
            panel = next((p for p in sb.panels if p.n == n), None)
            if panel is None:
                continue
            if part == "n":
                fixes.append({"id": pid, "before": panel.narration, "after": corrected, "issue": item.get("issue", "")})
                panel.narration = corrected
            elif part == "x":
                fixes.append({"id": pid, "before": panel.detail, "after": corrected, "issue": item.get("issue", "")})
                panel.detail = corrected
            elif part == "v":
                fixes.append({"id": pid, "before": panel.visual, "after": corrected, "issue": item.get("issue", "")})
                panel.visual = corrected
            elif part == "b" and panel.board:
                if board_flags is not None:
                    board_flags.setdefault(n, []).append(f"fact-check: {str(item.get('issue') or '').strip()} Fix: {corrected}".strip())
                fixes.append({"id": pid, "before": "", "after": corrected, "issue": item.get("issue", "")})
            elif part.startswith("b.") and panel.board:
                board_updates.setdefault(n, {})[part[2:]] = (corrected, str(item.get("issue") or ""))
            elif part.startswith("d"):
                try:
                    i = int(part[1:])
                except ValueError:
                    continue
                if i < len(panel.dialogue):
                    fixes.append({"id": pid, "before": panel.dialogue[i].text, "after": corrected, "issue": item.get("issue", "")})
                    panel.dialogue[i].text = corrected
        for n, updates in board_updates.items():
            panel = next(p for p in sb.panels if p.n == n)
            before = {str(path): text for path, text in _board_strings(panel.board)}
            new = _set_board_strings(panel.board, {k: v for k, (v, _) in updates.items()})
            if new is panel.board:
                continue
            if len(board_issues(new, paper.body)) > len(board_issues(panel.board, paper.body)):   # a correction must not break the board
                print(f"[StoryboarderV2] fact-check correction of board {n} refused: it breaks the board's checks")
                continue
            panel.board = new
            after = {str(path): text for path, text in _board_strings(new)}
            fixes += [{"id": f"{n}.b.{k}", "before": before[k], "after": after[k], "issue": issue}
                      for k, (_, issue) in updates.items() if k in before and after.get(k) != before[k]]
        if fixes:
            print(f"[StoryboarderV2] fact-check ({language}) corrected {len(fixes)} lines: {[f['id'] for f in fixes]}")
        sb.brief["factcheck" if language == "en-US" else f"factcheck_{language}"] = fixes
        return len(fixes)

    async def translate(self, sb: StoryboardV2, language: str, _retry: bool = False) -> StoryboardV2:
        if language == "en-US" or language not in BUDGETS:
            sb.language = language if language in BUDGETS else "en-US"
            return sb
        english = {f"{p.n}.d{i}": b.text for p in sb.panels for i, b in enumerate(p.dialogue)}
        english.update({f"{p.n}.n": p.narration for p in sb.panels if p.narration})
        data, usage = await _ask_json(self.translate_engine, "", translate_prompt(sb, language), thinking=False, temperature=0.3, max_tokens=24000, key="panels")
        self._add_usage(usage)
        if _text(data.get("title")) and not _retry:
            sb.title = _text(data["title"])
        if _text(data.get("summary")) and not _retry:
            sb.summary = _text(data["summary"])
        if isinstance(data.get("card"), dict) and not _retry:
            sb.card = _translated_card(sb.card, data["card"], language, _latin_keep(sb.brief))
        ch_titles = {str(c.get("id")): str(c.get("title") or "").strip() for c in _as_list(data.get("chapters")) if isinstance(c, dict)}
        for c in sb.chapters:
            if ch_titles.get(c.id):
                c.title = ch_titles[c.id]
        by_n: dict[int, dict] = {}
        for t in _panel_items(data.get("panels")):
            try:
                by_n[int(t.get("n"))] = t
            except (TypeError, ValueError):
                continue
        keep = _latin_keep(sb.brief)
        applied = 0
        rejected: list[int] = []
        for p in sb.panels:
            t = by_n.get(p.n)
            if not t:
                rejected.append(p.n)
                continue
            td = [b for b in _as_list(t.get("dialogue")) if isinstance(b, dict)]
            speakers_ok = len(td) == len(p.dialogue) and all(
                _match_cast(str(b.get("speaker") or ""), {orig.speaker}) == orig.speaker or not b.get("speaker") for b, orig in zip(td, p.dialogue))
            texts = [_text(b.get("text")) for b in td]
            # the narration is checked like a bubble; an absent key keeps the English, and none is added where there was none
            narration = _text(t["narration"]) if p.narration and t.get("narration") is not None else None
            lines = texts + ([narration] if narration is not None else [])
            if not speakers_ok or any(not x or not _has_script(x, language, keep) for x in lines):
                rejected.append(p.n)
                continue
            originals = [b.text for b in p.dialogue] + ([p.narration] if narration is not None else [])
            dropped = [o for o, x in zip(originals, lines) if not _figures_kept(o, x)]
            if sb.version >= 3 and dropped:
                if not _retry:   # a number lost in translation is lost evidence; the re-request usually keeps it
                    rejected.append(p.n)
                    continue
                print(f"[StoryboarderV2] WARNING: panel {p.n}: the translation still drops a number of {dropped[0][:60]!r}")
            if _text(t.get("title")):
                p.title = _text(t["title"])
            if narration:
                p.narration = narration
            for b, text in zip(p.dialogue, texts):
                b.text = text
                applied += 1
            tl = _labels(t.get("labels"))
            if tl and len(tl) == len(p.labels):
                p.labels = tl[:2]
            detail = _text(t.get("detail"))
            if p.detail and detail and _has_script(detail, language, keep):   # reader text: an untranslated detail keeps the English
                p.detail = detail
            if p.board and isinstance(t.get("board"), dict):
                p.board = _set_board_strings(p.board, _board_translation(p.board, t["board"], language, keep))
        if rejected and not _retry:
            print(f"[StoryboarderV2] translation rejected for panels {rejected}; re-requesting once")
            sub = StoryboardV2(title=sb.title, summary=sb.summary, theme=sb.theme, language=sb.language, chapters=sb.chapters,
                               panels=[p for p in sb.panels if p.n in rejected], brief=sb.brief, card={}, version=sb.version)
            try:
                await self.translate(sub, language, _retry=True)
            except Exception as e:  # the re-request is best effort: the first pass is kept
                _reraise_account(e)
                print(f"[StoryboarderV2] WARNING: re-request failed ({type(e).__name__}: {e}); panels {rejected} kept their English text")
        elif rejected:
            print(f"[StoryboarderV2] WARNING: panels {rejected} kept their English text (translation failed twice)")
        if not _retry:
            await self._translate_leftovers(sb, language)
            await self._fix_translation(sb, language, english)
        print(f"[StoryboarderV2] translated {applied} bubbles to {language}")
        sb.language = language
        return sb

    async def _fix_translation(self, sb: StoryboardV2, language: str, english: dict[str, str]) -> int:
        """Bubbles and narration whose translation is stronger than the English, drops its hedge, transliterates a
        name or renames a prop (translation_problems) are corrected in one call; a correction must keep the line's
        numbers. Returns the number of lines corrected (recorded in brief["translation_fixes_<language>"])."""
        names = frozenset(n for p in sb.panels for n in p.characters)
        items = []
        for pid, source in english.items():
            line = _line_at(sb, pid)
            found = translation_problems(source, line or "", language, names) if line and line != source else []
            if found:
                items.append({"id": pid, "english": source, "translation": line, "problems": found})
        if not items:
            return 0
        try:
            data, usage = await _ask_json(self.translate_engine, "", translation_fix_prompt(items, language), thinking=False,
                                          temperature=0.2, max_tokens=8000, key="items")
        except Exception as e:  # the translation stays as it is; the translation's fact-check still reads it
            _reraise_account(e)
            print(f"[StoryboarderV2] translation fixes skipped: {type(e).__name__}: {e}")
            self._skip("translation fixes", e)
            return 0
        self._add_usage(usage)
        keep, fixed = _latin_keep(sb.brief), []
        for item in _as_list(data.get("items")):
            if not isinstance(item, dict):
                continue
            pid, text = str(item.get("id") or ""), _text(item.get("text"))
            before = _line_at(sb, pid)
            if not text or pid not in english or not _has_script(text, language, keep) or not _figures_kept(english[pid], text):
                continue
            if _number_keys(text) - _number_keys(english[pid]):   # a correction of the wording never brings a number of its own
                continue
            if _set_line_at(sb, pid, text):
                fixed.append({"id": pid, "before": before, "after": text})
        if fixed:
            print(f"[StoryboarderV2] translation ({language}): corrected {len(fixed)} lines that misread their English: {[f['id'] for f in fixed]}")
        sb.brief[f"translation_fixes_{language}"] = fixed
        return len(fixed)

    async def backport(self, english: StoryboardV2, localized: StoryboardV2, paper: PaperText) -> int:
        """The translation's fact-check findings correct the English line they came from when it has the same problem:
        in the v3 runs, 30 of the 38 problems the Chinese fact-check found were already in the English script, which two
        English fact-checks had passed. One call to the checker; a correction may not bring a number the paper lacks.
        Returns the number of English lines changed (recorded in english.brief["backport"])."""
        fixes = localized.brief.get(f"factcheck_{localized.language}") or []
        items = []
        for f in fixes:
            pid = str(f.get("id") or "")
            line = _line_at(english, pid)
            if line:
                items.append({"id": pid, "english": line, "issue": str(f.get("issue") or ""), "corrected_translation": str(f.get("after") or "")})
        if not items:
            return 0
        try:
            data, usage = await _ask_json(self.check_engine, "", backport_prompt(items) + paper.body, thinking=False, temperature=0.1,
                                          max_tokens=6000, key="items", paper=paper.body, cache_key=self.cache_key)
        except Exception as e:  # the English stays as it is
            _reraise_account(e)
            print(f"[StoryboarderV2] back-port of the {localized.language} fact-check skipped: {type(e).__name__}: {e}")
            self._skip("back-port of the translation's fact-check", e)
            return 0
        self._add_usage(usage)
        changed = []
        for item in _as_list(data.get("items")):
            if not isinstance(item, dict):
                continue
            pid, text = str(item.get("id") or ""), _text(item.get("text"))
            before = _line_at(english, pid)
            if not text or text.upper().strip(" .") == "KEEP" or before is None or text == before:
                continue
            if CJK_RE.search(text) or not _numbers_supported(text, paper.body):
                continue
            if _set_line_at(english, pid, text):
                changed.append({"id": pid, "before": before, "after": text})
        if changed:
            print(f"[StoryboarderV2] the {localized.language} fact-check corrected {len(changed)} English lines too: {[c['id'] for c in changed]}")
        english.brief["backport"] = english.brief.get("backport", []) + changed
        return len(changed)

    async def _translate_leftovers(self, sb: StoryboardV2, language: str) -> None:
        """Details, card lines and board strings the translation left in English get one small re-request (a board string
        must keep its numbers); what still fails stays English with a warning."""
        keep = _latin_keep(sb.brief)
        items = {f"{p.n}.x": p.detail for p in sb.panels if p.detail and not _has_script(p.detail, language, keep)}
        for p in sb.panels:
            items.update({f"{p.n}.b.{path}": text for path, text in _board_strings(p.board) if not _has_script(text, language, keep)})
        for key, value in sb.card.items():
            for i, text in enumerate([value] if isinstance(value, str) else value):
                if not _has_script(text, language, keep):
                    items[f"card.{key}.{i}"] = text
        if not items:
            return
        prompt = (f"Translate each text into natural {LANG_NAMES[language]} for readers new to the topic. Keep technical terms, "
                  f"names and numbers exact; use full-width punctuation. Return JSON only: {{\"items\": [{{\"id\": \"...\", \"text\": \"...\"}}]}}\n\n"
                  + json.dumps([{"id": k, "text": v} for k, v in items.items()], ensure_ascii=False))
        try:
            data, usage = await _ask_json(self.translate_engine, "", prompt, thinking=False, temperature=0.2, max_tokens=8000, key="items")
        except Exception as e:  # best effort: the English stays readable
            _reraise_account(e)
            print(f"[StoryboarderV2] WARNING: {len(items)} reader texts kept their English ({type(e).__name__}: {e})")
            self._skip("translation of reader texts", e)
            return
        self._add_usage(usage)
        done = 0
        for t in _as_list(data.get("items")):
            if not isinstance(t, dict) or str(t.get("id")) not in items:
                continue
            text, pid = _text(t.get("text")), str(t["id"])
            if not text or not _has_script(text, language, keep):
                continue
            n_str, _, rest = pid.partition(".")
            if rest.startswith("b."):
                panel = next((p for p in sb.panels if str(p.n) == n_str and p.board), None)
                if panel is not None and _numbers(text) == _numbers(items[pid]):
                    new = _set_board_strings(panel.board, {rest[2:]: text})
                    if new is not panel.board:
                        panel.board, done = new, done + 1
                continue
            if pid.endswith(".x"):
                panel = next((p for p in sb.panels if f"{p.n}.x" == pid), None)
                if panel is not None:
                    panel.detail, done = text, done + 1
            else:
                _, key, index = pid.split(".")
                if isinstance(sb.card.get(key), str):
                    sb.card[key], done = text, done + 1
                elif isinstance(sb.card.get(key), list) and index.isdigit() and int(index) < len(sb.card[key]):
                    sb.card[key][int(index)], done = text, done + 1
        if done < len(items):
            print(f"[StoryboarderV2] WARNING: {len(items) - done} reader texts kept their English after a re-request")

    async def build(self, paper: PaperText, cast: Cast, language: str, panel_range: tuple[int, int], on_stage=None) -> StoryboardV2:
        if on_stage:
            on_stage("analyzing", "Reading the paper")
        brief = await self.analyze(paper)
        if on_stage:
            on_stage("analyzing", "Checking the brief against the paper")
        await self.check_brief(brief, paper)
        if on_stage:
            on_stage("storyboarding", "Writing the manga script")
        sb = await self.write(brief, cast, panel_range, language)
        await self.fit(sb, "en-US")
        if uncovered(sb):
            if on_stage:
                on_stage("storyboarding", "Adding panels for key points the script missed")
            if await self.cover(sb, cast, panel_range):
                await self.fit(sb, "en-US")
        if on_stage:
            on_stage("storyboarding", "Fact-checking the script against the paper")
        await self.factcheck(sb, paper)
        await self.fit(sb, "en-US")
        if language != "en-US":
            if on_stage:
                on_stage("translating", f"Translating to {LANG_NAMES.get(language, language)}")
            sb = await self.translate(sb, language)
            await self.fit(sb, language)
            if on_stage:
                on_stage("translating", "Fact-checking the translation against the paper")
            if await self.factcheck(sb, paper, language):
                await self.fit(sb, language)
        return sb

    # ------------------------------------------------------------------ v3 story layer

    async def analyze_v3(self, paper: PaperText, abstract: str) -> dict:
        """The brief plus the abstract's claims (quotes verified against the abstract, every sentence covered) and the
        paper's own example inputs (verified in the paper)."""
        for attempt in range(2):   # everything after depends on the brief: a reply no repair could parse is asked once more
            try:
                brief, usage = await _ask_json(self.text_engine, SYSTEM_WRITER, brief_prompt_v3(paper, abstract), thinking=True,
                                               temperature=0.2, max_tokens=16000, key="abstract_claims", paper=paper.body,
                                               cache_key=self.cache_key)
                break
            except ValueError as e:   # json.JSONDecodeError is a ValueError
                if attempt:
                    raise
                print(f"[StoryboarderV2] the brief reply could not be read ({e}); asking once more")
        self._add_usage(usage)
        if not brief.get("title") and paper.title:
            brief["title"] = paper.title
        brief["abstract_claims"] = verify_claims(brief.get("abstract_claims"), abstract, paper.body)
        brief["examples"] = paper_examples(brief.get("examples"), paper.body)
        brief["abstract"] = abstract
        print(f"[StoryboarderV2] abstract of {len(abstract)} chars → {len(brief['abstract_claims'])} claims; "
              f"{len(brief['examples'])} example inputs from the paper")
        return brief

    async def write_v3(self, brief: dict, claims: list[dict], cast: Cast, panel_range: tuple[int, int], paper: PaperText) -> StoryboardV2:
        panels: list[PanelV2] = []
        for attempt in range(2):   # a reply without panels, or one no repair could parse, is asked once more: the brief is paid for
            try:
                data, usage = await _ask_json(self.text_engine, SYSTEM_STORY, story_prompt(brief, claims, cast, panel_range, paper),
                                              thinking=True, temperature=0.7, max_tokens=24000, key="panels", paper=paper.body,
                                              cache_key=self.cache_key)
            except ValueError as e:   # json.JSONDecodeError is a ValueError ("Expecting ',' delimiter" in a 20-panel reply)
                if attempt:
                    raise
                print(f"[StoryboarderV2] the story reply could not be read ({e}); asking once more")
                continue
            self._add_usage(usage)
            for i, item in enumerate(_panel_items(data.get("panels")), start=1):
                try:
                    panels.append(_panel_from_reply(item, i))
                except Exception as e:  # keep going on a malformed panel
                    print(f"[StoryboarderV2] skipping malformed panel {i}: {e}")
            if panels:
                break
            print(f"[StoryboarderV2] the story reply had no panels (attempt {attempt + 1})")
        if not panels:
            raise ValueError("story model returned no panels")
        numbers = [p.n for p in panels]
        if len(set(numbers)) == len(numbers):
            panels.sort(key=lambda p: p.n)          # trust the model's numbering when it is unique; else keep reply order
        _normalize_v3_panels(panels, cast, claims, _ledger(paper.body))
        example = _running_example(data.get("running_example"), brief, panels, paper.body)
        brief["running_example"] = example
        if not panel_range[0] <= len(panels) <= panel_range[1]:
            print(f"[StoryboarderV2] the story has {len(panels)} panels (asked for {panel_range[0]}-{panel_range[1]})")
        return StoryboardV2(title=str(data.get("title") or brief.get("title") or "Paper manga"),
                            summary=str(data.get("summary") or brief.get("key_idea") or ""), theme=cast.theme, language="en-US",
                            chapters=_chapters(panels), panels=panels, brief=brief, card=paper_card(brief), version=3,
                            abstract_claims=claims, running_example=example["text"])

    def _replace_panels(self, sb: StoryboardV2, cast: Cast, items, allowed: set[int], keep_boards: frozenset = frozenset(),
                        ledger=None) -> int:
        """Put the model's replacement panels in place by "n" (only numbers in ``allowed``); a reply without usable numbers
        is matched in order when it holds exactly one panel per allowed number. A panel in ``keep_boards`` (its board was
        valid) keeps that board when the reply drops it: the problems were elsewhere, and a recap must not vanish because
        a rewrite left out an unchanged field. Returns the number of panels replaced."""
        items = _panel_items(items)
        by_n: dict[int, dict] = {}
        for item in items:
            try:
                n = int(item.get("n"))
            except (TypeError, ValueError):
                continue
            if n in allowed and n not in by_n:
                by_n[n] = item
        if not by_n and items and len(items) == len(allowed):
            by_n = dict(zip(sorted(allowed), items))
        count = 0
        for i, old in enumerate(sb.panels):
            item = by_n.get(old.n)
            if item is None:
                continue
            try:
                new = _panel_from_reply(item, old.n)
            except Exception as e:  # keep the old panel
                print(f"[StoryboarderV2] skipping malformed replacement for panel {old.n}: {e}")
                continue
            if not (new.dialogue or new.narration or new.board):
                continue
            new.n, new.beat, new.title = old.n, new.beat or old.beat, new.title or old.title
            new.scene, new.visual = new.scene or old.scene, new.visual or old.visual
            new.characters = new.characters or list(old.characters)
            if "claims" not in item and "abstract_claims" not in item:   # a reply that forgot the field keeps the claims
                new.claims = list(old.claims)
            if old.n in keep_boards and not (isinstance(new.board, dict) and new.board):
                new.kind, new.board = "board", old.board
            sb.panels[i] = new
            count += 1
        if count:
            _normalize_v3_panels(sb.panels, cast, sb.abstract_claims, ledger)
        return count

    def _demote_boards(self, sb: StoryboardV2, paper_text: str) -> None:
        """After the fix rounds, a board that is still invalid keeps what the paper supports (_salvage_board: problems
        the renderer copes with stay, unsupported entries go). Only when nothing usable is left is the panel drawn as a
        scene, with the board's title as narration unless the title holds a number the paper lacks. A scene panel loses
        a stray board."""
        for p in sb.panels:
            board = p.board if isinstance(p.board, dict) and p.board else None
            title = _plain(str(board.get("title") or "")) if board else ""
            if not _numbers_supported(title, paper_text):
                title = ""   # what made the board fail must not move into the narration
            if p.kind == "board":
                problems = board_issues(board, paper_text) if board else ["board: missing"]
                if not problems:
                    continue
                salvaged = _salvage_board(board, paper_text) if board else None
                if salvaged is not None:
                    print(f"[StoryboarderV2] panel {p.n}: board kept after the fix rounds without what the paper does not "
                          f"support ({len(problems)} problems before)")
                    p.board = salvaged
                    continue
                print(f"[StoryboarderV2] panel {p.n}: the board is still invalid; drawn as a scene ({'; '.join(problems)[:400]})")
                said = _content_tokens(" ".join(str(board.get(k) or "") for k in ("title", "subtitle", "note"))) if board else set()
                own = _content_tokens(p.narration)
                if own and said and len(own & said) >= 0.5 * len(own) and p.dialogue:
                    p.narration = ""   # a caption of the chart ("Attention weights for the verb 'making'") without the chart
                p.kind, p.board = "scene", None   # its title would caption a chart nobody sees ("Lower is better.")
                if not p.dialogue and not p.narration and title:
                    p.narration = title if title[-1] in ".!?" else f"{title}."
            elif p.board:
                p.board = None

    async def fix_v3(self, sb: StoryboardV2, cast: Cast, paper: PaperText, rounds: int = 2, extra: Optional[dict] = None) -> int:
        """Lint, then one model call that rewrites the flagged panels (same n), at most ``rounds`` times; a board that is
        still invalid afterwards becomes a scene panel. ``extra`` ({n: [problem]}, e.g. whole-board findings of the
        fact-check) joins the first round. Returns the number of panels replaced."""
        replaced = 0
        for r in range(rounds):
            problems = lint_v3(sb, paper.body)
            if r == 0:
                for n, found in (extra or {}).items():
                    problems.setdefault(n, []).extend(found)
            numbers = {p.n for p in sb.panels}
            problems = {n: v for n, v in problems.items() if n in numbers and v}
            if not problems:
                break
            print(f"[StoryboarderV2] lint round {r + 1}: problems on panels {sorted(problems)}")
            needs_paper = any(x.startswith(("evidence", "claim", "fact-check")) for found in problems.values() for x in found)
            try:
                data, usage = await _ask_json(self.text_engine, SYSTEM_STORY,
                                              fix_prompt(sb, cast, problems, paper.body if needs_paper else "", _tables_section(paper.body)),
                                              thinking=False, temperature=0.3, max_tokens=12000, key="panels", paper=paper.body,
                                              cache_key=self.cache_key)
            except Exception as e:  # the script stays as it is; invalid boards are salvaged or demoted below
                _reraise_account(e)
                print(f"[StoryboarderV2] lint fix skipped: {type(e).__name__}: {e}")
                self._skip("lint fix round", e)
                break
            self._add_usage(usage)
            replaced += self._replace_panels(sb, cast, data.get("panels"), set(problems), _valid_boards(sb.panels, paper.body, set(problems)),
                                             ledger=_ledger(paper.body))
        self._demote_boards(sb, paper.body)
        return replaced

    async def _run_quiz(self, quiz, sb: StoryboardV2, questions: Optional[list] = None) -> dict:
        """services.abstract_quiz.run_quiz on the teaching panels (the claims board and the recap only list the claims, so a
        reader must learn them from the story itself), normalized: float score (None on failure), failed claim ids, its
        usage counted."""
        taught = copy.copy(sb)
        taught.panels = _teaching_panels(sb.panels)
        result = await quiz.run_quiz(self.text_engine, self.fast_engine, taught, sb.abstract_claims,
                                     abstract_text=str(sb.brief.get("abstract") or ""), questions=questions)
        if not isinstance(result, dict):
            raise ValueError("the abstract quiz returned no result")
        if isinstance(result.get("usage"), dict):
            self._add_usage(result["usage"])
        out = {k: v for k, v in result.items() if k != "usage"}
        try:
            out["score"] = None if result.get("score") is None else float(result["score"])
        except (TypeError, ValueError):
            out["score"] = None
        failed: list[int] = []
        for x in _as_list(result.get("failed_claim_ids", result.get("failed_claims", result.get("failed")))):
            if isinstance(x, dict):   # failed questions instead of ids
                failed += _claim_ids(x.get("claim_ids", x.get("claims", x.get("claim_id", x.get("claim")))))
            else:
                failed += _claim_ids(x)
        out["failed_claim_ids"] = list(dict.fromkeys(failed))
        out.setdefault("questions", [])
        return out

    async def _revise_for_quiz(self, quiz, sb: StoryboardV2, cast: Cast, paper: PaperText, panel_range: tuple[int, int], result: dict) -> bool:
        """One revision round for the claims the test reader missed (services.abstract_quiz.revise_prompt): revised panels
        come back with their "n", new ones with "after" and are placed inside their chapter by _insert_position (a few at
        most). Then lint/fix, fact-check and fit run again. The script only changes when the reply applies."""
        failed = [c for c in sb.abstract_claims if c.get("id") in set(result["failed_claim_ids"])]
        if not failed:
            return False
        room = max(0, min(len(failed), panel_range[1] + QUIZ_EXTRA_PANELS - len(sb.panels)))
        for attempt in range(2):   # a reply with no usable panel (BERT's left six claims untaught) is asked once more
            data, usage = await _ask_json(self.text_engine, SYSTEM_STORY, quiz.revise_prompt(sb, failed, cast, paper_text=paper.body, quiz=result),
                                          thinking=True, temperature=0.4, max_tokens=12000, key="panels", paper=paper.body,
                                          cache_key=self.cache_key)
            self._add_usage(usage)
            items = [x for key in ("panels", "replace", "new_panels", "revised_panels", "revisions") for x in _panel_items(data.get(key))]
            if any(isinstance(x, dict) and (x.get("dialogue") or x.get("narration") or x.get("board")) for x in items):
                break
            print(f"[StoryboarderV2] the quiz revision reply held no usable panel (keys {list(data)[:5]}); asking once more")
        work = copy.copy(sb)
        work.panels = panels = copy.deepcopy(sb.panels)
        numbers = {p.n for p in panels}
        last = max((p.n for p in panels if p.beat != "recap"), default=0)
        for item in items:   # a new panel numbered past the script ("n": 19) instead of placed ("after": 12) still counts
            try:
                n = int(item.get("n")) if item.get("after") is None and item.get("n") is not None else None
            except (TypeError, ValueError):
                n = None
            if n is not None and n not in numbers:
                item["after"] = max(1, min(n - 1, last))
        replaced = self._replace_panels(work, cast, [x for x in items if x.get("after") is None], numbers,
                                        _valid_boards(panels, paper.body), ledger=_ledger(paper.body))
        inserts = []
        for i, item in enumerate(x for x in items if x.get("after") is not None):
            if len(inserts) >= room:
                break
            try:
                new, after = _panel_from_reply(item, 0), int(item.get("after"))
            except (TypeError, ValueError) as e:  # keep going on a malformed panel
                print(f"[StoryboarderV2] skipping malformed quiz panel: {e}")
                continue
            if not (new.dialogue or new.narration or new.board):
                continue
            new.beat = _normalize_beat(new.beat, "proof")
            new.chapter = BEAT_CHAPTER[new.beat]
            inserts.append((_insert_position(panels, new, after), i, new))
        if not replaced and not inserts:   # the reader still misses these claims: say so in the job's quality record
            print(f"[StoryboarderV2] WARNING: the quiz revision changed nothing ({len(items)} panels in the reply)")
            self._skip("quiz revision", ValueError(f"the reply changed no panel ({len(items)} panels in it); claims "
                                                   f"{[c['id'] for c in failed]} stay untaught"))
            return False
        for pos, _i, new in sorted(inserts, key=lambda t: (t[0], t[1]), reverse=True):
            panels.insert(pos, new)
        _normalize_v3_panels(panels, cast, sb.abstract_claims, _ledger(paper.body))
        sb.panels = panels
        _finish_v3(sb)
        print(f"[StoryboarderV2] quiz revision for claims {[c['id'] for c in failed]}: {replaced} panels revised, {len(inserts)} added")
        await self.fix_v3(sb, cast, paper, rounds=2)
        flags: dict[int, list[str]] = {}
        await self.factcheck(sb, paper, board_flags=flags)
        if flags:
            await self.fix_v3(sb, cast, paper, rounds=1, extra=flags)
        await self.fit(sb, "en-US")
        return True

    async def quiz_gate(self, sb: StoryboardV2, cast: Cast, paper: PaperText, panel_range: tuple[int, int]) -> dict:
        """The abstract quiz on the visible text of the teaching panels (services.abstract_quiz). When any claim fails (the
        owner's bar is the whole abstract, not a share of it), one revision round for the failed claims, then the quiz runs
        again. Never raises: a quiz that fails leaves the script as it is and records the error in sb.quiz."""
        quiz = _optional("abstract_quiz")
        if quiz is None or not sb.abstract_claims:
            sb.quiz = {"score": None, "failed_claim_ids": [], "questions": [],
                       "error": "abstract quiz unavailable" if quiz is None else "no abstract claims"}
            return sb.quiz
        try:
            first = await self._run_quiz(quiz, sb)
            if first["score"] is None:   # intermittent (BERT's re-run lost its gate this way; a replay of the same script ran)
                print(f"[StoryboarderV2] the abstract quiz could not run ({first.get('error')}); running it once more")
                first = await self._run_quiz(quiz, sb)
        except Exception as e:  # the quiz is a gate, never a reason to fail the job
            _reraise_account(e)
            print(f"[StoryboarderV2] WARNING: abstract quiz failed: {type(e).__name__}: {e}")
            self._skip("abstract quiz", e)
            sb.quiz = {"score": None, "failed_claim_ids": [], "questions": [], "error": f"{type(e).__name__}: {e}"}
            return sb.quiz
        sb.quiz = first
        print(f"[StoryboarderV2] abstract quiz: score {first['score']}, failed claims {first['failed_claim_ids']}")
        if first["score"] is None:   # run_quiz never raises: a quiz that could not run comes back with its error
            self._skip("abstract quiz", RuntimeError(first.get("error") or "no score"))
            return sb.quiz
        if not first["failed_claim_ids"]:
            return sb.quiz
        try:
            if not await self._revise_for_quiz(quiz, sb, cast, paper, panel_range, first):
                return sb.quiz
        except Exception as e:  # keep the script and the first result
            _reraise_account(e)
            print(f"[StoryboarderV2] WARNING: quiz revision failed: {type(e).__name__}: {e}")
            self._skip("quiz revision", e)
            sb.quiz = {**first, "revision_error": f"{type(e).__name__}: {e}"}
            return sb.quiz
        before = {"score": first["score"], "failed_claim_ids": first["failed_claim_ids"]}
        try:
            after = await self._run_quiz(quiz, sb, questions=first.get("questions") or None)   # same questions: comparable
            if after["score"] is None:
                raise RuntimeError(after.get("error") or "no score")
            sb.quiz = {**after, "before": before}
        except Exception as e:  # the script was revised; its new score is unknown
            _reraise_account(e)
            self._skip("abstract quiz after the revision", e)
            sb.quiz = {**first, "revised": True, "error": f"quiz after the revision failed: {type(e).__name__}: {e}"}
        print(f"[StoryboarderV2] abstract quiz after revision: score {sb.quiz.get('score')}")
        return sb.quiz

    async def build_v3(self, paper: PaperText, cast: Cast, language: str, panel_range: tuple[int, int], on_stage=None) -> StoryboardV2:
        """The v3 story layer: abstract → brief with verified abstract claims → a story with boards → deterministic lint and
        fix rounds → fact-check (boards included) → abstract quiz gate → translation (boards included), fit and the
        translation's fact-check. Panels end up numbered 1..N and chaptered by their beats."""
        def stage(status: str, message: str) -> None:
            if on_stage:
                on_stage(status, message)

        rng = story_range(panel_range)
        abstract = str(getattr(paper, "abstract", "") or "").strip() or find_abstract(paper.body)
        stage("analyzing", "Reading the paper and its abstract")
        brief = await self.analyze_v3(paper, abstract)
        stage("analyzing", "Checking the brief against the paper")
        await self.check_brief(brief, paper)
        claims = [c for c in _as_list(brief.pop("abstract_claims", [])) if isinstance(c, dict)]
        stage("storyboarding", "Writing the story")
        sb = await self.write_v3(brief, claims, cast, rng, paper)
        stage("storyboarding", "Checking every board, number and abstract claim")
        await self.fix_v3(sb, cast, paper)
        await self.fit(sb, "en-US")
        stage("storyboarding", "Fact-checking the script against the paper")
        flags: dict[int, list[str]] = {}
        await self.factcheck(sb, paper, board_flags=flags)
        if flags:
            await self.fix_v3(sb, cast, paper, rounds=1, extra=flags)
        await self.fit(sb, "en-US")
        stage("storyboarding", "Testing whether a reader learns the abstract from the panels")
        await self.quiz_gate(sb, cast, paper, rng)
        scrubbed = scrub_unsupported(sb, paper.body)
        _honest_recap(sb)
        _finish_v3(sb)
        sb.brief["lint"] = {str(n): found for n, found in lint_v3(sb, paper.body).items()}   # what is left, for the job report
        if scrubbed:
            sb.brief["scrubbed"] = scrubbed
        if language != "en-US":
            stage("translating", f"Translating to {LANG_NAMES.get(language, language)}")
            sb = await self.translate(sb, language)
            await self.fit(sb, language)
            stage("translating", "Fact-checking the translation against the paper")
            if await self.factcheck(sb, paper, language):
                await self.fit(sb, language)
        _finish_v3(sb)
        return sb


def panel_range_for(choice: str, config_min: int, config_max: int) -> tuple[int, int]:
    rng = PANEL_RANGES.get(choice or "auto")
    return rng if rng else (config_min, config_max)
