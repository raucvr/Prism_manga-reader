"""
Panel QA judge — a vision LLM transcribes the generated panel and checks it against the script.

Verdict drives targeted regeneration: a panel is retried when a bubble/narration is missing or not exact,
an expected character is missing or the wrong species, an off-script character appears, stray/misspelled text
or a wrong bubble tail is reported, or the overall score is low. Labels alone never trigger a retry.
A judge failure never blocks (and never re-bills) the pipeline: the result says ``available=False``.

Matching is content-based: expected texts are paired with the judge's items by text similarity, best pairs first
(the judge may reorder or omit items), and panels are looked up by ``panel_index``.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import re
from dataclasses import dataclass, field

from engines.base import BaseEngine, GenerationConfig, ImageContent, Message
from services.storyboarder_v2 import PanelV2, extract_json, scrub_surrogates

LAYOUT_TEXT = {1: "one panel", 2: "two panels side by side (left = panel 1, right = panel 2)"}
SPECIES = {"papi": "brown dog professor", "kumo": "light-blue cloud creature with two green leaves", "nezu": "brown hedgehog"}
NOISE_EXACT = {"", "none", "n/a", "na", "no issues", "no issue", "no critical issues", "nothing", "-"}
NOISE_PREFIXES = ("no critical issue", "no issues", "no issue found", "no problems", "only minor", "minor:", "all text is correct", "all expected text is present")
POSITIVE_RE = re.compile(r"\b(correct|correctly|properly|(?<!than )expected|match|matches|accurate|accurately)\b")
_PROBLEM = (r"incorrect\w*|wrong|missing|misspell\w*|gibberish|typos?|cut(?:[- ]?off)?|duplicat\w*|extra|extraneous|unexpected|additional"
            r"|stray|garbled|unreadable|illegible|issues?|errors?|problems?|defects?|mistakes?|overlap\w*|differ\w*|artifacts?")
NEGATIVE_RE = re.compile(rf"\b(?:{_PROBLEM}|not|instead|twice|outside)\b")
# a negated list of problems is praise, not a problem: "no cut-off or gibberish text", "nothing is misspelled",
# "there aren't any typos", "typos: none"
_ABSENT = (rf"(?:(?:other|any|visible|obvious|major|minor|critical|text|character|label|spelling|word|of|the)s?[\s/]+){{0,4}}"
           rf"(?:(?:is|are|was|were)\s+)?(?:{_PROBLEM})"
           r"(?:[\s/]+(?:text|content|characters?|words?|labels?|found|detected|present|visible|checks?)){0,3}")
_SEP = r"\s*(?:,|/|\bor\b|\band\b|\bnor\b)\s*"
_NEGATOR = (r"no|without|nothing|none|zero|free (?:of|from)"
            r"|(?:not|cannot|\w+n't)(?:\s+(?:have|has|contain|contains|show|shows|see|find))?(?:\s+(?:a single|one|any))?")
# the "typos: none" form has no leading keyword, so its repetitions are bounded (it is tried at every word)
NEGATED_RE = re.compile(rf"\b(?:{_NEGATOR})\s+{_ABSENT}(?:{_SEP}{_ABSENT})*\b"
                        rf"|\b{_ABSENT}(?:{_SEP}{_ABSENT}){{0,3}}\s*[:=\-–—]+\s*(?=(?:none|nothing|no|n/?a|0|zero|ok|okay|passed|clean)\b)")
MISSING_RE = re.compile(r"\b(?:no|without|never|\w+n't)\s+\w")   # what NEGATED_RE leaves ("has no speech tail", "doesn't match")
CONTRAST_RE = re.compile(r"\b(but|however|though|although|except|yet|whereas)\b")
EXCEPT_RE = re.compile(r"\b(?:except|apart from|other than|besides|aside from|save for)\b")
TYPO_RE = re.compile(r"(?<!not )(?<!n't )\b(misspell\w*|typos?|gibberish|garbled|unreadable|illegible)\b(?!-free)")
# a count is a duplicate only as "3 times" / "x2"; "two green leaves" describes Kumo ("two Kumo creatures": see _evaluate)
_COUNT = r"(?:two|three|four|\d+)"
DUPLICATE_RE = re.compile(rf"\b(?:second|another|other|duplicat\w*|extra|copy|copies|clones?|twins?|double|twice|multiple|again"
                          rf"|{_COUNT}\s+(?:times|copies|instances|versions)|x\s?\d+)\b|×\s?\d")
# what may surround the expected cast in an extra entry that reports nobody else ("None besides Papi and Kumo")
EXCLUDING_RE = re.compile(r"\b(?:besides|except(?: for)?|other than|apart from|aside from|but|only|just|and|or"
                          r"|(?:the )?(?:expected|listed|planned|scripted) (?:characters?|cast))\b|[(),&/]")
PLACEHOLDER_RE = re.compile(
    r"(?:none|nothing|nil|no|n/?a|not applicable|[-–—]+|无|没有|なし)(?: (?:found|visible|detected|present|seen|observed|noted|identified))?"
    r"|(?:there (?:is|are) )?no (?:\w+ ){0,3}?(?:characters?|creatures?|people|persons?|humans?|animals?|texts?|misspell\w*|typos?"
    r"|issues?|problems?|errors?|defects?)(?: (?:found|visible|detected|present|seen|observed|noted|identified))?")
CLAUSE_RE = re.compile(r"[,;:(]|\s[-–—]\s")
QUOTED_RE = re.compile(r"'[^']+'|\"[^\"]+\"|`[^`]+`|「[^」]+」")
SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)")
# "On a scale of 1-10: 8", "Score (1-10): 8", "scale of 10: 4", "8 out of 10", "8/10": the scale is not the score
SCALE_RE = re.compile(r"\b(?:on\s+)?(?:a\s+)?scale\s+(?:of\s+|from\s+)?(?:\d+\s*(?:-|–|to)\s*)?\d+\b"
                      r"|[(\[]?\s*\b\d+\s*(?:-|–|to)\s*10\b\s*[)\]]?(?:\s+scale\b)?|\bout\s+of\s+10\b|/\s*10\b", re.I)
# a wrong bubble tail reported only in issues[] ("the tail of '…' is incorrectly pointing at Kumo"): one clause must name
# the tail and the fault, and must not be negated ("none of the tails are wrong", "the caption has no tail")
TAIL_CLAUSE_SPLIT = re.compile(r"[;,]|(?<!\d)\.|\.(?!\d)|\b(?:but|however|though|although|while|whereas)\b")
# the bubble text a note quotes is not the judge's own words: its periods, commas and negations must not count
QUOTED_LINE_RE = re.compile(r"\"[^\"]*\"|“[^”]*”|‘[^’]*’|「[^」]*」|『[^』]*』|\*[^*]+\*|(?<!\w)'[^']+'(?!\w)")
TAIL_FAULT_RE = re.compile(r"\btails?\b.*\b(?:incorrect\w*|wrong\w*|misattribut\w*|instead of|rather than)\b"
                           r"|\b(?:incorrect\w*|wrong\w*|misattribut\w*)\b.*\btails?\b")
TAIL_NEGATED_RE = re.compile(r"\b(?:no|none|not|never|nothing|without|neither|nor)\b|n't\b")


# asked without the script: a judge that is shown the expected text tends to read it into the picture
# ("上辿った" passed as "上回った", a dropped "用" passed as exact), a blind reader reports what is drawn
BLIND_PROMPT = ("The lettering in this manga image was drawn by an image AI and often has wrong, missing, doubled, look-alike or "
                "invented characters. Your transcription is used to find them, so copy what is drawn, never what was probably meant. "
                "Transcribe every visible text glyph by glyph (letters, kanji, kana, hanzi, digits, superscripts, symbols, punctuation). "
                "Do not fix spelling, grammar or missing characters; do not complete text that is cut off or hidden; keep each "
                "character's exact form (never convert between simplified, traditional and Japanese forms). Give one item per speech "
                "bubble, narration box, label or sign, with its lines joined in order, and no speaker names or descriptions. "
                'Return JSON only: {"texts": ["..."]} in reading order.')


@dataclass
class QAResult:
    score: float
    text_exact: int
    text_total: int
    characters_ok: bool
    issues: list[str] = field(default_factory=list)
    needs_retry: bool = False
    transcription: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    layout_ok: bool = True        # every bubble tail points at its speaker and there is no stray/misspelled text
    available: bool = True        # False: no usable verdict (judge error, unparseable JSON, no panel data)
    # lines that did not match: {"panel": 1-based index in the group, "kind": "bubble"|"narration", "index": bubble index,
    # "expected", "judge": what the judge says is drawn (None if it passed the line), "blind": the script-free reading}
    mismatches: list[dict] = field(default_factory=list)
    low_score: bool = False       # the judge's own score is under the retry threshold (a missing score is not low)
    blind_texts: list[str] | None = None   # the script-free transcription, when one was made
    layout_issues: list[str] = field(default_factory=list)   # the issues behind layout_ok=False (tails, stray text)
    label_misses: list[str] = field(default_factory=list)    # labels the judge says are wrong (confirmed by the blind reading)

    def to_public(self) -> dict:
        return {"score": self.score, "text_exact": self.text_exact, "text_total": self.text_total, "issues": self.issues[:6],
                "available": self.available}

    def feedback(self) -> str:
        return "; ".join(self.issues[:4]) if self.issues else "text or characters did not match the script"


def _norm(s: str) -> str:
    s = str(s).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() in ("true", "yes", "1")


def _falsy(v) -> bool:
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v == 0
    return str(v).strip().lower() in ("false", "no", "0")


PUNCT_FOLD = str.maketrans({"，": ",", "。": ".", "！": "!", "？": "?", "：": ":", "；": ";", "（": "(", "）": ")",
                            "“": '"', "”": '"', "‘": "'", "’": "'", "、": ",", "～": "~", "－": "-",
                            "…": "...", "—": "-", "–": "-", "‐": "-"})
FULLWIDTH_ASCII = {c: c - 0xFEE0 for c in range(0xFF01, 0xFF5F)}   # "１２Ａ" -> "12A"


LINE_BREAK_HYPHEN = re.compile(r"(?<=[A-Za-z])[-\u2010\u00ad]\s+(?=[a-z])")   # "trans-\nformer" or, lines joined, "trans- former"


def _canon(text, join_hyphens: bool = False) -> str:
    """Comparison form: width-folded punctuation and ASCII, no whitespace, no single final period (bubbles often omit
    it). ``join_hyphens`` also drops a hyphen at a line break ("trans-\nformer"); a real one ("Self-\nattention") is
    kept by the default."""
    s = str(text or "")
    if join_hyphens:
        s = LINE_BREAK_HYPHEN.sub("", s)
    s = s.translate(FULLWIDTH_ASCII).translate(PUNCT_FOLD)
    s = re.sub(r"\s+", "", s).lower()
    return s[:-1] if s.endswith(".") and not s.endswith("..") else s


def _readings(text) -> set[str]:
    """Both readings of a hyphen at a line break."""
    return {_canon(text), _canon(text, join_hyphens=True)}


def _same_text(expected: str, rendered) -> bool:
    return isinstance(rendered, str) and bool(rendered.strip()) and _canon(expected) in _readings(rendered)


def _as_list(value) -> list:
    """A list-shaped verdict field: a bare string becomes a one-item list, any other non-list is ignored."""
    return value if isinstance(value, list) else [value] if isinstance(value, str) else []


def _entry_text(entry) -> str:
    """A verdict list entry as text: a string, a number, or an object's text/name/description."""
    if isinstance(entry, dict):
        entry = next((entry[k] for k in ("text", "rendered", "name", "description", "issue", "note")
                      if isinstance(entry.get(k), str) and entry[k].strip()), "")
    elif isinstance(entry, (int, float)) and not isinstance(entry, bool):
        entry = str(entry)
    return entry.strip() if isinstance(entry, str) else ""


def _is_placeholder(entry: str) -> bool:
    """True for a list entry that only says there is nothing to report ("None found", "No stray text", "n/a")."""
    n = _norm(entry).rstrip(".")
    if n in NOISE_EXACT or PLACEHOLDER_RE.fullmatch(n):
        return True
    # "None (all characters match)" is a placeholder, "None found; 'Transfomer' on the chart" is not
    parts = CLAUSE_RE.split(n, maxsplit=1)
    if len(parts) < 2 or not PLACEHOLDER_RE.fullmatch(parts[0].strip()) or CONTRAST_RE.search(n) or QUOTED_RE.search(parts[1]):
        return False
    return not NEGATIVE_RE.search(NEGATED_RE.sub(" ", parts[1]))


def _is_noise(note: str) -> bool:
    """True for empty, placeholder or purely positive notes ("tail correctly points to papi, no misspelled text")."""
    n = _norm(note).rstrip(".")
    affirmed = NEGATED_RE.sub(" ", n)
    if TYPO_RE.search(affirmed) or CONTRAST_RE.search(n):      # "..., but the tail points at kumo"
        return False
    if n.startswith(NOISE_PREFIXES) or _is_placeholder(n):
        return True
    return bool(POSITIVE_RE.search(n)) and not NEGATIVE_RE.search(affirmed) and not MISSING_RE.search(affirmed)


def _panel_verdict(p: dict) -> dict | None:
    """The usable parts of one panel object of the verdict; None when it holds nothing checkable (text_items: ["ok"])."""
    v = {
        "items": [x for x in _as_list(p.get("text_items")) if isinstance(x, dict)],
        "characters": [x for x in _as_list(p.get("characters")) if isinstance(x, dict)],
        "transcription": [t for t in map(_entry_text, _as_list(p.get("transcription"))) if t],
        "extras": [t for t in map(_entry_text, _as_list(p.get("extra_characters"))) if not _is_placeholder(t)],
        "stray": [t for t in map(_entry_text, _as_list(p.get("stray_text"))) if not _is_placeholder(t)],
        "notes": [t for t in map(_entry_text, _as_list(p.get("issues"))) if not _is_noise(t)],
    }
    return v if any(v.values()) or p.get("text_items") == [] else None


def _who(e: dict) -> str:
    return f"{e['speaker']}'s bubble" if e["kind"] == "bubble" else "the narration box"


def _closest(expected: str, texts: list[str]) -> str | None:
    """The run of consecutive transcribed items most like the expected line (a reader may split one bubble into its
    drawn lines), to tell the next attempt what went wrong."""
    target = _canon(expected)
    windows = ["\n".join(texts[i:j]) for i in range(len(texts)) for j in range(i + 1, min(i + 4, len(texts)) + 1)]
    scored = [(difflib.SequenceMatcher(None, _canon(w), target).ratio(), w) for w in windows]
    best = max(scored, default=(0.0, ""))
    return re.sub(r"\s+", " ", best[1]).strip()[:120] if best[0] >= 0.4 else None


SHEET_CAPTION_RE = re.compile(r"\b(?:soft spikes|name\s*:|occupation\s*:|species\s*:|profession\s*:)")
IDEOGRAPH_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
# never adopted when changed: negation, numerals and magnitudes, counters and units, direction and comparison, quantifiers
MEANING_CRITICAL = set("不没沒未非無无否别別勿" "〇零一二两兩三四五六七八九十百千万萬亿億兆" "倍割成半"
                       "秒分時时日週周月年回次個个台層层件人" "上下高低增増减減升降多少大小快慢早晚长長短强強弱前后後"
                       "全都每各某部只仅僅最更较較約约" "左右内外入出正负負反逆深浅淺优劣胜败勝真假对错錯串并训测编解过欠可必")


def wording_gate(expected: str, drawn: str) -> bool:
    """A drawn line may replace the script line only if it differs by same-length substitutions of ideographs (kana,
    digits, symbols, Latin and ?/! identical) and no changed run touches a meaning-critical character (莫大→巨大 passes;
    上回る→下回る, 三倍→两倍, 能用→不能用 do not)."""
    a, b = _canon(expected), _canon(drawn)
    if a == b or IDEOGRAPH_RE.sub("", a) != IDEOGRAPH_RE.sub("", b):
        return False
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace" or i2 - i1 != j2 - j1 or set(a[i1:i2] + b[j1:j2]) & MEANING_CRITICAL:
            return False
    return True


NUMBER_FACT_RE = re.compile(r"[+\-−]?\d+(?:[.,]\d+)?|[〇零一二两兩三四五六七八九十百千万萬亿億兆]+|[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+")
OTHER_FACT_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*|[%‰×÷±<>≈=≠≤≥~]")


def same_facts(a: str, b: str) -> bool:
    """The guard for a line rewritten so the image model can letter it: the same numbers in the same order (digits with
    their sign, CJK numerals, superscripts), the same Latin names and terms and math/percent symbols, and the same
    sentence-final ? or ! ("34层优于18层" -> "18层优于34层" and "+1.5" -> "-1.5" fail)."""
    fa, fb = (str(t).translate(FULLWIDTH_ASCII).translate(PUNCT_FOLD).lower().strip() for t in (a, b))
    ending = lambda t: t[-1:] if t[-1:] in "?!" else ""
    return (NUMBER_FACT_RE.findall(fa) == NUMBER_FACT_RE.findall(fb)
            and sorted(OTHER_FACT_RE.findall(fa)) == sorted(OTHER_FACT_RE.findall(fb)) and ending(fa) == ending(fb))


def agreed_alternatives(result: QAResult) -> list[dict]:
    """Mismatched lines whose drawn wording the judge and the script-free reading agree on and that pass wording_gate:
    candidates for adopting the drawn wording (Seedream swaps some rare words for common ones every time)."""
    joined = _readings("\n".join(result.blind_texts)) if result.blind_texts else set()
    out = []
    for m in result.mismatches:
        drawn = m.get("judge")
        if not drawn or not joined:
            continue
        seen = any(_canon(drawn) in j for j in joined) and not any(_canon(m["expected"]) in j for j in joined)
        if seen and wording_gate(m["expected"], drawn):
            out.append(m)
    return out


def adopt_wording(expected: str, drawn: str) -> str:
    """``expected`` with the ideographs the picture shows instead; the script keeps its own punctuation and spacing."""
    slots = [i for i, c in enumerate(expected) if IDEOGRAPH_RE.match(c)]
    drawn_ideographs = IDEOGRAPH_RE.findall(drawn)
    if len(slots) != len(drawn_ideographs):
        return expected
    chars = list(expected)
    for i, c in zip(slots, drawn_ideographs):
        chars[i] = c
    return "".join(chars)


def _unavailable(reason: str, raw: dict | None = None) -> QAResult:
    return QAResult(score=0.0, text_exact=0, text_total=0, characters_ok=True, issues=[f"qa unavailable: {reason[:120]}"],
                    needs_retry=False, raw=raw or {}, available=False)


class PanelQA:
    def __init__(self, vision_engine: BaseEngine, sheets: dict[str, ImageContent], language: str, retry_threshold: float = 6.0,
                 cast_names: list[str] | None = None, cast_looks: dict[str, str] | None = None, blind_check: bool = False):
        self.engine = vision_engine
        self.blind_check = blind_check   # also transcribe the image without the script and require both readings to agree
        self.sheets = sheets
        self.language = language
        self.retry_threshold = retry_threshold
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        # names recognised in the judge's character lists (themes without sheets pass their cast explicitly)
        self.cast_names = list(sheets.keys()) if cast_names is None else list(cast_names)
        # how the characters without a sheet look, so the judge can recognise them instead of reporting them as extras
        self.cast_looks = {str(k).strip().lower(): str(v).strip().rstrip(".") for k, v in (cast_looks or {}).items()
                           if str(v or "").strip() and str(k).strip().lower() not in sheets}

    def _expected(self, panels: list[PanelV2]) -> list[dict]:
        out = []
        for i, p in enumerate(panels, start=1):
            items = [{"kind": "bubble", "speaker": b.speaker, "text": b.text} for b in p.dialogue]
            if p.narration:
                items.append({"kind": "narration", "text": p.narration})
            for lab in p.labels[:2]:
                items.append({"kind": "label", "text": lab})
            out.append({"panel_index": i, "characters": p.characters, "expected_text": items})
        return out

    def _prompt(self, panels: list[PanelV2]) -> str:
        n_sheets = len(self.sheets)
        sheet_desc = "; ".join(f"image {i + 1} = {name} ({SPECIES.get(name, 'see sheet')})" for i, name in enumerate(self.sheets))
        look_desc = "; ".join(f"{name} = {look}" for name, look in self.cast_looks.items())
        gen_idx = n_sheets + 1
        return f"""You are a strict QA judge for an educational manga. {('Reference character sheets: ' + sheet_desc + '. ') if n_sheets else ''}{('Character descriptions: ' + look_desc + '. ') if look_desc else ''}Image {gen_idx} is the generated page containing {LAYOUT_TEXT.get(len(panels), str(len(panels)) + ' panels')}.{(' Judge only image ' + str(gen_idx) + ': text written on the reference sheets (names, notes) is not part of the page unless it is actually drawn on the page.') if n_sheets else ''}

EXPECTED (JSON): {json.dumps(self._expected(panels), ensure_ascii=False)}

For each panel (use the same panel_index):
1. transcription: all visible text.
2. text_items: for EVERY expected text item (same order; copy the expected text into "text"): is it present and EXACT (character for character, ignoring line breaks, curly vs straight quotes, full-width vs half-width punctuation and a missing final period; a missing or different "?" or "!" is NOT exact); what the image actually shows ("rendered"); and for speech bubbles, does the bubble's tail point at the expected speaker ("speaker_ok"; use true for narration boxes and labels).
3. characters: for each expected character, is it present and does it match its reference sheet / description (species, colors, face)?
4. extra_characters: any unexpected people, children, animals or creatures ([] if none).
5. stray_text: visible text that is NOT one of the expected texts and is misspelled, gibberish or unreadable, e.g. "Transfomer" on a chart ([] if none).
6. issues: real problems only (wrong tail, text outside bubbles, cut-off or duplicated text...); [] if there are none. Never list things that are correct.
Score the page 1-10 for "ready to show a reader" (10 = flawless; punctuation-width or final-period differences do not lower the score). Use JSON booleans (true/false) and JSON lists ([] when empty), never strings.

Output JSON only:
{{"panels": [{{"panel_index": 1, "transcription": ["..."], "text_items": [{{"text": "<expected text>", "found": true, "exact": true, "speaker_ok": true, "rendered": "<what the image shows>", "note": ""}}], "characters": [{{"name": "papi", "present": true, "matches_reference": true, "note": ""}}], "extra_characters": [], "stray_text": [], "issues": []}}], "overall": 9, "worst_problem": "..."}}"""

    async def judge(self, image: ImageContent, panels: list[PanelV2]) -> QAResult:
        if not self.blind_check:
            return await self._verdict(image, panels)
        result, blind = await asyncio.gather(self._verdict(image, panels), self._blind_transcription(image))
        return self._cross_check(result, panels, blind)

    async def _blind_transcription(self, image: ImageContent) -> list[str] | None:
        """What the image says, read without the script (None when the call or its JSON fails: the verdict still counts)."""
        config = GenerationConfig(temperature=0.0, max_tokens=1500)
        config.thinking = False
        try:
            response = await self.engine.generate_text([Message.user(BLIND_PROMPT, images=[image])], config)
            self.usage["prompt_tokens"] += int(response.usage.get("prompt_tokens", 0) or 0)
            self.usage["completion_tokens"] += int(response.usage.get("completion_tokens", 0) or 0)
            data = extract_json(response.content, key="texts")
        except Exception:  # noqa: BLE001
            return None
        texts = [t for t in map(_entry_text, _as_list(data.get("texts") if isinstance(data, dict) else None)) if t]
        return texts or None

    def _cross_check(self, result: QAResult, panels: list[PanelV2], blind: list[str] | None) -> QAResult:
        """A line the judge passed counts as exact only if the blind transcription contains it too (same canonical form)."""
        if blind is None or not result.available:
            return result
        joined = _readings("\n".join(blind))
        expected_len = sum(len(_canon(e["text"])) for p in self._expected(panels) for e in p["expected_text"] if e["kind"] != "label")
        if len(_canon("".join(blind))) < 0.25 * expected_len:   # a reader that saw almost nothing is less likely than a glitch
            return result
        result.blind_texts = list(blind)
        for m in result.mismatches:
            m["blind"] = _closest(m["expected"], blind)
        unconfirmed = []
        for p in self._expected(panels):
            bubble_i = -1
            for e in p["expected_text"]:
                bubble_i += e["kind"] == "bubble"
                head = f'{_who(e)} must read exactly "{e["text"]}"'
                if e["kind"] == "label" or any(_canon(e["text"]) in j for j in joined) or any(i.startswith(head) for i in result.issues):
                    continue
                shown = _closest(e["text"], blind)
                unconfirmed.append(f"{head} (image shows: {shown})" if shown else f"{head} (not found in the image)")
                result.mismatches.append({"panel": p["panel_index"], "kind": e["kind"], "index": bubble_i if e["kind"] == "bubble" else None,
                                          "expected": e["text"], "judge": None, "blind": shown})
        # a label is short, so only a miss both readers agree on counts (the judge alone often misreads small props text)
        def label_seen(lab: str) -> bool:   # its own item, not a word inside a bubble that happens to contain it
            c = _canon(lab)
            return any(c in _canon(t) and len(_canon(t)) <= len(c) + 4 for t in blind)

        wrong_labels = [lab for lab in result.label_misses if not label_seen(lab)]
        for lab in wrong_labels:
            note = f'the label must read exactly "{lab}"' + (f" (image shows: {_closest(lab, blind)})" if _closest(lab, blind) else "")
            result.layout_issues.append(note)
            result.issues = list(dict.fromkeys(result.issues + [note]))
        if wrong_labels:
            result.layout_ok, result.needs_retry = False, True
        copied = [t for t in blind if self.sheets and SHEET_CAPTION_RE.search(_norm(t))
                  and not any(_canon(t) in _canon(e["text"]) for p in self._expected(panels) for e in p["expected_text"])]
        if copied:   # the blind reader sees only the page, so this cannot be the judge confusing the sheet with it
            note = f"remove the caption copied from the reference sheet: {copied[0][:60]}"
            result.layout_ok, result.needs_retry = False, True
            result.layout_issues.append(note)
            result.issues = list(dict.fromkeys(result.issues + [note]))
        if not unconfirmed:
            return result
        result.text_exact = max(0, result.text_exact - len(unconfirmed))
        result.issues = list(dict.fromkeys(unconfirmed + result.issues))
        result.needs_retry = True
        return result

    async def _verdict(self, image: ImageContent, panels: list[PanelV2]) -> QAResult:
        images = list(self.sheets.values()) + [image]
        config = GenerationConfig(temperature=0.0, max_tokens=3000)
        config.thinking = False
        try:
            response = await self.engine.generate_text([Message.user(self._prompt(panels), images=images)], config)
            self.usage["prompt_tokens"] += int(response.usage.get("prompt_tokens", 0) or 0)
            self.usage["completion_tokens"] += int(response.usage.get("completion_tokens", 0) or 0)
            try:
                data = extract_json(response.content, key="panels")
            except (ValueError, json.JSONDecodeError):
                # one cheap text-only repair round instead of shipping the panel unchecked
                repair_cfg = GenerationConfig(temperature=0.0, max_tokens=3000)
                repair_cfg.thinking = False
                fixed = await self.engine.generate_text([Message.user(
                    "Return the following QA verdict as strictly valid JSON with the same content and keys, no commentary:\n\n" + scrub_surrogates(response.content[:20000]))], repair_cfg)
                self.usage["prompt_tokens"] += int(fixed.usage.get("prompt_tokens", 0) or 0)
                self.usage["completion_tokens"] += int(fixed.usage.get("completion_tokens", 0) or 0)
                data = extract_json(fixed.content, key="panels")
            return self._evaluate(data, panels)
        except Exception as e:  # a judge failure must never block or re-bill the pipeline
            return _unavailable(str(e))

    # ------------------------------------------------------------------ evaluation
    @staticmethod
    def _match_items(expected: list[dict], items: list[dict]) -> list[dict | None]:
        """Pair each expected text with a judge item: the most similar normalised texts first (difflib >= 0.5, each item
        used once), so a missing line cannot take the item of a similar line or label. Items without any text fall back
        to their position, since the prompt asks for script order."""
        texts = [_norm(it.get("text") or it.get("rendered") or "") for it in items]
        pairs = []
        for ei, e in enumerate(expected):
            target = _norm(e["text"])
            for i, text in enumerate(texts):
                if text:
                    score = 1.0 if text == target else difflib.SequenceMatcher(None, text, target).ratio()
                    if score >= 0.5:
                        pairs.append((-score, ei, i))
        matched: list[dict | None] = [None] * len(expected)
        used: set[int] = set()
        for _neg, ei, i in sorted(pairs):      # ties: the earlier expected text, then the earlier item
            if matched[ei] is None and i not in used:
                matched[ei] = items[i]
                used.add(i)
        for i, text in enumerate(texts):
            if not text and i < len(expected) and matched[i] is None:
                matched[i] = items[i]
        return matched

    @staticmethod
    def _panel_results(data: dict, n: int) -> list[dict | None]:
        """Usable panel verdicts for expected panels 1..n, looked up by panel_index (0-based numbering is shifted);
        verdicts numbered otherwise fill the remaining panels in order."""
        res_panels = [p for p in _as_list(data.get("panels")) if isinstance(p, dict)]
        indices = []
        for pos, p in enumerate(res_panels, start=1):
            m = re.search(r"\d+", str(p.get("panel_index", "")))
            indices.append(int(m.group()) if m else pos)
        if indices and sorted(indices) == list(range(len(indices))):
            indices = [i + 1 for i in indices]
        found = [(i, v) for i, v in zip(indices, map(_panel_verdict, res_panels)) if v is not None]
        by_index: dict[int, dict] = {}
        for i, v in found:
            by_index.setdefault(i, v)
        claimed = {id(by_index[k]) for k in range(1, n + 1) if k in by_index}
        spare = [v for _i, v in sorted(found, key=lambda t: t[0]) if id(v) not in claimed]
        return [by_index[k] if k in by_index else (spare.pop(0) if spare else None) for k in range(1, n + 1)]

    def _cast_in(self, text: str, known=()) -> list[str]:
        """Cast members named in ``text``, in order of appearance ("Papi (dog professor)" -> ["papi"]);
        ``known`` adds the panels' own characters."""
        n = _norm(text)
        names = dict.fromkeys(_norm(c) for c in [*self.cast_names, *known] if str(c).strip()) or SPECIES
        hits = [(m.start(), c) for c in names if (m := re.search(rf"\b{re.escape(c)}\b", n))]
        return [c for _pos, c in sorted(hits)]

    def _evaluate(self, data: dict, panels: list[PanelV2]) -> QAResult:
        expected = self._expected(panels)
        results = self._panel_results(data, len(expected))
        if not any(results):
            return _unavailable("judge returned no usable panel verdict", raw=data)
        total = sum(e["kind"] != "label" for p in expected for e in p["expected_text"])
        known = [c for p in panels for c in p.characters]
        m = SCORE_RE.search(SCALE_RE.sub(" ", str(data.get("overall", ""))))
        score = float(m.group(1)) if m else -1.0
        low_score = 0 <= score < self.retry_threshold

        exact = tail_errors = stray = 0
        characters_ok = True
        mismatches: list[dict] = []
        label_misses: list[str] = []
        # feedback order = retry prompt priority: text, then characters, then tails/stray text, then free-form notes
        text_issues: list[str] = []
        char_issues: list[str] = []
        layout_issues: list[str] = []
        notes: list[str] = []
        transcription: list[str] = []
        for p_exp, res in zip(expected, results):
            if res is None:   # the judge described only the other panel of the strip: none of these lines is confirmed
                text_issues += [f"{_who(e)} must read exactly \"{e['text']}\" (panel {p_exp['panel_index']} not found in the image)"
                                for e in p_exp["expected_text"] if e["kind"] != "label"]
                continue
            transcription += res["transcription"]
            # labels take part in the matching (a missing bubble cannot borrow a label's item) but are never scored;
            # an empty text_items list confirms nothing, so only what the judge transcribed can count
            items = res["items"] or [{"rendered": t} for t in res["transcription"]]
            bubble_i = -1
            for e, r in zip(p_exp["expected_text"], self._match_items(p_exp["expected_text"], items)):
                if e["kind"] == "label":
                    if r is not None and not ((_truthy(r.get("found")) and _truthy(r.get("exact"))) or _same_text(e["text"], r.get("rendered"))):
                        label_misses.append(e["text"])
                    continue
                bubble_i += e["kind"] == "bubble"
                if r is None or not ((_truthy(r.get("found")) and _truthy(r.get("exact")))
                                     or (not _falsy(r.get("found")) and _same_text(e["text"], r.get("rendered")))):
                    drawn = r.get("rendered") if r is not None and isinstance(r.get("rendered"), str) else None
                    mismatches.append({"panel": p_exp["panel_index"], "kind": e["kind"], "index": bubble_i if e["kind"] == "bubble" else None,
                                       "expected": e["text"], "judge": drawn if drawn and not _same_text(e["text"], drawn) else None,
                                       "blind": None})
                if r is None:
                    text_issues.append(f"{_who(e)} must read exactly \"{e['text']}\" (not found in the image)")
                elif (_truthy(r.get("found")) and _truthy(r.get("exact"))) or (not _falsy(r.get("found")) and _same_text(e["text"], r.get("rendered"))):
                    exact += 1
                else:
                    if r.get("rendered") and not _same_text(e["text"], r.get("rendered")):   # found=false + copied text: say why
                        shown = f"image shows: {str(r['rendered'])[:120]}"
                    else:
                        shown = str(r.get("note") or "missing or altered")[:120]
                    text_issues.append(f"{_who(e)} must read exactly \"{e['text']}\" ({shown})")
                if r is not None and e["kind"] == "bubble" and _falsy(r.get("speaker_ok")):
                    tail_errors += 1
                    layout_issues.append(f"the tail of {e['speaker']}'s bubble \"{e['text'][:40]}\" must point at {e['speaker']}")
            expected_chars = {_norm(c) for c in p_exp["characters"]}
            for c in res["characters"]:
                named = self._cast_in(str(c.get("name", "")), known)
                name = named[0] if named else None
                note = str(c.get("note") or "").strip()[:120]
                if name in expected_chars:
                    if _falsy(c.get("present")):
                        characters_ok = False
                        char_issues.append(f"{name} is missing from the panel")
                    elif _falsy(c.get("matches_reference")):
                        characters_ok = False
                        ref = "the reference sheet" if name in self.sheets else "its description"
                        char_issues.append(f"{name} does not match {ref} {note}".strip())
                elif name and _truthy(c.get("present")):
                    characters_ok = False
                    char_issues.append(f"remove the unexpected character: {name}")
            for extra in res["extras"]:
                n, named = _norm(extra), self._cast_in(extra, known)
                if named and set(named) <= expected_chars:
                    names = "|".join(map(re.escape, named))
                    head = CLAUSE_RE.split(n, maxsplit=1)[0]     # who it is; a description may follow ("Papi (dog and mentor)")
                    # the judge repeating an expected character ("Papi (dog professor)", "Professor Papi", "Papi and Kumo"),
                    # not "a human child next to Papi", "a second Papi", "Papi's student" or "Kumo and a bird"
                    itself = (re.match(rf"\W*(?:(?:the|a|an)\s+)?(?:\w+\s+)?(?:{names})\b", n) and not DUPLICATE_RE.search(n)
                              and not re.search(rf"\b(?:{names})(?:'s\s+\w|-(?:like|style|looking|shaped)\b)|\b{_COUNT}\s+(?:{names})\b"
                                                rf"|(?:\band|\bplus|&)\s+(?!(?:the\s+)?(?:{names})\b)", head))
                    # or a placeholder whose only exception is the expected cast ("None besides Papi and Kumo")
                    if itself or _is_placeholder(EXCLUDING_RE.sub(" ", re.sub(rf"\b(?:{names})\b", " ", n))):
                        continue
                elif not named and _is_noise(extra):
                    continue
                characters_ok = False
                char_issues.append(f"remove the unexpected character: {extra[:120]}")
            for st in res["stray"]:      # every entry is a problem by definition: no positive-note filter
                stray += 1
                layout_issues.append(f"remove or correct the stray text: {st[:80]}")
            for note in res["notes"]:
                low = _norm(note)
                # a misspelling reported only in issues[] ("nothing is misspelled except 'Transfomer'" reports one too)
                if TYPO_RE.search(NEGATED_RE.sub(" ", low)) or (EXCEPT_RE.search(low) and TYPO_RE.search(low)):
                    stray += 1
                    layout_issues.append(note[:200])
                elif any(TAIL_FAULT_RE.search(c) and not TAIL_NEGATED_RE.search(c)
                         for c in TAIL_CLAUSE_SPLIT.split(QUOTED_LINE_RE.sub(" ", low))):
                    tail_errors += 1
                    layout_issues.append(note[:200])
                else:
                    notes.append(note[:200])
        worst = _entry_text(data.get("worst_problem"))
        if low_score and worst and not _is_noise(worst):     # otherwise a low-score retry gets no specific feedback
            notes.append(worst[:200])

        layout_ok = tail_errors == 0 and stray == 0
        needs_retry = exact < total or not characters_ok or not layout_ok or low_score
        issues = list(dict.fromkeys(text_issues + char_issues + layout_issues + notes))
        return QAResult(score=max(score, 0.0), text_exact=exact, text_total=total, characters_ok=characters_ok, issues=issues,
                        needs_retry=needs_retry, transcription=transcription, raw=data, layout_ok=layout_ok, mismatches=mismatches,
                        low_score=low_score, layout_issues=list(dict.fromkeys(layout_issues)), label_misses=label_misses)
