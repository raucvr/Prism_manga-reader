"""
Tables read from the PDF page geometry, for the evidence ledger (services.ledger), which charts cite by cell.

Deterministic and local: PyMuPDF characters (font, size, baseline) and vector rules only, no ML and no LLM, about a
second per paper. Per table caption:

- Words. Characters become words with ^superscript and _subscript marks, bold and italic flags and a baseline (a mark
  stacked over a subscript is written after it: "44.0 ± 4.0^†"). Rules are thin lines, thin rectangles and hairline
  images; a Word table draws one border segment per cell, and the joints between segments mark merged header cells.
- Region. Captions ("Table 3:", "Table A.1.", "Figure 4" when the figure is a grid of numbers) anchor the tables. The
  region grows from the caption towards the side most captions of the paper use (the other side when that is clearly
  the table) through a chain of horizontal rules and of rows that line up, and stops at body text, notes, another
  caption, a plot or an image. A table that runs off the bottom of a page continues at the top of the next page when
  its rows line up with the columns read so far; a repeated header there is dropped.
- Grid. Rows are clustered by baseline. Columns are the x-extents of the body cells: a wide cell that covers cells of
  other rows is a merged cell, not a column. The header ends at a rule or at the first row of numbers; a header cell
  spans the columns under its cmidrule or between its cell-border joints, or the run of sub-headers it is centred on.
- Rows. Wrapped lines of a label or of a text cell join their row; a label-only row followed by rows with values is a
  section heading, emitted as a row without cells; a parenthesised row under a coefficient is its standard error and
  is folded into it ("0.120*** (0.012)"); leading columns that name the row (W/A, Year, Dataset, a size column that
  tells duplicate names apart) are folded into the label (" | "); a value spanning columns is repeated in each.
- Confidence starts at 1.0 and every structural doubt costs some (no rules, merged cells, values off their column,
  cells holding two numbers, rows squeezed between rows, headers missing or ambiguous). The ledger keeps the tables at
  or above MIN_CONFIDENCE: a value under the wrong row or column is far worse than a table left out.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from functools import cached_property
from pathlib import Path
from typing import Optional, Union

MIN_CONFIDENCE = 0.90

CAPTION_RE = re.compile(r"^\s*(Table|TABLE|Tab\.|Figure|FIGURE|Fig\.|Exhibit|EXHIBIT)\s*"
                        r"((?:[A-Z]\.?)?\d+(?:\.\d+)?[a-z]?|[IVX]+)(?!\d)\s*(.?)")
BOLD_FONT = re.compile(r"(bold|black|heavy|semibold|demibold|cmbx|-bd\b|-b$)", re.I)
# one printed value: 23.75, -0.682***, 1.4 ± 1.1, 2.3 · 10^19, 84.6/83.4, 0.019 (0.021), 4.4 (+0.3), 55^a
NUMERIC_RE = re.compile(
    r"^[\(\[]?[<>≤≥~≈]?\s*[-+−–]?\s*\$?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?\s*%?"
    r"(?:\s*(?:±|\+/-|\+−)\s*\d+(?:\.\d+)?%?){0,3}"
    r"(?:\s?/\s?[-+−–]?\d+(?:\.\d+)?%?)?"
    r"(?:\s*[·×x]\s*10\^?\{?[-−–]?\d+\}?)?"
    r"(?:[eE]\^?[-+−]?\d+)?"
    r"(?:\s*\^\{?[+]\d+(?:\.\d+)?\}?\s*_\{?[-−]\d+(?:\.\d+)?\}?)?"
    r"\s*[\)\]]?\s*(?:\^\{?[^\s{}]{1,6}\}?|[*†‡§¶#]{1,3})?"
    r"(?:\s*\(\s*[-+−–]?\d[\d,]*(?:\.\d+)?%?\s*\)(?:\^?[*†‡]{1,3})?)?$")
NUM_UNIT_RE = re.compile(r"^[~≈<>≤≥]?[-+−]?\$?\d[\d,]*(?:\.\d+)?\s?[A-Za-z%×µ]{1,4}\.?$")   # 7B, 2.5M, 110 GeV
PAREN_NUM = re.compile(r"^[\(\[]\s*[-−–]?\d[\d,]*(?:\.\d+)?%?\s*[\)\]](?:\^?[*†‡]+)?$")    # (0.012): a standard error
COLNUM_RE = re.compile(r"^\(\d{1,2}\)$")                                               # (3): a model number
NOTE_START = re.compile(r"^(Notes?\b|Source[s]?:|\*|†|‡|§|¶|\^)", re.I)
PANEL_RE = re.compile(r"^\s*Panel\s+[A-Z0-9]+\b", re.I)
DASHES = {"-", "–", "—", "−", "--", "n/a", "N/A", "na", "NA", "–/–", "-/-", "×", "✗", "✓"}
# spacing accents and their combining forms
ACCENTS = {"ˆ": "\u0302", "˜": "\u0303", "¯": "\u0304", "˙": "\u0307", "¨": "\u0308", "´": "\u0301", "`": "\u0300"}
TABLE_LINE = re.compile(r"\s*(?:Table|TABLE|Tab\.)\s*[A-Z]?\.?\d")


def _is_numeric(s: str) -> bool:
    s = s.strip()
    if s.count("(") != s.count(")") or s.count("[") != s.count("]"):
        return False   # "(2016-", "2021)": pieces of a wrapped header
    return bool(s) and (s in DASHES or (any(ch.isdigit() for ch in s) and bool(NUMERIC_RE.match(s))))


def _value_like(s: str) -> bool:
    s = s.strip()
    return bool(s) and (_is_numeric(s) or bool(NUM_UNIT_RE.match(s)))


def _numberish(s: str) -> bool:
    """Looser than a value: mostly digits ("84.6/83.4", "110–600", "1.4 ± 1.1 (stat)")."""
    s = s.strip()
    if not s:
        return False
    if _is_numeric(s):
        return True
    digits = sum(ch.isdigit() for ch in s)
    return digits >= 1 and digits >= 2 * sum(ch.isalpha() for ch in s)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


# ----------------------------------------------------------------------------- words

class _Tok:
    """A word: text with ^/_ script marks; vertically its box is the em box around the baseline."""
    __slots__ = ("text", "x0", "y0", "x1", "y1", "base", "size", "bold", "italic", "small", "lid", "n")

    def __init__(self, text, x0, y0, x1, y1, base, size, bold=False, italic=False, small=False, lid=0, n=1):
        self.text, self.x0, self.y0, self.x1, self.y1 = text, x0, y0, x1, y1
        self.base, self.size, self.bold, self.italic, self.small, self.lid, self.n = base, size, bold, italic, small, lid, n

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    def moved(self, dy: float) -> "_Tok":
        return _Tok(self.text, self.x0, self.y0 + dy, self.x1, self.y1 + dy, self.base + dy, self.size, self.bold,
                    self.italic, self.small, self.lid, self.n)


def _span_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or bool(span.get("char_flags", 0) & 8) or bool(BOLD_FONT.search(span["font"]))


def _script_text(chars: list[dict]) -> str:
    """The text of a word with ^ / _ before superscript / subscript runs ("10^{19}", "N_D"). A superscript stacked over
    the subscript that follows it is written after it, and a subscript "±0.2" is the spread of the value before it:
    "44.9_{±0.2}^†" is written "44.9 ± 0.2^†". An accent drawn over the next letter follows it ("ˆµ" is "µ̂")."""
    chars = list(chars)
    for i in range(len(chars) - 1):
        a, b = chars[i], chars[i + 1]
        if a["c"] in ACCENTS and b["c"].isalpha() and b["x0"] < a["x1"] - 0.2:
            chars[i], chars[i + 1] = b, dict(a, c=ACCENTS[a["c"]], role=b["role"])
    runs: list[list] = []
    for c in chars:
        if runs and runs[-1][0] == c["role"]:
            runs[-1][1].append(c)
        else:
            runs.append([c["role"], [c]])
    for i in range(len(runs) - 1):
        (r1, a), (r2, b) = runs[i], runs[i + 1]
        if r1 == "sup" and r2 == "sub" and b[0]["x0"] < a[-1]["x1"] - 0.1:
            runs[i], runs[i + 1] = runs[i + 1], runs[i]
    out = []
    for i, (role, cs) in enumerate(runs):
        seg = "".join(c["c"] for c in cs)
        if role == "n":
            out.append(seg)
        elif role == "sub" and seg.startswith("±"):
            out.append(" ± " + seg[1:].strip())
        else:
            mark = "^" if role == "sup" else "_"
            out.append(mark + ("{" + seg + "}" if i + 1 < len(runs) and len(seg) > 1 else seg))
    return "".join(out)


def _page_words(page) -> tuple[list[_Tok], list[dict], int]:
    """(words, horizontal and rotated text lines for the captions, number of rotated lines) of a page."""
    import pymupdf

    raw = page.get_text("rawdict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP)
    toks: list[_Tok] = []
    lines: list[dict] = []
    rotated = 0
    lid = 0
    for b in raw["blocks"]:
        if b.get("type") != 0:
            continue
        for lno, ln in enumerate(b["lines"]):
            chars = []
            for s in ln["spans"]:
                bold, italic, sup = _span_bold(s), bool(s["flags"] & 2), bool(s["flags"] & 1)
                for c in s["chars"]:
                    ch = c["c"]
                    if ch in ("​", "﻿", "�") or "" <= ch <= "":
                        continue
                    x0, y0, x1, y1 = c["bbox"]
                    chars.append({"c": ch, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "oy": c["origin"][1],
                                  "size": s["size"], "bold": bold, "italic": italic, "sup": sup})
            if not chars:
                continue
            text = "".join(c["c"] for c in chars).strip()
            dx, dy = ln["dir"]
            if not (abs(dy) < 0.05 and dx > 0):
                rotated += 1
                lines.append({"text": text, "bbox": tuple(ln["bbox"]), "size": max(c["size"] for c in chars),
                              "bold": chars[0]["bold"], "line": lno, "rot": True, "spans": ln["spans"]})
                continue
            ink = [c for c in chars if not c["c"].isspace()]
            if not ink:
                continue
            size = max(c["size"] for c in ink)
            base = statistics.median(c["oy"] for c in ink if c["size"] >= 0.9 * size)
            for c in chars:
                c["role"] = "n"
                if not c["c"].isspace() and c["size"] < 0.88 * size:
                    d = base - c["oy"]
                    if d > 0.18 * size or (c["sup"] and d > 0.05 * size):
                        c["role"] = "sup"
                    elif -d > 0.1 * size:
                        c["role"] = "sub"
            lines.append({"text": text, "bbox": tuple(ln["bbox"]), "size": size, "bold": ink[0]["bold"], "line": lno,
                          "rot": False, "spans": ln["spans"]})
            word: list[dict] = []
            prev = None
            for c in chars + [None]:
                if c is not None and not c["c"].isspace() and not (
                        prev is not None and (c["x0"] - prev["x1"] > 0.3 * size or c["x0"] - prev["x1"] < -0.5 * size)):
                    word.append(c)
                    prev = c
                    continue
                if word:
                    toks.append(_word(word, lid))
                word, prev = ([c], c) if c is not None and not c["c"].isspace() else ([], None)
            lid += 1
    return _attach_scripts(_dedupe(toks)), lines, rotated


def _word(chars: list[dict], lid: int) -> _Tok:
    """A word from its characters; size, box and style come from its normal (not script) characters."""
    normal = [c for c in chars if c["role"] == "n"] or chars
    size = max(c["size"] for c in normal)
    body = [c for c in normal if c["size"] >= 0.9 * size] or normal
    base = statistics.median(c["oy"] for c in body)
    y0 = max(min(c["y0"] for c in body), base - 0.85 * size)
    y1 = min(max(c["y1"] for c in body), base + 0.3 * size)
    return _Tok(_script_text(chars), min(c["x0"] for c in chars), y0, max(c["x1"] for c in chars), y1, base, size,
                bold=sum(c["bold"] for c in normal) * 2 > len(normal), italic=sum(c["italic"] for c in normal) * 2 > len(normal),
                small=all(c["role"] != "n" for c in chars), lid=lid, n=len(chars))


def _dedupe(toks: list[_Tok]) -> list[_Tok]:
    """Overprinted duplicates (a poor man's bold) are dropped; the word that stays is bold."""
    toks.sort(key=lambda t: (round(t.base), t.x0))
    out, seen = [], defaultdict(list)
    for t in toks:
        k = round(t.base / 2)
        dup = next((o for kk in (k, k - 1, k + 1) for o in seen[(t.text, kk)]
                    if abs(o.x0 - t.x0) < 1.2 and abs(o.base - t.base) < 1.2), None)
        if dup is not None:
            dup.bold = True
            continue
        seen[(t.text, k)].append(t)
        out.append(t)
    return out


def _attach_scripts(toks: list[_Tok]) -> list[_Tok]:
    """A free-standing script word joins the word right before it: MuPDF may start the next cell's line with a
    superscript, or put a mark ("†") on a line of its own, where it is not smaller than its line."""
    def mark(s, t):
        return len(s.text) <= 3 and s.size < 0.85 * t.size and s.base < t.base - 0.15 * t.size

    keep = [t for t in toks if not t.small]
    for s in [t for t in toks if t.small] + [t for t in keep if len(t.text) <= 3]:
        best = None
        for t in keep:
            gap = s.x0 - t.x1
            if t is not s and -1.0 <= gap <= 1.8 and t.y0 - 0.6 * t.size <= s.yc <= t.y1 + 0.3 * t.size and (s.small or mark(s, t)):
                if best is None or abs(gap) < abs(s.x0 - best.x1):
                    best = t
        if best is None:
            if s.small:
                keep.append(s)
            continue
        if not s.small:
            keep.remove(s)
        text = s.text if s.text[:1] in "^_" else ("^" if s.base < best.base - 0.15 * best.size else "_") + s.text
        best.text += text
        best.x1 = max(best.x1, s.x1)
    keep.sort(key=lambda t: (t.lid, t.x0))
    return keep


# ----------------------------------------------------------------------------- rules and pages

class _Rule:
    """A horizontal (or vertical) rule at ``c`` from ``a`` to ``b``; ``joints`` are where its segments meet."""
    __slots__ = ("c", "a", "b", "w", "joints")

    def __init__(self, c, a, b, w, joints=()):
        self.c, self.a, self.b, self.w, self.joints = c, a, b, w, list(joints)

    @property
    def length(self) -> float:
        return self.b - self.a


def _white(color) -> bool:
    return color is not None and len(color) >= 3 and min(color[:3]) > 0.92


def _page_rules(page):
    """(horizontal rules, vertical rules, non-rule graphics with their path counts, images, every shape) of a page."""
    H, V, graphics, images, shapes = [], [], [], [], []
    try:
        drawings = page.get_cdrawings()   # plain tuples: much faster than get_drawings() on pages with plots
    except Exception:  # noqa: BLE001 - a broken content stream: no rules
        drawings = []
    for p in drawings:
        w, stroke, fill, kind = p.get("width") or 0.0, p.get("color"), p.get("fill"), p.get("type") or ""
        stroked, filled = "s" in kind, "f" in kind
        if (stroked and _white(stroke) and (not filled or _white(fill))) or (filled and not stroked and _white(fill)):
            continue
        other = 0
        for it in p["items"]:
            if it[0] == "l":
                (ax, ay), (bx, by) = it[1], it[2]
                if abs(ay - by) < 0.8 and abs(ax - bx) > 2:
                    H.append(_Rule((ay + by) / 2, min(ax, bx), max(ax, bx), max(w, 0.1)))
                elif abs(ax - bx) < 0.8 and abs(ay - by) > 2:
                    V.append(_Rule((ax + bx) / 2, min(ay, by), max(ay, by), max(w, 0.1)))
                else:
                    other += 1
            elif it[0] in ("re", "qu"):
                pts = ((it[1][0], it[1][1]), (it[1][2], it[1][3])) if it[0] == "re" else it[1]
                x0, x1 = min(x for x, _ in pts), max(x for x, _ in pts)
                y0, y1 = min(y for _, y in pts), max(y for _, y in pts)
                rw, rh = x1 - x0, y1 - y0
                if rh > 2.6 and rw > 2.6:
                    shapes.append((x0, y0, x1, y1))
                if filled and not stroked and fill is not None and min(fill[:3]) > 0.6 and rh > 3 and rw > 3:
                    continue   # light shading
                if rh < 2.6 and rw > 2:
                    H.append(_Rule((y0 + y1) / 2, x0, x1, max(rh, w, 0.1)))
                elif rw < 2.6 and rh > 2:
                    V.append(_Rule((x0 + x1) / 2, y0, y1, max(rw, w, 0.1)))
                elif stroked:
                    H += [_Rule(y0, x0, x1, max(w, 0.1)), _Rule(y1, x0, x1, max(w, 0.1))]
                    V += [_Rule(x0, y0, y1, max(w, 0.1)), _Rule(x1, y0, y1, max(w, 0.1))]
                else:
                    other += 1
            else:
                other += 1
                shapes.append(tuple(p["rect"]))
        if other:
            graphics.append((tuple(p["rect"]), other))
    try:
        for info in page.get_image_info():
            x0, y0, x1, y1 = info["bbox"]
            if y1 - y0 < 2.6 and x1 - x0 > 2:
                H.append(_Rule((y0 + y1) / 2, x0, x1, y1 - y0))      # hairline images drawn as rules
            elif x1 - x0 < 2.6 and y1 - y0 > 2:
                V.append(_Rule((x0 + x1) / 2, y0, y1, x1 - x0))
            else:
                images.append((x0, y0, x1, y1))
    except Exception:  # noqa: BLE001
        pass
    return _merge_rules(H), _merge_rules(V), graphics, images, shapes


def _merge_rules(rules: list[_Rule], ctol: float = 0.9, gap: float = 1.5) -> list[_Rule]:
    """Collinear segments become one rule (segments that merely touch leave a joint: a Word cell border); a double
    rule becomes one thicker rule."""
    out: list[_Rule] = []
    for r in sorted(rules, key=lambda r: (round(r.c * 2) / 2, r.a)):
        for o in reversed(out[-12:]):
            if abs(o.c - r.c) <= ctol and r.a <= o.b + gap and r.b >= o.a - gap:
                if abs(r.a - o.b) <= gap and r.a >= o.b - 0.6:
                    o.joints.append((r.a + o.b) / 2)
                elif abs(o.a - r.b) <= gap and o.a >= r.b - 0.6:
                    o.joints.append((o.a + r.b) / 2)
                o.joints += r.joints
                o.a, o.b, o.w = min(o.a, r.a), max(o.b, r.b), max(o.w, r.w)
                break
        else:
            out.append(_Rule(r.c, r.a, r.b, r.w, r.joints))
    res: list[_Rule] = []
    for r in sorted(out, key=lambda r: (r.c, r.a)):
        if res and abs(res[-1].c - r.c) < 2.6 and _overlap(res[-1].a, res[-1].b, r.a, r.b) > 0.8 * min(res[-1].length, r.length):
            res[-1].w = max(res[-1].w, r.w) + 0.5
            res[-1].a, res[-1].b = min(res[-1].a, r.a), max(res[-1].b, r.b)
            continue
        res.append(r)
    return res


def _vector_bytes(page) -> int:
    """The size of a page's content streams, its form XObjects included: millions of paths make a plot."""
    try:
        doc = page.parent
        n = len(page.read_contents())
        for xref, *_ in page.get_xobjects():
            kind, length = doc.xref_get_key(xref, "Length")
            n += int(length) if kind == "int" else 0
        return n
    except Exception:  # noqa: BLE001
        return 0


class _Page:
    """The words of a page; its rules and graphics are read when first needed (pages near a caption)."""

    def __init__(self, page, pno: int):
        self.page, self.pno = page, pno
        self.width, self.height = page.rect.width, page.rect.height
        self.toks, self.lines, self.rotated = _page_words(page)

    @cached_property
    def _rules(self):
        if not any(TABLE_LINE.match(ln["text"]) for ln in self.lines) and _vector_bytes(self.page) > 1e6:
            return [], [], [((0.0, 0.0, self.width, self.height), 10**6)], [], []   # a big plot and no table: paths unread
        return _page_rules(self.page)

    H = property(lambda self: self._rules[0])
    V = property(lambda self: self._rules[1])
    graphics = property(lambda self: self._rules[2])
    images = property(lambda self: self._rules[3])
    shapes = property(lambda self: self._rules[4])

    @cached_property
    def words(self) -> list[_Tok]:
        """The words, two words of a line joined over an underscore drawn as a rule (typewriter "\\_": "intro_syuz")."""
        tiny = [h for h in self.H if h.length < 8]
        out: list[_Tok] = []
        for t in self.toks:
            p = out[-1] if out else None
            if p is not None and p.lid == t.lid and 0.5 < t.x0 - p.x1 < 0.8 * t.size and any(
                    p.x1 - 1 <= h.a and h.b <= t.x0 + 1 and h.length >= 0.5 * (t.x0 - p.x1)
                    and p.base - 0.15 * p.size <= h.c <= p.base + 0.4 * p.size for h in tiny):
                out[-1] = _Tok(p.text + "_" + t.text, p.x0, min(p.y0, t.y0), t.x1, max(p.y1, t.y1), p.base, p.size, p.bold,
                               p.italic, p.small, p.lid, p.n + t.n + 1)
                continue
            out.append(t)
        return out


def _layout(pages: list[_Page]) -> dict:
    """Body font size, text block (L, R, T, B), its middle and whether the paper is set in two columns."""
    sizes: Counter = Counter()
    for pm in pages:
        for t in pm.toks:
            if not t.small:
                sizes[round(t.size * 2) / 2] += t.n
    body = sizes.most_common(1)[0][0] if sizes else 10.0
    boxes = [ln["bbox"] for pm in pages for ln in pm.lines
             if not ln["rot"] and abs(ln["size"] - body) < 0.6 and ln["bbox"][2] - ln["bbox"][0] > 100]
    wide = [b for b in boxes if b[2] - b[0] > 120]
    if wide:
        xs0, xs1 = sorted(b[0] for b in wide), sorted(b[2] for b in wide)
        L, R = xs0[int(0.05 * len(xs0))], xs1[int(0.95 * (len(xs1) - 1))]
    else:
        L, R = 72, 540
    mid = (L + R) / 2
    left = sum(1 for b in boxes if b[2] < mid + 8)
    right = sum(1 for b in boxes if b[2] >= mid + 8 and b[0] > mid - 8)
    full = len(boxes) - left - right
    twocol = (left + right) > 1.2 * full and min(left, right) > 0.25 * (left + right)
    if boxes:
        ys0, ys1 = sorted(b[1] for b in boxes), sorted(b[3] for b in boxes)
        T, B = ys0[int(0.02 * (len(ys0) - 1))], ys1[int(0.98 * (len(ys1) - 1))]
    else:
        T, B = 72, 720
    return {"body": body, "L": L, "R": R, "mid": mid, "twocol": twocol, "T": T, "B": B}


# ----------------------------------------------------------------------------- captions

def _find_captions(pages: list[_Page], lay: dict) -> list[dict]:
    """The caption of every table (and figure) label: the best scoring line that starts with it ("Table 2:" at the
    start of a block, in the caption font, beats "Table 2 summarizes ..." in the text)."""
    cands = []
    for pm in pages:
        lines = [ln for ln in pm.lines if not ln["rot"]]
        for i, ln in enumerate(lines):
            m = CAPTION_RE.match(ln["text"])
            if not m:
                continue
            word, num, sep = m.group(1), m.group(2), m.group(3)
            kind = "table" if word.lower().startswith("tab") else ("exhibit" if word.lower().startswith("ex") else "figure")
            score = {":": 3, "|": 3, ".": 2, "—": 2, "–": 2, "-": 2, "": 1.5}.get(sep, -3 if sep == "(" and kind == "figure" else -4)
            if ln["size"] < lay["body"] - 0.3:
                score += 1
            elif ln["size"] > lay["body"] + 0.3:
                score += 0.5
            score += bool(ln["bold"]) + (ln["line"] == 0)
            label = {"tab": "Table", "fig": "Figure", "exh": "Exhibit"}[word[:3].lower()] + " " + num
            cands.append({"kind": kind, "num": num, "label": label, "sep": sep, "score": score, "page": pm.pno,
                          "bbox": list(ln["bbox"]), "size": ln["size"], "lidx": i})
    seps = Counter(c["sep"] for c in cands if c["score"] >= 3)
    if seps:   # the paper's own caption separator
        top = seps.most_common(1)[0][0]
        for c in cands:
            if c["sep"] not in (top, ""):
                c["score"] -= 1
    best: dict = {}
    for c in cands:
        key = (c["kind"], c["num"])
        if key not in best or c["score"] > best[key]["score"]:
            best[key] = c
    caps = [c for c in best.values() if c["score"] >= 2]
    for c in caps:
        c["text"], c["bbox"] = _caption_text(pages[c["page"]], c)
    caps.sort(key=lambda c: (c["page"], c["bbox"][1]))
    return caps


def _line_base(ln: dict) -> float:
    spans = [s for s in ln["spans"] if "".join(ch["c"] for ch in s.get("chars", [])).strip() or s.get("text", "").strip()]
    if not spans:
        return ln["bbox"][3]
    size = max(s["size"] for s in spans)
    return max(s["origin"][1] for s in spans if s["size"] >= 0.9 * size)


def _caption_text(pm: _Page, cap: dict) -> tuple[str, list[float]]:
    """The whole caption (its wrapped lines, by baseline pitch and alignment) and its box."""
    lines = [ln for ln in pm.lines if not ln["rot"]]
    first = lines[cap["lidx"]]
    texts, bb, size, base = [first["text"]], list(first["bbox"]), first["size"], _line_base(first)
    for ln in lines:   # the rest of the first line set as a line object of its own
        if ln is not first and abs(_line_base(ln) - base) < 1.5 and 0 <= ln["bbox"][0] - bb[2] < 15 \
                and abs(ln["size"] - size) < 0.6:
            texts.append(ln["text"])
            bb[2] = max(bb[2], ln["bbox"][2])
    x0, pitch = bb[0], None
    for _ in range(40):
        lim = (pitch + max(1.5, 0.15 * size)) if pitch is not None else 1.35 * size
        cands = [(_line_base(ln), ln) for ln in lines
                 if base + 0.6 * size <= _line_base(ln) <= base + lim and abs(ln["size"] - size) <= 0.7
                 and not (ln["bbox"][2] < bb[0] - 2 or ln["bbox"][0] > bb[2] + 2)]
        if not cands:
            break
        nb = min(c[0] for c in cands)
        same = sorted([ln for lb, ln in cands if abs(lb - nb) < 1.5], key=lambda ln: ln["bbox"][0])
        if any(base < h.c < nb - 0.5 * size and _overlap(h.a, h.b, bb[0], bb[2]) > 20 for h in pm.H):
            break   # a rule between two lines ends the caption
        if CAPTION_RE.match(same[0]["text"]):
            break
        sx0, sx1 = same[0]["bbox"][0], max(ln["bbox"][2] for ln in same)
        if not (abs(sx0 - x0) < 4 or abs((sx0 + sx1) / 2 - (bb[0] + bb[2]) / 2) < 6):
            break
        gaps = [b["bbox"][0] - a["bbox"][2] for a, b in zip(same, same[1:])]
        if len(same) >= 3 and sum(1 for g in gaps if g > 1.5 * size) >= 2:
            break   # a table row, not caption text
        pitch = nb - base if pitch is None else pitch
        texts.append(" ".join(ln["text"] for ln in same))
        bb[0], bb[2] = min(bb[0], *(ln["bbox"][0] for ln in same)), max(bb[2], *(ln["bbox"][2] for ln in same))
        bb[3] = max(bb[3], *(ln["bbox"][3] for ln in same))
        base = nb
    text = ""
    for t in (t.strip() for t in texts):
        text = text[:-1] + t if text.endswith("-") and t[:1].islower() else (text + " " + t).strip()
    return text, bb


# ----------------------------------------------------------------------------- rows and cells

class _Row:
    """One visual row: words sharing a baseline."""

    def __init__(self, toks: list[_Tok]):
        self.toks = sorted(toks, key=lambda t: t.x0)
        self.chunks: list[_Chunk] = []
        self.update()

    def update(self) -> None:
        ts = [t for t in self.toks if not t.small] or self.toks
        self.x0, self.x1 = min(t.x0 for t in self.toks), max(t.x1 for t in self.toks)
        self.y0, self.y1 = min(t.y0 for t in ts), max(t.y1 for t in ts)
        self.base, self.size = statistics.median(t.base for t in ts), statistics.median(t.size for t in ts)

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.toks)


def _cluster_rows(toks: list[_Tok]) -> list[_Row]:
    """Words into rows by baseline (within 0.38 em); script-only words join the nearest row."""
    rows: list[_Row] = []
    for t in sorted((t for t in toks if not t.small), key=lambda t: t.base):
        for r in reversed(rows[-3:]):
            if abs(t.base - r.base) <= 0.38 * min(t.size, r.size):
                r.toks.append(t)
                r.base = statistics.median(x.base for x in r.toks)
                r.size = statistics.median(x.size for x in r.toks)
                break
        else:
            rows.append(_Row([t]))
    for r in rows:
        r.toks.sort(key=lambda t: t.x0)
        r.update()
    for t in toks:   # script-only words join the nearest row
        if not t.small:
            continue
        near = [(abs(t.yc - r.yc), i) for i, r in enumerate(rows) if r.y0 - 0.5 * r.size <= t.yc <= r.y1 + 0.4 * r.size]
        if near:
            r = rows[min(near)[1]]
            r.toks.append(t)
            r.toks.sort(key=lambda x: x.x0)
        else:
            rows.append(_Row([t]))
    rows.sort(key=lambda r: r.base)
    return rows


class _Chunk:
    """The words of one cell of a row."""

    def __init__(self, toks: list[_Tok]):
        self.toks = toks
        ts = [t for t in toks if not t.small] or toks
        self.x0, self.x1 = min(t.x0 for t in toks), max(t.x1 for t in toks)
        self.y0, self.y1, self.size = min(t.y0 for t in ts), max(t.y1 for t in ts), max(t.size for t in ts)
        parts: list[str] = []
        for i, t in enumerate(toks):
            if parts and t.small and t.text[:1] in "^_":
                parts[-1] += t.text
            elif parts and t.text in ACCENTS and t.x0 < toks[i - 1].x1:
                parts[-1] += ACCENTS[t.text]   # an accent drawn over the letter before it: "β ˆ" is "β̂"
            else:
                parts.append(t.text)
        self.text = " ".join(parts)
        weight = sum(len(t.text) for t in ts)
        self.bold = sum(len(t.text) for t in ts if t.bold) * 2 > weight
        if not self.bold and ts[0].bold and _is_numeric(ts[0].text) and len(ts) > 1:
            self.bold = True   # a bold best value followed by "(mean±std)" in regular weight
        self.italic = sum(len(t.text) for t in ts if t.italic) * 2 > weight

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2


def _chunks(toks: list[_Tok], vx=(), factor: float = 0.5) -> list[_Chunk]:
    """Words grouped into cells: a gap wider than ``factor`` em, or a vertical rule (at ``vx``), separates cells."""
    groups: list[list[_Tok]] = []
    for t in sorted(toks, key=lambda t: t.x0):
        if groups:
            g = groups[-1]
            gx1, gsize = max(x.x1 for x in g), max(x.size for x in g)
            gap = t.x0 - gx1
            cross = any(gx1 - 0.3 <= v <= t.x0 + 0.3 for v in vx)
            if not cross and ((t.small and gap < 2.0) or gap <= factor * min(t.size, gsize)
                              or (t.size < 0.85 * gsize and gap <= 0.8 * gsize and t.text[:1] in "([±" and _is_numeric(g[-1].text))):
                g.append(t)   # the last case: "4.4 (+0.3)" with the delta in a smaller font
                continue
        groups.append([t])
    return [_Chunk(g) for g in groups]


# ----------------------------------------------------------------------------- the region of a table

def _caption_extent(cap: dict, lay: dict, pm: _Page) -> tuple[float, float]:
    """The x-range a table next to this caption may occupy: its text column, or the caption's own width when the
    caption is set in a narrow box (minipage, wrapfigure)."""
    x0, y0, x1, y1 = cap["bbox"]
    L, R, mid = lay["L"], lay["R"], lay["mid"]
    if lay["twocol"] and x1 <= mid + 12:
        col = (min(L, x0) - 4, mid + 6)
    elif lay["twocol"] and x0 >= mid - 12:
        col = (mid - 6, max(R, x1) + 4)
    else:
        col = (min(L, x0) - 4, max(R, x1) + 4)
    if y1 - y0 > 1.6 * cap["size"] and x1 - x0 < 0.8 * (col[1] - col[0]):
        return x0 - 6, x1 + 6
    if not lay["twocol"] or col[1] - col[0] > 0.7 * (R - L):   # tables may be wider than the text block
        col = (min(col[0], 0.5 * col[0]), max(col[1], pm.width - 0.5 * (pm.width - col[1])))
    return col


def _wordy(c: _Chunk) -> tuple[int, bool]:
    words = [t for t in c.toks if any(ch.isalpha() for ch in t.text) and not any(ch.isdigit() for ch in t.text)]
    return len(words), len(words) >= 0.7 * len(c.toks)


def _value_chunks(row: _Row) -> int:
    return sum(1 for c in _chunks(row.toks, (), 0.8) if _value_like(c.text) or c.text in DASHES)


def _body_like(row: _Row, lay: dict, ext_w: float) -> bool:
    """A line of running text (six or more plain words across most of the column)."""
    if row.size < lay["body"] - 0.45 or _value_chunks(row) >= 2:
        return False
    for c in _chunks(row.toks, (), 0.8):
        n, mostly = _wordy(c)
        if n >= 6 and mostly and c.x1 - c.x0 >= min(150, 0.5 * ext_w):
            return True
    return False


def _note_like(row: _Row, ext_w: float) -> bool:
    if NOTE_START.match(row.text.strip()):
        return True
    cs = _chunks(row.toks, (), 0.8)
    if len(cs) == 1:
        n, mostly = _wordy(cs[0])
        return n >= 8 and mostly and cs[0].x1 - cs[0].x0 >= 0.55 * ext_w
    return False


def _table_like(row: _Row, lay: dict) -> bool:
    """Two cells or more, or a short line with a number or in a small font."""
    if len(_chunks(row.toks, (), 0.6)) >= 2:
        return True
    if len(row.toks) <= 6 and (any(any(ch.isdigit() for ch in t.text) for t in row.toks) or row.size < lay["body"] - 0.45):
        return True
    return False


def _in_graphics(pm: _Page, r: _Row, pad: float = 3.0) -> bool:
    """The row overlaps a plot (non-rule vector paths) or an image."""
    for (x0, y0, x1, y1), k in pm.graphics:
        if k >= 1 and x0 - pad < r.x1 and x1 + pad > r.x0 and y0 - pad < r.y1 and y1 + pad > r.y0 and (x1 - x0) * (y1 - y0) > 30:
            return True
    return any(x0 - pad < r.x1 and x1 + pad > r.x0 and y0 - pad < r.y1 and y1 + pad > r.y0 for x0, y0, x1, y1 in pm.images)


def _pitch(rows: list[_Row], lay: dict) -> float:
    ys = sorted(r.base for r in rows)
    d = [b - a for a, b in zip(ys, ys[1:]) if b - a > 1]
    return statistics.median(d) if d else 1.3 * lay["body"]


def _aligned(row: _Row, accepted: list[_Row], need_value: bool = False) -> bool:
    """The row's cells line up (left, right or centre) with cells of the rows already accepted."""
    anchors, vanchors = [], []
    for r in accepted:
        for c in _chunks(r.toks, (), 0.5):
            anchors += (c.x0, c.x1, c.xc)
            if _value_like(c.text) or c.text in DASHES:
                vanchors += (c.x0, c.x1, c.xc)
    cs = _chunks(row.toks, (), 0.5)
    if len(cs) < 2:
        return False
    hit = sum(1 for c in cs if any(abs(c.x0 - a) < 2.5 or abs(c.x1 - a) < 2.5 or abs(c.xc - a) < 3.5 for a in anchors))
    vhit = sum(1 for c in cs if (_value_like(c.text) or c.text in DASHES)
               and any(abs(c.x0 - a) < 2.5 or abs(c.x1 - a) < 2.5 or abs(c.xc - a) < 2.5 for a in vanchors))
    if need_value and vanchors and vhit == 0 and not (hit >= 3 and hit >= 0.8 * len(cs)):
        return False
    return hit >= 2 and hit >= 0.4 * len(cs)


def _find_region(pm: _Page, cap: dict, side: str, lay: dict, caps_page: list[dict], extent=None,
                 context: Optional[list[_Row]] = None) -> Optional[dict]:
    """The rows of the table on one side of a caption: first the chain of long rules that starts near the caption
    (stopping where body text, a plot, an image or another caption comes between two rules), then the rows past it
    while they stay close and table-like. ``context``: the rows of the part above, for a continuation page. None
    when nothing table-like is there."""
    xa, xb = extent if extent else _caption_extent(cap, lay, pm)
    ext_w = xb - xa
    cx0, cy0, cx1, cy1 = cap["bbox"]
    toks = [t for t in pm.words if _overlap(t.x0, t.x1, xa, xb) >= 0.5 * max(0.1, t.x1 - t.x0)
            and not (cy0 - 1 <= t.yc <= cy1 + 1 and _overlap(t.x0, t.x1, cx0, cx1) > 0)]
    others = [c for c in caps_page if c is not cap]
    rows = _cluster_rows(toks)
    hr = [h for h in pm.H if _overlap(h.a, h.b, xa, xb) >= 0.6 * h.length]
    below = side == "below"
    edge = cy1 if below else cy0
    if below:
        cand = [r for r in rows if r.y0 >= edge - 1.5]
        hs = sorted((h for h in hr if h.c >= edge - 1), key=lambda h: h.c)
    else:
        cand = sorted((r for r in rows if r.y1 <= edge + 1.5), key=lambda r: -r.y1)
        hs = sorted((h for h in hr if h.c <= edge + 1), key=lambda h: -h.c)

    def dist(y):
        return y - edge if below else edge - y

    def near(r):
        return r.y0 if below else r.y1

    def far(r):
        return r.y1 if below else r.y0

    def hits_caption(r):
        return any(_overlap(r.y0, r.y1, c["bbox"][1], c["bbox"][3]) > 0.5 and _overlap(r.x0, r.x1, c["bbox"][0], c["bbox"][2]) > 0
                   for c in others)

    env: list[_Rule] = []
    last = edge
    for h in (h for h in hs if h.length >= 0.3 * ext_w or h.length >= 120):
        d = dist(h.c) - dist(last) if env else dist(h.c)
        lo, hi = sorted((last, h.c))
        between = [r for r in cand if lo - 0.5 <= r.yc <= hi + 0.5]
        if not env and d > 70:
            break
        if env and d > 330 and not (sum(_table_like(r, lay) for r in between) >= 0.5 * len(between) and all(
                not _body_like(r, lay, ext_w) and (_table_like(r, lay) or len(r.toks) <= 6) for r in between)):
            break   # a rule far down closes the table only when table rows and short lines are all that is between
        if env:   # between two rules: a plot frame or a picture means the table has ended
            gs = [rect for rect, k in pm.graphics if lo < (rect[1] + rect[3]) / 2 < hi and xa < (rect[0] + rect[2]) / 2 < xb]
            garea = sum(min((r[2] - r[0]) * (r[3] - r[1]), (xb - xa) * (hi - lo)) for r in gs)
            if (len(gs) >= 3 and garea > 0.08 * (xb - xa) * max(1.0, hi - lo)) or len(gs) >= 40 or \
                    any(im[1] < hi and im[3] > lo and im[0] < xb and im[2] > xa for im in pm.images):
                break
            rotated = any(ln["rot"] and lo < (ln["bbox"][1] + ln["bbox"][3]) / 2 < hi and xa < (ln["bbox"][0] + ln["bbox"][2]) / 2 < xb
                          for ln in pm.lines)
            if hi - lo > 6 and not rotated and not between:
                break   # two rules far apart with nothing between them: not one table
        if any(hits_caption(r) for r in between) or (env and any(_note_like(r, ext_w) for r in between)):
            break   # notes under the table's last rule, closed by a rule of their own
        nbody = sum(1 for r in between if _body_like(r, lay, ext_w))
        nval = sum(1 for r in between if _value_chunks(r) >= 2)
        if nbody and (nval < 2 or nbody > 0.8 * len(between)):
            break
        if not env and between and not all(_table_like(r, lay) for r in between):
            break
        env.append(h)
        last = h.c
    env_far = env[-1].c if env else None
    # booktabs: a thick last rule after thinner middle rules closes the table
    closed = len(env) >= 2 and env[-1].w >= 0.9 * max(h.w for h in env) and env[-1].w > min(h.w for h in env) + 0.15
    accepted: list[_Row] = []
    # lines that may be wrapped cells (small font, or a short label line): kept only when a table row follows
    pending: list[_Row] = []

    def label_line(r):   # one short text cell inside a text column of the rows so far
        cs = _chunks(r.toks, (), 0.8)
        if len(cs) != 1 or len(r.toks) > 6 or _numberish(cs[0].text):
            return False
        texts = [[d for d in _chunks(a.toks, (), 0.8) if not _numberish(d.text)] for a in accepted + (context or [])]
        return sum(1 for ds in texts if any(_overlap(d.x0, d.x1, cs[0].x0, cs[0].x1) > 0.5 * (cs[0].x1 - cs[0].x0) for d in ds)) >= 2
    lastpos = edge
    for r in cand:
        if hits_caption(r):
            break
        if env_far is not None and dist(r.yc) < dist(env_far):
            accepted.append(r)
            lastpos = far(r)
            continue
        gap = dist(near(r)) - dist(lastpos if not env or dist(lastpos) > dist(env_far) else env_far)
        if not accepted and dist(near(r)) > 45 and not env:
            break
        if gap > max(2.4 * lay["body"], 1.9 * _pitch(accepted, lay)):
            break
        if (_body_like(r, lay, ext_w) and not _aligned(r, accepted, need_value=True)) or _note_like(r, ext_w) \
                or _in_graphics(pm, r) or (closed and env_far is not None):
            break
        tsize = statistics.median(x.size for x in accepted or context) if accepted or context else None
        if tsize is not None and r.size > tsize + 0.6:
            break
        if not (_table_like(r, lay) and not (len(env) >= 2 and not _aligned(r, accepted, need_value=True))):
            if (accepted or context) and len(pending) < 3 and r.size <= tsize + 0.3 and (r.size < lay["body"] - 0.45 or label_line(r)):
                pending.append(r)
                lastpos = far(r)
                continue
            break
        accepted += pending + [r]
        pending = []
        lastpos = far(r)
    if not accepted:
        return None
    accepted.sort(key=lambda r: r.y0)
    y0, y1 = min(r.y0 for r in accepted), max(r.y1 for r in accepted)
    if env:
        y0, y1 = min(y0, min(h.c for h in env)), max(y1, max(h.c for h in env))
    return {"rows": accepted, "xa": xa, "xb": xb, "y0": y0, "y1": y1, "env": env, "closed": closed,
            "H": [h for h in pm.H if y0 - 2 <= h.c <= y1 + 2 and _overlap(h.a, h.b, xa, xb) > 0.5 * h.length],
            "V": [v for v in pm.V if _overlap(v.a, v.b, y0, y1) > 2 and xa - 2 <= v.c <= xb + 2]}


def _region_quality(reg: Optional[dict]) -> float:
    """How table-like a region is: rows, multi-cell rows, numbers and rules."""
    if not reg:
        return 0.0
    rows = reg["rows"]
    n = len(rows)
    if n < 2 and not reg["env"]:
        return 0.0
    multi = sum(1 for r in rows if len(_chunks(r.toks, (), 0.5)) >= 2)
    nums = sum(1 for r in rows for t in r.toks if _numberish(t.text))
    q = (min(n, 25) * 0.4 if n >= 2 else 0.0) + 3 * multi / max(1, n) + 4 * nums / max(1, sum(len(r.toks) for r in rows))
    return q + (2 + min(len(reg["env"]), 3) if reg["env"] else 0.0)


# ----------------------------------------------------------------------------- the grid

BARE_NUM = re.compile(r"^[-+−]?\d[\d,]*(?:\.\d+)?%?(?:\*{1,3}|\^\S{1,4})?$")


def _gap_factor(rows: list[_Row]) -> float:
    """The gap (in em) that separates two cells: word spaces are about 0.3 em, column gaps at least 0.6 em; the
    widest empty valley between 0.3 and 0.65 em."""
    vals = []
    for r in rows:
        ts = sorted((t for t in r.toks if not t.small), key=lambda t: t.x0)
        vals += [g for a, b in zip(ts, ts[1:]) if 0.3 <= (g := (b.x0 - a.x1) / max(1.0, min(a.size, b.size))) <= 0.65]
    vals = sorted(vals + [0.3, 0.65])
    best = (0.0, 0.5)
    for a, b in zip(vals, vals[1:]):
        if b - a > best[0]:
            best = (b - a, (a + b) / 2)
    return min(max(best[1], 0.36), 0.6)


def _cell_chunks(rows: list[_Row], V: list[_Rule], notes: list[str], pen: list) -> None:
    """Cells of every row: split at column gaps and vertical rules; runs of bare numbers ("51.6 52.2") are separate
    cells; justified text stretched inside a text column is one cell again."""
    factor = _gap_factor(rows)
    vlines = [v for v in V if v.length > 6]
    for r in rows:
        r.chunks = _chunks(r.toks, [v.c for v in vlines if v.a - 1 <= r.yc <= v.b + 1 and r.x0 - 2 < v.c < r.x1 + 2], factor)
    n_bare = 0
    for r in rows:
        out = []
        for c in r.chunks:
            ts = c.toks
            if len(ts) >= 2 and all(BARE_NUM.match(t.text) for t in ts) and \
                    not any(re.fullmatch(r"\d{3}", b.text) and re.fullmatch(r"\d{1,3}", a.text) for a, b in zip(ts, ts[1:])):
                out += [_Chunk([t]) for t in ts]
                n_bare += 1
            else:
                out.append(c)
        r.chunks = out
    if n_bare:
        notes.append(f"{n_bare} space-separated run(s) of bare numbers treated as separate cells")
        pen.append((0.05, "space-separated numbers split"))

    def wordy(c):
        return all(any(ch.isalpha() for ch in t.text) and not any(ch.isdigit() for ch in t.text) for t in c.toks)
    zones: list[list] = []
    for a, b in sorted((c.x0, c.x1) for r in rows if len(r.chunks) >= 2 for c in r.chunks if len(c.toks) >= 4 and wordy(c)):
        if zones and a <= zones[-1][1]:
            zones[-1][1], zones[-1][2] = max(zones[-1][1], b), zones[-1][2] + 1
        else:
            zones.append([a, b, 1])
    zones = [z for z in zones if z[2] >= 2]
    if not zones:
        return
    for r in rows:
        out = []
        for c in r.chunks:
            if out and wordy(c) and wordy(out[-1]):
                z = [z for z in zones if z[0] - 2 <= out[-1].x0 and out[-1].x1 <= z[1] + 2]
                if z and z[0][0] - 2 <= c.x0 and c.x1 <= z[0][1] + 2:
                    out[-1] = _Chunk(out[-1].toks + c.toks)
                    continue
            out.append(c)
        r.chunks = out


def _is_data_row(r: _Row) -> bool:
    """Numbers in at least half of the cells after the label (a row of column numbers is no data)."""
    cs = r.chunks
    if not cs or all(COLNUM_RE.match(c.text) for c in cs):
        return False
    rest = cs[1:] if len(cs) > 1 and not _is_numeric(cs[0].text) else cs
    nnum = sum(1 for c in rest if _is_numeric(c.text) and c.text not in DASHES)
    return nnum >= 1 and nnum + sum(1 for c in rest if c.text in DASHES) >= 0.5 * len(rest)


def _decimal_row(r: _Row) -> bool:
    cs = r.chunks[1:] if len(r.chunks) > 1 else r.chunks
    return bool(cs) and all(_is_numeric(c.text) and "." in c.text for c in cs)


def _header_count(rows: list[_Row], full: list[_Rule], x0: float, x1: float, pen: list) -> int:
    """How many rows the header has: up to the full-width rule under it (a rule further down extends it only over
    header-like rows), else up to the first row of numbers."""
    width = max(1.0, x1 - x0)

    def header_like(r):
        cs = r.chunks
        if len(cs) == 1 and abs(cs[0].xc - (x0 + x1) / 2) < 0.08 * width:
            return False   # a centred single text is a section row
        return not any(_value_like(c.text) for c in cs[1:] if c.text not in DASHES) and not _decimal_row(r)

    n = None
    for h in (h for h in full if rows[0].y0 + 1 < h.c < rows[-1].y1 - 1):
        above = [r for r in rows if r.yc < h.c]
        if not above or len(above) >= len(rows):
            continue
        if n is None:
            ndata = sum(1 for r in above[1:] if _is_data_row(r))
            if (len(above) <= 6 or (len(above) <= 10 and ndata == 0)) and not _decimal_row(above[-1]) and \
                    (len(above) == 1 or (not _is_data_row(above[-1]) and ndata <= len(above) / 3)):
                n = len(above)
                continue
            break
        new = above[n:]
        if 0 < len(new) <= 3 and all(header_like(r) and len(r.chunks) >= 2 for r in new):
            n = len(above)
            continue
        break
    if n is not None:
        return n
    n = next((i for i, r in enumerate(rows) if _is_data_row(r)), len(rows))
    if n >= len(rows):
        pen.append((0.25, "no numeric data rows; header guessed as first row"))
        return 1 if len(rows) >= 3 else 0
    if n > 6:
        pen.append((0.3, "could not locate header rows"))
        return 1
    return n


def _find_bands(body_rows: list[_Row]) -> list[list[float]]:
    """Column bands: the merged x-extents of the body cells, leaving out cells that cover cells of other rows
    (merged cells, widest first); a label set left of every band gets a band of its own."""
    items = [(ri, c) for ri, r in enumerate(body_rows) for c in r.chunks
             if len(r.chunks) > 1 or _numberish(c.text) or c.text in DASHES]
    if not items:
        items = [(ri, c) for ri, r in enumerate(body_rows) for c in r.chunks]
    spanning: set[int] = set()
    for _ in range(6):
        by_row = defaultdict(list)
        for ri, c in items:
            if id(c) not in spanning:
                by_row[ri].append(c)
        new = {id(c) for ri, c in items if id(c) not in spanning and any(
            rj != ri and sum(1 for d in cs if _overlap(c.x0, c.x1, d.x0, d.x1) > 0.5) >= 2 for rj, cs in by_row.items())}
        if not new:
            break
        widest = max(c.x1 - c.x0 for ri, c in items if id(c) in new)
        spanning |= {id(c) for ri, c in items if id(c) in new and c.x1 - c.x0 >= 0.6 * widest}
    bands: list[list[float]] = []
    for a, b in sorted((c.x0, c.x1) for ri, c in items if id(c) not in spanning):
        if bands and a < bands[-1][1] - 0.3:
            bands[-1][1] = max(bands[-1][1], b)
        else:
            bands.append([a, b])
    inset = {id(c) for _, c in items}
    extra: list[list[float]] = []
    for a, b in sorted((c.x0, c.x1) for r in body_rows for c in r.chunks
                       if id(c) not in inset and not _numberish(c.text) and bands and c.x1 < bands[0][0]
                       and not any(_overlap(c.x0, c.x1, ba, bb) > 0 for ba, bb in bands)):
        if extra and a < extra[-1][1]:
            extra[-1][1] = max(extra[-1][1], b)
        else:
            extra.append([a, b])
    return extra + bands


def _bands_of(c, bands: list[list[float]], minov: float = 0.3) -> list[int]:
    return [k for k, (a, b) in enumerate(bands) if _overlap(c.x0, c.x1, a, b) > minov]


def _resplit(body_rows: list[_Row], bands: list[list[float]]) -> int:
    """A cell over two bands whose words fall into separate bands is two cells ("LaMDA 420M" typeset tight, or two
    numbers a space apart)."""
    model = {}
    for k in range(len(bands)):
        cs = [c for r in body_rows for c in r.chunks if _bands_of(c, bands) == [k] and _value_like(c.text)]
        if len(cs) >= 2:
            model[k] = (statistics.median(c.x0 for c in cs), statistics.median(c.x1 for c in cs))
    n = 0
    for r in body_rows:
        out = []
        for c in r.chunks:
            ks = _bands_of(c, bands)
            if len(ks) == 2 and len(c.toks) >= 2 and ks[1] in model:
                tail = [t for t in c.toks if _bands_of(t, bands, 0.0) == [ks[1]]]
                head = [t for t in c.toks if t not in tail]
                if tail and head and all(_bands_of(t, bands, 0.0) == [ks[0]] for t in head):
                    tc = _Chunk(tail)
                    if _value_like(tc.text) and (abs(tc.x0 - model[ks[1]][0]) < 1.5 or abs(tc.x1 - model[ks[1]][1]) < 1.5):
                        out += [_Chunk(head), tc]
                        n += 1
                        continue
            if len(ks) >= 2 and len(c.toks) >= 2:
                parts = defaultdict(list)
                for t in c.toks:
                    kt = _bands_of(t, bands, 0.0)
                    if len(kt) != 1:
                        break
                    parts[kt[0]].append(t)
                else:
                    subs = [_Chunk(parts[k]) for k in sorted(parts)]
                    if len(parts) >= 2 and all(_numberish(sc.text) or sc.text in DASHES for sc in subs):
                        out += subs
                        n += 1
                        continue
            out.append(c)
        r.chunks = out
    return n


def _assign(body_rows: list[_Row], bands: list[list[float]], pen: list, notes: list[str]) -> list[dict]:
    """Each body row's cells per band; a cell over several bands (or a narrow value centred on two) is a span."""
    nb = len(bands)
    colstats = {}
    for k in range(nb):
        cs = [c for r in body_rows for c in r.chunks if _bands_of(c, bands) == [k] and (_value_like(c.text) or _numberish(c.text))]
        if len(cs) >= 3:
            colstats[k] = tuple(statistics.median(v) for v in zip(*((c.x0, c.x1, c.xc) for c in cs)))

    def misaligned(c, k):
        return k in colstats and all(abs(v - m) > 2.5 for v, m in zip((c.x0, c.x1, c.xc), colstats[k]))

    body, n_misal = [], 0
    for r in body_rows:
        cells: list[list[_Chunk]] = [[] for _ in range(nb)]
        spans = []
        single = len(r.chunks) == 1 and not _is_numeric(r.chunks[0].text)
        for c in r.chunks:
            ks = _bands_of(c, bands)
            if not ks:
                if single:
                    spans.append((0, nb - 1, c))   # a free-floating text row: a section heading
                    continue
                d = [min(abs(c.x0 - b), abs(c.x1 - a), abs(c.xc - (a + b) / 2)) for a, b in bands]
                ks = [d.index(min(d))]
                pen.append((0.1, f"chunk {c.text!r} outside any column"))
            if len(ks) == 1 and misaligned(c, ks[0]) and (_value_like(c.text) or _numberish(c.text)):
                k = ks[0]
                for lo, hi in ((k - 1, k), (k, k + 1)):
                    union = (bands[lo][0], bands[hi][1]) if 0 <= lo and hi < nb else None
                    if union and abs(c.xc - (union[0] + union[1]) / 2) < max(3.0, 0.08 * (union[1] - union[0])):
                        ks = [lo, hi]   # a narrow value centred over two columns: a merged cell
                        break
                else:
                    n_misal += 1
            if len(ks) == 1:
                cells[ks[0]].append(c)
            else:
                spans.append((ks[0], ks[-1], c))
        body.append({"row": r, "cells": cells, "spans": spans})
    if n_misal:
        pen.append((min(0.3, 0.1 * n_misal), f"{n_misal} value(s) not aligned with their column"))
        notes.append(f"{n_misal} value(s) do not line up with the rest of their column")
    return body


def _layout_sections(body: list[dict], x0: float, x1: float) -> None:
    """A row of one text cell that spans the table, is centred, or sits off the labels' indentation is a section."""
    width = max(1.0, x1 - x0)
    lefts = Counter(round(b["cells"][0][0].x0) for b in body if b["cells"][0] and len(b["row"].chunks) > 1)
    mode = lefts.most_common(1)[0][0] if lefts else None
    for b in body:
        if len(b["row"].chunks) != 1 or _is_numeric(b["row"].chunks[0].text):
            continue
        c = b["row"].chunks[0]
        centred = abs(c.xc - (x0 + x1) / 2) < 0.06 * width
        in0 = bool(b["cells"][0])
        off = mode is not None and abs(c.x0 - mode) > 6 and sum(lefts.values()) >= 3 and lefts[mode] >= 0.6 * sum(lefts.values())
        if b["spans"] or PANEL_RE.match(c.text) or (centred and not in0) or (centred and in0 and off):
            b["section"] = c.text


def _split_header_runs(hdr_rows: list[_Row], bands: list[list[float]], part: list[_Rule]) -> None:
    """Header cells that merely abut in a tight table are split when every piece sits over its own column, the gaps
    are wider than a word space, and nothing is below them or each piece continues as a wrapped header below."""
    mids = [(bands[k][1] + bands[k + 1][0]) / 2 for k in range(len(bands) - 1)]
    for hi, r in enumerate(hdr_rows):
        out = []
        for c in r.chunks:
            ks = _bands_of(c, bands, 0.5)
            below = [r2 for r2 in hdr_rows[hi + 1:] if any(_overlap(d.x0, d.x1, c.x0, c.x1) > 0 for d in r2.chunks)]
            cmid = any(c.y1 - 1.5 <= p.c <= c.y1 + 0.9 * c.size + 3 and _overlap(p.a, p.b, c.x0, c.x1) > 0.3 * (c.x1 - c.x0)
                       and len([k for k, (a, b) in enumerate(bands) if _overlap(a, b, p.a, p.b) > 0.4 * max(1, b - a)]) >= 2
                       for p in part)
            pieces = []
            if len(ks) >= 2 and len(c.toks) >= 2 and not cmid:
                groups = defaultdict(list)
                for t in c.toks:
                    groups[sum(1 for m in mids if t.xc > m)].append(t)
                zs = sorted(groups)
                if len(zs) >= 2 and zs == list(range(zs[0], zs[-1] + 1)) and set(zs) == set(ks):
                    pieces = [_Chunk(groups[z]) for z in zs]
                    for z, pc in zip(zs, pieces):
                        a, b = bands[z]
                        centred = abs(pc.xc - (a + b) / 2) <= 0.25 * max(pc.x1 - pc.x0, b - a) + 2
                        if not (centred or abs(pc.x0 - a) < 2 or abs(pc.x1 - b) < 2):
                            pieces = []
                            break
                wrapped = bool(below)
                if pieces and below:
                    nxt = below[0].chunks
                    if not all(any(abs(d.xc - pc.xc) <= max(2.5, 0.15 * max(d.x1 - d.x0, pc.x1 - pc.x0)) or abs(d.x0 - pc.x0) < 1.5
                                   for d in nxt) for pc in pieces):
                        pieces = []
                if pieces:
                    min_gap = 0.27 if wrapped else (0.12 if len(pieces) >= 3 and not below else 0.36)
                    if any(p2.x0 < p1.x1 or p2.x0 - p1.x1 < min_gap * min(p1.size, p2.size) for p1, p2 in zip(pieces, pieces[1:])):
                        pieces = []
            out += pieces or [c]
        r.chunks = out


def _header_entries(hdr_rows: list[_Row], bands: list[list[float]], H: list[_Rule], part: list[_Rule]):
    """band -> [(header row, text, number of bands the cell spans, chunk)], and the header texts placed nowhere. A
    cell spans the bands between the joints of the cell border under it (Word), under its cmidrule, or the run of
    next-row cells it is centred over; otherwise it belongs to the bands it overlaps (the best one when it is not
    centred over them)."""
    entries: dict[int, list] = defaultdict(list)
    unplaced: list[str] = []
    for hi, r in enumerate(hdr_rows):
        claim = {}
        for c in r.chunks:
            ks = None
            for p in H:
                if p.joints and c.y1 - 1.5 <= p.c <= c.y1 + 0.9 * c.size + 3 and p.a - 1 <= c.xc <= p.b + 1:
                    js = sorted([p.a] + [j for j in p.joints if p.a < j < p.b] + [p.b])
                    for ja, jb in zip(js, js[1:]):
                        if ja - 0.5 <= c.xc <= jb + 0.5:
                            kk = [k for k, (a, b) in enumerate(bands) if _overlap(a, b, ja, jb) > 0.6 * max(1, b - a)]
                            ks = kk if len(kk) >= 2 else None
                            break
                    if ks:
                        break
            for p in ([] if ks else part):
                if c.y1 - 1.5 <= p.c <= c.y1 + 0.9 * c.size + 3 and _overlap(p.a, p.b, c.x0, c.x1) > 0.3 * min(p.length, c.x1 - c.x0):
                    ks = [k for k, (a, b) in enumerate(bands) if _overlap(a, b, p.a, p.b) > 0.4 * max(1, b - a)]
                    break
            wrap_head = any(_overlap(d.x0, d.x1, c.x0, c.x1) > 0 and (abs(d.xc - c.xc) < 2.5 or abs(d.x0 - c.x0) < 1.5)
                            and len({k for e in r2.chunks for k in _bands_of(e, bands, 0.5)}) / max(1, len(bands)) < 0.7
                            for r2 in hdr_rows[hi + 1:] for d in r2.chunks)
            ri = next((i for i in range(hi + 1, len(hdr_rows))
                       if any(_overlap(d.x0, d.x1, c.x0, c.x1) > 0 for d in hdr_rows[i].chunks)), None)
            if not ks and ri is not None and not wrap_head:
                # the sub-headers are in the nearest row with cells under this one ("Model" between two header lines
                # is no sub-header); a sub-header under another header cell of the rows between is not in its run
                nxt = hdr_rows[ri].chunks
                others = [d for r2 in hdr_rows[hi:ri] for d in r2.chunks if d is not c]
                best = None
                for a in range(len(nxt)):
                    for b in range(a, len(nxt)):
                        ux0, ux1 = nxt[a].x0, nxt[b].x1
                        if _overlap(c.x0, c.x1, ux0, ux1) <= 0 or any(_overlap(d.x0, d.x1, ux0 + 0.5, ux1 - 0.5) > 0 for d in others):
                            continue
                        dd = abs(c.xc - (ux0 + ux1) / 2)
                        if dd <= 0.06 * (ux1 - ux0) + 2.0 and (best is None or (b - a + 1, -dd) > best[0]):
                            best = ((b - a + 1, -dd), a, b)
                if best is not None and best[0][0] >= 2:
                    run = nxt[best[1]:best[2] + 1]
                    kr = sorted({k for d in run for k in _bands_of(d, bands, 0.0)})
                    if len(kr) >= 2 and kr == list(range(kr[0], kr[-1] + 1)) and all(_bands_of(d, bands, 0.0) for d in run):
                        ks = kr
            if not ks:
                ov = {k: _overlap(a, b, c.x0 - 1, c.x1 + 1) for k, (a, b) in enumerate(bands)}
                ks = [k for k, o in ov.items() if o > 0]
                if len(ks) > 1:
                    ua, ub = bands[ks[0]][0], bands[ks[-1]][1]
                    if abs(c.xc - (ua + ub) / 2) > 0.22 * (ub - ua):
                        ks = [max(ks, key=lambda k: ov[k] / max(1, bands[k][1] - bands[k][0]))]
            if not ks:
                d = [abs(c.xc - (a + b) / 2) for a, b in bands]
                k = d.index(min(d))
                a, b = bands[k]
                if min(abs(c.x0 - b), abs(c.x1 - a), abs(c.xc - (a + b) / 2)) < max(8, 0.6 * (c.x1 - c.x0)):
                    ks = [k]
                else:
                    unplaced.append(c.text)
                    continue
            for k in ks:
                a, b = bands[k]
                sc = _overlap(a, b, c.x0, c.x1) / max(1, b - a) + (1.0 if len(ks) == 1 else 0)
                if k not in claim or sc > claim[k][0]:
                    claim[k] = (sc, c, len(ks))
        for k, (_, c, ns) in claim.items():
            entries[k].append((hi, c.text, ns, c))
    # Word tables: a top header that starts where a sub-header starts covers the columns up to the next top header
    for hi in range(len(hdr_rows) - 1):
        tops = sorted({id(e[3]): e for k in entries for e in entries[k] if e[0] == hi}.values(), key=lambda e: e[3].x0)
        subs = hdr_rows[hi + 1].chunks
        if len(tops) < 2 or not subs:
            continue
        aligned = [e for e in tops if any(abs(e[3].x0 - d.x0) < 2.0 for d in subs)]
        if len(aligned) < 2 or len(aligned) < len(tops) - 1:
            continue
        sub_bands = sorted({k for d in subs for k in _bands_of(d, bands, 0.5)})
        for ti, e in enumerate(aligned):
            start = min(k for k, (a, b) in enumerate(bands) if b >= e[3].x0 - 2)
            stop = (min(k for k, (a, b) in enumerate(bands) if b >= aligned[ti + 1][3].x0 - 2) if ti + 1 < len(aligned)
                    else (max(sub_bands) + 1 if sub_bands else len(bands)))
            for k in range(start, stop):
                if k in sub_bands and not any(x[0] == hi for x in entries.get(k, [])):
                    entries[k].append((hi, e[1], max(2, e[2]), e[3]))
    return entries, unplaced


def _label_bands(body: list[dict], entries: dict, nb: int) -> list[int]:
    """The leading bands that hold row labels: a text first band, then text bands without a header of their own,
    and short text attribute columns ("Prompting"), as long as numeric columns follow them."""
    def texts(k):
        return [t for b in body if not b.get("section") and (t := " ".join(c.text for c in b["cells"][k]))]

    def numeric(k):
        tk = texts(k)
        return bool(tk) and sum(1 for t in tk if _value_like(t) or _numberish(t) or t in DASHES) >= 0.5 * len(tk)

    t0 = texts(0)
    if not t0:
        return []
    out = [0]
    if sum(1 for t in t0 if not _is_numeric(t)) < 0.5 * len(t0):
        # a leading column of numbers (Pos., Year, Distance) names the row only together with a text column after it
        t1 = texts(1) if nb > 2 else []
        if not t1 or sum(1 for t in t1 if not _is_numeric(t)) < 0.6 * len(t1) or not any(numeric(k) for k in range(2, nb)):
            return []
        out = [0, 1]
    for k in range(len(out), nb):
        tk = texts(k)
        if not tk:
            break
        own = [e for e in entries.get(k, []) if e[2] == 1]
        if not own and sum(1 for t in tk if not _is_numeric(t)) >= 0.6 * len(tk):
            out.append(k)
        elif own and len(tk) >= 2 and sum(len(t.split()) for t in tk) <= 4 * len(tk) and \
                sum(1 for t in tk if not _value_like(t) and t not in DASHES and not _numberish(t)) >= 0.8 * len(tk):
            out.append(k)
        else:
            break
    while len(out) > 1 and not any(numeric(k) for k in range(out[-1] + 1, nb)):
        out.pop()
    return [0] if len(out) == nb and nb > 1 else out


def _header_paths(entries: dict, data_bands: list[int], hdr_rows: list[_Row], H: list[_Rule],
                  split: Optional[int] = None) -> list[list[str]]:
    """One header path per data band, top level first. Lines of one cell that wraps (a single-band entry under a
    single-band entry, unless both header rows are full rows or either is a spanning row; full rows too when they
    are a line apart with no rule between them) are joined; header rows from ``split`` on always start a level."""
    n_hdr = len(hdr_rows)
    cov = Counter(e[0] for k in data_bands for e in entries.get(k, []))
    full_row = {hi: cov[hi] >= 0.7 * len(data_bands) for hi in range(n_hdr)}
    tight = {hi for hi in range(n_hdr - 1) if hdr_rows[hi + 1].base - hdr_rows[hi].base <= 1.3 * hdr_rows[hi].size
             and not any(hdr_rows[hi].y1 - 1 < h.c < hdr_rows[hi + 1].y0 + 1 for h in H)}
    spanning_row = {e[0] for k in data_bands for e in entries.get(k, []) if e[2] > 1}
    columns = []
    for k in data_bands:
        path, prev = [], None
        for e in sorted(entries.get(k, []), key=lambda e: e[0]):
            single = prev is not None and prev[2] == 1 and e[2] == 1 and not COLNUM_RE.match(prev[1]) \
                and not (split is not None and prev[0] < split <= e[0])
            wrapped = single and (prev[1].endswith("-") or e[1][:1].islower() or (e[0] == prev[0] + 1 and prev[0] in tight))
            if wrapped or (single and not (full_row.get(prev[0]) and full_row.get(e[0]))
                           and prev[0] not in spanning_row and e[0] not in spanning_row):
                path[-1] = (path[-1] + " " + e[1]).strip()
            else:
                path.append(e[1])
            prev = e
        columns.append([re.sub(r"(\w)- (\w)", r"\1-\2", p) for p in path])
    return columns


def _join_label(a: str, b: str) -> str:
    return a + b if re.search(r"\w-$", a) else (a + " " + b).strip()   # a word-final hyphen stays: "Teacher-synthesized"


def _relabel(o: dict) -> None:
    o["label"] = " | ".join(q for q in o["_labparts"] if q)


CONTINUES = ("-", ",", "/", "&", "(", " and", " of", " for", " with", " the")


def _join_label_blocks(out: list[dict], n_lab: int, rpitch: float, notes: list[str]) -> None:
    """Label-only lines of a multi-line row label join their data row. Top-aligned cells: each line no more than a
    row step below the previous line of a label in the same column and style (a line further down, or italic under
    an upright label, is a section heading: "2007–2014" under "R-squared"). Cells centred on their label block: the
    clearly nearest row; a line as close to the row above as to the row below, or a whole line away (a cell of three
    lines), follows the words: one that breaks a word or a phrase continues downward ("Kitchener-"), the line after
    such a line, or one in lower case, continues upward."""
    data_rows = [o for o in out if any(o["cells"])]
    own = {id(d): list(d["_labparts"]) for d in data_rows}
    lab_only = [o for o in out if o["label"] and not any(o["cells"]) and not o["_spans"]]
    left_aligned = []
    for li in range(n_lab):
        xs = [round(o["_labx"][li][0]) for o in out if li < len(o["_labx"]) and o["_labx"][li]]
        left_aligned.append(len(xs) >= 3 and sum(n for _, n in Counter(xs).most_common(2)) >= 0.7 * len(xs))
    def line_near(d, li, sign):   # a label-only line of column li within 1.6 em above (sign 1) or below (-1) row d
        return any(x["_labparts"][li] and 0 < sign * (d["_row"].base - x["_row"].base) < 1.6 * x["_row"].size for x in lab_only)
    centred = any(not own[id(d)][li] and line_near(d, li, 1) and line_near(d, li, -1) for d in data_rows for li in range(n_lab))

    def same_column(x, y, li):
        ax, bx = x["_labx"][li] if li < len(x["_labx"]) else None, y["_labx"][li] if li < len(y["_labx"]) else None
        if not ax or not bx:
            return True
        return abs(ax[0] - bx[0]) < 2.0 or (not left_aligned[li] and abs((ax[0] + ax[1]) / 2 - (bx[0] + bx[1]) / 2) < 2.0)

    assign = defaultdict(list)
    owner: dict[int, dict] = {}   # row -> the data row whose label it is part of
    last: dict[int, float] = {}   # data row -> baseline of the last line of its label so far
    for idx, o in enumerate(out):
        if any(o["cells"]):
            owner[id(o)] = o
            continue
        if o["_spans"] or not o["label"] or o["label"].rstrip().endswith(":"):
            continue   # "Relative to first attempt:" introduces the rows below
        ob, size = o["_row"].base, o["_row"].size
        lis = [li for li, p in enumerate(o["_labparts"]) if p]
        rows = [d for d in data_rows if d["group"] == o["group"]]
        if not centred:
            above = [d for d in rows if d["_row"].base < ob]
            A = max(above, key=lambda d: d["_row"].base) if above else None
            if A is not None and all(own[id(A)][li] for li in lis) and all(same_column(A, o, li) for li in lis) \
                    and ob - last.get(id(A), A["_row"].base) <= 1.15 * rpitch and o["_lstyle"] == A["_lstyle"] \
                    and not any(A["_row"].base < d["_row"].base < ob for d in data_rows):
                assign[id(A)].append(o)
                last[id(A)] = ob
                owner[id(o)] = A
            continue
        cand = sorted(((abs(d["_row"].base - ob), d) for d in rows if abs(d["_row"].base - ob) <= 1.15 * rpitch), key=lambda t: t[0])
        if not cand:
            continue
        d = cand[0][1]
        if cand[0][0] > min(0.75 * rpitch, 1.6 * size) or (len(cand) > 1 and cand[1][0] - cand[0][0] <= 0.3 * size):
            prev = out[idx - 1] if idx else None
            text = o["_labparts"][lis[0]].rstrip()
            above_text = (prev["_labparts"][lis[0]] if prev is not None and lis[0] < len(prev["_labparts"]) else "").rstrip()
            down = text.endswith(CONTINUES)
            if down == (above_text.endswith(CONTINUES) or text[:1].islower()):
                continue   # no telling which row this line belongs to
            below = [x for _, x in cand if x["_row"].base > ob]
            if down:
                d = min(below, key=lambda x: x["_row"].base) if below else None
            else:
                d = owner.get(id(prev)) if prev is not None else None
            if d is None:
                continue
        if not all(same_column(d, o, li) for li in lis if own[id(d)][li]):
            continue
        assign[id(d)].append(o)
        owner[id(o)] = d
    if not assign:
        return
    n = 0
    for d in data_rows:
        extra = assign.get(id(d))
        if not extra:
            continue
        lines = sorted(extra + [d], key=lambda x: x["_row"].base)
        for li in range(n_lab):
            joined = ""
            for p in (x["_labparts"][li] for x in lines if x["_labparts"][li]):
                joined = _join_label(joined, p) if joined else p
            d["_labparts"][li] = joined
        _relabel(d)
        n += len(extra)
    drop = {id(x) for v in assign.values() for x in v}
    out[:] = [o for o in out if id(o) not in drop]
    notes.append(f"{n} line(s) of multi-line row labels joined to the row they belong to")


def _merge_wrapped_rows(out: list[dict], n_lab: int, notes: list[str]) -> None:
    """The second line of a row with wrapped cells: a row without numbers right under a row with numbers, nearer to
    it than to the row below, empty wherever that row has a number, with its label (if any) in the same column and
    style, continues that row's label and text cells ("receivable (DAR)" and "payment for services rendered" under
    "Days in accounts"). Without a label it may only hold text-column cells."""
    if not out:
        return
    text_col = []
    for j in range(len(out[0]["cells"])):
        vals = [o["cells"][j] for o in out if o["cells"][j]]
        text_col.append(bool(vals) and sum(1 for v in vals if not _numberish(v)) >= 0.6 * len(vals))
    n, i = 0, 1
    while i < len(out):
        o, pv = out[i], out[i - 1]
        nx = out[i + 1] if i + 1 < len(out) and out[i + 1]["group"] == o["group"] else None
        filled = [j for j, c in enumerate(o["cells"]) if c]
        up = o["_row"].base - pv["_row"].base
        closer = up < nx["_row"].base - o["_row"].base - 0.5 if nx is not None else up <= 1.3 * o["_row"].size
        lab = [li for li in range(n_lab) if o["_labparts"][li]]
        fits = all(pv["_labparts"][li] and pv["_labx"][li] and o["_labx"][li] and abs(pv["_labx"][li][0] - o["_labx"][li][0]) < 2.0
                   for li in lab) and (not lab or o["_lstyle"] == pv["_lstyle"])
        if pv["group"] == o["group"] and filled and closer and fits and not o["_spans"] and not pv["_spans"] \
                and any(_numberish(c) for c in pv["cells"] if c) and (lab or all(text_col[j] and pv["cells"][j] for j in filled)) \
                and all(text_col[j] and not (pv["cells"][j] and _numberish(pv["cells"][j])) for j in filled):
            for li in lab:
                pv["_labparts"][li] = _join_label(pv["_labparts"][li], o["_labparts"][li])
            for j in filled:
                pv["cells"][j] = _join_label(pv["cells"][j], o["cells"][j]) if pv["cells"][j] else o["cells"][j]
            _relabel(pv)
            del out[i]
            n += 1
            continue
        i += 1
    if n:
        notes.append(f"{n} wrapped line(s) of label and text cells joined to the row above")


def _merge_stacked_cells(out: list[dict], rpitch: float, notes: list[str]) -> None:
    """Cells of two lines printed around their row ("110–582" above, "113–532 (*)" below a row centred on them): a
    row without a label of its own (a multirow label spread onto it counts as none), at most half a row step from a
    fuller row (two values more) that is clearly the nearest, whose cells all sit in columns that row leaves empty,
    joins it."""
    joins = defaultdict(list)
    for o in out:
        filled = [j for j, c in enumerate(o["cells"]) if c]
        if (o["label"] and not o.get("_filled")) or o["_spans"] or not filled:
            continue
        near = sorted(((abs(d["_row"].base - o["_row"].base), d) for d in out if d is not o and d["group"] == o["group"]
                       and sum(1 for j, c in enumerate(d["cells"]) if c and j not in filled) >= 2), key=lambda t: t[0])
        if not near or near[0][0] > 0.75 * rpitch or (len(near) > 1 and near[1][0] - near[0][0] <= 0.3 * o["_row"].size):
            continue
        d = near[0][1]
        if all(not d["cells"][j] for j in filled) and (not o["label"] or o["label"] == d["label"]):
            joins[id(d)].append(o)
    if not joins:
        return
    for d in out:
        if parts := joins.get(id(d)):
            lines = sorted(parts + [d], key=lambda x: x["_row"].base)
            d["cells"] = [" ".join(x["cells"][j] for x in lines if x["cells"][j]) for j in range(len(d["cells"]))]
    drop = {id(x) for v in joins.values() for x in v}
    out[:] = [o for o in out if id(o) not in drop]
    notes.append(f"{len(drop)} line(s) of two-line cells joined to the row they are printed around")


def _spread_centred_labels(out: list[dict], rpitch: float, notes: list[str], pen: list) -> None:
    """A label-only line half a row from the rows above and below it is a multirow label: it fills the empty label
    slots of the rows it is centred on."""
    n, i = 0, 0
    while i < len(out):
        o = out[i]
        if not any(o["cells"]) and not o["_spans"] and o["label"] and 0 < i < len(out) - 1:
            pv, nx = out[i - 1], out[i + 1]
            ob = o["_row"].base
            if pv["group"] == o["group"] == nx["group"] and ob - pv["_row"].base < 0.8 * rpitch and nx["_row"].base - ob < 0.8 * rpitch:
                grp = [x for x in out if x["group"] == o["group"] and x is not o]
                above = [x for x in grp if x["_row"].base < ob]
                below = [x for x in grp if x["_row"].base > ob]
                k = min(len(above), len(below))
                for li, part in enumerate(o["_labparts"]):
                    for x in (above[len(above) - k:] + below[:k]) if part else []:
                        if not x["_labparts"][li]:
                            x["_labparts"][li] = part
                            x["_filled"] = True
                            _relabel(x)
                del out[i]
                n += 1
                continue
        i += 1
    if n:
        notes.append(f"{n} vertically centred multirow label(s) applied to the rows they span")
        pen.append((0.03, "multirow labels between rows"))


def _merge_fragments(out: list[dict], notes: list[str], pen: list) -> None:
    """Lines of multi-line text cells set as rows of their own (text only, no label) join the nearest labelled row."""
    lab_rows = [o for o in out if o["label"] and any(o["cells"])]
    if not lab_rows or len(lab_rows) >= len(out):
        return
    for o in lab_rows:
        o["_ext"] = [o["_row"].base, o["_row"].base]

    def fragment(o):
        ne = [c for c in o["cells"] if c]
        return not o["label"] and ne and not o["_spans"] and all(not _numberish(c) and not _value_like(c) for c in ne)

    n, changed = 0, True
    while changed:
        changed = False
        for o in list(out):
            if not fragment(o):
                continue
            b0 = o["_row"].base
            dist = sorted(((0.0 if x["_ext"][0] <= b0 <= x["_ext"][1] else min(abs(b0 - x["_ext"][0]), abs(b0 - x["_ext"][1])), x)
                           for x in lab_rows if x["group"] == o["group"]), key=lambda t: t[0])
            if not dist or dist[0][0] >= 1.6 * o["_row"].size or (len(dist) > 1 and dist[1][0] - dist[0][0] <= 0.25 * o["_row"].size):
                continue
            tgt = dist[0][1]
            for j, c in enumerate(o["cells"]):
                if c and tgt["cells"][j]:
                    tgt["cells"][j] = c + " " + tgt["cells"][j] if b0 < tgt["_row"].base else tgt["cells"][j] + " " + c
                elif c:
                    tgt["cells"][j] = c
            tgt["_ext"] = [min(tgt["_ext"][0], b0), max(tgt["_ext"][1], b0)]
            out[:] = [x for x in out if x is not o]
            n += 1
            changed = True
    if n:
        notes.append(f"{n} line(s) of multi-line text cells merged into the nearest labelled row")
        pen.append((0.05, "multi-line text cells merged"))


def _flag_squeezed(out: list[dict], rpitch: float, notes: list[str], pen: list) -> None:
    """A row with numbers half a line from neighbours with numbers holds pieces of multi-line cells: which row its
    values belong to is ambiguous."""
    n = 0
    for i in range(1, len(out) - 1):
        o, pv, nx = out[i], out[i - 1], out[i + 1]
        ob = o["_row"].base
        if pv["group"] == o["group"] == nx["group"] and ob - pv["_row"].base < 0.75 * rpitch and nx["_row"].base - ob < 0.75 * rpitch \
                and any(_numberish(c) for c in o["cells"] if c) \
                and (any(_numberish(c) for c in pv["cells"] if c) or any(_numberish(c) for c in nx["cells"] if c)):
            o["_squeezed"] = True
            n += 1
    if n:
        notes.append(f"{n} row(s) sit between two rows (multi-line cells); their values may belong to a neighbouring row")
        pen.append((min(0.4, 0.15 * n), "rows squeezed between rows"))


def _section_rows(out: list[dict], notes: list[str]) -> None:
    """A short label-only row followed by rows with values (in its block, or styled bold/italic) heads a section."""
    n, i = 0, 0
    while i < len(out) - 1:
        o, nx = out[i], out[i + 1]
        if o["label"] and len(o["label"].split()) <= 7 and not any(o["cells"]) and not o["_spans"] and any(nx["cells"]) and \
                (nx["group"] == o["group"] or (o["_row"].toks and (o["_row"].toks[0].bold or o["_row"].toks[0].italic))):
            for x in out[i + 1:]:
                if x.get("_section") not in (None, o.get("_section")) or (x["label"] and not any(x["cells"]) and not x["_spans"]):
                    break
                x["_section"] = o["label"]
            del out[i]
            n += 1
            continue
        i += 1
    if n:
        notes.append(f"{n} label-only row(s) treated as section headings for the rows below (row key 'section')")


def _inner_header(out: list[dict], i: int) -> bool:
    """A text row (no label, or among the first rows) right above a row of values: a header row inside the body."""
    o = out[i]
    ne = [c for c in o["cells"] if c]
    return i + 1 < len(out) and (not o["label"] or (i <= 1 and len(ne) >= 2)) and bool(ne) \
        and not any(_value_like(c) or _numberish(c) or c in DASHES for c in ne) and not o["_spans"] \
        and any(_value_like(c) for c in out[i + 1]["cells"] if c) and (len(ne) >= 2 or len(o["cells"]) == 1)


def _inner_headers(out: list[dict], columns: list[list[str]], notes: list[str], pen: list) -> None:
    """A text row without label right above rows of values (e.g. "# layers  # params") extends the column headers."""
    n, i = 0, 0
    while i < len(out) - 1:
        o = out[i]
        if _inner_header(out, i):
            if o["label"]:
                notes.append(f"label-column header {o['label']!r} found inside the body")
            for j, c in enumerate(o["cells"]):
                if c:
                    columns[j] = list(columns[j]) + [c]
            del out[i]
            n += 1
            continue
        i += 1
    if n:
        notes.append(f"{n} header row(s) found inside the body were appended to the column headers")
        pen.append((0.05, "header row inside body"))


def _continued_labels(out: list[dict], notes: list[str], pen: list) -> list[dict]:
    """A label line that continues the label above (lower case, or the label above ends with an operator) joins it;
    a row of parenthesised values right under a row of numbers is its standard error (stat "se")."""
    merged: list[dict] = []
    for o in out:
        prev = merged[-1] if merged else None
        has_data, lab = any(o["cells"]), o["label"]
        se_cells = all(not c or PAREN_NUM.match(c) for c in o["cells"])
        prev_values = prev is not None and any(c and not PAREN_NUM.match(c) and _is_numeric(c) for c in prev["cells"])
        if prev is not None and lab and prev["group"] == o["group"]:
            step = o["_row"].base - prev["_row"].base
            tight, near = step < 1.45 * o["_row"].size, step < 2.1 * o["_row"].size
            ends_op = prev["label"].rstrip().endswith(("-", ",", "/", "&", "and", "of", "the", "for", "×", "x", "+", "*", "·",
                                                       "with", "("))
            cont = lab[:1].islower() or ends_op
            if tight and cont and not has_data:
                prev["label"] = _join_label(prev["label"], lab)
                prev["_cont"] = True
                continue
            if (tight or (near and ends_op) or (near and lab[:1].islower())) and cont and has_data and se_cells and prev_values:
                prev["label"] = _join_label(prev["label"], lab)
                o["label"], o["stat"] = prev["label"], "se"
                merged.append(o)
                continue
        if prev is not None and not lab and has_data and prev["group"] == o["group"] and se_cells and prev_values:
            o["label"], o["stat"] = prev["label"], "se"
        merged.append(o)
    if any(o.get("stat") == "se" for o in merged):
        notes.append("stacked rows: parenthesised standard-error rows follow their coefficient row")
        pen.append((0.05, "stacked coefficient/SE rows"))
    return merged


def _second_label_lines(out: list[dict], n_lab: int) -> None:
    """The second line of a label typeset on the next row ("(text-davinci-002)") joins the label above."""
    for i in range(1, len(out)):
        o, pv = out[i], out[i - 1]
        if pv["group"] != o["group"]:
            continue
        for li in range(n_lab):
            part, above = o["_labparts"][li], pv["_labparts"][li]
            if part and above and part != above and above[:1] != "(" and not o.get("_filled") and \
                    (part[:1] == "(" or (part[:1].islower() and not any(o["cells"]))) and \
                    o["_row"].base - pv["_row"].base < 1.45 * o["_row"].size and (n_lab == 1 or li < n_lab - 1):
                pv["_labparts"][li] = _join_label(above, part)
                o["_labparts"][li] = ""
                _relabel(pv)
                _relabel(o)


def _fill_multirow(out: list[dict], n_lab: int, notes: list[str], pen: list) -> None:
    """A label printed once in a rule-delimited block of rows (a multirow cell) is the label of every row of it;
    with one label column only when it is vertically centred on the block."""
    groups = defaultdict(list)
    for o in out:
        groups[o["group"]].append(o)
    filled = 0
    for rows in groups.values():
        if not 2 <= len(rows) <= 12:
            continue
        for li in range(max(1, n_lab - (1 if n_lab > 1 else 0))):
            vals = [o["_labparts"][li] for o in rows]
            ne = [v for v in vals if v]
            if len(ne) != 1:
                continue
            idx = vals.index(ne[0])
            sub_ok = n_lab > 1 and all(o["_labparts"][li + 1] for o in rows if li + 1 < len(o["_labparts"]))
            if n_lab == 1 and not sub_ok and (abs(idx - (len(vals) - 1) / 2) > 0.6 or any(o.get("stat") for o in rows)):
                continue
            for o in rows:
                o["_labparts"][li] = ne[0]
                _relabel(o)
            filled += 1
    if filled:
        notes.append(f"multirow labels filled down in {filled} row group(s)")
        pen.append((0.04, "multirow labels filled"))


MULTI_NUM = re.compile(r"(?<![\d.,–-])[-+−]?\d(?:[\d,]*\d)?(?:\.\d+)?%?\*{0,3}\s+[-+−]?\d(?:[\d,]*\d)?(?:\.\d+)?%?(?![\d.–-])")


def _two_numbers(c: str) -> bool:
    """A cell holding two numbers a space apart: two columns read as one."""
    if not c or re.search(r"[±·×/(\[{]|10\^", c):
        return False
    c = re.sub(r"(?<![\d.,])(\d{1,3})(?:\s\d{3})+(?![\d.,])", lambda m: m.group(0).replace(" ", ""), c)   # "20 000"
    return bool(MULTI_NUM.search(c))


def _data_entries(entries: dict, label_bands: list[int], data_bands: list[int]) -> None:
    """A header over the label column and some data columns describes the labels: it leaves the data columns."""
    lab = {id(e[3]) for k in label_bands for e in entries.get(k, [])}
    for k in data_bands:
        if k in entries:
            entries[k] = [e for e in entries[k] if id(e[3]) not in lab or e[2] == 1]


def _label_heads(entries: dict, label_bands: list[int]) -> list[str]:
    return [" ".join(e[1] for e in sorted(entries.get(k, []), key=lambda e: e[0])) for k in label_bands]


def _build_rows(body: list[dict], label_bands: list[int], data_bands: list[int], full: list[_Rule], pitch: float,
                pen: list):
    """(rows, spanning cells [(text, first column, last column)], cells that got two values) from the assigned body.
    A rule or a gap of more than a row and a half starts a new row group; a layout section row sets the section."""
    out: list[dict] = []
    span_cells: list[tuple[str, int, int]] = []
    group, section, prev_row, n_conflict = 0, None, None, 0
    for b in body:
        r = b["row"]
        if prev_row is not None and (any(prev_row.y1 - 1 < h.c < r.y0 + 1 for h in full) or r.base - prev_row.base > 1.55 * pitch):
            group += 1
        prev_row = r
        if b.get("section"):
            section = b["section"]
            group += 1 if out else 0
            continue
        lab_parts = [" ".join(c.text for c in b["cells"][k]) for k in label_bands]
        cells, bold = [], []
        for k in data_bands:
            cs = b["cells"][k]
            n_conflict += sum(1 for c in cs if _value_like(c.text) or _numberish(c.text)) > 1
            cells.append(" ".join(c.text for c in cs))
            bold.append(bool(cs) and all(c.bold for c in cs))
        spans = []
        for ka, kb, c in b["spans"]:
            if ka in label_bands and all(not b["cells"][k] for k in range(ka + 1, kb + 1) if k not in label_bands) and \
                    kb in label_bands + [label_bands[-1] + 1 if label_bands else -1]:
                li = label_bands.index(ka)   # a long label running into the first data column
                lab_parts[li] = (lab_parts[li] + " " + c.text).strip()
                pen.append((0.08, f"label {c.text[:30]!r} overlaps a data column"))
                continue
            dk = [k for k in range(ka, kb + 1) if k in data_bands]
            if not dk:
                lab_parts[-1] = (lab_parts[-1] + " " + c.text).strip()
                continue
            j0, j1 = data_bands.index(dk[0]), data_bands.index(dk[-1])
            if cells[j0]:
                cells[j0] += " " + c.text
                n_conflict += 1
            else:
                cells[j0], bold[j0] = c.text, c.bold
            spans.append([j0, j1])
            span_cells.append((c.text, j0, j1))
        toks = [t for k in label_bands for c in b["cells"][k] for t in c.toks]
        out.append({"label": " | ".join(p for p in lab_parts if p), "group": group, "cells": cells, "bold": bold,
                    "_row": r, "_labparts": lab_parts, "_spans": spans, "_section": section,
                    "_lstyle": (sum(t.bold for t in toks) * 2 > len(toks), sum(t.italic for t in toks) * 2 > len(toks)),
                    "_labx": [((b["cells"][k][0].x0, b["cells"][k][-1].x1) if b["cells"][k] else None) for k in label_bands]})
    return out, span_cells, n_conflict


def _header_doubts(columns: list[list[str]], unplaced: list[str], has_header: bool, pen: list) -> None:
    """Header text placed over no column, header paths that repeat or are the tail of another, columns without one."""
    if unplaced:
        pen.append((min(0.3, 0.1 * len(unplaced)), f"header text not placed: {unplaced[:3]}"))
    dups = [p for p, n in Counter(tuple(c) for c in columns if c).items() if n > 1]
    shorter = sum(1 for c1 in columns if c1 and any(len(c2) > len(c1) and tuple(c2[-len(c1):]) == tuple(c1) for c2 in columns))
    if dups or shorter:
        pen.append((min(0.3, 0.1 * (len(dups) + shorter)), "ambiguous column headers (duplicate or incomplete header paths)"))
    if has_header and len(columns) > 1 and (n := sum(1 for c in columns if not c)):
        pen.append((0.05 * n, "some data columns have no header"))
    if not any(columns):
        pen.append((0.15, "no column headers detected (column meaning unknown)"))


def _span_cost(span_cells: list[tuple[str, int, int]], columns: list[list[str]]) -> float:
    """What the merged cells cost: nothing for a text cell (it binds no number), little for a number spanning
    exactly the sub-columns of one header group (the usual multicolumn), more for any other."""
    def grouped(j0, j1):
        parent = columns[j0][:-1]
        return bool(parent) and all(columns[j][:-1] == parent for j in range(j0, j1 + 1)) and \
            not (j0 > 0 and columns[j0 - 1][:-1] == parent) and not (j1 + 1 < len(columns) and columns[j1 + 1][:-1] == parent)
    return sum(0.02 if grouped(j0, j1) else 0.1 for text, j0, j1 in span_cells if any(ch.isdigit() for ch in text))


def _parse_grid(rows: list[_Row], H: list[_Rule], V: list[_Rule], lay: dict, notes: list[str]):
    """(header paths, rows, meta) of one table (or panel) from its visual rows; None when no grid can be built."""
    pen: list = []
    rows = sorted(rows, key=lambda r: r.base)
    x0, x1 = min(r.x0 for r in rows), max(r.x1 for r in rows)
    width = max(1.0, x1 - x0)
    _cell_chunks(rows, V, notes, pen)
    full = sorted((h for h in H if h.length >= 0.85 * width), key=lambda h: h.c)
    part = sorted((h for h in H if 4 < h.length < 0.85 * width), key=lambda h: h.c)
    n_hdr = _header_count(rows, full, x0, x1, pen)
    hdr_rows, body_rows = rows[:n_hdr], rows[n_hdr:]
    if not body_rows:
        return None
    bands = _find_bands(body_rows)
    if not bands:
        return None
    for v in (v for v in V if v.length > 6):   # a vertical rule splits a band
        bands = [half for a, b in bands for half in (([a, v.c - 0.1], [v.c + 0.1, b]) if a + 1 < v.c < b - 1 else ([a, b],))]
    if n := _resplit(body_rows, bands):
        notes.append(f"{n} run(s) of numbers separated only by spaces were split at column gaps")
        pen.append((0.04 * min(n, 3), "numbers split at column gaps"))
    body = _assign(body_rows, bands, pen, notes)
    _layout_sections(body, x0, x1)
    _split_header_runs(hdr_rows, bands, part)
    entries, unplaced = _header_entries(hdr_rows, bands, H, part)
    label_bands = _label_bands(body, entries, len(bands))
    data_bands = [k for k in range(len(bands)) if k not in label_bands]
    if not data_bands:
        return None
    label_header = " ".join(dict.fromkeys(e[1] for k in label_bands for e in entries.get(k, [])))
    _data_entries(entries, label_bands, data_bands)
    columns = _header_paths(entries, data_bands, hdr_rows, H)
    drows = [b["row"] for b in body if any(b["cells"][k] for k in data_bands) or b["spans"]]
    pitch = _pitch(drows if len(drows) >= 3 else [b["row"] for b in body], lay)
    out, span_cells, n_conflict = _build_rows(body, label_bands, data_bands, full, pitch, pen)

    steps = [d for o1, o2 in zip(out, out[1:]) if o1["group"] == o2["group"] and any(o1["cells"]) and any(o2["cells"])
             and (d := o2["_row"].base - o1["_row"].base) > 1]
    rpitch = statistics.median(steps) if steps else pitch
    if label_bands:
        _join_label_blocks(out, len(label_bands), rpitch, notes)
    _merge_wrapped_rows(out, len(label_bands), notes)
    _spread_centred_labels(out, rpitch, notes, pen)
    _merge_stacked_cells(out, rpitch, notes)
    _merge_fragments(out, notes, pen)
    _flag_squeezed(out, rpitch, notes, pen)
    _section_rows(out, notes)
    if (first := next((i for i in range(min(2, len(out))) if _inner_header(out, i)), None)) is not None:
        # the header's last row sits in the body, under a data row (ATLAS: "σ × B ... 39 ... 50"): the header is read
        # again with it, and a lone header word of the row above, among numbers, is the first line of its cell ("FWHM")
        o = out.pop(first)
        rows2 = hdr_rows + [o["_row"]]
        entries, unplaced = _header_entries(rows2, bands, H, part)
        _data_entries(entries, label_bands, data_bands)
        columns = _header_paths(entries, data_bands, rows2, H, split=len(hdr_rows))
        if first == 1:
            pv = out[0]
            for j, c in enumerate(pv["cells"]):
                if c and not _numberish(c) and o["cells"][j] and columns[j] and \
                        all(_numberish(x) for jj, x in enumerate(pv["cells"]) if x and jj != j):
                    columns[j][-1] = f"{c} {columns[j][-1]}"
                    pv["cells"][j] = ""
        notes.append("a header row found under the first body row: header read again with it")
        pen.append((0.05, "header row inside body"))
    _header_doubts(columns, unplaced, bool(hdr_rows), pen)
    _inner_headers(out, columns, notes, pen)
    out = _continued_labels(out, notes, pen)
    _second_label_lines(out, len(label_bands))
    if label_bands:
        _fill_multirow(out, len(label_bands), notes, pen)
    if n := sum(1 for o in out for c in o["cells"] if _two_numbers(c)):
        notes.append(f"{n} cell(s) contain two numbers separated only by a space (columns may be merged)")
        pen.append((min(0.45, 0.15 * n), "cells holding two numbers"))
    if span_cells:
        notes.append(f"{len(span_cells)} merged cell(s) spanning several columns")
        if p := _span_cost(span_cells, columns):
            pen.append((min(0.25, p), "merged cells"))
    if n_conflict:
        pen.append((min(0.45, 0.15 * n_conflict), f"{n_conflict} cell(s) received two text chunks"))
        notes.append(f"{n_conflict} cell(s) had two separate text pieces joined")
    return columns, out, {"pen": pen, "label_header": label_header, "bands": bands, "data_bands": data_bands,
                          "label_heads": _label_heads(entries, label_bands)}


# ----------------------------------------------------------------------------- tables

def _split_panels(rows: list[_Row]) -> list[tuple[Optional[str], list[_Row]]]:
    """[(panel title, rows)]: "Panel A" rows split the table when each panel has header rows of its own (else the
    panels share the header and stay one table, the panel rows becoming sections)."""
    idx = [i for i, r in enumerate(rows) if PANEL_RE.match(r.text)]

    def own_header(i):
        cs = _chunks(rows[i + 1].toks, (), 0.5) if i + 1 < len(rows) else []
        return len(cs) >= 2 and (not any(_value_like(c.text) for c in cs[1:]) or all(COLNUM_RE.match(c.text) for c in cs))
    if not idx or not any(own_header(i) for i in idx):
        return [(None, rows)]
    segs = [(None, rows[:idx[0]])] if idx[0] > 0 else []
    return segs + [(rows[a].text.strip(), rows[a + 1:b]) for a, b in zip(idx, idx[1:] + [len(rows)])]


def _find_continuation(pages: list[_Page], pno: int, reg: dict, lay: dict, by_page: dict, max_pages: int = 10) -> list:
    """[(page index, region)] of the parts of a table that runs off the bottom of its page (no closing rule): the
    region at the top of the next page(s) whose multi-cell rows line up with the columns read so far."""
    anchors = [x for r in reg["rows"] for c in _chunks(r.toks, (), 0.5)[1:] for x in (c.x0, c.x1, c.xc)]
    conts, cur = [], reg
    while anchors and len(conts) < max_pages and not cur.get("closed") and cur["y1"] >= lay["B"] - 2.5 * lay["body"] \
            and pno + 1 < len(pages):
        top = {"bbox": [cur["xa"], lay["T"] - 8, cur["xb"], lay["T"] - 7], "size": lay["body"]}
        nreg = _find_region(pages[pno + 1], top, "below", lay, by_page.get(pno + 1, []), extent=(cur["xa"], cur["xb"]),
                            context=cur["rows"])
        if not nreg or not nreg["rows"] or nreg["rows"][0].y0 > lay["T"] + 4 * lay["body"]:
            break
        multi = [cs for r in nreg["rows"] if len(cs := _chunks(r.toks, (), 0.5)) >= 2]
        aligned = val_aligned = strong = 0
        for cs in multi:
            hit = [c for c in cs[1:] if any(abs(c.x1 - a) < 2.5 or abs(c.x0 - a) < 2.5 or abs(c.xc - a) < 3 for a in anchors)]
            aligned += bool(hit)
            strong += len(hit) >= 3 and len(hit) == len(cs) - 1
            val_aligned += any(_value_like(c.text) or _numberish(c.text) for c in hit)
        if not multi or aligned < 0.6 * len(multi) or (val_aligned == 0 and strong == 0):
            break
        conts.append((pno + 1, nreg))
        cur, pno = nreg, pno + 1
    return conts


def _area(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _build_tables(cap: dict, reg: dict, pm: _Page, lay: dict) -> list[dict]:
    """The table (one per panel when the panels have headers of their own) of a caption's region, with confidence."""
    segs = _split_panels(reg["rows"])
    common = None
    if len(segs) > 1 and segs[0][0] is None:
        common, segs = segs[0][1], segs[1:]
    results = []
    for si, (title, prow) in enumerate(segs):
        if not prow:
            continue
        notes: list[str] = []
        rows = list(common) + list(prow) if common is not None and title is not None else prow
        ys0, ys1 = min(r.y0 for r in rows) - 3, max(r.y1 for r in rows) + 3
        res = _parse_grid(rows, [h for h in reg["H"] if ys0 - 4 <= h.c <= ys1 + 4], reg["V"], lay, notes)
        if res is None:
            continue
        columns, out, meta = res
        pen = list(meta["pen"])
        if sum(1 for c in columns if not c) >= 2:   # rotated column headers just above the body
            top = min(r.y0 for r in rows)
            lim = cap["bbox"][3] if cap["bbox"][3] <= top else top - 120
            rot = [ln for ln in pm.lines if ln["rot"] and lim - 1 <= ln["bbox"][1] and ln["bbox"][3] <= top + 4]
            nrot = 0
            for j, k in enumerate(meta["data_bands"]):
                a, b = meta["bands"][k]
                hits = [ln for ln in rot if a - 3 <= (ln["bbox"][0] + ln["bbox"][2]) / 2 <= b + 3]
                if not columns[j] and len(hits) == 1:
                    columns[j] = [hits[0]["text"]]
                    nrot += 1
                    ys0 = min(ys0, hits[0]["bbox"][1])
            if nrot:
                notes.append(f"{nrot} rotated column header(s) read from vertical text")
                pen = [x for x in pen if "no column headers" not in x[1] and "have no header" not in x[1]]
                if n := sum(1 for c in columns if not c):
                    pen.append((0.05 * n, "some data columns have no header"))
        if not reg["env"]:
            pen.append((0.1, "no horizontal rules: region found from text alignment only"))
        conf = 1.0 - sum(p for p, _ in pen)
        vals = [c for o in out for c in o["cells"] if c]
        if sum(1 for c in vals if _numberish(c)) < 0.3 * max(1, len(vals)):
            contlike = any(not o["label"] and any(c and not _numberish(c) for c in o["cells"]) for o in out)
            texts = vals + [o["label"] for o in out if o["label"]]
            prose = sum(len(t.split()) for t in texts) / max(1, len(texts)) > 5 or len(meta["label_header"].split()) > 8 or \
                sum(1 for t in vals if t[:1].islower()) > 0.15 * max(1, len(vals))
            conf = min(conf, 0.25 if prose else (0.35 if contlike else 0.75))
            notes.append("mostly text cells (not a numeric results table)" + ("; prose cells, grid unreliable" if prose else "")
                         + ("; multi-line text cells are not merged" if contlike and not prose else ""))
        if len(out) < 2:
            conf = min(conf, 0.4)
        if cap["kind"] == "figure":
            conf -= 0.05
            notes.append("table captioned as a figure")
        if title:
            notes.append(f"panel: {title}")
        if len(segs) > 1:
            notes.append("table has several panels with different columns; each panel is a separate entry")
        conf = max(0.05, min(0.98, conf))
        first = [r for r in rows if r.y0 < 9000] or rows   # continuation rows are shifted 10000 pt down per page
        bx0, bx1 = min(r.x0 for r in first), max(r.x1 for r in first)
        by0, by1 = min(min(r.y0 for r in first), ys0 + 3), max(r.y1 for r in first)
        for h in reg["env"]:
            if ys0 - 4 <= h.c <= min(ys1 + 4, 9000):
                bx0, bx1, by0, by1 = min(bx0, h.a), max(bx1, h.b), min(by0, h.c), max(by1, h.c)
        notes += [why for p, why in pen if p >= 0.07 and why not in " ".join(notes)]
        t = {"id": ("T" if cap["kind"] != "figure" else "F") + cap["num"] + (f".{chr(65 + si)}" if len(segs) > 1 else ""),
             "label": cap["label"], "page": pm.pno + 1, "bbox": [round(v, 1) for v in (bx0, by0, bx1, by1)],
             "caption": cap["text"], "columns": columns, "rows": out, "confidence": round(conf, 2), "notes": notes,
             "label_heads": meta["label_heads"]}
        if title:
            t["panel"] = title
        if reg.get("cont") and any(r.y0 > 9000 for r in rows):
            t["notes"].append("table continues on page(s) " + ", ".join(str(p) for p in reg["cont"]))
            t["confidence"] = round(max(0.05, t["confidence"] - 0.05), 2)
        results.append(t)
    return results


def _figure_table_ok(t: dict, pm: _Page) -> bool:
    """A figure caption holds a table only when it reads like one: labelled rows of numbers under headers, no plot."""
    if len(t["columns"]) < 2 or len(t["rows"]) < 3 or t["confidence"] < 0.4:
        return False
    vals = [c for r in t["rows"] for c in r["cells"] if c]
    if sum(1 for r in t["rows"] if r["label"]) < 0.6 * len(t["rows"]) or sum(1 for c in t["columns"] if c) < 0.5 * len(t["columns"]) \
            or sum(1 for c in vals if _numberish(c)) < 0.5 * max(1, len(vals)):
        return False
    x0, y0, x1, y1 = t["bbox"]
    return sum(1 for s in pm.shapes if s[0] > x0 - 30 and s[2] < x1 + 30 and s[1] > y0 - 30 and s[3] < y1 + 30
               and (s[2] - s[0]) * (s[3] - s[1]) < 0.5 * (x1 - x0) * (y1 - y0)) <= 4


def _caption_tables(c: dict, pages: list[_Page], lay: dict, prior: dict, claimed: list, by_page: dict) -> list[dict]:
    """The tables of one caption: its region on the side the paper's tables use (the other side when that one is
    clearly the table, or is already another caption's), with the parts on the next pages."""
    pm = pages[c["page"]]
    figure = c["kind"] == "figure"
    (ra, qa), (rb, qb) = c["reg"]["above"], c["reg"]["below"]
    if qa <= 0 and qb <= 0:
        return []
    if qa > 0 and qb > 0:
        side = prior["figure" if figure else "table"]
        if (side == "above" and qb > 2.2 * qa) or (side == "below" and qa > 2.2 * qb):
            side = "below" if side == "above" else "above"
    else:
        side = "above" if qa > qb else "below"
    reg = ra if side == "above" else rb
    box = (reg["xa"], reg["y0"], reg["xb"], reg["y1"])
    if any(p == c["page"] and _area(b, box) > 0.5 * min(_area(b, b), _area(box, box)) for p, b in claimed):
        other, oq = (rb, qb) if side == "above" else (ra, qa)
        if not (other and oq > 0):
            return []
        reg = other
    x0, y0, x1, y1 = reg["xa"], reg["y0"], reg["xb"], reg["y1"]
    area = (x1 - x0) * (y1 - y0)
    img = sum(_area(im, (x0, y0, x1, y1)) for im in pm.images)
    if figure:
        paths = sum(k for g, k in pm.graphics if g[2] > x0 and g[0] < x1 and g[3] > y0 and g[1] < y1)
        numbers = sum(1 for r in reg["rows"] for t in r.toks if _numberish(t.text))
        if paths > 6 or img > 0.05 * area or len(reg["rows"]) < 3 or numbers < 6:
            return []
    elif img > 0.25 * area and len(reg["rows"]) < 3:
        return []   # a table that is an image
    if pm.rotated > 20 and len(reg["rows"]) < 2:
        return []
    reg = dict(reg, cont=[])
    if not figure and (conts := _find_continuation(pages, c["page"], reg, lay, by_page)):
        rows, H, V = list(reg["rows"]), list(reg["H"]), list(reg["V"])
        head = {re.sub(r"\s+", "", r.text) for r in reg["rows"][:5]}
        for ci, (pno, nreg) in enumerate(conts, start=1):
            dy = 10000.0 * ci   # the parts are stacked below the first one
            crow = list(nreg["rows"])
            while crow and re.sub(r"\s+", "", crow[0].text) in head:
                crow.pop(0)   # the header repeated on the next page
            if not crow:
                continue
            rows += [_Row([t.moved(dy) for t in r.toks]) for r in crow]
            H += [_Rule(h.c + dy, h.a, h.b, h.w) for h in nreg["H"]]
            V += [_Rule(v.c, v.a + dy, v.b + dy, v.w) for v in nreg["V"]]
            reg["cont"].append(pno + 1)
            claimed.append((pno, (nreg["xa"], nreg["y0"], nreg["xb"], nreg["y1"])))
        reg.update(rows=rows, H=H, V=V)
    ts = _build_tables(c, reg, pm, lay)
    if figure:
        ts = [t for t in ts if _figure_table_ok(t, pm)]
    if ts:
        claimed.append((c["page"], box))
    return [t for t in ts if t["confidence"] >= 0.3]


def _extract(doc) -> list[dict]:
    """The parsed tables of a document, before the ledger conventions (_finish)."""
    pages = [_Page(doc[i], i) for i in range(doc.page_count)]
    lay = _layout(pages)
    caps = _find_captions(pages, lay)
    by_page = defaultdict(list)
    for c in caps:
        by_page[c["page"]].append(c)
    for c in caps:
        c["reg"] = {}
        for side in ("below", "above"):
            try:
                reg = _find_region(pages[c["page"]], c, side, lay, by_page[c["page"]])
            except Exception:  # noqa: BLE001
                reg = None
            c["reg"][side] = (reg, _region_quality(reg))
    votes = {"table": Counter(), "figure": Counter()}   # which side of their captions the paper's tables are
    for c in caps:
        qa, qb = c["reg"]["above"][1], c["reg"]["below"][1]
        k = "figure" if c["kind"] == "figure" else "table"
        if qa > 6 and qb < 2:
            votes[k]["above"] += 1
        elif qb > 6 and qa < 2:
            votes[k]["below"] += 1
    prior = {k: v.most_common(1)[0][0] if v else ("below" if k == "table" else "above") for k, v in votes.items()}
    tables, claimed = [], []
    for c in caps:
        try:
            tables += _caption_tables(c, pages, lay, prior, claimed, by_page)
        except Exception as e:  # noqa: BLE001 - one table that cannot be read never costs the others
            print(f"[pdf_tables] {c['label']} not read: {type(e).__name__}: {e}")
    return tables


STUB_VALUE = re.compile(r"^(?:\d+|\d+(?:\.\d+)?\s?[KMBTkmbt]|\d+\s?/\s?\d+|\d+%|(?:19|20)\d\d)$")


def _names_rows(values: list[str]) -> bool:
    """Values that name a row rather than measure it: integers, sizes ("7B", "1.7M"), bit widths ("3/16"), rates
    ("80%"), years, words ("7 TeV")."""
    vals = [v for v in values if v and v not in DASHES]
    return bool(vals) and all(STUB_VALUE.match(v) or not _numberish(v) for v in vals)


def _fold_stubs(columns: list[list[str]], rows: list[dict], heads: list[str]) -> int:
    """Leading columns that name the rows join the label (" | "): with no label column, the first column and the
    columns of its header group or of words; else the one or two leading columns of names, filled on every row whose
    label repeats in its section, that make the labels unique ("GPTQ | 3/16", "ResNet | 110"). Returns how many
    columns were folded."""
    data = [r for r in rows if any(r["cells"])]
    if not data or len(columns) < 2:
        return 0

    def fold():
        for r in rows:
            r["label"] = " | ".join(x for x in (r["label"], r["cells"][0]) if x)
            del r["cells"][0], r["bold"][0]
        heads.append(" › ".join(columns.pop(0)))

    if not any(r["label"] for r in data):
        group = columns[0][0] if len(columns[0]) > 1 else None
        fold()
        n = 1
        while len(columns) > 1 and ((group and len(columns[0]) > 1 and columns[0][0] == group)
                                    or not any(_numberish(r["cells"][0]) for r in data if r["cells"][0])):
            fold()
            n += 1
        return n
    seen = Counter((r["section"], r["label"]) for r in data)
    repeated = [r for r in data if seen[(r["section"], r["label"])] > 1]
    if not repeated:
        return 0
    for k in (1, 2):
        if len(columns) <= k:
            break
        if all(_names_rows([r["cells"][j] for r in data]) and all(r["cells"][j] for r in repeated) for j in range(k)):
            keys = Counter((r["section"], r["label"], *r["cells"][:k]) for r in data)
            if max(keys.values()) == 1:
                for _ in range(k):
                    fold()
                return k
    return 0


def _finish(t: dict) -> Optional[dict]:
    """The ledger's view of a parsed table: a value spanning columns is repeated in each; a standard-error row is
    folded into its coefficient ("0.120*** (0.029)"); a column-number row ("(1) (2) (3)") under the header becomes
    its last level; leading columns that name the rows join the label; a row with values but no label is dropped (it
    costs confidence: its values belong to a row the table did not show); each section starts with a row that has
    the section name as label and no cells; header paths are unique and never empty. None when no row is left."""
    columns = [list(p) for p in t["columns"]]
    notes, pen = list(t["notes"]), 0.0
    rows = []
    for o in t["rows"]:
        r = {"label": o["label"], "group": o["group"], "section": o.get("_section") or "", "cells": list(o["cells"]),
             "bold": list(o["bold"]), "se": o.get("stat") == "se"}
        for a, b in o["_spans"]:
            for j in range(a + 1, b + 1):
                if not r["cells"][j]:
                    r["cells"][j], r["bold"][j] = r["cells"][a], r["bold"][a]
        rows.append(r)
    out: list[dict] = []
    for r in rows:
        if not r["se"]:
            out.append(r)
            continue
        prev = out[-1] if out else None
        if prev is None or prev.get("folded") or prev["group"] != r["group"] or not any(prev["cells"]):
            pen += 0.05   # a standard error without its coefficient
            continue
        prev["cells"] = [f"{c} {e}" if c and e else c for c, e in zip(prev["cells"], r["cells"])]
        prev["folded"] = True
    rows, seen_data = [], False
    for r in out:
        filled = [c for c in r["cells"] if c]
        if not r["label"] and filled and all(COLNUM_RE.match(c) for c in filled):
            if not seen_data:
                for j, c in enumerate(r["cells"]):
                    if c:
                        columns[j].append(c)
            continue
        seen_data = seen_data or bool(filled)
        rows.append(r)
    heads = list(t["label_heads"])
    if n := _fold_stubs(columns, rows, heads):
        notes.append(f"{n} column(s) that name the rows folded into the row labels")
    orphans = [r for r in rows if not r["label"] and any(r["cells"])]
    if orphans:
        pen += min(0.3, 0.1 * len(orphans))
        notes.append(f"{len(orphans)} row(s) with values but no label left out")
    rows = [r for r in rows if r["label"]]
    while rows and not any(rows[-1]["cells"]):
        rows.pop()   # label-only lines after the last row of values: notes, not sections
    if not rows:
        return None
    stray = [r for r in rows if not any(r["cells"])]   # section headings are not rows here: lines of a label not joined
    if stray:
        pen += min(0.3, 0.1 * len(stray))
        notes.append(f"{len(stray)} label line(s) not joined to a row left out")
        rows = [r for r in rows if any(r["cells"])]
    final, section, gmap = [], "", {}
    for r in rows:
        g = gmap.setdefault(r["group"], len(gmap))
        if r["section"] and r["section"] != section and r["section"] != r["label"]:
            final.append({"label": r["section"], "group": g, "section": r["section"], "cells": [""] * len(columns),
                          "bold": [False] * len(columns)})
        section = r["section"]
        final.append({"label": r["label"], "group": g, "section": r["section"], "cells": r["cells"], "bold": r["bold"]})
    counts = Counter(tuple(p) for p in columns)
    for j, p in enumerate(columns):
        if not any(x.strip() for x in p):
            columns[j] = [f"column {j + 1}"]
        elif counts[tuple(p)] > 1:
            columns[j] = p[:-1] + [f"{p[-1]} (column {j + 1})"]
    out_t = {"id": t["id"], "label": t["label"], "page": t["page"], "bbox": t["bbox"], "caption": t["caption"],
             "columns": columns, "rows": final, "confidence": round(max(0.05, t["confidence"] - pen), 2),
             "notes": "; ".join(notes)}
    if any(heads):
        out_t["label_header"] = " | ".join(heads)
    return out_t


def extract_tables(source: Union[bytes, str, Path]) -> list[dict]:
    """Every table of a PDF (bytes or a path) whose grid could be read, in page order; [] when the PDF cannot be
    read. Never raises. A table: {"id", "label", "page" (1-based), "bbox", "caption", "columns" (one header path per
    data column, top level first), "rows": [{"label", "group", "section", "cells", "bold"}] (a row with a label and
    no cells is a section heading), "confidence", "notes"}."""
    try:
        import pymupdf

        doc = pymupdf.open(str(source)) if isinstance(source, (str, Path)) else pymupdf.open(stream=bytes(source), filetype="pdf")
    except Exception as e:  # noqa: BLE001
        print(f"[pdf_tables] PDF not opened: {type(e).__name__}: {e}")
        return []
    out = []
    try:
        for t in _extract(doc):
            try:
                if (f := _finish(t)) is not None:
                    out.append(f)
            except Exception as e:  # noqa: BLE001
                print(f"[pdf_tables] {t['label']} not read: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[pdf_tables] tables not read: {type(e).__name__}: {e}")
    finally:
        doc.close()
    return out
