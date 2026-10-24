"""
Paper text extraction (v2)

Why a new module: pdfplumber's default ``x_tolerance=3`` glues words together on LaTeX papers
("Thedominantsequencetransductionmodels…") and the old parser fed references/appendix into the LLM.
This extractor uses PyMuPDF (fallback: pdfplumber with ``x_tolerance=1``), reads two-column pages column by column
while keeping table rows together, repairs hyphenation, drops running headers / arXiv stamps, stops at the
References section and keeps figure/table captions. The tables themselves are read from the page geometry
(services.pdf_tables) and appended to the text as the evidence ledger (services.ledger), which charts cite by cell.
"""

from __future__ import annotations

import io
import math
import re
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from itertools import accumulate
from pathlib import Path
from statistics import median
from typing import NamedTuple, Union


@dataclass
class PaperText:
    title: str
    body: str                                           # ends with the ledger section when tables were read
    captions: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    tables: list[dict] = field(default_factory=list)    # services.pdf_tables output, as read (saved with a job)


REFERENCE_HEADING = re.compile(r"^\s*(?:\d+\.?\s*)?(references|bibliography|reference list)\s*$", re.I)
ACK_HEADING = re.compile(r"^\s*(?:\d+\.?\s*)?acknowledg(?:e)?ments?\s*$", re.I)
# run-in "Acknowledgements We thank …": capitalised, so a sentence wrapped onto "acknowledgment that …" is not one
ACK_RUN_IN = re.compile(r"^\s*(?:\d+\.?\s*)?(?:Acknowledg(?:e)?ments?|ACKNOWLEDG(?:E)?MENTS?)\b[.:]?(?=\s|$)")
ARXIV_STAMP = re.compile(r"arXiv:\d{4}\.\d{4,5}v\d+\s*\[[^\]]+\]\s*\d{1,2}\s+\w{3}\s+\d{4}")
CAPTION = re.compile(r"^(?:Figure|Fig\.|Table)\s*\d+[.:]\s*(.+)$", re.I)
CITATION_NOISE = re.compile(r"\s*\[(?:\d+(?:[,–-]\s*\d+)*)\]")


def _read_pages_pymupdf(data: bytes) -> tuple[list[str], dict]:
    import pymupdf  # PyMuPDF

    doc = pymupdf.open(stream=data, filetype="pdf")
    body = _body_family(doc)
    pages = [_fix_ligatures(_with_figure_sentences(_page_text_reading_order(page, body), page.get_text())) for page in doc]
    meta = dict(doc.metadata or {})
    meta["_largest_font_line"] = _fix_ligatures(_largest_font_line(doc[0])) if len(doc) else ""
    return pages, meta


class _Line(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    horizontal: bool


MARGIN = 0.08      # top/bottom share of the page that holds running heads, footers and page numbers
CROSS_TOL = 0.05   # a full-width line reaches this share of the page width past the gutter on both sides


def _page_lines(page, body: str = "") -> tuple[list[_Line], float, float]:
    import pymupdf

    d = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
    lines = []
    for block in d.get("blocks", []):
        for ln in block.get("lines", []):
            spans = ln.get("spans") or []
            text = _join_spans(spans, body)
            if not text.strip():
                continue
            x0, y0, x1, y1 = ln["bbox"]
            dx, dy = ln.get("dir", (1.0, 0.0))
            horizontal = dx > 0 and abs(dy) < 0.05
            if horizontal:
                # rows are matched on a box derived from the baseline: tall glyphs (radicals, sums, fractions)
                # stretch the bbox and would merge neighbouring lines
                main = max(spans, key=lambda s: len(s.get("text", "").strip()))
                base, size = main["origin"][1], main["size"]
                y0, y1 = base - 0.75 * size, base + 0.25 * size
            lines.append(_Line(x0, y0, x1, y1, text, horizontal))
    return lines, float(d.get("width") or 1.0), float(d.get("height") or 1.0)


SUPERSCRIPT = 1  # PyMuPDF span flag
NUMBER_RUN = re.compile(r"^[−–-]?\d+(?:\.\d+)?$")
ALWAYS_MATH = re.compile(r"^(?:CM(?:MI|SY|EX)\d|MSAM|MSBM|.*Math|txsy|txmi|rsfs)", re.I)   # never a body text font
ROMAN_MATH = re.compile(r"^(?:CMR\d|CMBX\d|LMRoman\d)", re.I)                         # math digits, or the body font of CM papers
SCI_BASE = re.compile(r"(?:^|[^\d.,])10$")
OPERATOR_NEXT = re.compile(r"^\s*(?:[=<>≤≥≈×·+−÷]|-\s*\d)")


def _font_family(font: str) -> str:
    """'CMR10' -> 'CMR', 'NimbusRomNo9L-Regu' -> 'NimbusRomNo9L', 'LMRoman10-Regular' -> 'LMRoman'."""
    return re.sub(r"\d+$", "", re.split(r"[-+,]", str(font or ""))[0])


def _body_family(doc, pages: int = 12) -> str:
    """The font family of most of the running text, so a superscript set in it can be told apart from math."""
    counts: Counter[str] = Counter()
    for page in list(doc)[:pages]:
        try:
            blocks = page.get_text("dict").get("blocks", [])
        except Exception:
            continue
        for block in blocks:
            for line in block.get("lines", []):
                for sp in line.get("spans", []):
                    if not sp.get("flags", 0) & SUPERSCRIPT and sp.get("size", 0) >= 8:
                        counts[_font_family(sp.get("font", ""))] += len(sp.get("text", "").strip())
    return counts.most_common(1)[0][0] if counts else ""


def _math_font(font: str, body: str) -> bool:
    return bool(ALWAYS_MATH.match(font)) or (bool(ROMAN_MATH.match(font)) and _font_family(font) != body)


def _join_spans(spans: list[dict], body: str = "") -> str:
    """Span texts of one line. A run of superscript spans attached to a number keeps its meaning: an exponent (of 10, in
    a math font, signed, or followed by an operator) becomes "^" ("2.3 · 10^19", "10^−9", "10000^(2i/dmodel)", "n^2",
    "2^16 = 65,536"), while other digits after a number are a footnote or citation mark ("4096[2]", which _clean_body
    removes) instead of "40962". ``body`` is the document's running-text font family: in a Computer Modern paper the
    footnote marks are set in CMR too, so that family alone does not mean math."""
    text, i = "", 0
    while i < len(spans):
        if not spans[i].get("flags", 0) & SUPERSCRIPT:
            nxt = spans[i].get("text", "")
            if text.endswith("\x00"):   # a word straight after an exponent gets its space back
                text = text[:-1] + (" " if nxt[:1].isalnum() else "")
            text += nxt
            i += 1
            continue
        j, run = i, ""
        while j < len(spans) and spans[j].get("flags", 0) & SUPERSCRIPT:
            run += spans[j].get("text", "")
            j += 1
        core, trail = run.strip(), run[len(run.rstrip()):]
        attached = bool(core) and run[:1] == core[:1] and text[-1:].strip() != ""
        math = any(_math_font(spans[k].get("font", ""), body) for k in range(i, j))
        after = spans[j].get("text", "") if j < len(spans) else ""
        exponent = math or core[:1] in "−–-" or SCI_BASE.search(text) or OPERATOR_NEXT.match(trail + after)
        if attached and text[-1:].isdigit() and NUMBER_RUN.match(core) and exponent:
            text += "^" + core + (trail or "\x00")
        elif attached and text[-1:].isdigit() and math and re.match(r"^[\w./+−–-]+$", core):
            text += "^(" + core + ")" + trail
        elif attached and text[-1:].isalpha() and math and NUMBER_RUN.match(core):
            text += "^" + core + (trail or "\x00")
        elif attached and text[-1:].isdigit() and re.fullmatch(r"\d+(?:[,–-]\d+)*", core):
            text += "[" + core + "]" + trail
        else:
            text += run
        i = j
    return text.replace("\x00", "")


def _rows_text(lines: list[_Line]) -> list[str]:
    """Rows as PyMuPDF's ``sort=True`` builds them: lines whose tops or bottoms are within 3 pt share a row (a table
    row stays together). Cells start at the offset of their x position in one character width per region, so a
    value lines up with its column header even in a row with empty cells. Rotated lines (arXiv stamp, axis labels)
    stay rows of their own instead of being glued into body lines."""
    rows: list[list] = []
    for ln in sorted((ln for ln in lines if ln.horizontal), key=lambda ln: (ln.y1, ln.x0)):
        if rows and (abs(ln.y0 - rows[-1][0]) <= 3 or abs(ln.y1 - rows[-1][1]) <= 3):
            rows[-1][0], rows[-1][1] = min(rows[-1][0], ln.y0), max(rows[-1][1], ln.y1)
            rows[-1][2].append(ln)
        else:
            rows.append([ln.y0, ln.y1, [ln]])
    rows += [[ln.y0, ln.y1, [ln]] for ln in lines if not ln.horizontal]
    rows.sort(key=lambda r: r[1])
    horizontal = [ln for ln in lines if ln.horizontal]
    origin = min((ln.x0 for ln in horizontal), default=0.0)
    char_w = median([(ln.x1 - ln.x0) / len(ln.text.strip()) for ln in horizontal] or [1.0]) or 1.0
    out = []
    for _, _, parts in rows:
        parts.sort(key=lambda ln: ln.x0)
        if len(parts) == 1:
            out.append(parts[0].text.strip())
            continue
        for i in range(1, len(parts)):  # stacked parts read top to bottom: a numerator before its denominator
            for j in range(i, 0, -1):
                a, b = parts[j - 1], parts[j]
                if not (_stacked(a, b) and b.y0 + b.y1 + 2 < a.y0 + a.y1):
                    break
                parts[j - 1], parts[j] = b, a
        text, x, prev = "", origin, None
        for ln in parts:
            gap = ln.x0 - x
            if text and gap < 1.5 * char_w:  # MuPDF split one visual line: keep at most the word gap
                n = int(gap > 0 or _stacked(prev, ln) or prev.text[-1:].isspace() or ln.text[:1].isspace())
            else:
                n = max(round((ln.x0 - origin) / char_w) - len(text), 2 if text else 0)
            text += " " * n + ln.text.strip()
            x, prev = ln.x1, ln
        out.append(text)
    return out


def _stacked(a: _Line, b: _Line) -> bool:
    """One above the other (a fraction's numerator and denominator, stacked figure labels), not side by side."""
    return min(a.x1, b.x1) - max(a.x0, b.x0) > 0.5 * min(a.x1 - a.x0, b.x1 - b.x0)


def _crosses(ln: _Line, gutter: float, tol: float) -> bool:
    return ln.x0 < gutter - tol and ln.x1 > gutter + tol


def _is_left(ln: _Line, gutter: float) -> bool:
    return ln.x0 + ln.x1 < 2 * gutter


def _is_prose(ln: _Line, width: float) -> bool:
    """A line of running text, not a table cell, figure label or equation fragment."""
    return ln.horizontal and len("".join(ln.text.split())) >= 20 and ln.x1 - ln.x0 >= 0.15 * width


def _gutter(lines: list[_Line], width: float) -> float:
    """Between the running text on the left and on the right: first the x in the middle 40% of the page that the
    fewest prose lines cross (centre of the widest such stretch), then the midpoint between the typical right edge
    of the prose left of it and the typical left edge of the prose right of it."""
    prose = [ln for ln in lines if _is_prose(ln, width)]
    tol, lo, hi = CROSS_TOL * width, int(0.3 * width), int(0.7 * width)
    diff = [0] * (hi - lo + 2)
    for ln in prose:
        a, b = max(lo, math.floor(ln.x0 + tol) + 1), min(hi, math.ceil(ln.x1 - tol) - 1)
        if a <= b:
            diff[a - lo] += 1
            diff[b - lo + 1] -= 1
    counts = list(accumulate(diff[:-1]))
    best = min(counts)
    runs, start = [], None
    for i, c in enumerate(counts + [best + 1]):
        if c == best and start is None:
            start = i
        elif c != best and start is not None:
            runs.append((i - start, start))
            start = None
    length, start = max(runs)
    gutter = lo + start + (length - 1) / 2
    sides = [ln for ln in prose if not _crosses(ln, gutter, tol)]
    left = [ln.x1 for ln in sides if _is_left(ln, gutter)]
    right = [ln.x0 for ln in sides if not _is_left(ln, gutter)]
    if left and right and median(left) < median(right):
        return (median(left) + median(right)) / 2
    return gutter


def _column_text(lines: list[_Line], gutter: float, width: float) -> set[_Line]:
    """Prose lines on one side of the gutter with another prose line right above or below on that side: paragraphs
    in a column or beside a figure. Table cells, equations and a paragraph's lone last line do not qualify."""
    tol = CROSS_TOL * width
    found: set[_Line] = set()
    for left in (True, False):
        side = sorted((ln for ln in lines if _is_prose(ln, width) and not _crosses(ln, gutter, tol)
                       and _is_left(ln, gutter) == left), key=lambda ln: ln.y0)
        for a, b in zip(side, side[1:]):
            overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
            if -2 <= b.y0 - a.y1 <= a.y1 - a.y0 and overlap >= 0.5 * min(a.x1 - a.x0, b.x1 - b.x0):
                found.update((a, b))
    return found


def _page_edges(lines: list[_Line], height: float) -> tuple[list[_Line], list[_Line]]:
    """Running heads at the top and footers / page numbers at the bottom: the lines within the top or bottom 8% of
    the page that a clear gap separates from everything else (a column may start or end that close to the edge)."""
    gap = 0.012 * height
    top: list[_Line] = []
    ordered, reach = sorted(lines, key=lambda ln: ln.y0), -math.inf
    for i, ln in enumerate(ordered):
        if i and ln.y0 - reach >= gap:
            top = ordered[:i] if reach <= MARGIN * height else []
            break
        reach = max(reach, ln.y1)
    bottom: list[_Line] = []
    ordered, reach = sorted(lines, key=lambda ln: -ln.y1), math.inf
    for i, ln in enumerate(ordered):
        if i and reach - ln.y1 >= gap:
            bottom = ordered[:i] if reach >= (1 - MARGIN) * height else []
            break
        reach = min(reach, ln.y0)
    return top, bottom


def _two_sided(lines: list[_Line], column_text: set[_Line], gutter: float, total: int = 0) -> bool:
    left = sum(1 for ln in lines if ln in column_text and _is_left(ln, gutter))
    right = sum(1 for ln in lines if ln in column_text and not _is_left(ln, gutter))
    return left >= 2 and right >= 2 and left + right >= total


def _page_text_reading_order(page, body: str = "") -> str:
    """Column-aware page text.

    PyMuPDF's ``sort=True`` merges left/right column lines that share a baseline. A page is read column by column
    only when both sides of its gutter carry running text (two-column papers, text beside a figure). There, lines
    crossing the gutter (titles, full-width captions and tables) split the page into bands, and a band is read left
    side then right side if both of its sides hold running text. Everything else (other pages, bands of table cells,
    equations or figure labels) is read row by row like ``sort=True``, which keeps table rows together.
    """
    try:
        lines, width, height = _page_lines(page, body)
    except Exception:
        return page.get_text("text", sort=True)
    horizontal = [ln for ln in lines if ln.horizontal]
    gutter = _gutter(horizontal, width)
    column_text = _column_text(horizontal, gutter, width)
    if not _two_sided(horizontal, column_text, gutter, total=8):
        return "\n".join(_rows_text(lines))
    top, bottom = _page_edges(lines, height)  # running heads, footers and page numbers stay at the page edges
    edge = set(top + bottom)
    main = [ln for ln in lines if ln not in edge]
    body = [ln for ln in main if ln.horizontal]
    tol = CROSS_TOL * width
    spans: list[list] = []  # [y0, y1, lines] of the full-width stretches, top to bottom
    for ln in sorted((ln for ln in body if _crosses(ln, gutter, tol)), key=lambda ln: ln.y0):
        if spans and ln.y0 <= spans[-1][1] + 3:
            spans[-1][1] = max(spans[-1][1], ln.y1)
            spans[-1][2].append(ln)
        else:
            spans.append([ln.y0, ln.y1, [ln]])
    starts = [s[0] for s in spans]
    bands: list[list[_Line]] = [[] for _ in range(len(spans) + 1)]
    for ln in main:
        if ln.horizontal and _crosses(ln, gutter, tol):
            continue
        mid = (ln.y0 + ln.y1) / 2
        k = bisect_right(starts, mid)
        if k and mid <= spans[k - 1][1]:
            spans[k - 1][2].append(ln)  # e.g. a table cell in a row whose other cells cross the gutter
        else:
            bands[k].append(ln)
    # one line each, so a head and the page number on its baseline are matched separately by _strip_running_headers
    out = [ln.text.strip() for ln in sorted(top, key=lambda ln: (ln.y1, ln.x0))]
    for k, band in enumerate(bands):
        if _two_sided(band, column_text, gutter):
            out += _rows_text([ln for ln in band if _is_left(ln, gutter)])
            out += _rows_text([ln for ln in band if not _is_left(ln, gutter)])
        else:
            out += _rows_text(band)
        if k < len(spans):
            out += _rows_text(spans[k][2])
    out += [ln.text.strip() for ln in sorted(bottom, key=lambda ln: (ln.y1, ln.x0))]
    return "\n".join(out)


def _largest_font_line(page) -> str:
    """The text set in the largest font on the first page is almost always the paper title."""
    try:
        blocks = page.get_text("dict").get("blocks", [])
    except Exception:
        return ""
    best_size, best_lines = 0.0, []
    for block in blocks:
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            size = max(s.get("size", 0) for s in spans)
            text = "".join(s["text"] for s in spans).strip()
            if len(text) < 3 or ARXIV_STAMP.search(text):
                continue
            if size > best_size + 0.5:
                best_size, best_lines = size, [text]
            elif abs(size - best_size) <= 0.5:
                best_lines.append(text)
    title = " ".join(best_lines).strip()
    return title if 8 <= len(title) <= 200 else ""


def _read_pages_pdfplumber(data: bytes) -> tuple[list[str], dict]:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = [_fix_ligatures(page.extract_text(x_tolerance=1) or "") for page in pdf.pages]
        meta = pdf.metadata or {}
    return pages, meta


def _strip_running_headers(pages: list[str]) -> list[str]:
    """Remove lines that repeat on ≥3 pages at the top/bottom (journal headers, page footers), page numbers and
    arXiv stamps."""
    if len(pages) < 3:
        return pages

    def key(ln: str) -> str:  # a two-part head on one baseline is one row with a wide gap
        return " ".join(ln.split())

    counter: Counter[str] = Counter()
    for page in pages:
        lines = [key(ln) for ln in page.split("\n") if ln.strip()]
        edge = set(lines[:3] + lines[-3:])
        for ln in edge:
            if 3 <= len(ln) <= 120:
                counter[ln] += 1
    repeated = {ln for ln, n in counter.items() if n >= 3 and not ln.isdigit()}
    cleaned = []
    for page in pages:
        lines = page.split("\n")
        filled = [i for i, ln in enumerate(lines) if ln.strip()]
        edge = set(filled[:3] + filled[-3:])  # page numbers; a digit-only line mid-page is a table cell
        out = []
        for i, ln in enumerate(lines):
            s = key(ln)
            if s in repeated or (s.isdigit() and i in edge) or ARXIV_STAMP.search(s):
                continue
            out.append(ln)
        cleaned.append("\n".join(out))
    return cleaned


def _detect_title(pages: list[str], meta: dict) -> str:
    title = _fix_ligatures(meta.get("title") or "").strip()
    if 8 <= len(title) <= 200 and not title.lower().endswith((".pdf", ".tex", ".dvi")) and " " in title:
        return title
    if meta.get("_largest_font_line"):
        return meta["_largest_font_line"]
    for ln in pages[0].split("\n") if pages else []:
        s = ln.strip()
        if 10 <= len(s) <= 160 and not ARXIV_STAMP.search(s) and "@" not in s and not s.lower().startswith(("provided proper", "preprint", "under review")):
            return s
    return ""


YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def _looks_like_bibliography(lines: list[str]) -> bool:
    window = " ".join(lines[:40])
    return (len(YEAR.findall(window)) >= 5 or window.count("arXiv") >= 2 or window.count("et al") >= 3
            or len(re.findall(r"\bIn\s+(?:Proceedings|Advances|International|Conference)", window)) >= 2)


# the first appendix heading after the bibliography: "Appendix", "A   Additional Details for BERT", "A. Object Detection"
APPENDIX_HEADING = re.compile(r"^\s*(?:Appendix\b|Appendices\b|APPENDIX\b|Supplementary\s+(?:Material|Information)\b"
                              r"|A(?:\.|\s)\s*[A-Z][A-Za-z-]+(?:\s+[A-Za-z-]+){0,8}\s*$)")


REFERENCE_LINE = re.compile(r"^\s*\[\d+\]|\b(?:In|Proceedings of)\s+(?:Proceedings|CVPR|ICCV|ECCV|NIPS|NeurIPS|ICML|ICLR|ACL|EMNLP"
                            r"|NAACL|AAAI|IJCAI)\b|\barXiv\b|\bet al\.|\b(?:19|20)\d{2}[a-z]?\.\s*$")


def _reference_like(lines: list[str]) -> bool:
    """Mostly reference entries ([12] ..., In Proceedings ..., arXiv, a year ending the line); prose that merely
    mentions dataset years (VOC 2007, COCO 2015) is not."""
    filled = [ln for ln in lines if ln.strip()]
    return bool(filled) and sum(bool(REFERENCE_LINE.search(ln)) for ln in filled) >= 0.25 * len(filled)


# an untitled-appendix heading such as "Attention Visualizations" (Attention Is All You Need): 1-6 capitalised words
GENERIC_HEADING = re.compile(r"^\s*[A-Z][A-Za-z-]+(?:\s+(?:[A-Z][A-Za-z-]+|of|and|for|the|on|in|to|with)){0,5}\s*$")


def _appendix_after(lines: list[str], start: int) -> int | None:
    """Index of the appendix heading that follows the bibliography (which starts at ``start``), if any: appendices hold
    training details, ablations, result tables and the paper's own worked examples the story needs."""
    for i in range(start + 1, len(lines)):
        ln = lines[i].strip()
        if (len(ln) < 80 and "," not in ln and (APPENDIX_HEADING.match(ln) or GENERIC_HEADING.match(ln))
                and not REFERENCE_LINE.search(ln) and not _reference_like(lines[i + 1:i + 41])
                and sum(len(x) for x in lines[i:]) >= 1500):
            return i
    return None


CAPTION_START = re.compile(r"^\s*(?:Figure|Fig\.|Table)\s+\d+[.:]")


def _captions_after(lines: list[str], start: int) -> list[str]:
    """Figure/table captions printed after the bibliography (a figure page with no appendix heading)."""
    out, i = [], start + 1
    while i < len(lines):
        if CAPTION_START.match(lines[i]):
            para = [lines[i].strip()]
            i += 1
            while i < len(lines) and len(lines[i].strip()) > 25 and not CAPTION_START.match(lines[i]):
                para.append(lines[i].strip())
                i += 1
            out.append(" ".join(para))
            continue
        i += 1
    return out


TOKEN_LINE = re.compile(r"^\s*\S{1,20}\s*$")
WORD_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’-]*|[,.;:!?\-–—]")


def _token_sentences(raw: str, min_run: int = 8) -> list[str]:
    """Sentences a figure prints one word per line (attention maps), in the PDF's drawing order, each once."""
    lines, out, i = raw.split("\n"), [], 0
    while i < len(lines):
        j = i
        while j < len(lines) and TOKEN_LINE.match(lines[j]):
            j += 1
        if j - i >= min_run:
            joined = re.sub(r"(?:\s*<pad>)+", "", " ".join(t.strip() for t in lines[i:j]))
            ended = "<EOS>" in joined
            for sentence in re.split(r"\s*<EOS>\s*", joined):
                sentence = sentence.strip()
                words = sentence.split()
                alpha = [w for w in words if WORD_TOKEN.fullmatch(w)]
                # a sentence of words (not a column of table cells, axis ticks or repeated block labels such as "Trm"),
                # that ends like a sentence; each sentence once even if the figure prints it with a different tail
                if (len(alpha) >= 6 and len(alpha) >= 0.75 * len(words) and len({w.lower() for w in alpha}) >= 0.6 * len(alpha)
                        and (ended or words[-1] in (".", "!", "?")) and not any(o[:40] == sentence[:40] for o in out)):
                    out.append(sentence)
            i = j
        else:
            i += 1
    return out


def _dedupe_examples(text: str) -> str:
    """Several figure pages repeat the same example sentence: keep it once."""
    seen, out = set(), []
    for ln in text.split("\n"):
        if ln.startswith("Example sentence shown in a figure:"):
            key = ln[:80]
            if key in seen:
                continue
            seen.add(key)
        out.append(ln)
    return "\n".join(out)


def _with_figure_sentences(text: str, raw: str) -> str:
    """Reading-order extraction scrambles a figure's vertical word list; replace those scrambled lines with the
    sentences in drawing order (the paper's own worked examples are the best running examples for the story)."""
    sentences = _token_sentences(raw)
    if not sentences:
        return text
    vocab = {w.lower() for s in sentences for w in s.split()} | {"<eos>", "<pad>"}
    kept = []
    for ln in text.split("\n"):
        words = ln.split()
        if (len(words) >= 4 and sum(w.lower() in vocab for w in words) >= 0.9 * len(words)
                and sum(bool(WORD_TOKEN.fullmatch(w)) or w in ("<EOS>", "<pad>") for w in words) >= 0.8 * len(words)):
            continue   # a scrambled figure line (never a table row: those are mostly numbers)
        kept.append(ln)
    return "\n".join(kept + [f"Example sentence shown in a figure: {s}" for s in sentences])


def _collapse_token_runs(text: str, min_run: int = 8) -> str:
    """Figures that show a sentence word by word (attention maps) extract as one token per line: join such runs into
    one line, once, so the paper's own example sentences survive as readable text."""
    lines, out, seen, i = text.split("\n"), [], set(), 0
    while i < len(lines):
        j = i
        while j < len(lines) and TOKEN_LINE.match(lines[j]):
            j += 1
        if j - i >= min_run:
            joined = " ".join(t.strip() for t in lines[i:j])
            joined = re.sub(r"(?:\s*<pad>)+", "", joined).strip()
            # a figure repeats its sentence (source and target sides, several heads): keep each sentence once
            for sentence in re.split(r"\s*<EOS>\s*", joined):
                if len(sentence) > 20 and sentence not in seen:
                    seen.add(sentence)
                    out.append(sentence + (" <EOS>" if "<EOS>" in joined else ""))
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _cut_tail(text: str) -> tuple[str, int | None]:
    """Cut at the References heading (confirmed by bibliography-looking lines) and drop a directly preceding
    Acknowledgments section, but keep an appendix that follows the bibliography; fall back to any heading past 30% of
    the text."""
    lines = text.split("\n")
    n = len(lines)
    fallback = None
    ack_idx = None
    for i, ln in enumerate(lines):
        if ACK_HEADING.match(ln):
            ack_idx = i
            if fallback is None and i >= n * 0.3:
                fallback = i
            continue
        if ACK_RUN_IN.match(ln):  # only cuts when a confirmed References heading follows within 80 lines
            ack_idx = i
            continue
        if REFERENCE_HEADING.match(ln):
            if i >= n * 0.15 and _looks_like_bibliography(lines[i + 1:i + 41]):
                cut = i
                if ack_idx is not None and 0 < i - ack_idx <= 80:
                    cut = ack_idx
                appendix = _appendix_after(lines, i)
                if appendix is not None:
                    kept = lines[:cut] + [""] + lines[appendix:]
                else:
                    kept = lines[:cut] + ([""] + _captions_after(lines, i) if _captions_after(lines, i) else [])
                return "\n".join(kept), cut
            if fallback is None and i >= n * 0.3:
                fallback = i
    if fallback is not None:
        return "\n".join(lines[:fallback]), fallback
    return text, None


LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"}


def _fix_ligatures(text: str) -> str:
    for lig, plain in LIGATURES.items():                # "diﬃcult" → "difficult" (keeps ², ×, √ untouched)
        text = text.replace(lig, plain)
    return text


def _clean_body(text: str) -> str:
    text = re.sub(r"-\n[ \t]*(?=[a-z])", "", text)    # de-hyphenate line breaks
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # [12], [3, 4] citation markers; one that ends a table cell becomes padding so the next cell stays in its column
    text = CITATION_NOISE.sub(lambda m: " " * len(m.group(0)) if m.string.startswith("  ", m.end()) else "", text)
    return text.strip()


def _collect_captions(text: str) -> list[str]:
    captions = []
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        m = CAPTION.match(ln.strip())
        if not m:
            continue
        cap = ln.strip()
        # captions often wrap onto the next line
        if i + 1 < len(lines) and lines[i + 1].strip() and not CAPTION.match(lines[i + 1].strip()):
            cap += " " + lines[i + 1].strip()
        captions.append(cap[:300])
    return captions[:40]


def extract_paper(source: Union[bytes, str, Path], max_body_chars: int = 120_000, filename: str = "") -> PaperText:
    """Extract clean body text, title and captions from a PDF (bytes or path)."""
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
    try:
        pages, meta = _read_pages_pymupdf(data)
        backend = "pymupdf"
    except Exception:  # pragma: no cover - fallback path
        pages, meta = _read_pages_pdfplumber(data)
        backend = "pdfplumber"

    pages = _strip_running_headers(pages)
    full = _dedupe_examples(_collapse_token_runs("\n".join(pages)))
    body, cut_line = _cut_tail(full)
    captions = _collect_captions(body)
    body = _clean_body(body)
    title = _detect_title(pages, meta) or (Path(filename).stem if filename else "")
    if len(body) > max_body_chars:  # table alignment padding must not push paper text past the cap
        body = re.sub(r" {3,}", "  ", body)
    truncated = len(body) > max_body_chars
    if truncated:
        body = body[:max_body_chars]
    tables, kept = _read_tables(data)
    if kept:
        from services.ledger import with_ledger
        body = with_ledger(body, kept)

    return PaperText(
        title=title,
        body=body,
        captions=captions,
        stats={
            "backend": backend,
            "pages": len(pages),
            "chars_total": len(full),
            "chars_body": len(body),
            "references_cut_at_line": cut_line,
            "truncated": truncated,
            "tables": len(kept.tables) if kept else 0,
        },
        tables=tables,
    )


def _read_tables(data: bytes):
    """(every table services.pdf_tables reads, the ledger of those it is sure about); ([], None) when it fails or finds
    none: the tables are an aid, the paper text never depends on them."""
    try:
        from services import pdf_tables
        from services.ledger import Ledger
        tables = pdf_tables.extract_tables(data)
        kept = Ledger.from_tables(tables, min_confidence=pdf_tables.MIN_CONFIDENCE)
        return tables, (kept if kept else None)
    except Exception as e:  # noqa: BLE001
        print(f"[paper_extractor] tables not read: {type(e).__name__}: {e}")
        return [], None


def extract_text_only(text: str, title: str = "") -> PaperText:
    """Wrap already-extracted text (legacy /api/manga endpoints) in a PaperText."""
    body, _ = _cut_tail(_fix_ligatures(text))
    return PaperText(title=title, body=_clean_body(body), captions=_collect_captions(body), stats={"backend": "text", "chars_body": len(body)})
