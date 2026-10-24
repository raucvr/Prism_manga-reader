"""Review group "extract" (finding S1): column-aware PDF reading order, bibliography cut, ligatures.

S1 as reported: PyMuPDF ``get_text(sort=True)`` merges same-baseline lines of the left and right column, so
two-column papers came out scrambled, and their "References" heading (merged into a body line) never matched,
so the bibliography was never cut. The fix under test (services/paper_extractor.py):
``_page_text_reading_order`` reads a page column by column when both sides of its gutter carry running text; lines
crossing the gutter split the page into bands, and everything that is not two-sided running text is read row by row
like ``sort=True``. Running heads, footers and page numbers stay at the page edges. ``_cut_tail`` confirms
"References" with bibliography-looking lines and starts the cut at a directly preceding Acknowledgments heading (also
a run-in one); ligatures are replaced in every page, so in the body, captions and title as well.

The two-column PDFs are synthesised with PyMuPDF line by line (like LaTeX output), so nothing has to be
downloaded. Every body word is a numbered token (``L0007`` in a left column, ``R0008`` in a right column) handed
out in true reading order, so the reading order and any cross-column line merging can be checked exactly.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from services.paper_extractor import LIGATURES, extract_paper, extract_text_only

from .fakes import FIXTURES, image_bytes

# Optional real two-column paper (arXiv 1512.03385, CVPR layout). Like the other fixture PDFs it is not committed:
#   curl -L -o backend/tests/fixtures/resnet.pdf https://arxiv.org/pdf/1512.03385v1
# or point PRISM_RESNET_PDF at a local copy.
RESNET_PDF = Path(os.environ.get("PRISM_RESNET_PDF") or FIXTURES / "resnet.pdf")

W, H = 612, 792            # US letter
LX, RX = 54, 318           # x of the left / right column (240 pt wide, 24 pt gutter)
FS, LEAD = 9, 11           # body font size / line pitch
TOKEN = re.compile(r"\b([LR])(\d{4})\b")
LIGATURE_CHARS = "".join(LIGATURES)


# ----------------------------------------------------------------------------- synthetic PDF builder

def wrap(tokens: list[str], width: int = 52) -> list[str]:
    lines, cur = [], ""
    for tok in tokens:
        if cur and len(cur) + 1 + len(tok) > width:
            lines.append(cur)
            cur = tok
        else:
            cur = f"{cur} {tok}".strip()
    if cur:
        lines.append(cur)
    return lines


def bib_entry(k: int) -> list[str]:
    return [f"[{k}] A. Author{k}, B. Writer, et al. A study of things {k}.",
            f"In Proceedings of the Conference on Stuff, {2000 + k}."]


class Synth:
    """Builds a PDF page by page; text is placed one line at a time with ``insert_text``."""

    def __init__(self):
        self.doc = pymupdf.open()
        self.page = None
        self.k = 0  # last token number handed out

    def new_page(self) -> "Synth":
        self.page = self.doc.new_page(width=W, height=H)
        return self

    def words(self, n: int, tag: str) -> list[str]:
        out = []
        for _ in range(n):
            self.k += 1
            out.append(f"{tag}{self.k:04d}")
        return out

    def line(self, x: float, y: float, text: str, size: float = FS, font: str = "tiro") -> None:
        self.page.insert_text((x, y), text, fontsize=size, fontname=font)

    def lines(self, x: float, y: float, lines: list[str], indent: float = 0) -> float:
        for i, text in enumerate(lines):
            self.line(x + (indent if i == 0 else 0), y, text)
            y += LEAD
        return y

    def paragraph(self, x: float, y: float, tokens: list[str], indent: float = 8) -> float:
        """An indented paragraph (a new paragraph starts a new PyMuPDF text block)."""
        return self.lines(x, y, wrap(tokens), indent) + 3

    def column(self, x: float, y: float, until: float, tag: str) -> tuple[float, int, int]:
        """Paragraphs of 45 tokens until the column reaches ``until``; returns (y, first token, last token)."""
        first = self.k + 1
        while y < until:
            y = self.paragraph(x, y, self.words(45, tag))
        return y, first, self.k

    def heading(self, x: float, y: float, text: str) -> float:
        self.line(x, y + 6, text, size=FS + 1, font="hebo")
        return y + 6 + LEAD + 4

    def centered(self, y: float, text: str, size: float, font: str = "hebo") -> None:
        self.line((W - pymupdf.get_text_length(text, font, size)) / 2, y, text, size=size, font=font)

    def right_aligned(self, y: float, text: str, size: float = 7) -> None:
        self.line(W - 54 - pymupdf.get_text_length(text, "tiro", size), y, text, size=size)

    def tobytes(self) -> bytes:
        return self.doc.tobytes()


def numbers(text: str) -> list[int]:
    return [int(n) for _, n in TOKEN.findall(text)]


def mixed_lines(text: str) -> list[str]:
    """Lines that contain words from both a left and a right column (sort=True line merging)."""
    return [ln for ln in text.splitlines() if {"L", "R"} <= {tag for tag, _ in TOKEN.findall(ln)}]


CAPTION_LINES = [
    "Figure 2: A double-column figure caption that spans the full text width of the page, as figure* floats do in LaTeX.",
    "It wraps onto a second line, and the text below it starts a new band that again has two columns of text.",
]


@pytest.fixture(scope="module")
def two_column():
    """Three two-column pages: title + abstract, a full-width caption mid-page, Acknowledgments + References."""
    s = Synth()
    marks = {}
    for pno in range(3):
        s.new_page()
        s.line(LX, 40, "Synthetic Two-Column Paper", size=7)            # running head, left part
        s.right_aligned(40, "Conf '26, October 2026, Springfield")        # running head, right part
        s.line(W / 2 - 3, 760, str(pno + 1))                              # page number
        if pno == 0:
            s.centered(90, "A Synthetic Study of Column Order", 16)
            y = s.heading(LX, 120, "Abstract")
            y = s.paragraph(LX, y, s.words(70, "L"))
            y = s.heading(LX, y, "1 Introduction")
            s.column(LX, y, 690, "L")
            s.column(RX, 120, 700, "R")
        elif pno == 1:
            yl, _, _ = s.column(LX, 70, 280, "L")
            yr, _, marks["top_right_last"] = s.column(RX, 70, 280, "R")
            y = s.lines(LX, max(yl, yr) + 12, CAPTION_LINES) + 12
            _, marks["bottom_left_first"], _ = s.column(LX, y, 690, "L")
            s.column(RX, y, 690, "R")
        else:
            y, _, marks["last_body"] = s.column(LX, 70, 520, "L")
            y = s.heading(LX, y, "Acknowledgments")
            s.lines(LX, y, ["We thank the ACKMARK foundation for its generous", "support of this work."], indent=8)
            y = s.heading(RX, 70, "References")
            for k in range(1, 15):
                y = s.lines(RX, y, bib_entry(k)) + 2
    data = s.tobytes()
    return SimpleNamespace(data=data, n=s.k, marks=marks, paper=extract_paper(data))


# ----------------------------------------------------------------------------- S1: reading order (fixed)

def test_s1_two_column_pages_are_read_left_column_then_right_column(two_column):
    body = two_column.paper.body
    assert two_column.paper.stats["backend"] == "pymupdf"
    # every token exactly once, in reading order: page by page, left column before right column
    assert numbers(body) == list(range(1, two_column.n + 1))


def test_s1_no_line_merges_the_two_columns(two_column):
    assert mixed_lines(two_column.paper.body) == []


def test_s1_full_width_caption_splits_the_page_into_bands(two_column):
    body, marks = two_column.paper.body, two_column.marks
    caption = body.find("Figure 2: A double-column figure caption")
    assert body.find(f"R{marks['top_right_last']:04d}") < caption < body.find(f"L{marks['bottom_left_first']:04d}")
    assert any(c.startswith("Figure 2: A double-column figure caption") for c in two_column.paper.captions)


def test_s1_two_part_running_head_is_still_stripped(two_column):
    body = two_column.paper.body
    assert "Synthetic Two-Column Paper" not in body and "Conf '26" not in body


# ----------------------------------------------------------------------------- S1: bibliography cut (fixed)

def test_s1_bibliography_in_the_right_column_is_cut_with_the_acknowledgments(two_column):
    paper, last = two_column.paper, two_column.marks["last_body"]
    assert paper.stats["references_cut_at_line"] is not None
    assert "References" not in [ln.strip() for ln in paper.body.splitlines()]
    assert "Proceedings of the Conference" not in paper.body and "Author1," not in paper.body
    # the Acknowledgments section right before the References heading goes too...
    assert "ACKMARK" not in paper.body and "Acknowledgments" not in paper.body
    # ...and the body ends exactly with the last word of the paper text
    assert paper.body.rstrip().endswith(f"L{last:04d}")


def _body(n: int, tag: str = "Body") -> list[str]:
    return [f"{tag} sentence {i} explains the method in plain words." for i in range(n)]


def _bib(n: int) -> list[str]:
    return [f"[{k}] A. Author and B. Writer. A study of things {k}. In Proceedings of the Conference, {1990 + k}."
            for k in range(1, n + 1)]


def test_s1_confirmed_references_heading_is_cut_even_before_30_percent():
    # short body, bibliography, long appendix (like the chain-of-thought paper): the heading sits at ~19 %
    lines = _body(20) + ["References"] + _bib(25) + [f"Appendix line {i} with more experiments." for i in range(60)]
    paper = extract_text_only("\n".join(lines))
    assert "Body sentence 19 " in paper.body
    assert "A study of things" not in paper.body          # the bibliography is cut...
    assert "Appendix line 59 with more experiments." in paper.body   # ...but the appendix after it is kept (results, ablations)


def test_s1_references_line_without_bibliography_after_it_is_not_the_cut():
    lines = _body(40) + ["References"] + _body(45, "More") + ["References"] + _bib(20)
    paper = extract_text_only("\n".join(lines))
    assert "More sentence 44 " in paper.body           # the unconfirmed heading did not cut the body short
    assert "A study of things" not in paper.body        # the confirmed one did cut the bibliography


def test_s1_acknowledgments_directly_before_references_start_the_cut():
    lines = _body(60) + ["Acknowledgments", "We thank the ACKMARK foundation."] + ["References"] + _bib(20)
    paper = extract_text_only("\n".join(lines))
    assert "Body sentence 59 " in paper.body
    assert "ACKMARK" not in paper.body and "Acknowledgments" not in paper.body
    assert "A study of things" not in paper.body


def test_s1_distant_acknowledgments_heading_does_not_move_the_cut():
    # an Acknowledgments heading 90 lines before the References heading is not "directly preceding" it
    lines = _body(30) + ["Acknowledgments", "We thank the ACKMARK foundation."] + _body(90, "Late") + ["References"] + _bib(20)
    paper = extract_text_only("\n".join(lines))
    assert "Late sentence 89 " in paper.body and "ACKMARK" in paper.body
    assert "A study of things" not in paper.body


# ----------------------------------------------------------------------------- S1: ligatures

def test_s1_ligatures_are_replaced_in_text_bodies():
    text = "\n".join(["The classiﬁer is eﬃcient and ﬂexible; eﬀects diﬀer; baﬄing ﬅ.",
                      "Keep x² × √d untouched."] + _body(5))
    body = extract_text_only(text).body
    assert "The classifier is efficient and flexible; effects differ; baffling st." in body
    assert "Keep x² × √d untouched." in body
    assert not any(ch in body for ch in LIGATURE_CHARS)


@pytest.fixture(scope="module")
def ligature_pdf() -> bytes:
    """One page set in PyMuPDF's built-in Droid Sans Fallback, which has the ﬁ/ﬂ/ﬃ glyphs (LaTeX PDFs use them)."""
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    page.insert_font(fontname="dsf", fontbuffer=pymupdf.Font("cjk").buffer)
    page.insert_text((72, 90), "Uniﬁed ﬂow for Eﬃcient Classiﬁcation", fontsize=18, fontname="dsf")
    y = 140
    for _ in range(4):
        page.insert_text((72, y), "The classiﬁer is eﬃcient and ﬂexible on every benchmark we tried.", fontsize=10, fontname="dsf")
        y += 14
    page.insert_text((72, y + 20), "Figure 1: The uniﬁed classiﬁer is eﬃcient.", fontsize=10, fontname="dsf")
    data = doc.tobytes()
    assert "ﬁ" in pymupdf.open(stream=data, filetype="pdf")[0].get_text()  # the PDF really carries ligatures
    return data


def test_s1_ligatures_are_replaced_in_the_pdf_body(ligature_pdf):
    body = extract_paper(ligature_pdf).body
    assert "The classifier is efficient and flexible" in body
    assert not any(ch in body for ch in LIGATURE_CHARS)


def test_s1_ligatures_are_replaced_in_title_and_captions(ligature_pdf):
    paper = extract_paper(ligature_pdf)
    assert paper.title == "Unified flow for Efficient Classification"
    assert paper.captions and paper.captions[0].startswith("Figure 1: The unified classifier is efficient.")
    assert not any(ch in "".join(paper.captions) for ch in LIGATURE_CHARS)


# ----------------------------------------------------------------------------- S1: real papers

def test_s1_single_column_papers_keep_their_section_order(attention_paper, cot_paper):
    """Single-column pages are read row by row and pages with text beside a figure column by column; paragraphs
    must not move."""
    expected = {
        "attention": (attention_paper, [
            "Recurrent neural networks, long short-term memory", "The goal of reducing sequential computation",
            "Most competitive neural sequence transduction models have an encoder-decoder structure",
            "In this section we compare various aspects of self-attention layers",
            "This section describes the training regime for our models",
            "On the WMT 2014 English-to-German translation task, the big transformer model",
            "In this work, we presented the Transformer"]),
        "cot": (cot_paper, [
            "The NLP landscape has recently been revolutionized", "The goal of this paper is to endow language models",
            "We begin by considering math word problems", "Although chain of thought is particularly suitable",
            "Our final experimental evaluation considers symbolic reasoning",
            "We have explored chain-of-thought prompting as a simple"]),
    }
    for name, (paper, phrases) in expected.items():
        flat = re.sub(r"\s+", " ", paper.body)
        positions = [flat.find(p) for p in phrases]
        assert -1 not in positions, (name, dict(zip(phrases, positions)))
        assert positions == sorted(positions), (name, positions)


@pytest.fixture(scope="module")
def resnet_paper():
    if not RESNET_PDF.exists():
        pytest.skip("resnet.pdf not downloaded (see the RESNET_PDF comment in this file)")
    return extract_paper(RESNET_PDF)


def test_s1_resnet_is_read_column_by_column_and_its_bibliography_is_cut(resnet_paper):
    body = resnet_paper.body
    lines = [ln.strip() for ln in body.splitlines()]
    assert resnet_paper.title == "Deep Residual Learning for Image Recognition"
    # page 1: the first abstract line is a line of its own (sort=True glued the right column onto it) ...
    assert "Deeper neural networks are more difficult to train. We" in lines
    # ... and the left column (abstract, 1. Introduction) is read before the right column
    assert body.find("1. Introduction") < body.find("greatly benefited from very deep models")
    sections = ["1. Introduction", "2. Related Work", "3. Deep Residual Learning", "4. Experiments",
                "4.1. ImageNet Classification", "4.2. CIFAR-10 and Analysis", "4.3. Object Detection on PASCAL and MS COCO"]
    assert all(s in lines for s in sections)
    assert [lines.index(s) for s in sections] == sorted(lines.index(s) for s in sections)
    # the bibliography is cut right after the last body sentence; the appendix after it is kept
    assert resnet_paper.stats["references_cut_at_line"] is not None
    assert "The details are in the appendix.\n\nA. Object Detection Baselines" in body
    assert "Bengio" not in body and "ImageNet localization task in ILSVRC 2015." in body
    assert not any(ch in body for ch in LIGATURE_CHARS)


def test_s1_run_in_acknowledgements_paragraph_is_cut(attention_paper):
    body = attention_paper.body
    assert "We are grateful to Nal Kalchbrenner" not in body
    # the conclusion ends the main text; the untitled appendix ("Attention Visualizations") follows the bibliography
    assert "tensor2tensor" in body and body.index("tensor2tensor") < body.index("Attention Visualizations")
    assert "The Law will never be perfect" in body


# ----------------------------------------------------------------------------- S1: single-block columns, page edges

def _single_block_column_page(layout: str) -> tuple[bytes, int]:
    s = Synth().new_page()
    if layout == "long_paragraph":
        # the left column is one paragraph continued from the previous page: one text block, no indentation
        s.lines(LX, 70, wrap(s.words(500, "L")))
    else:
        # the left column holds a photo (no text) and a four-line caption: again a single text block
        s.page.insert_image(pymupdf.Rect(LX, 70, LX + 240, 520), stream=image_bytes(64, 64, (120, 160, 200)))
        s.lines(LX, 540, ["Figure 4. Qualitative results on held-out validation",
                          "images: every row shows the input picture, the",
                          "baseline output and our output, which keeps much",
                          "finer detail around the object boundaries."])
    s.column(RX, 70, 690, "R")
    return s.tobytes(), s.k


@pytest.mark.parametrize("layout", ["long_paragraph", "raster_figure"])
def test_s1_column_made_of_one_text_block_is_not_merged_with_the_other_column(layout):
    data, n = _single_block_column_page(layout)
    paper = extract_paper(data)
    assert numbers(paper.body) == list(range(1, n + 1))
    assert mixed_lines(paper.body) == []
    if layout == "raster_figure":
        caption_lines = [ln for ln in paper.body.splitlines()
                         if any(w in ln for w in ("Qualitative results", "every row shows", "baseline output", "finer detail"))]
        assert len(caption_lines) == 4 and not any(TOKEN.search(ln) for ln in caption_lines)
        assert paper.captions and not TOKEN.search(paper.captions[0])


def test_s1_references_heading_on_a_fallback_page_still_cuts_the_bibliography():
    s = Synth().new_page()
    s.column(LX, 70, 690, "L")
    s.column(RX, 70, 690, "R")
    s.new_page()
    s.lines(LX, 70, wrap(s.words(420, "L")))           # last body column: one paragraph, one text block
    last = s.k
    y = s.heading(RX, 70 + 4 * LEAD, "References")
    for k in range(1, 13):
        y = s.lines(RX, y, bib_entry(k)) + 2
    paper = extract_paper(s.tobytes())
    assert paper.stats["references_cut_at_line"] is not None
    assert "Proceedings of the Conference" not in paper.body
    assert paper.body.rstrip().endswith(f"L{last:04d}")


def test_s1_one_sided_running_head_and_footer_are_stripped():
    s = Synth()
    for _ in range(5):
        s.new_page()
        s.right_aligned(40, "Running Head: Column Order Study")
        s.line(LX, 760, "Workshop on Synthetic Layouts 2026", size=7)
        s.column(LX, 70, 680, "L")
        s.column(RX, 70, 680, "R")
    body = extract_paper(s.tobytes()).body
    assert "Running Head" not in body
    assert "Workshop on Synthetic Layouts" not in body


def test_running_head_with_the_page_number_on_its_baseline_is_stripped():
    """IEEE journal style: head and page number share a baseline (the number on the outer side), so as one row they
    differ on every page; the snapshot's block text stripped them, the row text must too."""
    s = Synth()
    head = "JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015"
    for pno in range(1, 6):
        s.new_page()
        if pno % 2:
            s.line(LX, 40, head, size=7)
            s.right_aligned(40, str(pno))
        else:
            s.line(LX, 40, str(pno), size=7)
            s.right_aligned(40, head)
        s.column(LX, 70, 680, "L")
        s.column(RX, 70, 680, "R")
    body = extract_paper(s.tobytes()).body
    assert "JOURNAL OF LATEX" not in body
    assert not [ln for ln in body.splitlines() if ln.strip().isdigit()]
    assert numbers(body) == list(range(1, s.k + 1))
