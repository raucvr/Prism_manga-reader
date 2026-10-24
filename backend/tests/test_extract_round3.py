"""Round 3, group "extract": the S1 follow-ups and the two column-path regressions, checked on the real papers.

Regressions fixed in services/paper_extractor.py:
- The two-column heuristic fired on single-column pages (tables, display math, author grids). Attention Table 2 came
  out one cell per line with its header row below the rows, and a real brief then misattributed a FLOPs figure.
  Now a page is read column by column only when both sides of its gutter carry running text, and bands of table
  cells, equations or figure labels are read row by row like ``sort=True``.
- Column-path text put every integer table cell on a line of its own, where the page-number filter deleted it
  (ResNet's CIFAR-10 table lost its depth column). Digit-only lines are now dropped only at the page edges.

Tests that need a PDF skip when it is not downloaded (see tests/fixtures/README.md).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pymupdf
import pytest

from services.paper_extractor import (LIGATURES, _read_pages_pymupdf, _strip_running_headers, extract_paper,
                                      extract_text_only)

from .fakes import COT_PDF, FIXTURES

RESNET_PDF = Path(os.environ.get("PRISM_RESNET_PDF") or FIXTURES / "resnet.pdf")
LIGATURE_CHARS = "".join(LIGATURES)


@pytest.fixture(scope="module")
def resnet_paper():
    if not RESNET_PDF.exists():
        pytest.skip("resnet.pdf not downloaded - see tests/fixtures/README.md")
    return extract_paper(RESNET_PDF)


def _line_with(text: str, *needles: str) -> str:
    line = next((ln for ln in text.splitlines() if all(n in ln for n in needles)), None)
    assert line is not None, f"no line contains {needles}"
    return line


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


# ----------------------------------------------------------------------------- tables keep their rows

def test_attention_table2_keeps_each_model_row_on_one_line(attention_paper):
    """Single-column page: the row reads BLEU EN-DE, BLEU EN-FR, then the training cost (exponents kept as 10^19)."""
    body = attention_paper.body
    assert re.search(r"Transformer \(big\)\s+28\.4\s+41\.8\s+2\.3 · 10\^19$", _line_with(body, "Transformer (big)"))
    assert re.search(r"26\.30\s+41\.16\s+1\.8 · 10\^20\s+1\.1 · 10\^21$", _line_with(body, "GNMT + RL Ensemble"))
    assert re.search(r"27\.3\s+38\.1\s+3\.3 · 10\^18$", _line_with(body, "Transformer (base model)"))


def test_attention_table2_values_stay_under_their_column_header(attention_paper):
    """Rows with empty cells only say EN-DE or EN-FR through their position, so values must line up with the header
    row (which must come before the rows: the column path had moved it below them)."""
    lines = attention_paper.body.splitlines()
    header = next(i for i, ln in enumerate(lines) if ln.split() == ["EN-DE", "EN-FR", "EN-DE", "EN-FR"])
    assert header < next(i for i, ln in enumerate(lines) if "ByteNet" in ln and "23.75" in ln)
    centres = [m.start() + 2.5 for m in re.finditer(r"EN-(?:DE|FR)", lines[header])]  # BLEU DE, FR; cost DE, FR

    def column(line: str, value: str) -> int:
        centre = line.index(value) + len(value) / 2
        return min(range(4), key=lambda k: abs(centres[k] - centre))

    body = attention_paper.body
    assert column(_line_with(body, "ByteNet", "23.75"), "23.75") == 0
    deep_att = _line_with(body, "Deep-Att + PosUnk", "39.2")
    assert (column(deep_att, "39.2"), column(deep_att, "1.0 · 10^20")) == (1, 3)
    ensemble = _line_with(body, "GNMT + RL Ensemble")
    assert [column(ensemble, v) for v in ("26.30", "41.16", "1.8 · 10^20", "1.1 · 10^21")] == [0, 1, 2, 3]


def test_numeric_superscripts_after_a_digit_keep_their_exponent(attention_paper, resnet_paper):
    """'2.3 · 10^19' must not read as '2.3 · 1019' (Table 2 training costs, ResNet FLOPs)."""
    assert "1.0 · 10^20" in _line_with(attention_paper.body, "Deep-Att + PosUnk", "39.2")
    assert "1019" not in attention_paper.body.replace("10^19", "")
    assert "3.6×10^9" in resnet_paper.body


def test_attention_projection_formula_stays_on_its_line(attention_paper):
    """Single-column display math is not re-paired into fragments (the column path scrambled W_i^Q, W_i^K, W_i^V)."""
    line = _line_with(attention_paper.body, "parameter matrices")
    assert line.index("∈Rdmodel×dk") < line.index("∈Rdmodel×dv")


def _fraction_pdf() -> bytes:
    """'... by 1/sqrt(d_k).' set like LaTeX: the numerator raised above the line, the denominator lowered below it."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    lead = "To counteract this effect, we scale the dot products by"
    page.insert_text((72, 100), lead, fontsize=10)
    x = 72 + pymupdf.get_text_length(lead, "helv", 10)
    page.insert_text((x + 9, 96), "1", fontsize=7)
    page.insert_text((x + 4, 103), "sqrt(d_k).", fontsize=7)
    return doc.tobytes()


def test_inline_fraction_keeps_its_numerator_before_the_denominator():
    """The numerator shares a row with its denominator; ordered by x alone it came after it ('by sqrt(d_k).1'), which
    reads as scaling by sqrt(d_k) instead of dividing by it."""
    line = _line_with(extract_paper(_fraction_pdf()).body, "scale the dot products")
    assert _flat(line).endswith("by 1 sqrt(d_k).")


def test_attention_scaling_factor_is_one_over_sqrt_dk(attention_paper):
    flat = _flat(attention_paper.body)
    assert "we scale the dot products by 1 √dk ." in flat
    assert "except for the scaling factor of 1 √dk . Additive attention" in flat


def test_resnet_cifar10_table_keeps_its_depth_column(resnet_paper):
    body = resnet_paper.body
    for depth, params, error in [("20", "0.27M", "8.75"), ("32", "0.46M", "7.51"), ("44", "0.66M", "7.17"),
                                 ("56", "0.85M", "6.97"), ("110", "1.7M", "6.43"), ("1202", "19.4M", "7.93")]:
        assert re.search(rf"^\s*ResNet\s+{depth}\s+{re.escape(params)}\s+{re.escape(error)}\b", body, re.M), depth
    # the architecture table next to it keeps its filter counts and layer counts
    assert re.search(r"^\s*# filters\s+16\s+32\s+64\s*$", body, re.M)
    assert re.search(r"^\s*# layers\s+1\+2n\s+2n\s+2n\s*$", body, re.M)


def test_resnet_full_width_table_on_a_two_column_page_is_read_row_by_row(resnet_paper):
    """Table 1 spans both columns: its rows must not be split into a left half and a right half."""
    body = resnet_paper.body
    assert re.search(r"FLOPs\s+1\.8×10\^9\s+3\.6×10\^9\s+3\.8×10\^9\s+7\.6×10\^9\s+11\.3×10\^9", body)
    assert re.search(r"conv1\s+112×112\s+7×7, 64, stride 2", body)


def test_text_table_on_a_single_column_page_keeps_its_rows():
    """A table whose right column holds multi-line prose cells is not two-column text (CoT appendix Table 12)."""
    if not COT_PDF.exists():
        pytest.skip("chain_of_thought.pdf not downloaded - see tests/fixtures/README.md")
    page = pymupdf.open()
    page.insert_pdf(pymupdf.open(COT_PDF), from_page=28, to_page=28)
    body = extract_paper(page.tobytes()).body
    assert re.search(r"GSM8K\s+1,319\s+Josh decides to try flipping a house", body)
    assert re.search(r"AQuA\s+254\s+A car is being driven", body)


# ----------------------------------------------------------------------------- reading order

def test_resnet_two_column_paragraphs_are_in_order(resnet_paper):
    """Multi-line sentences from both columns of several pages, in reading order (sort=True glued lines of the two
    columns together, so none of these sentences survived)."""
    flat = _flat(resnet_paper.body)
    sentences = [
        "We present a residual learning framework to ease the training of networks that are substantially deeper "
        "than those used previously.",                                                        # p1, left column
        "Driven by the significance of depth, a question arises: Is learning better networks as easy as stacking "
        "more layers?",                                                                       # p1, right column
        "We show that: 1) Our extremely deep residual nets are easy to optimize, but the counterpart “plain” nets "
        "(that simply stack layers) exhibit higher training error when the depth increases;",  # p2
        "We adopt residual learning to every few stacked layers.",                             # p3
        "We use a weight decay of 0.0001 and momentum of 0.9,",                                # p7, right column
        "The details are in the appendix.",                                                    # p8, last sentence
    ]
    positions = [flat.find(s) for s in sentences]
    assert -1 not in positions, dict(zip(sentences, positions))
    assert positions == sorted(positions)


def test_cot_text_beside_a_figure_is_not_interleaved_with_figure_labels(cot_paper):
    """Wrapped text next to a figure (CoT pages 2 and 6) reads as one paragraph, before the figure's labels."""
    flat = _flat(cot_paper.body)
    for sentence in [
        "This work explores how the reasoning ability of large language models can be unlocked by a simple method "
        "motivated by two ideas.",
        "Another intuition is that chain of thought allows the model to spend more computation (i.e., intermediate "
        "tokens) on harder problems.",
        "In this final subsection, we evaluate robustness to chains of thought written by different annotators.",
    ]:
        assert sentence in flat, sentence


# ----------------------------------------------------------------------------- References, page furniture, ligatures

def test_resnet_bibliography_is_cut_and_the_arxiv_stamp_dropped(resnet_paper):
    body = resnet_paper.body
    assert resnet_paper.stats["references_cut_at_line"] is not None
    assert "The details are in the appendix.\n\nA. Object Detection Baselines" in body      # the appendix follows the body
    assert "Bengio" not in body and "arXiv:1512.03385" not in body and "[cs.CV]" not in body


def test_attention_arxiv_stamp_is_not_glued_into_body_lines(attention_paper):
    """sort=True glued the rotated stamp's words onto body lines ('the training costs of thearXiv:1706.03762v7')."""
    body = attention_paper.body
    assert "arXiv:1706.03762" not in body and "[cs.CL]" not in body
    assert "a small fraction of the training costs of the best models from the literature" in _flat(body)


def test_page_numbers_are_stripped_at_the_page_edges():
    if not RESNET_PDF.exists():
        pytest.skip("resnet.pdf not downloaded - see tests/fixtures/README.md")
    pages, _ = _read_pages_pymupdf(RESNET_PDF.read_bytes())
    for number, page in enumerate(_strip_running_headers(pages), start=1):
        edge = [ln.strip() for ln in page.splitlines() if ln.strip()]
        assert str(number) not in edge[:3] + edge[-3:], number


def test_resnet_captions_have_no_ligatures(resnet_paper):
    captions = resnet_paper.captions
    assert any(c.startswith("Table 6. Classification error on the CIFAR-10 test set.") for c in captions)
    assert not any(ch in "".join(captions) for ch in LIGATURE_CHARS)


def _body(n: int, tag: str = "Body") -> list[str]:
    return [f"{tag} sentence {i} explains the method in plain words." for i in range(n)]


def _bib(n: int) -> list[str]:
    return [f"[{k}] A. Author and B. Writer. A study of things {k}. In Proceedings of the Conference, {1990 + k}."
            for k in range(1, n + 1)]


@pytest.mark.parametrize("heading", ["Acknowledgements We thank the ACKMARK team for", "Acknowledgments. We thank the",
                                     "ACKNOWLEDGMENTS: We thank the", "6 Acknowledgement We thank the"])
def test_run_in_acknowledgments_right_before_references_are_cut(heading):
    lines = _body(60) + [heading, "ACKMARK team for comments."] + ["References"] + _bib(20)
    body = extract_text_only("\n".join(lines)).body
    assert "Body sentence 59 " in body
    assert "ACKMARK" not in body and "A study of things" not in body


def test_run_in_acknowledgments_far_before_references_do_not_move_the_cut():
    lines = (_body(20) + ["Acknowledgments of prior work are given in ACKMARK Section 2."] + _body(90, "Late")
             + ["References"] + _bib(20))
    body = extract_text_only("\n".join(lines)).body
    assert "ACKMARK" in body and "Late sentence 89 " in body
    assert "A study of things" not in body


def test_lowercase_acknowledgment_starting_a_wrapped_line_is_not_a_heading():
    """Run-in headings are capitalised; a sentence wrapped onto 'acknowledgment that ...' must not become the cut."""
    lines = (_body(60) + ["7 Conclusion", "Our results are, in effect, an explicit",
                          "acknowledgment that scale alone does not fix reasoning.", "CONCLUSIONMARK Future work follows."]
             + ["References"] + _bib(20))
    body = extract_text_only("\n".join(lines)).body
    assert "acknowledgment that scale alone" in body and "CONCLUSIONMARK" in body
    assert "A study of things" not in body


# ----------------------------------------------------------------------------- superscripts: exponents vs markers

def _superscript_pdf(tmp_path, lines) -> Path:
    """One line per entry; a (text, True) segment is set 7 pt and raised 3.6 pt like a LaTeX superscript."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    font = pymupdf.Font("tiro")
    y = 90
    for segs in lines:
        x = 72
        for text, sup in segs:
            size, rise = (7.0, 3.6) if sup else (10.0, 0.0)
            page.insert_text((x, y - rise), text, fontsize=size, fontname="tiro")
            x += font.text_length(text, fontsize=size)
        y += 22
    path = tmp_path / "sup.pdf"
    doc.save(path)
    return path


def test_exponents_of_ten_keep_a_caret_and_footnote_markers_after_numbers_are_dropped(tmp_path):
    path = _superscript_pdf(tmp_path, [
        [("We train with a batch size of 4096", False), ("2", True), (" sentence pairs per step.", False)],
        [("This effect was first reported in 2019", False), ("12", True), (" and later confirmed.", False)],
        [("We use a learning rate of 10", False), ("-4", True), (" for all runs.", False)],   # base-14 Times has no U+2212
        [("The error rate fell from 3.6 to 2.9", False), ("7", True), (" in the follow-up study.", False)],
        [("The network has 13.6 · 10", False), ("6 ", True), ("parameters in total.", False)],
    ])
    body = _flat(extract_paper(path).body)
    assert "batch size of 4096 sentence pairs" in body
    assert "reported in 2019 and later" in body
    assert "learning rate of 10^-4 for all runs" in body
    assert "fell from 3.6 to 2.9 in the follow-up" in body
    assert "13.6 · 10^6 parameters" in body


def test_latex_negative_compound_and_power_exponents(attention_paper):
    body = _flat(attention_paper.body)
    assert "ϵ = 10^−9" in body                       # minus and digit come as two superscript spans
    assert "10000^(2i/dmodel)" in body               # the whole exponent, not 10000^2 · i/dmodel
    assert "O(n^2 · d)" in body


def _spans(*parts, body_size=10.0):
    """(text, font, superscript) triples as PyMuPDF line spans."""
    return [{"text": t, "font": f, "flags": 5 if sup else 4, "size": 7.0 if sup else body_size} for t, f, sup in parts]


@pytest.mark.parametrize("parts, body, expected", [
    ((("batches of 4096", "CMR10", False), ("2", "CMR7", True), (" tokens", "CMR10", False)), "CMR", "batches of 4096[2] tokens"),
    ((("took 3.5", "LMRoman10-Regular", False), ("1", "LMRoman7-Regular", True), (" days", "LMRoman10-Regular", False)), "LMRoman", "took 3.5[1] days"),
    ((("has 2", "Times-Roman", False), ("16", "Times-Roman", True), (" = 65,536 entries", "Times-Roman", False)), "Times", "has 2^16 = 65,536 entries"),
    ((("rate of 10", "Times-Roman", False), ("-4", "Times-Roman", True), (" for all", "Times-Roman", False)), "Times", "rate of 10^-4 for all"),
    ((("O(n", "CMMI10", False), ("2", "CMR7", True), (" · d)", "CMSY10", False)), "NimbusRomNo9L", "O(n^2 · d)"),
    ((("as 4096", "NimbusRomNo9L-Regu", False), ("2", "CMR7", True), (" pixels", "NimbusRomNo9L-Regu", False)), "NimbusRomNo9L", "as 4096^2 pixels"),
])
def test_superscripts_are_judged_against_the_body_font(parts, body, expected):
    """CM/LM papers set footnote marks in the body font (CMR7), so CMR is math only when the body is another family."""
    from services.paper_extractor import _join_spans
    assert _join_spans(_spans(*parts), body) == expected


def test_an_appendix_after_the_bibliography_is_kept_but_reference_lists_are_not():
    """Appendices hold training details, ablations and result tables (BERT A.2, CoT B); the bibliography never comes back."""
    from services.paper_extractor import extract_text_only
    body = [f"Body sentence {i} about the method and its results in plain words." for i in range(60)]
    bib = [f"[{i}] A. Author and B. Writer. A study of things {i}. In Proceedings of ICML, 2019." for i in range(30)]
    appendix = ["A   Additional Details for the Model"] + [f"Appendix sentence {i}: the training used 64 chips for 4 days." for i in range(40)]
    text = extract_text_only("\n".join(body + ["References"] + bib + appendix)).body
    assert "Body sentence 59" in text and "Appendix sentence 39" in text and "A study of things" not in text
    # an "A ..." line that is itself a reference entry is not an appendix heading
    titled = ["A Neural Conversational Model"] + [f"[{i}] O. Vinyals and Q. Le. A neural model {i}. arXiv preprint, 2015." for i in range(30)]
    assert "neural model" not in extract_text_only("\n".join(body + ["References"] + bib[:10] + titled)).body
