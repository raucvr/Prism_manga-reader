"""services.paper_extractor on the two real fixture papers and on synthetic text."""

import re

import pytest

from services.ledger import LEDGER_HEAD

from services.paper_extractor import extract_text_only


def glued_ratio(body: str) -> float:
    """Share of alphabetic tokens longer than 20 letters (pdfplumber's x_tolerance=3 de-spacing bug)."""
    words = re.findall(r"[A-Za-z]+", body)
    return sum(len(w) > 20 for w in words) / max(1, len(words))


def test_attention_title_is_detected(attention_paper):
    assert attention_paper.title == "Attention Is All You Need"
    assert attention_paper.stats["backend"] == "pymupdf"
    assert attention_paper.stats["pages"] == 15


def text_part(paper) -> str:
    """The paper's own text, without the tables appended after it (services.ledger)."""
    return paper.body.split("\n\n" + LEDGER_HEAD)[0]


def test_cot_title_is_detected(cot_paper):
    assert cot_paper.title.startswith("Chain-of-Thought Prompting")
    assert cot_paper.stats["pages"] == 43
    assert len(text_part(cot_paper)) <= 120_000   # the cap is on the paper's text; its tables follow it


@pytest.mark.parametrize("paper", ["attention_paper", "cot_paper"])
def test_body_words_are_not_glued_together(paper, request):
    body = request.getfixturevalue(paper).body
    assert len(re.findall(r"[A-Za-z]+", body)) > 3000  # a real body, not an empty extraction
    assert glued_ratio(body) < 0.005


def test_attention_body_stops_before_references(attention_paper):
    body = text_part(attention_paper)
    assert attention_paper.stats["references_cut_at_line"] is not None
    assert "References" not in [ln.strip() for ln in body.splitlines()]
    tail = body[-3000:]
    assert "Vaswani" not in tail
    # bibliography entries such as "[1] ... Layer normalization. arXiv preprint arXiv:1607.06450" are gone
    assert "Layer normalization" not in body and "arXiv preprint" not in body
    # ...while the conclusion right before them survives
    assert "tensor2tensor" in tail


@pytest.mark.parametrize("paper", ["attention_paper", "cot_paper"])
def test_captions_include_figure_1(paper, request):
    captions = request.getfixturevalue(paper).captions
    assert any(c.startswith("Figure 1:") for c in captions)
    assert len(captions) <= 40 and all(len(c) <= 300 for c in captions)


def test_attention_captions_cover_figures_and_tables(attention_paper):
    captions = attention_paper.captions
    assert captions[0].startswith("Figure 1: The Transformer - model architecture.")
    assert any(c.startswith("Table 2:") and "BLEU" in c for c in captions)


def _numbered_text(references_at: float, total: int = 100) -> str:
    lines = [f"Body sentence number {i} explains the method in detail." for i in range(total)]
    lines.insert(int(total * references_at), "References")
    return "\n".join(lines)


def test_text_only_cuts_at_references_after_30_percent():
    paper = extract_text_only(_numbered_text(0.6), title="Synthetic")
    assert paper.title == "Synthetic"
    assert "References" not in paper.body
    assert "number 59 " in paper.body and "number 60 " not in paper.body


def test_text_only_keeps_a_references_line_before_30_percent():
    paper = extract_text_only(_numbered_text(0.2))
    assert "References" in paper.body
    assert "number 99 " in paper.body


def test_text_only_repairs_hyphenation_drops_citations_and_collects_captions():
    filler = [f"Filler line {i} about attention." for i in range(10)]
    text = "\n".join(["The Transformer uses multi-head atten-", "tion everywhere [12] and is fast [3, 4].",
                      "Figure 1: The model architecture.", "Encoder on the left.", *filler])
    paper = extract_text_only(text)
    assert "multi-head attention everywhere and is fast." in paper.body
    assert "[12]" not in paper.body and "[3, 4]" not in paper.body
    assert paper.captions == ["Figure 1: The model architecture. Encoder on the left."]
