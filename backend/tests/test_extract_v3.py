"""Extraction for v3: untitled appendices after the bibliography and the paper's own figure example sentences."""
from services.paper_extractor import _appendix_after, _captions_after, _cut_tail, _token_sentences, _with_figure_sentences


def _refs(n: int) -> list[str]:
    return [f"[{i}] A. Author, B. Author. Some title of a paper. In Proceedings of NeurIPS, 2017." for i in range(1, n + 1)]


def test_untitled_appendix_heading_after_bibliography_is_kept():
    lines = ["Intro text."] * 50 + ["References"] + _refs(30) + ["Attention Visualizations"] + ["Figure 3: An example of attention " + "x" * 80] * 25
    text = "\n".join(lines)
    body, cut = _cut_tail(text)
    assert "Attention Visualizations" in body and "Figure 3:" in body
    assert "[12] A. Author" not in body


def test_reference_continuation_lines_are_not_headings():
    lines = ["References"] + _refs(10) + ["Neural Computation"] + _refs(30)
    assert _appendix_after(lines, 0) is None


def test_captions_after_bibliography_survive_without_heading():
    lines = ["References"] + _refs(5) + ["Figure 4: Two attention heads, also in layer 5 of 6, apparently involved",
                                          "in anaphora resolution. Note that the attentions are very sharp."]
    assert _captions_after(lines, 0) == ["Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Note that the attentions are very sharp."]


def test_figure_word_list_becomes_one_sentence_once():
    words = "The Law will never be perfect , but its application should be just - this is what we are missing , in my opinion . <EOS> <pad>".split()
    raw = "\n".join(words + words + ["Figure 4: caption text that is long enough to be a caption line"])
    assert _token_sentences(raw) == ["The Law will never be perfect , but its application should be just - this is what we are missing , in my opinion ."]


def test_table_columns_and_block_labels_are_not_sentences():
    cells = "\n".join(["80.6/80.1", "66.1", "82.3", "93.2", "35.0", "81.0", "86.0", "61.7", "74.0"])
    blocks = "\n".join(["Trm"] * 12)
    assert _token_sentences(cells) == [] and _token_sentences(blocks) == []


def test_scrambled_figure_lines_are_replaced_but_table_rows_stay():
    words = "It is in this spirit that a majority of American governments have passed new laws since 2009 making the registration or voting process more difficult . <EOS>".split()
    raw = "\n".join(words)
    text = "\n".join(["Attention Visualizations", "in this that majority governments passed since making or process is spirit",
                      "Pre-OpenAI SOTA 80.6/80.1 66.1 82.3 93.2 35.0 81.0 86.0 61.7 74.0"])
    out = _with_figure_sentences(text, raw)
    assert "in this that majority governments" not in out
    assert "Pre-OpenAI SOTA 80.6/80.1" in out
    assert "Example sentence shown in a figure: It is in this spirit that a majority of American governments" in out


def test_example_sentences_repeated_on_several_pages_are_kept_once():
    from services.paper_extractor import _dedupe_examples
    line = "Example sentence shown in a figure: The Law will never be perfect , but its application should be just"
    assert _dedupe_examples("\n".join([line, "Figure 4: x", line, "Figure 5: y"])).count(line) == 1
