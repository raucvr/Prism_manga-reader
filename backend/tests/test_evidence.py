"""services.evidence: the quote check and the row check on the real Attention (Table 2) and BERT paper texts; converted,
rounded, mislabeled and invented values fail; text numbers must occur in the paper."""

from pathlib import Path

import pytest

from services.evidence import normalize_text, quote_in_paper, verify_board, verify_text_numbers
from services.paper_extractor import extract_paper

FIXTURES = Path(__file__).resolve().parent / "fixtures"   # the PDFs are downloaded once (fixtures/README.md)
ATTENTION_PDF = FIXTURES / "attention_is_all_you_need.pdf"
BERT_PDF = FIXTURES / "bert.pdf"

# A few lines of the Attention paper as the extractor writes them (always available, unlike the job outputs above).
EXCERPT = """Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including
ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.8 after
training for 3.5 days on eight GPUs, a small fraction of the training costs of the
       GNMT + RL Ensemble                   26.30     41.16      1.8 · 10^20  1.1 · 10^21
       ConvS2S Ensemble                     26.36     41.29      7.7 · 10^19  1.2 · 10^21
       Transformer (big)                    28.4      41.8              2.3 · 10^19
"""


def bars(*rows, **extra):
    """A bar board from (label, value, quote or "", highlight) rows."""
    return {"kind": "bar", "title": "T", **extra,
            "bars": [{"label": lab, "value": v, "quote": q, "highlight": h} for lab, v, q, h in rows]}


@pytest.fixture(scope="module")
def attention():
    if not ATTENTION_PDF.exists():
        pytest.skip("tests/fixtures/attention_is_all_you_need.pdf is missing (see fixtures/README.md)")
    return extract_paper(ATTENTION_PDF).body   # the text the pipeline sees, tables read from the layout included


@pytest.fixture(scope="module")
def bert():
    if not BERT_PDF.exists():
        pytest.skip("tests/fixtures/bert.pdf is missing (see fixtures/README.md)")
    return extract_paper(BERT_PDF).body


# ----------------------------------------------------------------------------- normalization and quotes

def test_normalize_text_folds_dashes_quotes_hyphens_and_space():
    assert normalize_text("“Self‑Attention” — it’s English-\nto-German,   newstest2014") == \
        "\"selfattention\" - it's englishtogerman, newstest2014"


def test_quote_in_paper_tolerates_layout_but_not_other_numbers():
    assert quote_in_paper("Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task", EXCERPT)
    assert quote_in_paper("ConvS2S   Ensemble 26.36 41.29", EXCERPT)
    assert quote_in_paper("training for 3.5 days on eight GPUs", EXCERPT)     # across a line break
    assert quote_in_paper("Transformer (big) 28.4 41.8 2.3 × 10¹⁹", EXCERPT)    # the display form of 2.3 · 10^19
    assert not quote_in_paper("ConvS2S Ensemble 26.3", EXCERPT)                 # 26.36 in the paper
    assert not quote_in_paper("Our model achieves 28.5 BLEU", EXCERPT)
    assert not quote_in_paper("26.36", EXCERPT)                                 # a quote needs words
    assert not quote_in_paper("", EXCERPT)


def test_excerpt_quote_and_row_checks():
    good = bars(("ConvS2S ensemble", "26.36", "ConvS2S Ensemble 26.36 BLEU", False),    # quote not verbatim: row check
                ("Transformer big", "28.4", "Our model achieves 28.4 BLEU", True))
    assert verify_board(good, EXCERPT) == []
    bad = bars(("ConvS2S ensemble", "26.4", "", False), ("Transformer big", "28.4", "", True))
    problems = verify_board(bad, EXCERPT)
    assert len(problems) == 1 and "26.4" in problems[0] and "ConvS2S ensemble" in problems[0]


# ----------------------------------------------------------------------------- Attention: Table 2

def test_attention_table2_bleu_bars_pass(attention):
    spec = {"kind": "bar", "title": "WMT14 English → German BLEU", "unit": "BLEU", "source": "Table 2", "bars": [
        {"label": "ConvS2S Ens", "paper_label": "ConvS2S Ensemble", "value": "26.36", "quote": "ConvS2S Ensemble 26.36 BLEU"},
        {"label": "GNMT+RL Ens", "paper_label": "GNMT + RL Ensemble", "value": "26.30", "quote": "GNMT + RL Ensemble 26.30 BLEU"},
        {"label": "Transformer base", "value": "27.3", "highlight": True},
        {"label": "Transformer Big", "paper_label": "Transformer (big)", "value": "28.4",
         "quote": "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task", "highlight": True}]}
    assert verify_board(spec, attention) == []


def test_attention_english_french_single_models_pass(attention):
    spec = bars(("GNMT + RL", "39.92", "", False), ("ConvS2S", "40.46", "", False), ("MoE", "40.56", "", False),
                ("Transformer (big)", "41.8", "", True), title="WMT 2014 English-to-French (BLEU)")
    assert verify_board(spec, attention) == []


@pytest.mark.parametrize("big, ensemble", [("2.3 · 10^19", "1.2 · 10^21"), ("2.3 × 10¹⁹", "1.2 × 10²¹"), ("2.3·10^19", "1.2x10^21")])
def test_attention_training_cost_in_any_notation_passes(attention, big, ensemble):
    spec = bars(("ConvS2S Ensemble", ensemble, "", False), ("GNMT + RL Ensemble", "1.1 · 10^21", "", False),
                ("Transformer (big)", big, "", True), unit="FLOPs", higher_is_better=False)
    assert verify_board(spec, attention) == []


def test_attention_training_cost_on_a_stated_scale(attention):
    # the script's real case: 1.2 · 10^21 FLOPs written as "120" with a unit of 10^19
    spec = {"kind": "bar", "title": "Training Cost En-Fr", "unit": "FLOPs × 10^19", "higher_is_better": False, "bars": [
        {"label": "ConvS2S Ens", "paper_label": "ConvS2S Ensemble", "value": "120",
         "quote": "ConvS2S Ensemble training cost 1.2 · 10^21 FLOPs"},
        {"label": "GNMT+RL Ens", "paper_label": "GNMT + RL Ensemble", "value": "110",
         "quote": "GNMT + RL Ensemble training cost 1.1 · 10^21 FLOPs"},
        {"label": "Transformer Big", "paper_label": "Transformer (big)", "value": "2.3",
         "quote": "Transformer (big) training cost 2.3 · 10^19 FLOPs", "highlight": True}]}
    # a scale stated once in the unit is read: "120" at "FLOPs × 10^19" is exactly the paper's 1.2 · 10^21 (Table 2's row)
    assert verify_board(spec, attention) == []
    # a wrong scale is still caught: "12" at 10^19 is 1.2 · 10^20, not the 1.2 · 10^21 on the ConvS2S Ensemble row
    wrong = {**spec, "bars": [{**spec["bars"][0], "value": "12"}, spec["bars"][2]]}
    assert any('"ConvS2S Ens"' in p and "12" in p for p in verify_board(wrong, attention))


def test_attention_invented_baseline_fails(attention):
    spec = bars(("Prior best", "14.0", "", False), ("Transformer Big", "3.5", "Training took 3.5 days on 8 P100 GPUs", True),
                title="Training Time En-Fr", unit="Days on 8 GPUs", higher_is_better=False)
    problems = verify_board(spec, attention)
    assert len(problems) == 1 and "14.0" in problems[0] and '"Prior best"' in problems[0]


def test_attention_rounded_value_fails_even_where_the_number_exists_elsewhere(attention):
    # 26.4 is in the paper (Table 3, big model dev BLEU) but not on the ConvS2S Ensemble row
    problems = verify_board(bars(("ConvS2S Ensemble", "26.4", "", False), ("Transformer (big)", "28.4", "", True)), attention)
    assert len(problems) == 1 and "26.4" in problems[0]


def test_attention_trailing_zero_is_the_same_value(attention):
    assert verify_board(bars(("GNMT + RL Ensemble", "26.3", "", False), ("Transformer (big)", "28.4", "", True)), attention) == []


def test_attention_mislabeled_rows_fail(attention):
    # 1.2 · 10^21 belongs to the ConvS2S Ensemble row, and 41.29 is an ensemble, not the best single model
    swapped = bars(("GNMT + RL Ensemble", "1.2 · 10^21", "", False), ("Transformer (big)", "2.3 · 10^19", "", True))
    assert any("GNMT + RL Ensemble" in p for p in verify_board(swapped, attention))
    single = bars(("Best Single", "41.29", "", False), ("Transformer Big", "41.8", "", True))
    assert any("41.29" in p for p in verify_board(single, attention))


def test_attention_quote_must_contain_the_value(attention):
    spec = bars(("Transformer (big)", "28.5", "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German", True),
                ("ConvS2S Ensemble", "26.36", "", False))
    problems = verify_board(spec, attention)
    assert len(problems) == 1 and "28.5" in problems[0] and "does not contain" in problems[0]


def test_attention_table_boards(attention):
    table1 = {"kind": "table", "title": "Linking two words", "columns": ["Layer type", "Sequential operations", "Maximum path length"],
              "rows": [["Self-Attention", "O(1)", "O(1)"], ["Recurrent", "O(n)", "O(n)"], ["Convolutional", "O(1)", "O(log_k(n))"]],
              "highlight_rows": [0], "source": "Table 1"}
    assert verify_board(table1, attention) == []
    table4 = {"kind": "table", "title": "English constituency parsing", "columns": ["Parser", "Training", "WSJ 23 F1"],
              "rows": [{"cells": ["Dyer et al. (2016)", "WSJ only, discriminative", "91.7"]},
                       {"cells": ["Transformer (4 layers)", "semi-supervised", "92.7"]}], "highlight_rows": [1]}
    assert verify_board(table4, attention) == []
    mislabeled = {**table4, "rows": [{"cells": ["Transformer (4 layers)", "WSJ only", "92.7"]}]}
    assert any("92.7" in p for p in verify_board(mislabeled, attention))
    numbers_only = {"kind": "table", "title": "Heads (Table 3)", "columns": ["Heads", "BLEU (dev)"],
                    "rows": [["1", "24.9"], ["16", "25.8"]]}
    assert verify_board(numbers_only, attention) == []
    wrong_pair = {**numbers_only, "rows": [["8", "25.4"]]}    # 25.4 is the 32-head row
    problems = verify_board(wrong_pair, attention)
    assert problems and all("together with the row's other numbers" in p for p in problems)


def test_attention_text_numbers_on_flow_list_links(attention):
    recap = {"kind": "list", "title": "What the paper showed", "items": [
        "28.4 BLEU English→German: over 2 BLEU above the best, ensembles included", "41.8 BLEU English→French",
        "3.5 days on 8 GPUs", "+2.04 BLEU over ConvS2S"]}
    problems = verify_board(recap, attention)
    assert len(problems) == 1 and "2.04" in problems[0] and "item 4" in problems[0]
    flow = {"kind": "flow", "title": "The Transformer", "steps": ["Encoder: 6 layers", "Decoder: 6 layers", "WMT 2014 test"], "note": "d_model = 512"}
    assert verify_board(flow, attention) == []
    example = {"kind": "links", "title": "Attention, illustrated", "tokens": ["Mix", "137g", "butter"], "from": 0,
               "weights": [0, 0.7, 0.3], "illustrative": True}
    assert verify_board(example, attention) == []
    assert verify_board({**example, "illustrative": False}, attention) == []   # a links board is always an illustration


def test_verify_board_unusable_spec(attention):
    assert verify_board({"kind": "pie"}, attention) == ["the board spec is unusable, so its numbers cannot be checked (see board_problems)"]


# ----------------------------------------------------------------------------- BERT

def test_bert_abstract_results_pass(bert):
    glue = {"kind": "bar", "title": "GLUE Test Score", "unit": "%", "bars": [
        {"label": "Prior SOTA", "paper_label": "Pre-OpenAI SOTA", "value": "74.0", "quote": "Pre-OpenAI SOTA Average 74.0"},
        {"label": "OpenAI GPT", "value": "75.1", "quote": "OpenAI GPT Average 75.1"},
        {"label": "BERT LARGE", "paper_label": "BERTLARGE", "value": "80.5",
         "quote": "pushing the GLUE score to 80.5% (7.7% point absolute improvement)", "highlight": True}]}
    assert verify_board(glue, bert) == []
    mnli = bars(("OpenAI GPT", "82.1", "", False), ("BERT LARGE", "86.7", "MultiNLI accuracy to 86.7% (4.6% absolute improvement)", True))
    assert verify_board(mnli, bert) == []
    squad = {"kind": "bar", "title": "SQuAD v1.1 Test F1", "bars": [
        {"label": "Prior #1 Ensemble", "paper_label": "#1 Ensemble - nlnet", "value": "91.7"},
        {"label": "BERT LARGE", "paper_label": "BERTLARGE (Ens.+TriviaQA)", "value": "93.2", "highlight": True}]}
    assert verify_board(squad, bert) == []


def test_bert_invented_and_mislabeled_values_fail(bert):
    # 85.0 occurs in the paper (SWAG human expert) but not as a BERT GLUE score
    assert verify_board(bars(("OpenAI GPT", "75.1", "", False), ("BERT LARGE", "85.0", "", True)), bert)
    assert verify_board(bars(("OpenAI GPT", "75.1", "", False), ("BERT LARGE", "81.5", "GLUE score to 81.5%", True)), bert)


def test_bert_text_numbers(bert):
    flow = {"kind": "flow", "title": "BERT pipeline", "steps": ["Pre-train on 3.3B words", "Mask 15% of tokens", "Fine-tune on SQuAD v2.0"]}
    assert verify_board(flow, bert) == []
    assert verify_text_numbers("BERT pushes GLUE to 80.5% and MultiNLI to 86.7%, SQuAD v1.1 F1 to 93.2.", bert) == []
    assert verify_text_numbers("BERT beats GPT by 5.4 points.", bert) == ["the number 5.4 is not in the paper"]


# ----------------------------------------------------------------------------- verify_text_numbers

def test_verify_text_numbers(attention):
    assert verify_text_numbers("28.4 BLEU after 3.5 days on 8 GPUs, about 2 BLEU better", attention) == []
    assert verify_text_numbers("That is 14.0 days faster, +2.04 BLEU, and 14.0 again.", attention) == \
        ["the number 14.0 is not in the paper", "the number 2.04 is not in the paper"]
    assert verify_text_numbers("8张GPU跑3.5天就训练好了，只要以前的1/4成本。", attention) == []
    assert verify_text_numbers("Mix 137g butter with 3 eggs", attention) == ["the number 137 is not in the paper"]
    assert verify_text_numbers("Mix 137g butter with 3 eggs", attention, allowed="Mix 137g butter, then add 3 eggs") == []
    assert verify_text_numbers("", attention) == []


def test_verify_text_numbers_on_the_excerpt():
    assert verify_text_numbers("Transformer (big): 2.3 × 10¹⁹ FLOPs vs 1.2 · 10^21", EXCERPT) == []
    assert verify_text_numbers("1.2 · 10^20 FLOPs", EXCERPT) == ["the number 1.2 · 10^20 is not in the paper"]
    assert verify_text_numbers("ConvS2S and GPT-2 on WMT14", EXCERPT) == []


# ----------------------------------------------------------------------------- review fixes

def test_bare_numbers_drawn_on_a_board_are_checked_too():
    # "+2.04" alone is not translatable text, so it used to skip the check although the reader sees it on the board
    recap = {"kind": "list", "title": "What the paper showed", "items": ["28.4 BLEU", "+2.04", "41.8"]}
    problems = verify_board(recap, EXCERPT)
    assert len(problems) == 1 and "2.04" in problems[0] and "item 2" in problems[0]
    assert verify_board({**recap, "items": ["28.4 BLEU", "41.8"]}, EXCERPT) == []
    titled = {"kind": "flow", "title": "28.5", "steps": ["Encoder", "Decoder"]}
    assert any("28.5" in p and p.startswith("title") for p in verify_board(titled, EXCERPT))
    links = {"kind": "links", "title": "T", "tokens": ["Mix", "137", "butter"], "from": 0, "weights": [0, 0.7, 0.3]}
    assert verify_board({**links, "illustrative": True}, EXCERPT) == []           # the running example's own words
    assert verify_board({**links, "illustrative": False}, EXCERPT) == []          # always drawn and read as an illustration


def test_an_elided_quote_is_checked_piece_by_piece():
    elided = "Our model achieves 28.4 BLEU … improving over the existing best results"
    assert quote_in_paper(elided, EXCERPT) and quote_in_paper(elided.replace("…", "..."), EXCERPT)
    assert quote_in_paper("Our model achieves 28.4 BLEU [...] by over 2 BLEU", EXCERPT)
    assert not quote_in_paper("Our model achieves 28.4 BLEU … beating every ensemble", EXCERPT)   # a piece not in the paper
    assert not quote_in_paper("Our model achieves … 28.4", EXCERPT)                            # a piece without words
    spec = bars(("Prior best", "26.36", "improving over the existing best results … ConvS2S Ensemble 26.36 41.29", False),
                ("Transformer (big)", "28.4", "", True))
    assert verify_board(spec, EXCERPT) == []


def test_a_footnoted_table_value_and_a_model_size_are_supported():
    paper = "Table 1. GSM8K\nPrior best  N/A (finetuning) 55^(a) 57.4^(b)\nChain of thought 56.9 (+39.0)\nPaLM 540B with just eight exemplars"
    bar = {"kind": "bar", "title": "GSM8K", "bars": [
        {"label": "Finetuned GPT-3 + verifier", "value": "55", "quote": "Prior best N/A (finetuning) 55^(a)", "baseline": True},
        {"label": "PaLM 540B chain of thought", "value": "56.9", "quote": "Chain of thought 56.9 (+39.0)", "highlight": True}]}
    assert verify_board(bar, paper) == []
    assert verify_text_numbers("PaLM 540B beats it.", paper) == []


def test_a_chart_may_state_its_scale_once_in_the_unit():
    paper = "Model  Steps to 72.2%  Max accuracy\nInception  31.0 · 10^6  72.2%\nBN-x5  2.1 · 10^6  73.0%"
    for unit in ("million steps", "steps (millions)", "x10^6 steps"):
        chart = {"kind": "bar", "title": "Training steps to reach 72.2%", "unit": unit, "higher_is_better": False,
                 "bars": [{"label": "Inception", "value": "31.0", "quote": "Inception 31.0 · 10^6", "baseline": True},
                          {"label": "BN-x5", "value": "2.1", "quote": "BN-x5 2.1 · 10^6", "highlight": True}]}
        assert verify_board(chart, paper) == [], unit
    unscaled = {"kind": "bar", "title": "Training steps", "unit": "steps", "bars": [{"label": "Inception", "value": "31.0"}]}
    assert verify_board(unscaled, paper)                      # 31.0 steps is not what the paper reports
