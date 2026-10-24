"""Evidence ledger: tables read from the PDF layout, appended to the paper text, cited by boards as @T2.r9.c1."""

from services import ledger as L
from services.boards import normalize_board
from services.evidence import verify_board

# Attention Is All You Need, Table 2 (abridged) and Table 4 (a "Training" text column is the setting)
TABLES = [
    {"label": "Table 2", "page": 8,
     "caption": "Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models.",
     "columns": [["BLEU", "EN-DE"], ["BLEU", "EN-FR"], ["Training Cost (FLOPs)", "EN-DE"]],
     "rows": [
         {"label": "ByteNet [18]", "group": 0, "cells": ["23.75", "", ""]},
         {"label": "GNMT + RL [38]", "group": 0, "cells": ["24.6", "39.92", "2.3 · 10^19"]},
         {"label": "ConvS2S [9]", "group": 0, "cells": ["25.16", "40.46", "9.6 · 10^18"]},
         {"label": "MoE [32]", "group": 0, "cells": ["26.03", "40.56", "2.0 · 10^19"]},
         {"label": "GNMT + RL Ensemble [38]", "group": 1, "cells": ["26.30", "41.16", "1.8 · 10^20"]},
         {"label": "ConvS2S Ensemble [9]", "group": 1, "cells": ["26.36", "41.29", "7.7 · 10^19"]},
         {"label": "Transformer (base model)", "group": 2, "cells": ["27.3", "38.1", "3.3 · 10^18"]},
         {"label": "Transformer (big)", "group": 2, "cells": ["28.4", "41.8", "2.3 · 10^19"]},
     ], "confidence": 0.95},
    {"label": "Table 4", "page": 9, "caption": "Table 4: The Transformer generalizes well to English constituency parsing",
     "columns": [["Training"], ["WSJ 23 F1"]],
     "rows": [
         {"label": "Petrov et al. (2006) [29]", "cells": ["WSJ only, discriminative", "90.4"]},
         {"label": "Dyer et al. (2016) [8]", "cells": ["WSJ only, discriminative", "91.7"]},
         {"label": "Transformer (4 layers)", "cells": ["WSJ only, discriminative", "91.3"]},
         {"label": "McClosky et al. (2006) [26]", "cells": ["semi-supervised", "92.1"]},
         {"label": "Transformer (4 layers)", "cells": ["semi-supervised", "92.7"]},
     ], "confidence": 0.9},
    {"label": "Table 3", "page": 9, "caption": "Table 3: Variations on the Transformer architecture.",
     "columns": [["PPL (dev)"], ["BLEU (dev)"]],
     "rows": [{"label": "base", "cells": ["4.92", "25.8"]}, {"label": "big", "cells": ["4.33", "26.4"]}],
     "confidence": 0.3},
]
BODY = "The Transformer achieves 28.4 BLEU on the WMT 2014 English-to-German translation task."


def paper():
    return L.with_ledger(BODY, L.Ledger.from_tables(TABLES, min_confidence=0.5))


def test_section_round_trip_and_low_confidence_tables_left_out():
    text = paper()
    assert text.startswith(BODY) and L.LEDGER_HEAD in text
    led = L.ledger_of(text)
    assert [t.id for t in led.tables] == ["T2", "T4"]          # Table 3 is below the confidence floor
    t2 = led.table("T2")
    assert t2.columns[0] == "BLEU › EN-DE" and t2.rows[0].label == "ByteNet"   # citations dropped from row names
    assert [r.group for r in t2.rows] == [0, 0, 0, 0, 1, 1, 2, 2]
    c = led.cell("@T2.r8.c1")
    assert (c.value, c.primary, c.number, c.label, c.column) == ("28.4", "28.4", 28.4, "Transformer (big)", "BLEU › EN-DE")
    assert led.cell("T2.R8.C3").primary == "2.3 · 10^19"
    assert led.cell("@T2.r1.c2") is None and led.cell("@T9.r1.c1") is None   # an empty cell, an unknown table
    assert L.with_ledger(text, led) == text                    # appending again replaces the section


def test_row_lines_are_table_rows_for_the_evidence_check():
    led = L.ledger_of(paper())
    line = led.row_line("T2", 8)
    assert line == "[T2.r8]   Transformer (big)   c1: 28.4   c2: 41.8   c3: 2.3 · 10^19"


def test_bind_board_copies_values_quotes_and_source():
    led = L.ledger_of(paper())
    spec = {"kind": "bar", "title": "EN-DE BLEU", "source": "Figure 2", "bars": [
        {"label": "ConvS2S Ensemble", "value": "@T2.r6.c1", "baseline": True},
        {"label": "Transformer big", "value": "@T2.r8.c1", "highlight": True},
        {"label": "MoE", "value": "26.03"},                     # a plain number next to its name binds too
        {"label": "Ghost", "value": "@T2.r99.c1"}]}
    bound = L.bind_board(spec, led)
    bars = bound["bars"]
    assert [b["value"] for b in bars] == ["26.36", "28.4", "26.03"]
    assert [b["cell"] for b in bars] == ["T2.r6.c1", "T2.r8.c1", "T2.r4.c1"]
    assert bars[1]["paper_label"] == "Transformer (big)" and bars[1]["quote"] == led.row_line("T2", 8)
    assert bound["source"] == "Table 2" and bound["unbound"] == ['@T2.r99.c1 (bar "Ghost")']
    assert spec["bars"][0]["value"] == "@T2.r6.c1"             # the model's spec is not changed in place
    norm = normalize_board(bound)
    assert norm["bars"][0]["cell"] == "T2.r6.c1" and norm["unbound"]
    assert verify_board(norm, paper()) == []                     # every value comes from a cell


def test_a_changed_value_on_a_bound_bar_fails_the_check():
    text = paper()
    led = L.ledger_of(text)
    norm = normalize_board(L.bind_board({"kind": "bar", "title": "t", "bars": [
        {"label": "Transformer big", "value": "@T2.r8.c1", "highlight": True},
        {"label": "MoE", "value": "@T2.r4.c1"}]}, led))
    norm["bars"][0]["value"] = "29.4"
    norm["bars"][0]["quote"] = ""
    assert any("29.4" in x for x in verify_board(norm, text))


def test_table_cells_and_texts_are_bound():
    led = L.ledger_of(paper())
    bound = L.bind_board({"kind": "table", "title": "Parsing", "columns": ["Parser", "Training", "F1"],
                          "rows": [{"cells": ["Transformer (4 layers)", "@T4.r3.c1", "@T4.r3.c2"]}],
                          "note": "best single: @T4.r2.c2"}, led)
    row = bound["rows"][0]
    assert row["cells"] == ["Transformer (4 layers)", "WSJ only, discriminative", "91.3"]
    assert row["cell_ids"] == [None, "T4.r3.c1", "T4.r3.c2"] and row["quote"] == led.row_line("T4", 3)
    assert bound["note"] == "best single: 91.7" and bound["source"] == "Table 4"
    assert verify_board(normalize_board(bound), paper()) == []


def test_bind_text_and_unresolved():
    led = L.ledger_of(paper())
    assert L.bind_text("Big Transformer: @T2.r8.c1 BLEU!", led) == "Big Transformer: 28.4 BLEU!"
    assert L.bind_text("no citation here", led) == "no citation here"
    assert L.unresolved("see @T2.r1.c2 and @T2.r8.c1", led) == ["@T2.r1.c2"]


def test_match_needs_the_name_and_prefers_the_cited_column():
    led = L.ledger_of(paper())
    assert led.match("41.8", ["Transformer (big)"]).id == "T2.r8.c2"
    assert led.match("41.8", ["ConvS2S"]) is None               # the value is not on that row
    assert led.match("28.4", []) is None                         # no name: no guess


def test_strongest_rival_same_column_setting_and_no_variants():
    led = L.ledger_of(paper())
    # single models only: MoE beats the other single systems; the base Transformer is a variant
    assert led.strongest_rival("@T2.r8.c1").label == "MoE"
    # the claim includes ensembles: ConvS2S Ensemble
    assert led.strongest_rival("@T2.r8.c1", context="ensembles included").label == "ConvS2S Ensemble"
    # parsing: only rivals trained on WSJ alone count for the WSJ-only Transformer
    assert led.strongest_rival("@T4.r3.c2").label == "Dyer et al. (2016)"
    assert led.strongest_rival("@T4.r5.c2").label == "McClosky et al. (2006)"
    # lower is better: training cost
    assert led.strongest_rival("@T2.r8.c3", higher_is_better=False).label == "ConvS2S"


def test_strongest_rival_skips_ablation_tables_and_nameless_rows():
    raw = [{"label": "Table 5", "page": 3, "caption": "Table 5: Ablation over the pre-training tasks",
            "columns": [["MNLI"]], "rows": [{"label": "BERT base", "cells": ["84.4"]}, {"label": "No NSP", "cells": ["83.9"]}]},
           {"label": "Table 6", "page": 4, "caption": "Table 6: results", "columns": [["Acc"]],
            "rows": [{"label": "(A)", "cells": ["70.1"]}, {"label": "(B)", "cells": ["71.0"]}]}]
    led = L.Ledger.from_tables(raw)
    assert led.strongest_rival("@T5.r1.c1") is None
    assert led.strongest_rival("@T6.r2.c1") is None


def test_table_ids_from_labels():
    raw = [{"label": "Table 1", "page": 1, "columns": [["a"]], "rows": [{"label": "x", "cells": ["1.5"]}]},
           {"label": "Table 1", "page": 9, "columns": [["a"]], "rows": [{"label": "x", "cells": ["2.5"]}]},
           {"label": "Figure 4", "page": 5, "columns": [["a"]], "rows": [{"label": "x", "cells": ["3.5"]}]},
           {"label": "Table A.2", "page": 12, "columns": [["a"]], "rows": [{"label": "x", "cells": ["4.5"]}]},
           {"label": "Table 7", "page": 13, "columns": [["a"]], "rows": [{"label": "x", "cells": ["text only"]}]}]
    led = L.Ledger.from_tables(raw)
    assert [t.id for t in led.tables] == ["T1", "T1b", "F4", "TA2"]   # a table without a number is left out
    assert L.ledger_of(L.with_ledger("body", led)).cell("@T1b.r1.c1").primary == "2.5"


def test_negative_and_scientific_primary_values():
    led = L.Ledger.from_tables([{"label": "Table 1", "page": 1, "columns": [["delta"], ["cost"]],
                                 "rows": [{"label": "ours", "cells": ["−0.9", "1.0 · 10^20 (est.)"]}]}])
    assert led.cell("@T1.r1.c1").primary == "-0.9" and led.cell("@T1.r1.c1").number == -0.9
    assert led.cell("@T1.r1.c2").primary == "1.0 · 10^20"


def test_no_ledger_changes_nothing():
    spec = {"kind": "bar", "bars": [{"label": "a", "value": "@T2.r8.c1"}]}
    assert L.bind_board(spec, L.Ledger()) is spec
    assert L.ledger_of("no tables here").tables == [] and L.section("no tables") == ""


def test_section_heading_rows_separate_settings_and_the_papers_own_rows():
    raw = [{"label": "Table 2", "page": 4, "caption": "Table 2: SQuAD 1.1 results",
            "columns": [["Test", "F1"]],
            "rows": [{"label": "Published", "cells": [""]},
                     {"label": "BiDAF+ELMo", "cells": ["85.8"]},
                     {"label": "R.M. Reader (Ensemble)", "cells": ["88.5"]},
                     {"label": "Ours", "cells": [""]},
                     {"label": "BERT LARGE (Single)", "cells": ["90.9"]},
                     {"label": "Our full system", "cells": ["91.8"]}]},
           {"label": "Table 4", "page": 9, "caption": "Table 4: parsing", "columns": [["F1"]],
            "rows": [{"label": "WSJ only, discriminative", "cells": [""]},
                     {"label": "Dyer et al.", "cells": ["91.7"]},
                     {"label": "Transformer (4 layers)", "cells": ["91.3"]},
                     {"label": "Semi-supervised", "cells": [""]},
                     {"label": "Vinyals et al. (semi)", "cells": ["92.1"]}]}]
    led = L.Ledger.from_tables(raw)
    text = L.with_ledger("body", led)
    again = L.ledger_of(text)
    assert [r.cells for r in again.table("T2").rows][0] == {} and again.table("T2").rows[0].label == "Published"
    assert again.row_line("T2", 4) == "[T2.r4]   Ours"
    # BERT's rivals come from outside "Ours"; the ensemble needs the board to speak of ensembles
    assert again.strongest_rival("@T2.r5.c1").label == "BiDAF+ELMo"
    assert again.strongest_rival("@T2.r5.c1", context="ensembles included").label == "R.M. Reader (Ensemble)"
    # the WSJ-only Transformer is compared with WSJ-only parsers, not with the semi-supervised section
    assert again.strongest_rival("@T4.r3.c1").label == "Dyer et al."


def test_a_figure_reading_binds_to_its_row_only_within_the_charts_column():
    led = L.ledger_of(paper())
    # "28" read off a figure: the chart's other bars cite column c1, so Transformer (big)'s 28.4 there
    bound = L.bind_board({"kind": "bar", "title": "t", "bars": [
        {"label": "MoE", "value": "@T2.r4.c1"}, {"label": "Transformer big", "value": "28", "highlight": True}]}, led)
    assert bound["bars"][1]["cell"] == "T2.r8.c1" and bound["bars"][1]["value"] == "28.4"
    # without a cited column no guess is made, and a value far from the cell stays unbound
    assert "cell" not in L.bind_board({"kind": "bar", "title": "t", "bars": [{"label": "Transformer big", "value": "28"}]}, led)["bars"][0]
    far = L.bind_board({"kind": "bar", "title": "t", "bars": [
        {"label": "MoE", "value": "@T2.r4.c1"}, {"label": "Transformer big", "value": "30.1"}]}, led)
    assert "cell" not in far["bars"][1]


def test_a_row_printed_as_an_addition_keeps_the_method_it_adds_to():
    raw = [{"label": "Table 1", "page": 20, "caption": "Table 1: arithmetic reasoning", "columns": [["GSM8K"]],
            "rows": [{"label": "Prior best | N/A (finetuning)", "cells": ["55^a"]},
                     {"label": "Codex | Standard", "cells": ["19.7"]},
                     {"label": "Codex | Chain of thought", "cells": ["63.1 (+43.4)"]},
                     {"label": "Codex | + ext. calc", "cells": ["65.4"]},
                     {"label": "PaLM 540B | Standard", "cells": ["17.9"]},
                     {"label": "PaLM 540B | Chain of thought", "cells": ["56.9 (+39.0)"]},
                     {"label": "PaLM 540B | + ext. calc", "cells": ["58.6"]}]}]
    led = L.ledger_of(L.with_ledger("body", L.Ledger.from_tables(raw)))
    assert led.table("T1").rows[3].label == "Codex | Chain of thought + ext. calc"
    # chain of thought on another model (with or without a calculator) is the paper's method, not an earlier system
    assert led.strongest_rival("@T1.r6.c1").label == "Prior best | N/A (finetuning)"
