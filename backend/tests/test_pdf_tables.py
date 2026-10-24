"""Tables read from the PDF geometry (services.pdf_tables): synthetic PDFs drawn with PyMuPDF in the test, with exact
cells, and the fixture papers when they are downloaded."""

from pathlib import Path

import pytest

from services import ledger as L
from services import pdf_tables
from services.pdf_tables import MIN_CONFIDENCE, extract_tables

from .fakes import ATTENTION_PDF, COT_PDF

RESNET_PDF = Path(__file__).resolve().parent / "fixtures" / "resnet.pdf"
PROSE = "The systems are compared on the same data, and the text around each table is ordinary running prose."
LANGUAGES = ["English", "German", "French", "Italian", "Spanish", "Dutch", "Polish", "Czech", "Danish", "Finnish",
             "Greek", "Hungarian", "Latvian", "Maltese", "Swedish", "Slovak", "Irish", "Croatian", "Romanian"]


class Page:
    """A US Letter page drawn like a paper: 10 pt prose from x = 72 to the right margin, 9 pt tables."""

    def __init__(self, doc):
        self.page = doc.new_page(width=612, height=792)

    def prose(self, y: float, until: float) -> float:
        while y <= until:
            self.page.insert_text((72, y), PROSE, fontsize=10, fontname="helv")
            y += 12
        return y

    def text(self, x: float, y: float, s: str, align: str = "left", font: str = "helv", size: float = 9) -> None:
        import pymupdf

        w = pymupdf.get_text_length(s, fontname=font, fontsize=size)
        self.page.insert_text((x - w / 2 if align == "center" else x, y), s, fontsize=size, fontname=font)

    def row(self, y: float, label: str, *cells: str, xs=(330, 430), font: str = "helv") -> None:
        if label:
            self.text(110, y, label, font=font)
        for x, c in zip(xs, cells):
            if c:
                self.text(x, y, c, align="center")

    def rule(self, y: float, width: float = 0.4, x0: float = 100, x1: float = 520) -> None:
        self.page.draw_line((x0, y), (x1, y), width=width)


def _paper() -> bytes:
    """Six pages: a booktabs table with a spanning header (caption above), a table with section headings (caption
    below), a regression table with stacked standard errors, a table that runs onto the next page, prose."""
    import pymupdf

    doc = pymupdf.open()
    p = Page(doc)                                       # page 1: Table 1, caption above, "BLEU" over two columns
    y = p.prose(80, 300) + 14
    p.text(72, y, "Table 1: BLEU scores of the systems on two test sets.")
    p.rule(y + 8, 0.8)
    p.text(380, y + 20, "BLEU", align="center")
    p.rule(y + 23.5, 0.4, 300, 460)
    p.row(y + 33, "Model", "EN-DE", "EN-FR")
    p.rule(y + 37)
    for i, cells in enumerate((("ByteNet", "23.75", ""), ("ConvS2S", "25.16", "40.46"), ("Transformer (big)", "28.4", "41.8"))):
        p.row(y + 49 + 12 * i, *cells)
    p.rule(y + 80, 0.8)
    p.prose(y + 100, 740)

    p = Page(doc)                                       # page 2: Table 2, caption below, two italic section rows
    y = p.prose(80, 300) + 14
    p.rule(y, 0.8)
    p.row(y + 12, "Method", "Acc", "F1")
    p.rule(y + 16)
    rows = [("Supervised", None), ("BERT", ("84.6", "83.4")), ("RoBERTa", ("86.1", "85.0")),
            ("Zero-shot", None), ("GPT-3", ("70.2", "68.9")), ("Ours", ("75.4", "74.1"))]
    for i, (label, cells) in enumerate(rows):
        p.row(y + 28 + 12 * i, label, *(cells or ()), font="heit" if cells is None else "helv")
    p.rule(y + 28 + 12 * len(rows) - 7, 0.8)
    p.text(72, y + 28 + 12 * len(rows) + 10, "Table 2: Accuracy and F1 of the systems in two settings.")
    p.prose(y + 28 + 12 * len(rows) + 34, 740)

    p = Page(doc)                                       # page 3: Table 3, coefficients over standard errors
    y = p.prose(80, 300) + 14
    p.text(72, y, "Table 3: Effect of the treatment on the outcome.")
    p.rule(y + 8, 0.8)
    p.row(y + 20, "", "OLS", "IV")
    p.rule(y + 24)
    p.row(y + 36, "", "(1)", "(2)")
    for i, cells in enumerate((("Treatment", "0.120***", "0.095**"), ("", "(0.012)", "(0.030)"),
                               ("Income", "-0.031", "-0.027*"), ("", "(0.020)", "(0.011)"))):
        p.row(y + 48 + 12 * i, *cells)
    p.rule(y + 89)
    p.row(y + 101, "Observations", "1,204", "1,198")
    p.rule(y + 105, 0.8)
    p.prose(y + 125, 740)

    p = Page(doc)                                       # pages 4-5: Table 4 runs off the bottom of page 4
    y = p.prose(80, 500) + 14
    p.text(72, y, "Table 4: Score and number of test sentences per language.")
    p.rule(y + 8, 0.8)
    p.row(y + 20, "Language", "Score", "Count")
    p.rule(y + 24)
    yy = y + 36
    for i, lang in enumerate(LANGUAGES[:14]):
        p.row(yy + 12 * i, lang, f"{60 + i}.{i}", str(1000 + 10 * i))
    p = Page(doc)
    for i, lang in enumerate(LANGUAGES[14:]):
        p.row(84 + 12 * i, lang, f"{74 + i}.{i}", str(1140 + 10 * i))
    p.rule(84 + 12 * 5 - 7, 0.8)
    p.prose(84 + 12 * 5 + 20, 740)

    p = Page(doc)                                       # page 6: prose only
    p.prose(80, 740)
    return doc.tobytes()


@pytest.fixture(scope="module")
def tables():
    return {t["label"]: t for t in extract_tables(_paper())}


def _cells(t):
    return [(r["label"], r["cells"]) for r in t["rows"]]


def test_booktabs_table_with_a_spanning_header(tables):
    t = tables["Table 1"]
    assert t["page"] == 1 and t["caption"].startswith("Table 1: BLEU scores") and t["confidence"] >= MIN_CONFIDENCE
    assert t["columns"] == [["BLEU", "EN-DE"], ["BLEU", "EN-FR"]]
    assert _cells(t) == [("ByteNet", ["23.75", ""]), ("ConvS2S", ["25.16", "40.46"]), ("Transformer (big)", ["28.4", "41.8"])]


def test_section_heading_rows_and_a_caption_below_the_table(tables):
    t = tables["Table 2"]
    assert t["page"] == 2 and t["columns"] == [["Acc"], ["F1"]] and t["confidence"] >= MIN_CONFIDENCE
    assert _cells(t) == [("Supervised", ["", ""]), ("BERT", ["84.6", "83.4"]), ("RoBERTa", ["86.1", "85.0"]),
                         ("Zero-shot", ["", ""]), ("GPT-3", ["70.2", "68.9"]), ("Ours", ["75.4", "74.1"])]
    assert [r["section"] for r in t["rows"]] == ["Supervised"] * 3 + ["Zero-shot"] * 3


def test_standard_errors_fold_into_their_coefficients_and_column_numbers_into_the_header(tables):
    t = tables["Table 3"]
    assert t["columns"] == [["OLS", "(1)"], ["IV", "(2)"]]
    assert _cells(t) == [("Treatment", ["0.120*** (0.012)", "0.095** (0.030)"]), ("Income", ["-0.031 (0.020)", "-0.027* (0.011)"]),
                         ("Observations", ["1,204", "1,198"])]
    assert all(r["label"] for r in t["rows"])   # never a row of values without a name


def test_a_table_continues_on_the_next_page(tables):
    t = tables["Table 4"]
    assert t["page"] == 4 and t["columns"] == [["Score"], ["Count"]]
    assert [r["label"] for r in t["rows"]] == LANGUAGES
    assert t["rows"][0]["cells"] == ["60.0", "1000"] and t["rows"][-1]["cells"] == ["78.4", "1180"]
    assert "Table 5" not in tables and len(tables) == 4


def test_the_ledger_reads_the_tables_by_cell(tables):
    led = L.Ledger.from_tables(list(tables.values()), min_confidence=MIN_CONFIDENCE)
    assert led.cell("@T1.r3.c1").value == "28.4" and led.cell("@T1.r3.c1").column == "BLEU › EN-DE"
    t2 = led.table("T2")
    assert [r.label for r in t2.rows if not r.cells] == ["Supervised", "Zero-shot"]   # section headings
    assert L.Ledger._sections(t2)[6] == "Zero-shot"
    c = led.cell("@T3.r1.c1")
    assert (c.value, c.primary) == ("0.120*** (0.012)", "0.120")


def test_stub_columns_fold_into_the_label_and_header_paths_stay_unique():
    columns = [["W/A"], ["GSM8K"], ["GSM8K"]]
    rows = [{"label": lab, "section": "Qwen3", "cells": [wa, "1.0", "2.0"], "bold": [False] * 3}
            for lab, wa in (("GPTQ", "3/16"), ("OnQ", "3/16"), ("GPTQ", "2/16"), ("OnQ", "2/16"))]
    heads = ["Method"]
    assert pdf_tables._fold_stubs(columns, rows, heads) == 1
    assert [r["label"] for r in rows] == ["GPTQ | 3/16", "OnQ | 3/16", "GPTQ | 2/16", "OnQ | 2/16"]
    assert columns == [["GSM8K"], ["GSM8K"]] and heads == ["Method", "W/A"]
    # a column of measurements is never folded, however often the names repeat
    rows = [{"label": "(A)", "section": "", "cells": [v, "5.0"], "bold": [False] * 2} for v in ("4.92", "5.29")]
    assert pdf_tables._fold_stubs([["PPL"], ["BLEU"]], rows, []) == 0


def test_word_marks_after_the_number():
    chars = [{"c": c, "x0": x, "x1": x + 4, "role": role} for c, x, role in
             (("4", 0, "n"), ("4", 4, "n"), (".", 8, "n"), ("0", 10, "n"), ("†", 14, "sup"), ("±", 14, "sub"), ("4", 18, "sub"))]
    assert pdf_tables._script_text(chars) == "44.0 ± 4^†"


def test_never_raises():
    assert extract_tables(b"") == [] and extract_tables(b"not a pdf") == []
    import pymupdf

    doc = pymupdf.open()
    Page(doc).prose(80, 740)
    assert extract_tables(doc.tobytes()) == []


# ----------------------------------------------------------------------------- the fixture papers

def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path.name} not downloaded — see tests/fixtures/README.md")
    return path


def _row(t, label):
    return next(r for r in t["rows"] if r["label"] == label)


@pytest.fixture(scope="module")
def attention():
    return {t["label"]: t for t in extract_tables(_require(ATTENTION_PDF))}


def test_attention_table_2(attention):
    t = attention["Table 2"]
    assert t["confidence"] >= MIN_CONFIDENCE
    assert t["columns"] == [["BLEU", "EN-DE"], ["BLEU", "EN-FR"], ["Training Cost (FLOPs)", "EN-DE"], ["Training Cost (FLOPs)", "EN-FR"]]
    assert _row(t, "Transformer (big)")["cells"] == ["28.4", "41.8", "2.3 · 10^19", "2.3 · 10^19"]   # one cost printed over both columns
    assert _row(t, "ByteNet [18]")["cells"] == ["23.75", "", "", ""]
    assert [r["group"] for r in t["rows"]] == [0] * 5 + [1] * 3 + [2] * 2


def test_attention_paper_text_carries_the_ledger(attention_paper):
    led = L.ledger_of(attention_paper.body)
    assert attention_paper.stats["tables"] >= 2 and led.table("T2") is not None
    c = next(c for c in led.column("T2", 1) if c.label == "Transformer (big)")
    assert c.value == "28.4" and c.column == "BLEU › EN-DE"


def test_resnet_imagenet_table():
    t = next(t for t in extract_tables(_require(RESNET_PDF)) if t["label"] == "Table 3")
    assert t["confidence"] >= MIN_CONFIDENCE and t["columns"] == [["top-1 err."], ["top-5 err."]]
    assert _row(t, "ResNet-152")["cells"] == ["21.43", "5.71"] and _row(t, "ResNet-152")["bold"] == [True, True]
    assert _row(t, "GoogLeNet [44]")["cells"] == ["-", "9.15"]
    assert [r["group"] for r in t["rows"]] == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]


def test_chain_of_thought_table_with_a_multirow_label():
    t = next(t for t in extract_tables(_require(COT_PDF)) if t["label"] == "Table 1")
    assert t["confidence"] >= MIN_CONFIDENCE and t["columns"][0] == ["GSM8K"]
    assert _row(t, "PaLM 540B | Chain of thought")["cells"][0] == "56.9 (+39.0)"
    assert _row(t, "PaLM 540B | Standard")["cells"][0] == "17.9"
