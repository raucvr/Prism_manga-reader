"""services.fonts: the bundled font files (present, pinned, licensed), the resolver (config override, bundled main faces,
never an installed font for a main face, the fallback chain, installed CJK fonts only for Han characters), glyph
coverage (GB2312 levels 1 and 2, the symbols boards print) and characters no font has (ASCII stand-ins, emoji left out).
No network; nothing here needs an installed font."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

import services.boards as boards
from services import fonts
from services.fonts import FONT_DIR, bundled, has_glyph, resolve
from services.lettering import Face, FontSet, _pretty

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(FONT_DIR)))
# what boards and bubbles print: signs, comparisons, arrows, roots, powers and Greek
SYMBOLS = "−≈×√≤≥→±²³¹⁰⁻" + "αβγδεθλμπρστφω" + "ΔΣΩ"


def gb2312(level: int) -> list[str]:
    """The Han characters of GB2312 level 1 (rows 16-55, 3,755 characters) or level 2 (rows 56-87, 3,008)."""
    out = []
    for row in range(16, 56) if level == 1 else range(56, 88):
        for cell in range(1, 95):
            try:
                out.append(bytes([0xA0 + row, 0xA0 + cell]).decode("gb2312"))
            except UnicodeDecodeError:
                pass
    return out


@pytest.fixture
def no_installed_fonts(monkeypatch):
    """A machine without CJK fonts installed (Windows without them, Docker, a minimal Linux): only the bundled files."""
    monkeypatch.setattr(fonts, "system_cjk", lambda: ())
    for f in (boards._chain, boards._face):
        f.cache_clear()
    yield
    for f in (boards._chain, boards._face):
        f.cache_clear()


def test_bundled_files_are_present_pinned_and_licensed():
    assert fonts.problems() == ()                       # every file there, with the SHA-256 fonts.json pins
    for role in ("zh", "en", "digits", "symbols"):
        text, bold = bundled(role)
        assert text[0].startswith(FONT_DIR) and fonts.loads(text) and fonts.loads(bold), role
    with open(os.path.join(ROOT, "THIRD_PARTY_NOTICES.md"), encoding="utf-8") as f:
        notices = f.read()
    for name, info in fonts.manifest()["files"].items():
        assert os.path.getsize(os.path.join(FONT_DIR, name)) == info["bytes"], name
        assert os.path.isfile(os.path.join(FONT_DIR, info["license_file"])), name
        assert info["source"].startswith("https://") and name in notices, name


def test_a_missing_or_damaged_bundled_file_is_reported(monkeypatch, tmp_path):
    (tmp_path / "fonts.json").write_bytes(open(os.path.join(FONT_DIR, "fonts.json"), "rb").read())
    (tmp_path / "DejaVuSans.ttf").write_text("version https://git-lfs.github.com/spec/v1\n")   # a Git LFS pointer
    monkeypatch.setattr(fonts, "FONT_DIR", str(tmp_path))
    for f in (fonts.manifest, fonts.problems):
        f.cache_clear()
    try:
        found = fonts.problems()
        assert any("DejaVuSans.ttf is not the file" in p for p in found)
        assert any("ChillRoundM.otf is missing" in p for p in found)
        assert not fonts.status({})["ok"]
    finally:
        monkeypatch.undo()
        for f in (fonts.manifest, fonts.problems):
            f.cache_clear()


def test_the_bundled_zh_font_has_all_of_gb2312_levels_1_and_2():
    text, _ = bundled("zh")
    for level, count in ((1, 3755), (2, 3008)):
        chars = gb2312(level)
        lacking = "".join(c for c in chars if not has_glyph(text, c))
        assert len(chars) == count and not lacking, (level, lacking[:40])


def test_main_faces_are_the_bundled_files_and_the_chain_is_in_order(no_installed_fonts):
    plan = resolve({})
    for lang in ("zh", "en"):
        assert plan.files[lang] == dict(zip(("text", "bold"), bundled(lang)))
    assert [f["role"] for f in plan.fallbacks] == ["digits", "symbols", "cjk"]
    assert [f["role"] for f in plan.chain("zh")] == ["digits", "symbols"]          # the zh face is its own CJK font
    assert [f["role"] for f in plan.chain("en")] == ["digits", "symbols", "cjk"]   # Chinese words in English text


@pytest.mark.parametrize("lang", ["zh-CN", "en-US"])
@pytest.mark.parametrize("bold", [False, True])
def test_the_chain_draws_every_board_symbol_without_installed_fonts(no_installed_fonts, lang, bold):
    face = FontSet(config={}).face(lang, 40, bold=bold)
    board_face = boards.DefaultFonts().face(lang, 40, bold)
    for ch in SYMBOLS:
        assert face._pick(ch, False) is not None and board_face._index(ch) is not None, ch    # a real glyph, not a box


def test_text_in_both_languages_draws_without_boxes_or_stand_ins(no_installed_fonts):
    fs = FontSet(config={})
    for lang, text in (("zh-CN", "注意力机制：训练3.5天，BLEU 28.4分（+2.0），约10⁻³，“双向”编码器、掩码语言模型！"),
                       ("en-US", "Attention: 3.5 days, 28.4 BLEU (+2.0), about 10⁻³; it's “bidirectional” — masked LM!")):
        face = fs.face(lang, 40, emphasis=True)
        assert all(c.isspace() or c in "⁻³" or face._pick(c, False) is not None for c in text), lang


def test_chinese_digits_stand_as_tall_as_the_main_fonts_own(no_installed_fonts):
    face = FontSet(config={}).face("zh-CN", 100)

    def height(font) -> int:
        _, top, _, bottom = font.getbbox("8", anchor="ls")
        return bottom - top

    assert face._fonts[1].size > 100 and abs(height(face._fonts[1]) - height(face._fonts[0])) <= 2   # '80.5' next to 'GLUE'


def test_a_font_without_a_bold_file_gets_a_synthetic_bold(no_installed_fonts):
    fs = FontSet(config={})
    regular, bold = fs.face("zh-CN", 48), fs.face("zh-CN", 48, bold=True)
    assert fonts.bold_stroke(bold._fonts[0]) >= 1 and fonts.bold_stroke(regular._fonts[0]) == 0
    assert fonts.bold_stroke(fs.face("en-US", 48, bold=True)._fonts[0]) == 0     # Shantell Sans has a Bold file

    def ink(face) -> int:
        im = Image.new("L", (520, 90), 0)
        face.draw(ImageDraw.Draw(im), (10, 10), "论文宣称内容", 255)
        return int((np.asarray(im) > 127).sum())

    assert ink(bold) > 1.2 * ink(regular)
    assert bold.length("论文宣称内容") == regular.length("论文宣称内容")         # same advances: same line breaks
    assert fonts.bold_stroke(boards.DefaultFonts().face("zh", 48, True).fonts[0]) >= 1


def test_installed_cjk_fonts_come_last_and_draw_only_han_characters(monkeypatch):
    zh = bundled("zh")
    monkeypatch.setattr(fonts, "system_cjk", lambda: (zh,))          # the zh file stands in for an installed CJK font
    fs = FontSet(config={"zh": bundled("en")[0][0]})                 # a zh face without Han characters
    assert fs.fallbacks[-1]["role"] == "system" and fs.fallbacks[-1]["han_only"]
    face = fs.face("zh-CN", 40)
    assert face._pick("中", False) is face._fonts[-1]                # a Han character: the installed font draws it
    assert face._pick("（", False) is None and face.missing("（") == "("   # anything else: never the installed font


def test_a_configured_font_comes_first_and_bad_entries_are_skipped(no_installed_fonts, tmp_path):
    dejavu = bundled("symbols")[0][0]
    plan = resolve({"en": f"{dejavu}#0", "zh": str(tmp_path / "missing.ttf"), "fallback": [str(tmp_path / "nope.ttf")]})
    assert plan.files["en"] == {"text": (dejavu, 0), "bold": (dejavu, 0)}
    assert plan.files["zh"]["text"] == bundled("zh")[0]                       # a missing file: the bundled font
    assert [f["role"] for f in plan.fallbacks] == ["digits", "symbols", "cjk"]   # a fallback that does not load: skipped


def test_a_configured_fallback_does_not_take_the_digits_of_chinese_text(no_installed_fonts):
    hand = bundled("en")[0][0]
    face = FontSet(config={"fallback": hand}).face("zh-CN", 40)
    assert face._chain[1] == (hand, 0) and face._chain[2] == bundled("digits")[0]
    assert face._pick("7", False) is face._fonts[2]


def test_the_macos_preset_is_opt_in(no_installed_fonts):
    assert resolve({}).files == resolve({"preset": "nonsense"}).files       # unknown preset: the bundled fonts
    mac = resolve({"preset": "macos"})
    for lang in ("zh", "en"):
        wanted = [t for t, _ in fonts.preset("macos", lang) if fonts.loads(t)]
        assert mac.files[lang]["text"] == (wanted[0] if wanted else bundled(lang)[0])   # a Mac with them, else bundled


def test_missing_characters_get_ascii_stand_ins_boxes_or_nothing():
    ascii_only = str.isascii
    assert [fonts.missing(c, ascii_only) for c in "−≈×→≤≥√±⁺（"] == ["-", "~", "x", "->", "<=", ">=", "sqrt", "+/-", "+", "("]
    assert fonts.missing("😀", ascii_only) == "" and fonts.missing("✅", ascii_only) == ""   # emoji and pictographs
    assert fonts.missing("∮", ascii_only) == "∮" and fonts.missing("中", ascii_only) == "中"  # no stand-in: a visible box
    assert fonts.missing("\u200b", ascii_only) == ""


def test_a_face_never_drops_a_math_symbol():
    face = Face(fonts.PILLOW, [], 40, "en")             # what a broken install gets: Pillow's built-in font only
    drawn = "".join(t for _, t, _ in face.runs("loss −0.5, ≈1/5.2, √d × 8 → 3.5 ≤ 4 ≥ 2"))
    assert drawn == "loss -0.5, ~1/5.2, sqrtd x 8 -> 3.5 <= 4 >= 2"
    assert face.length("−0.5") == face.length("-0.5")


def test_emoji_are_never_drawn_even_when_a_fallback_font_has_one():
    face = FontSet(config={}).face("en-US", 36)
    assert has_glyph(bundled("symbols")[0], "😀")        # DejaVu Sans has this one, but not 🎉: neither is drawn
    assert face._pick("😀", False) is None and _pretty("ok 😀 🎉 done", face) == "ok done"
    assert boards.default_fonts().face("en", 36)._index("😀") is None


def test_every_face_uses_the_basic_layout_engine():
    face = FontSet(config={}).face("zh-CN", 40, emphasis=True)
    board_face = boards.default_fonts().face("en", 40, True)
    for font in face._fonts + face._strong_fonts + list(face._script) + board_face.fonts:
        assert font.layout_engine == ImageFont.Layout.BASIC, font.getname()


def test_status_reports_the_resolved_files(no_installed_fonts):
    status = fonts.status({})
    assert status["ok"] and status["problems"] == []
    assert status["zh"] == "ChillRoundM.otf#0" and status["en"] == "Shantell_Sans-Normal-Medium.otf#0"
    assert status["fallbacks"][:2] == ["AtkinsonHyperlegibleNext-Regular.ttf#0", "DejaVuSans.ttf#0"]
