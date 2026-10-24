"""services.cast: themes, reference sheets from config/character_images and fallbacks."""

import base64

import pytest

import services.cast as cast_module
from services.cast import KUMOMO_LOOKS, THEME_STYLE, build_cast, load_reference_sheets

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_kumomo_loads_three_reference_sheets_in_role_order():
    cast, sheets = build_cast("kumomo")
    assert (cast.theme, cast.style) == ("kumomo", THEME_STYLE["kumomo"])
    assert [(m.name, m.role) for m in cast.members] == [("papi", "mentor"), ("kumo", "student"), ("nezu", "skeptic")]
    assert [m.look for m in cast.members] == [KUMOMO_LOOKS[n] for n in ("papi", "kumo", "nezu")]
    assert list(sheets) == ["papi", "kumo", "nezu"]
    for sheet in sheets.values():
        assert sheet.is_base64 and sheet.mime_type == "image/png"
        assert base64.b64decode(sheet.data[:12]).startswith(PNG_SIGNATURE)


@pytest.mark.parametrize("theme", ["steampunk", "", None, "chiikawa", "ghibli"])   # the last two: themes of older versions
def test_unknown_theme_falls_back_to_kumomo(theme):
    cast, sheets = build_cast(theme)
    assert cast.theme == "kumomo" and cast.names() == ["papi", "kumo", "nezu"]
    assert list(sheets) == ["papi", "kumo", "nezu"]


def test_load_reference_sheets_orders_by_number_and_skips_non_images(tmp_path):
    for name in ["2. kumo.png", "1. Papi.PNG", "extra.webp", "notes.txt"]:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "3. folder.png").mkdir()
    assert [(order, name) for order, name, _ in load_reference_sheets(tmp_path)] == [(1, "papi"), (2, "kumo"), (999, "extra")]
    assert load_reference_sheets(tmp_path / "missing") == []


def test_kumomo_without_sheets_on_disk_uses_the_described_trio(monkeypatch):
    monkeypatch.setattr(cast_module, "load_reference_sheets", lambda *args, **kwargs: [])
    cast, sheets = build_cast("kumomo")
    assert sheets == {}
    assert [(m.name, m.role, m.look) for m in cast.members] == [
        (n, r, KUMOMO_LOOKS[n]) for n, r in (("papi", "mentor"), ("kumo", "student"), ("nezu", "skeptic"))]


def test_kumomo_prefers_the_caption_free_copy_of_a_sheet(tmp_path, monkeypatch):
    import hashlib
    (tmp_path / "clean").mkdir()
    (tmp_path / "1. papi.png").write_bytes(PNG_SIGNATURE + b"captioned")
    (tmp_path / "2. kumo.png").write_bytes(PNG_SIGNATURE + b"captioned")
    (tmp_path / "clean" / "1. papi.png").write_bytes(PNG_SIGNATURE + b"clean")
    (tmp_path / "clean" / "1. papi.png.source").write_text(hashlib.sha256(PNG_SIGNATURE + b"captioned").hexdigest())
    monkeypatch.setattr(cast_module, "load_reference_sheets", lambda *a, **k: load_reference_sheets(tmp_path))
    _, sheets = build_cast("kumomo")
    assert base64.b64decode(sheets["papi"].data).endswith(b"clean")       # captions removed: nothing to copy into the art
    assert base64.b64decode(sheets["kumo"].data).endswith(b"captioned")   # no clean copy: the sheet itself
    # a re-skinned cast: papi's sheet was replaced, so the old cast's clean copy no longer belongs to it
    (tmp_path / "1. papi.png").write_bytes(PNG_SIGNATURE + b"a new papi")
    cast_module._SHEET_CACHE.clear()
    _, sheets = build_cast("kumomo")
    assert base64.b64decode(sheets["papi"].data).endswith(b"a new papi")


def test_the_repo_ships_caption_free_sheets_and_nezu_is_described_as_drawn():
    clean = cast_module.CHARACTER_IMAGE_DIR / "clean"
    assert sorted(p.name for p in clean.iterdir() if p.suffix == ".png") == ["1. papi.png", "2. kumo.png", "3. nezu.png"]
    for sheet in clean.glob("*.png"):   # each clean copy names the sheet it was made from, so it is used for that sheet only
        assert cast_module._clean_sheet(cast_module.CHARACTER_IMAGE_DIR / sheet.name) == sheet
    assert "clipboard" not in KUMOMO_LOOKS["nezu"] and "berries" in KUMOMO_LOOKS["nezu"]
