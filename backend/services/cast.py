"""
The cast: Prism's own characters, ``kumomo``, whose looks are defined by the reference sheets in
``config/character_images`` (``"1. papi.png"`` → mentor, ``"2. kumo.png"`` → student, ``"3. nezu.png"`` → skeptic).
Replacing the sheets re-skins the manga; there are no themes borrowed from other works.
"""

from __future__ import annotations

import hashlib

import re
from pathlib import Path
from typing import Optional

from engines.base import ImageContent
from services.storyboarder_v2 import Cast, CastMember

CHARACTER_IMAGE_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "character_images"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ROLE_BY_ORDER = {1: "mentor", 2: "student", 3: "skeptic"}

VOICES = {
    "mentor": "patient, explains with one concrete metaphor at a time, always gives the exact number",
    "student": "curious, asks exactly what the reader is wondering, reacts with real emotion",
    "skeptic": "dry and sharp, asks the hard question, names the limitation, is won over only by evidence",
}

KUMOMO_LOOKS = {
    "papi": "a plump brown dog professor with a cream face patch, floppy ears and a small dark nose (no glasses)",
    "kumo": "a fluffy light-blue cloud creature with two small green leaves on its head, tiny paws and a curly tail",
    # as drawn on her sheet (berries and a tart, no clipboard: a clipboard in the art came back with scribbled glyphs)
    "nezu": "a brown hedgehog with soft spikes, dark eye patches and a ringed tail, often holding berries",
}

THEME_STYLE = {
    "kumomo": "Watercolor picture-book manga: soft pastel palette, thick clean ink outlines, gentle paper texture, exactly the art style of the attached reference sheet(s).",
}


def load_reference_sheets(image_dir: Path = CHARACTER_IMAGE_DIR) -> list[tuple[int, str, Path]]:
    """Return [(order, name, path)] for files named like ``"2. kumo.png"`` (unnumbered files sort last)."""
    sheets = []
    if not image_dir.exists():
        return sheets
    for f in sorted(image_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        m = re.match(r"^(\d+)\.\s*(.+)$", f.stem)
        order = int(m.group(1)) if m else 999
        name = (m.group(2) if m else f.stem).strip().lower()
        sheets.append((order, name, f))
    sheets.sort(key=lambda t: t[0])
    return sheets


_SHEET_CACHE: dict[Path, tuple[tuple[int, int], ImageContent]] = {}


def _reference_sheet(path: Path) -> ImageContent:
    """A sheet is ~1.6 MB of base64: read each file once per process (again only when it changes on disk),
    so all jobs share one copy instead of keeping their own."""
    stat = path.stat()
    version = (stat.st_mtime_ns, stat.st_size)
    cached = _SHEET_CACHE.get(path)
    if cached is None or cached[0] != version:
        cached = _SHEET_CACHE[path] = (version, ImageContent.from_file(path))
    return cached[1]


def _clean_sheet(path: Path) -> Optional[Path]:
    """The same sheet without its captions (clean/<name>: the image model copies writing it sees), but only while it
    still belongs to this sheet: clean/<name>.source holds the SHA-256 of the sheet it was made from, so a replaced
    sheet (a re-skinned cast) is used as it is instead of the old cast's clean copy."""
    clean = path.parent / "clean" / path.name
    source = clean.parent / f"{path.name}.source"
    if not clean.is_file() or not source.is_file():
        return None
    return clean if source.read_text(encoding="utf-8").strip() == hashlib.sha256(path.read_bytes()).hexdigest() else None


def build_cast(theme: str = "kumomo") -> tuple[Cast, dict[str, ImageContent]]:
    """Return the kumomo cast plus ``{name: reference sheet}``; any other theme name (an old job's) gets the same cast."""
    members, sheets = [], {}
    for order, name, path in load_reference_sheets():
        role = ROLE_BY_ORDER.get(order, "student")
        look = KUMOMO_LOOKS.get(name, f"the creature shown on {name}'s reference sheet")
        members.append(CastMember(name, role, look, VOICES.get(role, VOICES["student"])))
        sheets[name] = _reference_sheet(_clean_sheet(path) or path)
    if not members:  # no sheets on disk — fall back to the described kumomo trio
        for name, role in (("papi", "mentor"), ("kumo", "student"), ("nezu", "skeptic")):
            members.append(CastMember(name, role, KUMOMO_LOOKS[name], VOICES[role]))
    return Cast(theme="kumomo", style=THEME_STYLE["kumomo"], members=members), sheets
