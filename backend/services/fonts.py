"""
Fonts: the font files that letter the panels and draw the boards, the same on every machine.

The fonts ship with Prism in backend/assets/fonts; fonts.json names the file of each role and pins its source and
SHA-256, the licenses are in assets/fonts/LICENSES and THIRD_PARTY_NOTICES.md. resolve() returns
- per language (zh, en) the main face, text and bold: `lettering.fonts.<lang>` from config when it loads (an explicit
  override, logged), else the bundled file (zh ChillRound M, en Shantell Sans). Never an installed font, so every
  OS draws the same letters; Yuanti SC + Chalkboard SE, the macOS look before the fonts were bundled, is the opt-in
  `lettering.fonts.preset: macos`;
- the fallback chain for the glyphs a main face lacks: `lettering.fonts.fallback` from config, the bundled Atkinson
  Hyperlegible Next (Chinese text takes its digits from it: 1 / l / I and 0 / O stay distinct), DejaVu Sans (symbols,
  Greek, super/subscripts; √ and × always, see SYMBOLS_FIRST), the zh face (Chinese words in English text), then
  installed CJK fonts, for rare Han characters only (macOS PingFang / Hiragino, Windows Microsoft YaHei / SimHei /
  DengXian, Linux Noto CJK / WenQuanYi).
A character that no font of a chain has is drawn as missing() decides: an ASCII stand-in for a symbol ('−' -> '-':
dropping it would change a number), nothing for an emoji (never drawn, even when a fallback font has one), else a box.
A font without a bold file gets a synthetic bold, as browsers synthesize one (load(..., embolden=True)).
Every face is loaded with Pillow's BASIC layout engine: Raqm, where installed, measures text differently and would move
the line breaks. Pure Pillow, no network.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import threading
import unicodedata
from functools import lru_cache
from typing import Any, Callable, NamedTuple, Optional

from PIL import Image, ImageDraw, ImageFont

FontSpec = tuple[str, int]          # (font file, face index inside a .ttc); ("", 0) is Pillow's built-in font
PILLOW: FontSpec = ("", 0)
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")
LANGS = ("zh", "en")
PRESETS = ("macos",)

# FreeType faces must not be used from two threads at once; jobs letter panels (and draw their boards) in worker threads
FT_LOCK = threading.RLock()

# drawn by the symbols font (DejaVu Sans) even when the main font has them: hand fonts draw them ambiguously
# (Shantell Sans' root sign reads as a 'V', a CJK font's times sign as a small 'x')
SYMBOLS_FIRST = frozenset("√×")
# ASCII stand-ins for symbols that no font of a chain can draw: dropping one would change a number ("−0.5" -> "0.5")
STAND_INS = {"−": "-", "–": "-", "—": "-", "‐": "-", "≈": "~", "∼": "~", "×": "x", "·": ".", "⋅": ".", "÷": "/",
             "→": "->", "←": "<-", "↔": "<->", "⇒": "=>", "≤": "<=", "≥": ">=", "≠": "!=", "±": "+/-", "√": "sqrt",
             "∞": "inf", "′": "'", "″": "''"}

_LOGGED: set[str] = set()


def _log_once(message: str) -> None:
    """Print a message the first time only: FontSets are built per job and /health may ask often."""
    if message not in _LOGGED:
        _LOGGED.add(message)
        print(f"[fonts] {message}")


# ----------------------------------------------------------------------------- bundled files

@lru_cache(maxsize=1)
def manifest() -> dict:
    """assets/fonts/fonts.json: {"roles": {role: {"text": [file, index], "bold": [...]}}, "files": {file: {sha256, ...}}}."""
    try:
        with open(os.path.join(FONT_DIR, "fonts.json"), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def bundled(role: str) -> Optional[tuple[FontSpec, FontSpec]]:
    """(text, bold) of a bundled role: 'zh', 'en', 'digits' or 'symbols'; None when the manifest lacks it."""
    entry = (manifest().get("roles") or {}).get(role)
    if not isinstance(entry, dict):
        return None
    text, bold = _spec(entry.get("text"), FONT_DIR), _spec(entry.get("bold"), FONT_DIR)
    return (text, bold or text) if text else None


@lru_cache(maxsize=1)
def problems() -> tuple[str, ...]:
    """What is wrong with the bundled files: missing, or not the bytes fonts.json pins (a Git LFS pointer, a damaged
    copy). Checked once per process; each problem is logged."""
    files = manifest().get("files")
    files = files if isinstance(files, dict) else {}
    out = [] if files else ["assets/fonts/fonts.json is missing or unreadable"]
    for name, info in files.items():
        try:
            with open(os.path.join(FONT_DIR, name), "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            out.append(f"assets/fonts/{name} is missing")
            continue
        if digest != str(info.get("sha256") if isinstance(info, dict) else "").lower():
            out.append(f"assets/fonts/{name} is not the file fonts.json pins (its SHA-256 differs)")
    for problem in out:
        print(f"[fonts] WARNING: {problem}")
    return tuple(out)


# ----------------------------------------------------------------------------- loading and glyphs

@lru_cache(maxsize=1024)
def load(spec: FontSpec, size: int, embolden: bool = False):
    """The font at a pixel size, laid out by Pillow's BASIC engine (Pillow's built-in font, PILLOW, is BASIC too).
    embolden: a synthetic bold for a font without a bold file; draw it with stroke_width=bold_stroke(font)."""
    path, index = spec
    with FT_LOCK:
        if not path:
            font = ImageFont.load_default(size)
        else:
            font = ImageFont.truetype(path, size, index=index, layout_engine=ImageFont.Layout.BASIC)
    if embolden:
        font.bold_stroke = max(1, round(size / 48))   # FreeType's own emboldening: 1/24 em in all, half on each side
    return font


def bold_stroke(font) -> int:
    """The outline width that draws a synthetic bold (load(..., embolden=True)); 0 for every other font."""
    return getattr(font, "bold_stroke", 0)


@lru_cache(maxsize=64)
def digit_scale(main: FontSpec, digits: FontSpec) -> float:
    """How much larger to load the digits font so that its figures stand as tall as the main font's own: Atkinson
    Hyperlegible Next's are about a tenth shorter than the Chinese font's ('80.5' looked small after 'GLUE')."""
    def height(spec: FontSpec) -> int:
        with FT_LOCK:
            _, top, _, bottom = load(spec, 100).getbbox("8", anchor="ls")
        return bottom - top

    main_h, digits_h = height(main), height(digits)
    return min(1.25, max(0.8, main_h / digits_h)) if main_h > 0 and digits_h > 0 else 1.0


@lru_cache(maxsize=256)
def loads(spec: FontSpec) -> bool:
    """True when the file exists and FreeType opens that face of it."""
    if not spec[0] or not os.path.isfile(spec[0]):
        return False
    try:
        load(spec, 16)
        return True
    except (OSError, ValueError):
        return False


def _ink(spec: FontSpec, ch: str) -> bytes:
    im = Image.new("L", (80, 80), 0)
    with FT_LOCK:
        ImageDraw.Draw(im).text((16, 16), ch, font=load(spec, 32), fill=255)
    return im.tobytes()


@lru_cache(maxsize=64)
def _box(spec: FontSpec) -> bytes:
    return _ink(spec, "\ufdd0")         # a noncharacter: no font maps it, so this is the font's missing-glyph box


@lru_cache(maxsize=None)
def has_glyph(spec: FontSpec, ch: str) -> bool:
    """True when the font has a real glyph for ch: a missing one draws the same box as a noncharacter."""
    if ch.isspace():
        return True
    return _ink(spec, ch) != _box(spec)


def is_han(ch: str) -> bool:
    """A Chinese character: CJK unified ideographs (extensions included) and compatibility ideographs."""
    o = ord(ch)
    return 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x3134F


def is_emoji(ch: str) -> bool:
    """A pictograph (emoji, emoticons, flags, skin tones): never drawn, even when a fallback font has one."""
    return ord(ch) >= 0x1F000 and unicodedata.category(ch)[0] == "S"


def missing(ch: str, drawable: Callable[[str], bool]) -> str:
    """What to draw for a character that no font of a chain has (drawable(c): one of them has c): an ASCII stand-in
    (STAND_INS, else its compatibility form: '⁺' -> '+', '（' -> '('), nothing for an emoji, an invisible character or a
    pictograph such as '✅', else the character itself, whose box at least shows that a font is missing (a letter, or a
    math symbol without a stand-in: never dropped). Logged once per character."""
    if is_emoji(ch):
        _log_once(f"emoji {ch!r} (U+{ord(ch):04X}) left out")
        return ""
    sub = STAND_INS.get(ch) or unicodedata.normalize("NFKC", ch)
    if sub != ch and sub.strip() and all(drawable(c) for c in sub):
        _log_once(f"WARNING: no font has {ch!r} (U+{ord(ch):04X}); drawn as {sub!r}")
        return sub
    category = unicodedata.category(ch)
    if category[0] in "CM":
        return ""
    if category == "So":
        _log_once(f"WARNING: no font has {ch!r} (U+{ord(ch):04X}); left out")
        return ""
    _log_once(f"WARNING: no font has {ch!r} (U+{ord(ch):04X}); drawn as a box")
    return ch


# ----------------------------------------------------------------------------- resolution

def _spec(value: Any, base: str = "") -> Optional[FontSpec]:
    """'path', 'path#2', [path, 2] or {path, index} -> (path, index); a relative path is taken from base when given."""
    try:
        if isinstance(value, str) and value.strip():
            path, _, index = value.strip().partition("#")
            spec = (path.strip(), int(index or 0))
        elif isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], str):
            spec = (value[0], int(value[1]))
        elif isinstance(value, dict) and isinstance(value.get("path"), str):
            spec = (value["path"], int(value.get("index") or 0))
        else:
            return None
    except (TypeError, ValueError):
        return None
    path = os.path.expanduser(spec[0])
    return (os.path.join(base, path) if base and path and not os.path.isabs(path) else path), spec[1]


def _configured(value: Any) -> list[tuple[FontSpec, FontSpec]]:
    """Configured candidates of one role: {text, bold}, a single font spec, or a list of those."""
    single = isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[1], int)
    items = value if isinstance(value, list) and not single else [value]
    out = []
    for item in items:
        if isinstance(item, dict) and ("text" in item or "bold" in item):
            text = _spec(item.get("text")) or _spec(item.get("bold"))
            bold = _spec(item.get("bold")) or text
        else:
            text = bold = _spec(item)
        if text:
            out.append((text, bold))
    return out


def configured() -> dict:
    """`lettering.fonts` (or `manga_settings.lettering.fonts`) from config/api_config.yaml, if present."""
    try:
        from config_loader import get_config
        raw = get_config().get_raw_config() or {}
    except Exception:  # noqa: BLE001 - fonts must resolve even without a usable config
        return {}
    section = raw.get("lettering")
    if not isinstance(section, dict):
        manga = raw.get("manga_settings")
        section = manga.get("lettering") if isinstance(manga, dict) else None
    fonts = section.get("fonts") if isinstance(section, dict) else None
    return fonts if isinstance(fonts, dict) else {}


def _mac_asset(name: str) -> str:
    """macOS downloadable fonts (Yuanti; PingFang on recent versions) live in hashed MobileAsset folders."""
    hits = sorted(glob.glob(f"/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/{name}"))
    return hits[0] if hits else ""


def preset(name: str, lang: str) -> list[tuple[FontSpec, FontSpec]]:
    """(text, bold) candidates of a named look. 'macos': Yuanti SC (a downloadable font: Font Book fetches it) and
    Chalkboard SE, the macOS fonts of the manga before the fonts were bundled; empty where they are not installed."""
    if name != "macos":
        return []
    if lang == "zh":
        yuanti = _mac_asset("Yuanti.ttc")                         # 0 Yuanti SC, 2 Yuanti SC Bold
        return [((yuanti, 0), (yuanti, 2))] if yuanti else []
    chalk = "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc"  # 1 regular, 2 bold
    return [((chalk, 1), (chalk, 2))]


@lru_cache(maxsize=1)
def system_cjk() -> tuple[tuple[FontSpec, FontSpec], ...]:
    """Installed CJK fonts (text, bold), best first; they draw only the rare Han characters the zh face lacks."""
    pingfang = "/System/Library/Fonts/PingFang.ttc"
    pingfang = pingfang if os.path.isfile(pingfang) else _mac_asset("PingFang.ttc")
    hiragino = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    found = ([((pingfang, 3), (pingfang, 7))] if pingfang else []) + [((hiragino, 0), (hiragino, 2))]   # SC faces
    windows = os.path.join(os.environ.get("WINDIR") or "C:\\Windows", "Fonts")
    found += [((os.path.join(windows, text), 0), (os.path.join(windows, bold), 0))
              for text, bold in (("msyh.ttc", "msyhbd.ttc"), ("simhei.ttf", "simhei.ttf"), ("Deng.ttf", "Dengb.ttf"))]
    roots = ("/usr/share/fonts", "/usr/local/share/fonts", "~/.local/share/fonts", "~/.fonts")
    for text, index, bold in (("NotoSansCJK-Regular.ttc", 2, "NotoSansCJK-Bold.ttc"),      # 2: Noto Sans CJK SC
                              ("NotoSansCJKsc-Regular.otf", 0, "NotoSansCJKsc-Bold.otf"),
                              ("wqy-microhei.ttc", 0, "wqy-microhei.ttc"), ("wqy-zenhei.ttc", 0, "wqy-zenhei.ttc")):
        hits = [h for root in roots for h in sorted(glob.glob(os.path.join(os.path.expanduser(root), "**", text), recursive=True))]
        if hits:
            found.append(((hits[0], index), (os.path.join(os.path.dirname(hits[0]), bold), index)))
    return tuple((text, bold if loads(bold) else text) for text, bold in found if loads(text))


class Plan(NamedTuple):
    """Resolved fonts: per language {"text", "bold"} specs, and the fallback chain in order, each fallback a dict with
    "text", "bold", "role" ('config', 'digits', 'symbols', 'cjk' or 'system') and "han_only" (installed CJK fonts)."""
    files: dict
    fallbacks: list

    def chain(self, lang: str) -> list[dict]:
        """The fallbacks of a language's face, without its own main font (the zh face is a fallback of English text)."""
        main = self.files[lang]["text"]
        return [f for f in self.fallbacks if f["text"] != main]


def resolve(config: Optional[dict] = None) -> Plan:
    """The fonts for a `lettering.fonts` mapping ({} or None: the bundled set), as the module docstring describes."""
    config = config if isinstance(config, dict) else {}
    problems()
    name = str(config.get("preset") or "").strip().lower()
    if name and name not in PRESETS:
        _log_once(f"WARNING: unknown lettering.fonts.preset {name!r} (known: {', '.join(PRESETS)}); using the bundled fonts")
    files: dict[str, dict[str, FontSpec]] = {}
    for lang in LANGS:
        own = bundled(lang)
        candidates = ([(text, bold, "configured") for text, bold in _configured(config.get(lang))]
                      + [(text, bold, f"preset {name}") for text, bold in preset(name, lang)]
                      + ([(own[0], own[1], "bundled")] if own else []))
        for text, bold, source in candidates:
            if loads(text):
                files[lang] = {"text": text, "bold": bold if loads(bold) else text}
                if source != "bundled":
                    _log_once(f"{lang} font: {text[0]}#{text[1]} ({source})")
                break
            _log_once(f"WARNING: the {source} {lang} font {text[0] or '(none)'} does not load; trying the next one")
        else:
            files[lang] = {"text": PILLOW, "bold": PILLOW}
            _log_once(f"WARNING: no {lang} font loads (is backend/assets/fonts complete?); Pillow's built-in font has no CJK glyphs")
    chain = [(text, bold, "config") for text, bold in _configured(config.get("fallback"))]
    chain += [(found[0], found[1], role) for role in ("digits", "symbols") for found in [bundled(role)] if found]
    chain.append((files["zh"]["text"], files["zh"]["bold"], "cjk"))      # Chinese words in English text
    chain += [(text, bold, "system") for text, bold in system_cjk()]
    seen, fallbacks = set(), []
    for text, bold, role in chain:
        key = (os.path.basename(text[0]), text[1])         # the same font installed twice adds nothing
        if not loads(text):
            if role == "config":
                _log_once(f"WARNING: the configured fallback font {text[0] or '(none)'} does not load; skipped")
        elif key not in seen:
            seen.add(key)
            fallbacks.append({"text": text, "bold": bold if loads(bold) else text, "role": role, "han_only": role == "system"})
    return Plan(files, fallbacks)


def status(config: Optional[dict] = None) -> dict:
    """For /health: the file of each main face and fallback, and whether the bundled set is complete and intact.
    config: a `lettering.fonts` mapping; None reads the app config."""
    plan = resolve(configured() if config is None else config)

    def name(spec: FontSpec) -> str:
        return f"{os.path.basename(spec[0])}#{spec[1]}" if spec[0] else "pillow-default"

    return {**{lang: name(plan.files[lang]["text"]) for lang in LANGS}, "fallbacks": [name(f["text"]) for f in plan.fallbacks],
            "problems": list(problems()), "ok": not problems() and all(plan.files[lang]["text"] != PILLOW for lang in LANGS)}
