"""Regression tests for the 2026-10-22 review fixes S4, S5, S10, S11, S12 (storyboarder_v2 / panel_renderer).

Plain tests lock in behaviour that the fixes got right. Tests marked ``xfail(strict=True)`` describe the correct
behaviour for scenarios that are still broken; once the source is fixed they XPASS, which strict mode reports as a
failure: delete the marker then and keep the test as a regression test.
"""

import base64
import re

import pytest

from engines.base import ImageContent
from services.paper_extractor import PaperText
from services.panel_renderer import PanelRenderer, _ordinal, _reference_block, build_single_prompt
from services.storyboarder_v2 import (
    BUDGETS,
    Cast,
    CastMember,
    StoryboarderV2,
    StoryboardV2,
    _match_cast,
    _normalize_chapter,
    text_len,
)

from .fakes import (
    LOOKS,
    TINY_PNG_B64,
    FakeEngine,
    FakeImageEngine,
    PaperModel,
    b64,
    image_bytes,
    make_cast,
    make_panel,
    make_sheets,
    storyboard_payload,
)


def storyboard(panels, language="en-US") -> StoryboardV2:
    return StoryboardV2(title="T", summary="S", theme="kumomo", language=language, chapters=[], panels=panels)


def raw(n, chapter, title, *, speaker="papi", characters=("papi",), **extra) -> dict:
    """One panel of a storyboard model reply."""
    panel = {"n": n, "chapter": chapter, "title": title, "characters": list(characters), "scene": "s", "visual": "v",
             "dialogue": [{"speaker": speaker, "text": f"{title} line."}]}
    panel.update(extra)
    return panel


async def write(panels: list[dict], cast: Cast = None) -> StoryboardV2:
    engine = FakeEngine([{"title": "T", "summary": "S", "panels": panels}])
    return await StoryboarderV2(engine).write({}, cast or make_cast(), (1, 40), "en-US")


def storyboard_router(job_env, edit):
    """Text-engine router for job tests: the default PaperModel, with the storyboard reply passed through ``edit``."""
    base = job_env.paper

    def router(prompt, messages, config):
        if messages[-1].content.startswith("Write the manga script"):
            return edit()
        return base(prompt, messages, config)

    return router


REF_SENTENCE = re.compile(r"The (\S+) reference image is ([^:.]+)")


# ============================================================================= S4 reading order / chapters

async def test_s4_numbering_restarted_per_chapter_keeps_reply_order():
    sb = await write([raw(1, "hook", "H1"), raw(1, "problem", "P1"), raw(2, "problem", "P2"),
                      raw(1, "idea", "I1"), raw(2, "idea", "I2"), raw(3, "idea", "I3")])
    assert [(p.n, p.title, p.chapter) for p in sb.panels] == [
        (1, "H1", "hook"), (2, "P1", "problem"), (3, "P2", "problem"), (4, "I1", "idea"), (5, "I2", "idea"), (6, "I3", "idea")]
    assert [(c.id, c.panel_numbers) for c in sb.chapters] == [("hook", [1]), ("problem", [2, 3]), ("idea", [4, 5, 6])]


@pytest.mark.parametrize("numbers", [
    ["1.1", "2.1", "2.2", "3.1"],   # chapter.panel strings -> unparsable -> reply index
    [0, 1, 2, 3],                   # zero-based (0 falls back to the reply index 1 -> duplicate)
    [1, "2a", 2, 3],
    [1, 1, 1, 1],
    [None, None, 1, 2],
])
async def test_s4_duplicate_or_unparsable_numbers_keep_reply_order(numbers):
    sb = await write([raw(n, "how", t) for n, t in zip(numbers, "ABCD")])
    assert [(p.n, p.title) for p in sb.panels] == [(1, "A"), (2, "B"), (3, "C"), (4, "D")]


async def test_s4_unique_numbering_is_still_trusted():
    sb = await write([raw(3, "idea", "C"), raw(1, "hook", "A"), raw(2, "problem", "B")])
    assert [(p.n, p.title) for p in sb.panels] == [(1, "A"), (2, "B"), (3, "C")]


@pytest.mark.parametrize("label, expected", [
    ("Introduction", "hook"), ("motivation", "hook"), ("Background", "problem"), ("Key Idea", "idea"), ("solution", "idea"),
    ("Methods", "how"), ("How it works", "how"), ("Results", "evidence"), ("experiments", "evidence"),
    ("Limitations", "limits"), ("caveats", "limits"), ("Conclusion", "takeaway"), ("summary", "takeaway"),
    # the CHAPTER_PLAN titles the prompt shows next to the ids
    ("The old problem", "problem"), ("The key idea", "idea"), ("Does it work? The evidence", "evidence"),
    ("Limits and open questions", "limits"), ("Takeaway", "takeaway"),
    ("Chapter 1: Hook", "hook"), (" EVIDENCE ", "evidence"),
])
def test_s4_chapter_aliases_and_plan_titles_map_to_plan_ids(label, expected):
    assert _normalize_chapter(label, previous="limits") == expected


async def test_s4_unknown_chapter_ids_inherit_the_previous_chapter_instead_of_how():
    labels = ["prologue", "problem", "idea", "how", "evidence", "bonus round", "limits", "conclusion"]
    sb = await write([raw(i, c, c) for i, c in enumerate(labels, start=1)])
    assert [p.chapter for p in sb.panels] == ["hook", "problem", "idea", "how", "evidence", "evidence", "limits", "takeaway"]
    assert [(c.id, c.panel_numbers) for c in sb.chapters] == [
        ("hook", [1]), ("problem", [2]), ("idea", [3]), ("how", [4]), ("evidence", [5, 6]), ("limits", [7]), ("takeaway", [8])]


async def test_s4_job_keeps_the_reading_order_of_a_restarted_numbering(job_env):
    def edit():
        payload = storyboard_payload(4)
        for p, (n, chapter) in zip(payload["panels"], [(1, "hook"), (1, "problem"), (2, "problem"), (1, "conclusion")]):
            p["n"], p["chapter"] = n, chapter
        return payload

    job_env.text.router = storyboard_router(job_env, edit)
    state = await job_env.run(qa=False)

    assert state["status"] == "completed"
    assert [(p["n"], p["title"], p["chapter"]) for p in state["panels"]] == [
        (1, "Panel 1 title", "hook"), (2, "Panel 2 title", "problem"), (3, "Panel 3 title", "problem"), (4, "Panel 4 title", "takeaway")]
    assert [p["dialogue"][0]["text"] for p in state["panels"]] == [f"Line for panel {i} please." for i in range(1, 5)]
    assert [(c["id"], c["panel_numbers"]) for c in state["chapters"]] == [("hook", [1]), ("problem", [2, 3]), ("takeaway", [4])]


# ============================================================================= S5 speakers / narrator / emotions

@pytest.mark.parametrize("speaker", ["Papi (smiling)", "Professor Papi", "PAPI:", "Dr. Papi", "papi-sensei"])
async def test_s5_speaker_variants_resolve_to_the_named_cast_member(speaker):
    # the old fallback would have picked characters[0] == "kumo"
    sb = await write([raw(1, "hook", "Attention", speaker=speaker, characters=["kumo", "Nezu (arms crossed)", "Papi (dog professor)"])])
    p = sb.panels[0]
    assert [b.speaker for b in p.dialogue] == ["papi"]
    assert p.characters == ["kumo", "nezu", "papi"]


def test_s5_cast_names_match_whole_words_only():
    valid = {"papi", "kumo", "nezu"}
    assert _match_cast("papillon", valid) is None and _match_cast("kumomo", valid) is None
    assert _match_cast(" Nezu ", valid) == "nezu" and _match_cast("", valid) is None


@pytest.mark.parametrize("speaker", ["Narrator", "NARRATOR", " narration ", "Caption"])
async def test_s5_narrator_bubbles_become_narration(speaker):
    sb = await write([{"n": 1, "chapter": "hook", "characters": ["kumo", "papi"], "dialogue": [
        {"speaker": speaker, "text": "Meanwhile, in 2017..."}, {"speaker": "papi", "text": "Attention!"}]}])
    p = sb.panels[0]
    assert p.narration == "Meanwhile, in 2017..."
    assert [(b.speaker, b.text) for b in p.dialogue] == [("papi", "Attention!")]
    assert p.characters == ["kumo", "papi"]


async def test_s5_narrator_only_panel_has_no_speech_bubbles_in_the_prompt():
    sb = await write([{"n": 1, "chapter": "hook", "characters": ["kumo"], "dialogue": [{"speaker": "narrator", "text": "Google, 2017."}]}])
    p = sb.panels[0]
    assert (p.dialogue, p.narration, p.characters) == ([], "Google, 2017.", ["kumo"])
    prompt = build_single_prompt(p, make_cast(), "en-US", ["kumo"])
    assert "No speech bubbles in this panel." in prompt and 'Narration box along the top edge: "Google, 2017."' in prompt
    assert "says:" not in prompt


async def test_s5_emotions_of_absent_characters_never_reach_the_prompt():
    sb = await write([{"n": 1, "chapter": "hook", "characters": ["papi"],
                       "dialogue": [{"speaker": "papi", "text": "Hi."}, {"speaker": "kumo", "text": "Hello!"}],
                       "emotions": {"Papi": "calm", "kumo": "happy", "nezu": "skeptical", "dragon": "angry"}}])
    p = sb.panels[0]
    assert p.characters == ["papi", "kumo"]
    assert p.emotions == {"papi": "calm", "kumo": "happy"}
    prompt = build_single_prompt(p, make_cast(), "en-US", ["papi", "kumo"])
    assert "Emotions: papi is calm, kumo is happy." in prompt
    assert "nezu is" not in prompt and "dragon" not in prompt


@pytest.mark.parametrize("speaker", ["Narrator (V.O.)", "The narrator", "Narration box", "Caption box", "narrator:"])
async def test_s5_narrator_variants_become_narration(speaker):
    sb = await write([{"n": 1, "chapter": "hook", "characters": ["kumo", "papi"], "dialogue": [
        {"speaker": speaker, "text": "Meanwhile, in 2017..."}, {"speaker": "papi", "text": "Attention!"}]}])
    p = sb.panels[0]
    assert p.narration == "Meanwhile, in 2017..."
    assert [(b.speaker, b.text) for b in p.dialogue] == [("papi", "Attention!")]


async def test_s5_narrator_line_does_not_count_against_the_two_bubble_cap():
    sb = await write([{"n": 1, "chapter": "hook", "characters": ["papi", "kumo"], "dialogue": [
        {"speaker": "narrator", "text": "Meanwhile, in 2017..."},
        {"speaker": "papi", "text": "Attention!"},
        {"speaker": "kumo", "text": "Is all you need?"}]}])
    p = sb.panels[0]
    assert p.narration == "Meanwhile, in 2017..."
    assert [(b.speaker, b.text) for b in p.dialogue] == [("papi", "Attention!"), ("kumo", "Is all you need?")]


async def test_s5_narrator_text_is_not_dropped_when_narration_exists():
    sb = await write([
        {"n": 1, "chapter": "hook", "characters": ["papi"], "narration": "Google, 2017.", "dialogue": [
            {"speaker": "narrator", "text": "Training took 3.5 days on 8 GPUs."}, {"speaker": "papi", "text": "Hi."}]},
        {"n": 2, "chapter": "hook", "characters": ["kumo"], "dialogue": [
            {"speaker": "narrator", "text": "Part one."}, {"speaker": "caption", "text": "Part two."}]},
    ])
    first, second = sb.panels
    assert "Google, 2017." in first.narration and "3.5 days on 8 GPUs" in first.narration
    assert "Part one." in second.narration and "Part two." in second.narration


def test_s5_speaker_naming_two_members_resolves_to_the_leading_name():
    # write() passes set(cast.names()); explicit orders make the dependence on iteration order deterministic here
    for order in (["papi", "kumo", "nezu"], ["kumo", "papi", "nezu"], ["nezu", "kumo", "papi"]):
        assert _match_cast("kumo (to papi)", order) == "kumo"
        assert _match_cast("Papi, pointing at Kumo", order) == "papi"


async def test_s5_emotion_keys_written_like_speakers_are_kept():
    sb = await write([{"n": 1, "chapter": "hook", "characters": ["Papi (smiling)"],
                       "dialogue": [{"speaker": "Papi (smiling)", "text": "Hi."}], "emotions": {"Papi (smiling)": "calm"}}])
    assert sb.panels[0].emotions == {"papi": "calm"}


# ============================================================================= S10 fact-check sees the whole body

FILLER = "Scaled dot-product attention divides each score by the square root of the key dimension."
TAIL = "TAIL-MARKER: the big model reached 41.0 BLEU on WMT 2014 English-to-French."


def long_body(lines: int = 1500) -> str:
    return "\n".join(["HEAD-MARKER"] + [FILLER] * lines + [TAIL])


async def test_s10_factcheck_prompt_contains_the_whole_body():
    body = long_body()
    assert len(body) > 120_000
    engine = FakeEngine([{"problems": []}])
    sb = storyboard([make_panel(1, narration="The big model reached 41.0 BLEU.")])
    assert await StoryboarderV2(engine).factcheck(sb, PaperText("T", body, [])) == 0
    messages, _config = engine.calls[0]
    assert messages[0].content.endswith(body)   # the whole body, sent first (the provider caches it for the job's later calls)


async def test_s10_brief_and_factcheck_get_the_same_full_body():
    body = long_body()
    model = PaperModel(num_panels=3)
    engine = FakeEngine(router=model)
    await StoryboarderV2(engine).build(PaperText("T", body, []), make_cast(), "en-US", (3, 6))
    assert model.seen == ["brief", "briefcheck", "storyboard", "factcheck"]
    brief = next(p for p in engine.prompts if "write a compact, faithful BRIEF" in p)
    factcheck = next(p for p in engine.prompts if "meticulous fact-checker" in p)
    check = next(p for p in engine.prompts if "You verify the BRIEF" in p)
    assert brief.endswith(body)
    # the checks send the full body first (paper-first prompts share the provider's prefix cache)
    assert body in factcheck and body in check and factcheck.startswith("You work on one research paper")


async def test_s10_job_factcheck_sees_the_end_of_a_long_paper(job_env):
    text = long_body(lines=1100)
    assert len(text) > 90_000
    state = await job_env.run(text=text, qa=False)
    assert state["status"] == "completed"
    paper_txt = (job_env.output_dir / state["job_id"] / "paper.txt").read_text(encoding="utf-8")
    assert paper_txt.endswith(TAIL) and len(paper_txt) > 90_000
    factcheck = next(m for m, _ in job_env.text.calls if "meticulous fact-checker" in m[-1].content)
    assert factcheck[0].content.endswith(paper_txt)   # the whole paper, sent first


# ============================================================================= S11 label budgets / CJK width

def test_s11_every_language_has_a_label_budget():
    assert {lang: b["label"] for lang, b in BUDGETS.items()} == {"en-US": 3, "zh-CN": 6, "ja-JP": 8}


@pytest.mark.parametrize("language, ok, too_long", [
    ("en-US", "Scaled Attention", "Scaled Dot Product Attention"),
    ("zh-CN", "缩放注意力", "缩放点积注意力"),
    ("ja-JP", "スケール内積注意", "スケールド内積注意機構"),
])
def test_s11_violations_cover_labels(language, ok, too_long):
    sb = storyboard([make_panel(1, dialogue=[("papi", "好")], labels=[ok, too_long]),
                     make_panel(2, dialogue=[("papi", "好")], labels=["d_k", "a\\b"])], language)
    assert {v["id"]: (v["kind"], v["text"]) for v in StoryboarderV2(FakeEngine()).violations(sb, language)} == {
        "1.l1": ("label", too_long), "2.l0": ("label", "d_k"), "2.l1": ("label", "a\\b")}


async def test_s11_fit_rewrites_labels_and_strips_leftover_forbidden_characters():
    fast = FakeEngine([{"items": [{"id": "1.l0", "text": "Scaled Attention"}]}, {"items": []}])
    sb = storyboard([make_panel(1, dialogue=[("papi", "Short.")], labels=["Scaled Dot Product Attention", "d_k"])])
    assert await StoryboarderV2(FakeEngine(), fast_engine=fast).fit(sb, "en-US") == 1
    assert sb.panels[0].labels == ["Scaled Attention", "d k"]  # "d_k" was never rewritten -> "_" stripped, not truncated
    assert len(fast.calls) == 2
    first = fast.prompts[0]
    assert '"1.l0"' in first and '"1.l1"' in first and '"kind": "label"' in first and "label on a prop ≤ 3 words" in first


async def test_s11_chinese_fit_uses_the_character_label_budget():
    fast = FakeEngine([{"items": [{"id": "1.l0", "text": "注意力"}]}])
    sb = storyboard([make_panel(1, dialogue=[("papi", "好")], labels=["缩放点积注意力", "BLEU 28.4"])], "zh-CN")
    assert await StoryboarderV2(FakeEngine(), fast_engine=fast).fit(sb, "zh-CN") == 1
    assert sb.panels[0].labels == ["注意力", "BLEU 28.4"]
    assert "label on a prop ≤ 6 characters" in fast.prompts[0] and "BLEU 28.4" not in fast.prompts[0]


@pytest.mark.parametrize("text, expected", [
    ("Transformer纯用注意力", 11),   # 11 ASCII letters = 5.5, + 5 CJK -> ceil(10.5)
    ("BLEU 28.4", 4),                # whitespace ignored, 8 ASCII -> 4
    ("注意力ABC", 5),                 # 3 + 1.5 -> 5
    ("ＢＬＥＵ", 4),                   # full-width Latin letters count as full width
    ("注 意 力", 3),
    ("", 0),
])
def test_s11_cjk_length_counts_ascii_as_half_width(text, expected):
    assert text_len(text, "zh-CN") == expected
    assert text_len(text, "ja-JP") == expected


def test_s11_mixed_script_bubble_within_budget_is_not_flagged():
    line = "Transformer在WMT2014上达到了28.4的BLEU分数"
    assert len(line) > BUDGETS["zh-CN"]["bubble"] and text_len(line, "zh-CN") == 21
    sb = storyboard([make_panel(1, dialogue=[("papi", line)])], "zh-CN")
    assert StoryboarderV2(FakeEngine()).violations(sb, "zh-CN") == []


# ============================================================================= S12 reference wording / anchor limits

def big_cast(k: int):
    names = [f"char{i}" for i in range(k)]
    cast = Cast("kumomo", "Style.", [CastMember(n, "student", f"look {i}", "calm") for i, n in enumerate(names)])
    sheets = {n: ImageContent.from_base64(b64(image_bytes(2, 2, (10 * i, 0, 0))), "image/png") for i, n in enumerate(names)}
    return cast, sheets, names


def test_s12_single_sheet_with_style_anchor_uses_ordinals():
    panel = make_panel(1, characters=["kumo"], dialogue=[("kumo", "Why so slow?")])
    prompt = build_single_prompt(panel, make_cast(), "en-US", ["kumo"], anchor=True)
    assert f"The first reference image is kumo: {LOOKS['kumo']}." in prompt
    assert "The second reference image is the previous panel of this same manga" in prompt
    assert "The reference image is" not in prompt
    plain = build_single_prompt(panel, make_cast(), "en-US", ["kumo"])
    assert f"The reference image is kumo: {LOOKS['kumo']}." in plain and "previous panel" not in plain


async def test_s12_render_panel_refs_follow_the_prompt_ordinals():
    image = FakeImageEngine(fixed_size=(1, 1), fmt="PNG")
    sheets = make_sheets()
    anchor = ImageContent.from_base64(TINY_PNG_B64)
    await PanelRenderer(image, make_cast(), sheets, "en-US").render_panel(
        make_panel(1, characters=["kumo"], dialogue=[("kumo", "Hi.")]), anchor=anchor)
    (call,) = image.calls
    assert call["refs"] == [sheets["kumo"], anchor]
    assert REF_SENTENCE.findall(call["prompt"]) == [("first", "kumo"), ("second", "the previous panel of this same manga")]


async def test_s12_anchor_without_any_sheet():
    image = FakeImageEngine(fixed_size=(1, 1), fmt="PNG")
    anchor = ImageContent.from_base64(TINY_PNG_B64)
    await PanelRenderer(image, make_cast(), {}, "en-US").render_panel(
        make_panel(1, characters=["kumo"], dialogue=[("kumo", "Hi.")]), anchor=anchor)
    (call,) = image.calls
    assert call["refs"] == [anchor]
    assert REF_SENTENCE.findall(call["prompt"]) == [("first", "the previous panel of this same manga")]


async def test_s12_nine_sheets_plus_anchor_make_ten_references():
    cast, sheets, names = big_cast(9)
    image = FakeImageEngine(fixed_size=(1, 1), fmt="PNG")
    anchor = ImageContent.from_base64(TINY_PNG_B64)
    await PanelRenderer(image, cast, sheets, "en-US").render_panel(
        make_panel(1, characters=names, dialogue=[(names[0], "Hi.")], emotions={}), anchor=anchor)
    (call,) = image.calls
    assert call["refs"] == [sheets[n] for n in names] + [anchor]
    sentences = REF_SENTENCE.findall(call["prompt"])
    assert [who for _, who in sentences] == names + ["the previous panel of this same manga"]
    assert sentences[-1][0] == "tenth"


@pytest.mark.parametrize("k", [10, 11])
async def test_s12_ten_or_more_sheets_drop_the_anchor_without_index_error(k):
    cast, sheets, names = big_cast(k)
    image = FakeImageEngine(fixed_size=(1, 1), fmt="PNG")
    anchor = ImageContent.from_base64(TINY_PNG_B64)
    await PanelRenderer(image, cast, sheets, "en-US").render_panel(
        make_panel(1, characters=names, dialogue=[(names[0], "Hi.")]), anchor=anchor)  # used to raise IndexError
    (call,) = image.calls
    assert anchor not in call["refs"] and "previous panel" not in call["prompt"]
    sentences = REF_SENTENCE.findall(call["prompt"])
    assert [who for _, who in sentences] == names
    assert [nth for nth, _ in sentences][:10] == ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


def test_s12_ordinals_past_ten_do_not_raise():
    assert [_ordinal(i) for i in (0, 1, 9)] == ["first", "second", "tenth"]
    assert (_ordinal(10), _ordinal(14)) == ("#11", "#15")
    cast, _, names = big_cast(12)
    block = _reference_block(cast, names)
    assert "The #11 reference image is char10: look 10." in block and "The #12 reference image is char11: look 11." in block


async def test_s12_job_style_anchor_with_single_character_panels(job_env):
    job_env.settings.update(style_anchor=True, concurrency=1)

    def edit():
        payload = storyboard_payload(2)
        for p in payload["panels"]:
            p["characters"], p["emotions"] = ["kumo"], {"kumo": "curious"}
            p["dialogue"][0]["speaker"] = "kumo"
        return payload

    job_env.text.router = storyboard_router(job_env, edit)
    state = await job_env.run(qa=False)

    assert state["status"] == "completed"
    first, second = job_env.image.calls
    kumo_sheet = make_sheets()["kumo"]
    assert first["refs"] == [kumo_sheet] and "The reference image is kumo:" in first["prompt"] and "previous panel" not in first["prompt"]
    assert second["refs"][0] == kumo_sheet and len(second["refs"]) == 2
    panel_one = (job_env.output_dir / state["job_id"] / "panel_01.jpg").read_bytes()
    assert base64.b64decode(second["refs"][1].data) == panel_one  # the anchor is panel 1's image
    assert REF_SENTENCE.findall(second["prompt"]) == [("first", "kumo"), ("second", "the previous panel of this same manga")]
