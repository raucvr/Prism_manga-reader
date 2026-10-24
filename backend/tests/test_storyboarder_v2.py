"""services.storyboarder_v2: JSON extraction, panel normalisation, budgets, fit / write / translate."""

import pytest

from services.paper_extractor import PaperText
from services.storyboarder_v2 import (
    CHAPTER_PLAN,
    Chapter,
    PanelV2,
    StoryboarderV2,
    StoryboardV2,
    extract_json,
    panel_range_for,
    text_len,
)

from .fakes import FakeEngine, PaperModel, make_cast, make_panel


def storyboard(panels, language="en-US") -> StoryboardV2:
    return StoryboardV2(title="T", summary="S", theme="kumomo", language=language, chapters=[], panels=panels)


# ----------------------------------------------------------------------------- extract_json

@pytest.mark.parametrize("fence", ["```json", "```"])
def test_extract_json_from_code_fence(fence):
    reply = f'Here you go:\n{fence}\n{{"title": "T", "panels": [{{"n": 1}}]}}\n```\nEnjoy!'
    assert extract_json(reply) == {"title": "T", "panels": [{"n": 1}]}


def test_extract_json_ignores_surrounding_prose():
    assert extract_json('Sure! The brief is {"key_idea": "attention", "n": 2} - hope it helps.') == {"key_idea": "attention", "n": 2}


def test_extract_json_repairs_trailing_commas():
    assert extract_json('{"items": [{"id": "1.d0", "text": "ok",},],}') == {"items": [{"id": "1.d0", "text": "ok"}]}


def test_extract_json_without_an_object_raises():
    with pytest.raises(ValueError):
        extract_json("I could not do that.")


# ----------------------------------------------------------------------------- PanelV2.from_dict

def test_panel_from_dict_normalises_names_and_truncates_labels():
    p = PanelV2.from_dict({
        "n": "4", "chapter": " How ", "title": " Q, K and V ", "characters": ["Papi", " KUMO ", ""],
        "scene": "lab", "visual": "arrows", "narration": None,
        "dialogue": [{"speaker": " Papi ", "text": " Hello! "}, {"speaker": "kumo", "text": ""}, "not a bubble"],
        "on_image_labels": ["Query", "Key", "Value"], "emotions": {"Papi": "proud"},
    }, default_n=1)
    assert (p.n, p.chapter, p.title) == (4, "how", "Q, K and V")
    assert p.characters == ["papi", "kumo"]
    assert [(b.speaker, b.text) for b in p.dialogue] == [("papi", "Hello!")]
    assert p.narration == ""
    assert p.labels == ["Query", "Key"]
    assert p.emotions == {"papi": "proud"}


def test_panel_from_dict_tolerates_missing_fields_and_round_trips():
    p = PanelV2.from_dict({}, default_n=7)
    assert (p.n, p.chapter, p.title, p.characters, p.dialogue, p.narration, p.labels, p.emotions) == (7, "main", "", [], [], "", [], {})
    assert PanelV2.from_dict({"labels": ["QK"]}, 1).labels == ["QK"]  # legacy key
    full = make_panel(3, narration="Fact.", labels=["Query"], emotions={"papi": "proud"})
    assert PanelV2.from_dict(full.to_dict(), 99) == full


# ----------------------------------------------------------------------------- budgets

def test_violations_use_words_for_english_and_characters_for_chinese():
    writer = StoryboarderV2(FakeEngine())
    en = storyboard([
        make_panel(1, dialogue=[("papi", " ".join(["word"] * 14)), ("kumo", " ".join(["word"] * 15))], narration=" ".join(["fact"] * 21)),
        make_panel(2, dialogue=[("papi", "divide by sqrt of d_k"), ("kumo", "a \\frac b")], narration="Fine narration."),
    ])
    assert {v["id"] for v in writer.violations(en, "en-US")} == {"1.d1", "1.n", "2.d0", "2.d1"}

    zh = storyboard([make_panel(1, dialogue=[("papi", "注" * 22), ("kumo", "注" * 23)], narration="事" * 33)])
    assert {v["id"]: v["kind"] for v in writer.violations(zh, "zh-CN")} == {"1.d1": "bubble", "1.n": "narration"}
    assert writer.violations(zh, "en-US") == []  # a 23-character Chinese line is one "word"
    assert (text_len("注 意 力", "zh-CN"), text_len("one two  three", "en-US")) == (3, 3)


async def test_fit_applies_rewrites_and_strips_leftover_underscores():
    fast = FakeEngine([
        {"items": [{"id": "1.d0", "text": "Attention picks the words that matter."}, {"id": "1.n", "text": "We divide by root d_k."}]},
        {"items": [{"id": "1.n", "text": "We still say d_k here."}]},
    ])
    sb = storyboard([make_panel(1, dialogue=[("papi", " ".join(["long"] * 16)), ("kumo", "Short and fine.")],
                                narration="We divide by sqrt of d_k to keep things stable.")])
    writer = StoryboarderV2(FakeEngine(), fast_engine=fast)

    assert await writer.fit(sb, "en-US") == 3
    p = sb.panels[0]
    assert [b.text for b in p.dialogue] == ["Attention picks the words that matter.", "Short and fine."]
    assert p.narration == "We still say d k here."  # still had "_" after 2 rounds -> stripped, never truncated
    assert len(fast.calls) == 2
    first = fast.prompts[0]
    assert '"1.d0"' in first and '"1.n"' in first and '"1.d1"' not in first
    assert "speech bubble ≤ 14 words" in first
    assert fast.calls[0][1].thinking is False


async def test_fit_without_violations_makes_no_model_call():
    engine = FakeEngine()  # would raise on any call
    sb = storyboard([make_panel(1, dialogue=[("papi", "All good here.")], narration="Short fact.")])
    assert await StoryboarderV2(engine).fit(sb, "en-US") == 0
    assert engine.calls == []


# ----------------------------------------------------------------------------- write

async def test_write_renumbers_remaps_speakers_caps_dialogue_and_orders_chapters():
    reply = {"title": "Attention, Please!", "summary": "Attention replaces recurrence.", "panels": [
        {"n": 9, "chapter": "evidence", "title": "BLEU", "characters": ["nezu", "papi"], "scene": "s", "visual": "v",
         "dialogue": [{"speaker": "Nezu", "text": "Prove it."}, {"speaker": "papi", "text": "28.4 BLEU."}, {"speaker": "nezu", "text": "Fine."}]},
        {"n": 2, "chapter": "hook", "title": "Why", "characters": ["kumo", "professor"], "scene": "s", "visual": "v",
         "dialogue": [{"speaker": "professor", "text": "Translation used to be slow."}]},
        {"n": 5, "chapter": "bonus", "title": "Heads", "characters": ["dragon"], "scene": "s", "visual": "v", "dialogue": []},
        {"n": 7, "chapter": "problem", "title": "RNNs", "characters": [], "scene": "s", "visual": "v",
         "dialogue": [{"speaker": "kumo", "text": "Why so slow?"}]},
    ]}
    engine = FakeEngine([reply])
    sb = await StoryboarderV2(engine).write({"title": "brief title"}, make_cast(), (3, 6), "zh-CN")

    assert [(p.n, p.title) for p in sb.panels] == [(1, "Why"), (2, "Heads"), (3, "RNNs"), (4, "BLEU")]
    why, heads, rnns, bleu = sb.panels
    assert why.characters == ["kumo"] and [b.speaker for b in why.dialogue] == ["kumo"]  # unknown speaker -> cast member
    assert heads.chapter == "hook" and heads.characters == ["papi"]  # unknown chapter -> inherits the previous panel's; nobody valid -> first cast member
    assert rnns.characters == ["kumo"]  # empty characters -> the valid speaker
    assert [(b.speaker, b.text) for b in bleu.dialogue] == [("nezu", "Prove it."), ("papi", "28.4 BLEU.")]  # capped at 2
    assert [(c.id, c.panel_numbers) for c in sb.chapters] == [("hook", [1, 2]), ("problem", [3]), ("evidence", [4])]
    assert sb.chapters[0].title == dict(CHAPTER_PLAN)["hook"]
    assert (sb.title, sb.summary, sb.language, sb.theme) == ("Attention, Please!", "Attention replaces recurrence.", "en-US", "kumomo")

    prompt = engine.prompts[0]
    assert "Between 3 and 6 panels" in prompt and "- papi: the professor/mentor" in prompt
    assert engine.calls[0][1].thinking is True


async def test_write_without_panels_raises():
    with pytest.raises(ValueError, match="no panels"):
        await StoryboarderV2(FakeEngine([{"title": "x", "panels": []}])).write({}, make_cast(), (3, 6), "en-US")


# ----------------------------------------------------------------------------- factcheck

async def test_factcheck_applies_corrections_by_line_id():
    sb = storyboard([
        make_panel(1, dialogue=[("papi", "We divide by the number of queries."), ("kumo", "Why?")], narration="The base model cost 3.3e18 FLOPs."),
        make_panel(2, dialogue=[("nezu", "Prove it.")]),
    ])
    engine = FakeEngine([{"problems": [
        {"id": "1.d0", "issue": "wrong scaling", "corrected": "We divide by the square root of the key size."},
        {"id": "1.n", "issue": "that cost belongs to the big model", "corrected": "The big model cost 2.3e19 FLOPs."},
        {"id": "2.d5", "corrected": "no such bubble"}, {"id": "9.n", "corrected": "no such panel"},
        {"id": "x", "corrected": "malformed id"}, {"id": "2.d0", "corrected": ""},
    ]}])
    paper = PaperText("Attention Is All You Need", "PAPER BODY: scaled by the square root of dk", [])

    assert await StoryboarderV2(engine).factcheck(sb, paper) == 2
    assert sb.panels[0].dialogue[0].text == "We divide by the square root of the key size."
    assert sb.panels[0].narration == "The big model cost 2.3e19 FLOPs."
    assert sb.panels[1].dialogue[0].text == "Prove it."
    assert [(f["id"], f["before"]) for f in sb.brief["factcheck"]] == [
        ("1.d0", "We divide by the number of queries."), ("1.n", "The base model cost 3.3e18 FLOPs.")]
    assert '"id": "1.d1"' in engine.prompts[0] and "PAPER BODY: scaled by the square root of dk" in engine.prompts[0]


async def test_factcheck_is_best_effort():
    sb = storyboard([make_panel(1)])
    before = sb.panels[0].dialogue[0].text
    assert await StoryboarderV2(FakeEngine([RuntimeError("model overloaded")])).factcheck(sb, PaperText("T", "body", [])) == 0
    assert sb.panels[0].dialogue[0].text == before and "factcheck" not in sb.brief


# ----------------------------------------------------------------------------- translate / analyze / build

async def test_translate_applies_per_panel_translations_and_sets_language():
    sb = storyboard([
        make_panel(1, dialogue=[("papi", "Hello there."), ("kumo", "Hi!")], narration="Fact one.", labels=["Query"]),
        make_panel(2, dialogue=[("nezu", "Prove it.")], narration="Fact two.", labels=["Key", "Value"]),
        make_panel(3, dialogue=[("kumo", "Untouched.")], narration="Fact three."),
    ])
    sb.chapters = [Chapter("hook", "Why should anyone care?", [1, 2, 3])]
    engine = FakeEngine([{
        "title": "注意力", "summary": "摘要", "chapters": [{"id": "hook", "title": "为什么要关心？"}],
        "panels": [
            {"n": 1, "title": "第一格", "dialogue": [{"speaker": "papi", "text": "你好。"}, {"speaker": "kumo", "text": "嗨！"}], "narration": "事实一。",
             "labels": ["查询"]},
            {"n": 2, "dialogue": [{"speaker": "nezu", "text": "证明一下。"}], "labels": ["键"]},
        ],
    }, {  # panels missing from the first reply are re-requested once
        "panels": [{"n": 3, "dialogue": [{"speaker": "kumo", "text": "没动过。"}]}],
    }])

    assert await StoryboarderV2(engine).translate(sb, "zh-CN") is sb
    assert sb.language == "zh-CN"
    assert (sb.title, sb.summary, sb.chapters[0].title) == ("注意力", "摘要", "为什么要关心？")
    p1, p2, p3 = sb.panels
    assert (p1.title, [b.text for b in p1.dialogue], p1.narration) == ("第一格", ["你好。", "嗨！"], "事实一。")
    assert [b.speaker for b in p1.dialogue] == ["papi", "kumo"]
    assert (p2.title, p2.dialogue[0].text, p2.narration) == ("Panel 2", "证明一下。", "Fact two.")  # absent keys keep the English
    assert p3.dialogue[0].text == "没动过。"  # panel missing from the first reply, translated by the retry
    assert (p1.labels, p2.labels) == (["查询"], ["Key", "Value"])  # labels only replaced when the count matches
    assert "Simplified Chinese" in engine.prompts[0] and "each bubble ≤ 22 characters" in engine.prompts[0]


@pytest.mark.parametrize("language, expected", [("en-US", "en-US"), ("fr-FR", "en-US")])
async def test_translate_is_a_noop_for_english_and_unknown_languages(language, expected):
    engine = FakeEngine()
    sb = storyboard([make_panel(1)])
    await StoryboarderV2(engine).translate(sb, language)
    assert sb.language == expected and engine.calls == []


async def test_analyze_repairs_a_non_json_reply_and_fills_the_title():
    engine = FakeEngine(["Here is the brief: key idea = attention (sorry, no JSON)", '{"key_idea": "Attention replaces recurrence."}'])
    paper = PaperText(title="Attention Is All You Need", body="paper body text", captions=["Figure 1: The Transformer."])
    brief = await StoryboarderV2(engine).analyze(paper)
    assert brief == {"key_idea": "Attention replaces recurrence.", "title": "Attention Is All You Need"}
    (_, first_cfg), (_, repair_cfg) = engine.calls
    assert first_cfg.thinking is True and repair_cfg.thinking is False and repair_cfg.temperature == 0.0
    assert "Figure 1: The Transformer." in engine.prompts[0] and "paper body text" in engine.prompts[0]
    assert "strictly valid JSON" in engine.prompts[1]


async def test_build_runs_every_step_and_reports_stages():
    model = PaperModel(num_panels=3)
    stages = []
    sb = await StoryboarderV2(FakeEngine(router=model)).build(
        PaperText("T", "body", []), make_cast(), "zh-CN", (3, 6), on_stage=lambda status, message: stages.append(status))
    assert list(dict.fromkeys(stages)) == ["analyzing", "storyboarding", "translating"]
    assert model.seen == ["brief", "briefcheck", "storyboard", "factcheck", "translate", "factcheck"]
    assert sb.language == "zh-CN" and sb.panels[0].dialogue[0].text == "第1格的台词"


def test_panel_range_for():
    assert panel_range_for("short", 16, 32) == (12, 18)
    assert panel_range_for("long", 16, 32) == (28, 40)
    for choice in ("auto", "", None, "huge"):
        assert panel_range_for(choice, 16, 32) == (16, 32)
