"""Translation checks: a fixed glossary, deterministic drift rules (stronger claims, lost hedges, names, props) with one
correction call, and the translation's fact-check findings back-ported to the English line."""

from __future__ import annotations

from services.paper_extractor import PaperText
from services.storyboarder_v2 import (
    StoryboarderV2,
    _line_at,
    translate_prompt,
    translation_problems,
)

from .fakes import FakeEngine
from .test_story_v3 import v3_panel, v3_storyboard

NAMES = frozenset({"kumo", "nezu", "papi"})


def test_drift_rules():
    assert translation_problems("The model is hard to train.", "这个模型根本没法训练。", "zh-CN", NAMES) == [
        "stronger than the English ('根本'): keep the English line's strength"]
    assert translation_problems("It fails on every task.", "它在所有任务上都失败了。", "zh-CN", NAMES) == []   # the English says every
    assert any("hedge 'Most'" in x for x in translation_problems("Most old models were slow.", "旧模型很慢。", "zh-CN", NAMES))
    assert any("hedge 'over 2'" in x for x in translation_problems("It wins by over 2 BLEU.", "它领先2 BLEU。", "zh-CN", NAMES))
    assert translation_problems("It wins by over 2 BLEU.", "它领先超过2 BLEU。", "zh-CN", NAMES) == []
    assert any("Latin letters" in x for x in translation_problems("Kumo, a PhD student.", "库莫，博士生。", "zh-CN", NAMES))
    assert any("浆果挞" in x for x in translation_problems("I bet my berry tart.", "我赌我的蛋挞。", "zh-CN", NAMES))
    assert any("doubled" in x for x in translation_problems("First, look.", "首先先看。", "zh-CN", NAMES))
    assert translation_problems("The model is hard to train.", "Le modèle est parfait.", "fr-FR", NAMES) == []   # zh rules only


def test_the_prompt_fixes_the_casts_props():
    sb = v3_storyboard([v3_panel(1, dialogue=[("nezu", "I bet my berry tart.")])])
    assert "berry tart = 浆果挞" in translate_prompt(sb, "zh-CN")


async def test_drifted_lines_are_corrected_once_and_must_keep_their_numbers():
    sb = v3_storyboard([v3_panel(1, dialogue=[("kumo", "Most models need 3.5 days."), ("nezu", "It is hard to train.")])])
    translation = {"title": "标题", "summary": "摘要", "chapters": [], "card": {}, "panels": [
        {"n": 1, "title": "第1格", "dialogue": [{"speaker": "kumo", "text": "模型需要3.5天。"}, {"speaker": "nezu", "text": "根本没法训练。"}]}]}
    fixes = {"items": [{"id": "1.d0", "text": "大多数模型需要3.5天。"}, {"id": "1.d1", "text": "很难训练，大约要9天。"}]}
    engine = FakeEngine([translation, fixes])   # nothing left untranslated, so no leftovers call
    zh = await StoryboarderV2(engine).translate(sb, "zh-CN")
    assert zh.panels[0].dialogue[0].text == "大多数模型需要3.5天。"           # the hedge is back
    assert zh.panels[0].dialogue[1].text == "根本没法训练。"                  # a "fix" that adds a number is refused
    assert [f["id"] for f in zh.brief["translation_fixes_zh-CN"]] == ["1.d0"]
    assert "drops the English hedge 'Most'" in engine.prompts[-1] and "stronger than the English ('根本')" in engine.prompts[-1]


async def test_back_port_corrects_the_english_line_the_finding_came_from():
    paper = PaperText("T", "Gains were minimal on CSQA. The Transformer reaches 28.4 BLEU.")
    english = v3_storyboard([v3_panel(1, dialogue=[("papi", "Gains held across all tested categories.")], narration="All good.")])
    localized = v3_storyboard([v3_panel(1, dialogue=[("papi", "所有类别都有提升。")], narration="都很好。")])
    localized.language = "zh-CN"
    localized.brief["factcheck_zh-CN"] = [{"id": "1.d0", "before": "所有类别都有提升。", "after": "大多数类别有提升，CSQA提升很小。",
                                          "issue": "overgeneralises: the paper says CSQA gains were minimal"},
                                         {"id": "1.n", "before": "都很好。", "after": "结果不错。", "issue": "style"}]
    reply = {"items": [{"id": "1.d0", "text": "Gains held in most categories; CSQA gains were minimal."}, {"id": "1.n", "text": "KEEP"}]}
    engine = FakeEngine([reply])
    assert await StoryboarderV2(engine).backport(english, localized, paper) == 1
    assert _line_at(english, "1.d0") == "Gains held in most categories; CSQA gains were minimal."
    assert english.panels[0].narration == "All good." and english.brief["backport"][0]["id"] == "1.d0"
    # a back-port may not bring a number the paper lacks
    english2 = v3_storyboard([v3_panel(1, dialogue=[("papi", "Gains held everywhere.")])])
    engine2 = FakeEngine([{"items": [{"id": "1.d0", "text": "Gains of 99.9 points everywhere."}]}])
    assert await StoryboarderV2(engine2).backport(english2, localized, paper) == 0
