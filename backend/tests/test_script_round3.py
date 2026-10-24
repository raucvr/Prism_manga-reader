"""Round-3 regression tests for the script stage (services.storyboarder_v2): S3 translation acceptance, S5 speakers /
narrator / emotions, S8 reply shapes, and P2 (absolute claims and extra people in the storyboard and fact-check prompts).

Each test failed on the code before round 3 (P2: before the pending patch) and passes now.
"""

import json

import pytest

from services.paper_extractor import PaperText
from services.panel_renderer import build_single_prompt
from services.storyboarder_v2 import (
    PanelV2,
    StoryboarderV2,
    StoryboardV2,
    _has_script,
    _latin_keep,
    _match_cast,
    extract_json,
)

from .fakes import BRIEF, FakeEngine, PaperModel, make_cast, make_panel, storyboard_payload, translation_of


def storyboard(panels, brief=None) -> StoryboardV2:
    return StoryboardV2(title="T", summary="S", theme="kumomo", language="en-US", chapters=[], panels=panels, brief=dict(brief or {}))


def payload_of(prompt: str) -> dict:
    """The JSON payload at the end of ``translate_prompt``."""
    return json.loads(prompt.rsplit("\n\n", 1)[1])


async def write(panels: list[dict]) -> StoryboardV2:
    return await StoryboarderV2(FakeEngine([{"title": "T", "summary": "S", "panels": panels}])).write({}, make_cast(), (1, 40), "en-US")


def storyboard_router(job_env, edit):
    """Text-engine router for job tests: the default PaperModel, with the storyboard reply passed through ``edit``."""
    base = job_env.paper

    def router(prompt, messages, config):
        if messages[-1].content.startswith("Write the manga script"):
            return edit(storyboard_payload(base.num_panels))
        return base(prompt, messages, config)

    return router


ITEMS = [{"id": "1.d0", "text": "A shorter line."}, {"id": "1.n", "text": "A shorter fact."}]


# ============================================================================= S3 translation acceptance

@pytest.mark.parametrize("text, language", [
    ("ResNet-152？", "ja-JP"), ("ImageNet！", "zh-CN"), ("FLOPs？", "zh-CN"), ("DALL-E 2！", "zh-CN"),
])
def test_s3_model_names_kept_in_latin_script_count_as_translated(text, language):
    assert _has_script(text, language) is True
    assert _has_script(f"{text} is much faster", language) is False  # English words around the name are still caught


async def test_s3_names_from_the_brief_may_stay_latin_but_english_sentences_may_not():
    brief = {"key_idea": "The Transformer replaces recurrence with attention.", "glossary": [{"term": "BLEU score", "plain": "quality"}]}
    sb = storyboard([make_panel(1, dialogue=[("papi", "Transformer!")]), make_panel(2, dialogue=[("kumo", "What a BLEU score!")])], brief)
    engine = FakeEngine([{"panels": [{"n": 1, "dialogue": [{"speaker": "papi", "text": "Transformer！"}]},
                                     {"n": 2, "dialogue": [{"speaker": "kumo", "text": "BLEU score！"}]}]}])

    await StoryboarderV2(engine).translate(sb, "zh-CN")

    assert [p.dialogue[0].text for p in sb.panels] == ["Transformer！", "BLEU score！"]
    assert len(engine.calls) == 1
    keep = _latin_keep(sb.brief)
    assert not _has_script("Hello there.", "zh-CN", keep) and not _has_script("Transformer is so fast!", "zh-CN", keep)


async def test_s3_narration_only_panel_left_in_english_is_re_requested():
    engine = FakeEngine([
        {"panels": [{"n": 1, "dialogue": [{"speaker": "papi", "text": "你好。"}]},
                    {"n": 2, "dialogue": [], "narration": "Training took 3.5 days on eight GPUs."}]},
        {"panels": [{"n": 2, "dialogue": [], "narration": "在八块GPU上训练了3.5天。"}]},
    ])
    sb = storyboard([make_panel(1, dialogue=[("papi", "Hello there.")]),
                     make_panel(2, dialogue=[], narration="Training took 3.5 days on eight GPUs.")])

    await StoryboarderV2(engine).translate(sb, "zh-CN")

    assert [p.narration for p in sb.panels] == ["", "在八块GPU上训练了3.5天。"]
    assert [p["n"] for p in payload_of(engine.prompts[1])["panels"]] == [2]


async def test_s3_translation_never_adds_a_narration_the_panel_did_not_have():
    engine = FakeEngine([{"panels": [{"n": 1, "dialogue": [{"speaker": "papi", "text": "你好。"}], "narration": "这是模型自己加的事实。"}]}])
    sb = storyboard([make_panel(1, dialogue=[("papi", "Hello there.")], narration="")])
    await StoryboarderV2(engine).translate(sb, "zh-CN")
    assert (sb.panels[0].dialogue[0].text, sb.panels[0].narration) == ("你好。", "")


async def test_s3_failed_re_request_keeps_the_job_running_with_a_warning(job_env, capsys):
    class FlakyTranslator(PaperModel):
        """The first translation leaves panel 2 in English; the re-request then fails at the engine."""

        def __init__(self):
            super().__init__(num_panels=3)
            self.translations = 0

        def __call__(self, prompt, messages, config):
            user = messages[-1].content
            if not user.startswith("Translate this manga script"):
                return super().__call__(prompt, messages, config)
            self.translations += 1
            if self.translations > 1:
                return RuntimeError("ModelArk 500: service overloaded")
            reply = translation_of(payload_of(user))
            reply["panels"][1]["dialogue"][0]["text"] = "Line for panel 2 please."
            return reply

    model = FlakyTranslator()
    job_env.text.router = model
    state = await job_env.run(language="zh-CN", qa=False)

    assert state["status"] == "completed"
    assert model.translations == 2
    assert [p["dialogue"][0]["text"] for p in state["panels"]] == ["第1格的台词", "Line for panel 2 please.", "第3格的台词"]
    assert "re-request failed" in capsys.readouterr().out


# ============================================================================= S8 reply shapes

@pytest.mark.parametrize("reply", [
    pytest.param('Format example: {"id": "x"}\nAnswer: ' + json.dumps(ITEMS), id="list-after-an-unrelated-object"),
    pytest.param("As noted in [1], the rewritten lines are: " + json.dumps(ITEMS), id="citation-before-the-list"),
])
def test_s8_a_list_anywhere_in_the_reply_is_found(reply):
    assert extract_json(reply, key="items") == {"items": ITEMS}


def test_s8_keyed_lookup_prefers_the_object_with_the_key_and_never_wraps_noise():
    example = 'Example:\n```json\n["id", "text"]\n```\nAnswer: ' + json.dumps({"items": ITEMS})
    assert extract_json(example, key="items") == {"items": ITEMS}  # the fenced example list used to win
    # a citation is not a payload, and a truncated object is not mined for inner lists: both go to the repair round
    for reply in ("I checked panel [1]; every line is fine.", '{"title": "T", "mechanisms": [{"name": "attention"}], "evidence": [{"claim"'):
        with pytest.raises(ValueError):
            extract_json(reply, key="items")


def test_s8_from_dict_takes_single_items_where_lists_belong():
    p = PanelV2.from_dict({"characters": "Papi, Kumo", "dialogue": {"speaker": "Papi", "text": "Hello."},
                           "on_image_labels": [{"text": "Query"}, "Key"], "emotions": {"papi": "", "kumo": "happy"}}, 1)
    assert p.characters == ["papi", "kumo"]
    assert [(b.speaker, b.text) for b in p.dialogue] == [("papi", "Hello.")]
    assert p.labels == ["Key"]  # a dict is never drawn on the image as "{'text': 'Query'}"
    assert p.emotions == {"kumo": "happy"}  # an empty emotion would read "papi is ." in the prompt


async def test_s8_write_keeps_a_bubble_sent_as_a_dict_and_a_cast_list_sent_as_a_string():
    sb = await write([{"n": 1, "chapter": "hook", "title": "Hi", "characters": "kumo, papi",
                       "dialogue": {"speaker": "Papi", "text": "Attention is all you need."}}])
    p = sb.panels[0]
    assert [(b.speaker, b.text) for b in p.dialogue] == [("papi", "Attention is all you need.")]
    assert p.characters == ["kumo", "papi"]


async def test_s8_translate_accepts_a_single_bubble_sent_as_a_dict_and_ignores_dict_labels():
    engine = FakeEngine([{"panels": [{"n": 1, "dialogue": {"speaker": "papi", "text": "你好。"}, "labels": [{"text": "查询"}]}]}])
    sb = storyboard([make_panel(1, dialogue=[("papi", "Hello there.")], labels=["Query"])])
    await StoryboarderV2(engine).translate(sb, "zh-CN")
    assert (sb.panels[0].dialogue[0].text, sb.panels[0].labels) == ("你好。", ["Query"])
    assert len(engine.calls) == 1


# ============================================================================= S5 speakers / narrator / emotions

def test_s5_names_starting_at_the_same_place_prefer_the_longer_cast_name():
    for order in (["papi", "papi jr"], ["papi jr", "papi"]):
        assert _match_cast("papi jr (waving)", order) == "papi jr"
        assert _match_cast("papi, then papi jr", order) == "papi"


@pytest.mark.parametrize("speaker", ["Voice-over", "voiceover", "Text box", "Title card"])
async def test_s5_more_narrator_spellings_become_narration(speaker):
    sb = await write([
        {"n": 1, "chapter": "hook", "characters": ["kumo", "papi"], "dialogue": [
            {"speaker": speaker, "text": "Meanwhile, in 2017..."}, {"speaker": "papi", "text": "Attention!"}]},
        {"n": 2, "chapter": "hook", "characters": ["kumo", "papi"], "dialogue": [{"speaker": "Papi (narrator)", "text": "I narrate too."}]},
    ])
    first, second = sb.panels
    assert (first.narration, [(b.speaker, b.text) for b in first.dialogue]) == ("Meanwhile, in 2017...", [("papi", "Attention!")])
    assert (second.narration, [(b.speaker, b.text) for b in second.dialogue]) == ("", [("papi", "I narrate too.")])  # names a cast member


async def test_s5_narrator_lines_merge_in_order_without_duplicates():
    sb = await write([{"n": 1, "chapter": "hook", "characters": ["kumo"], "narration": "Google, 2017.", "dialogue": [
        {"speaker": "narrator", "text": "Google, 2017."}, {"speaker": "kumo", "text": "Wow."}, {"speaker": "caption", "text": "Eight GPUs."}]}])
    p = sb.panels[0]
    assert p.narration == "Google, 2017. Eight GPUs."
    assert [(b.speaker, b.text) for b in p.dialogue] == [("kumo", "Wow.")]


async def test_s5_duplicate_character_spellings_collapse_to_one():
    sb = await write([{"n": 1, "chapter": "hook", "characters": ["papi", "Papi (smiling)", "kumo"],
                       "dialogue": [{"speaker": "papi", "text": "Hi."}]}])
    p = sb.panels[0]
    assert p.characters == ["papi", "kumo"]
    assert "Characters in this panel: papi, kumo only." in build_single_prompt(p, make_cast(), "en-US", ["papi", "kumo"])


async def test_s5_job_draws_a_narrator_line_in_the_narration_box(job_env):
    def edit(payload):
        first = payload["panels"][0]
        first["narration"] = ""
        first["dialogue"] = [{"speaker": "Narrator (V.O.)", "text": "Meanwhile, in 2017..."}, first["dialogue"][0]]
        return payload

    job_env.text.router = storyboard_router(job_env, edit)
    state = await job_env.run(qa=False)

    assert state["status"] == "completed"
    panel = state["panels"][0]
    assert (panel["narration"], [b["speaker"] for b in panel["dialogue"]]) == ("Meanwhile, in 2017...", ["papi"])
    prompt = next(c["prompt"] for c in job_env.image.calls if "Line for panel 1 please." in c["prompt"])
    assert 'Narration box along the top edge: "Meanwhile, in 2017..."' in prompt and 'says: "Meanwhile' not in prompt


# ============================================================================= P2 absolute claims / extra people

async def test_p2_storyboard_prompt_bans_unsupported_absolute_claims_and_extra_people():
    engine = FakeEngine([storyboard_payload(3)])
    await StoryboarderV2(engine).write(BRIEF, make_cast(), (3, 6), "en-US")
    rules = engine.prompts[0].split("HARD RULES", 1)[1].split("OUTPUT:", 1)[0]
    assert "No absolute claims (all, every, always, never, nobody) unless the paper itself makes them" in rules
    assert "No claims about the paper's later impact, adoption or today's practice" in rules
    assert "single model or ensemble, dataset and split" in rules and "never invent a weakness" in rules
    assert "Metaphors use objects, places, diagrams and the cast only — never extra people, children, crowds or other creatures" in rules


async def test_p2_factcheck_prompt_flags_unsupported_absolute_claims():
    engine = FakeEngine([{"problems": [{"id": "1.d0", "issue": "absolute claim the paper does not make",
                                        "corrected": "Many translation systems now use attention."}]}])
    sb = storyboard([make_panel(1, dialogue=[("papi", "Every phone on Earth uses this now!")])])

    assert await StoryboarderV2(engine).factcheck(sb, PaperText("T", "We propose the Transformer.", [])) == 1

    instructions = engine.prompts[0].split("LINES:", 1)[0]
    assert "claims impact, adoption or present-day status the paper does not state" in instructions and "every phone uses this" in instructions
    assert "single model vs ensemble" in instructions and "limitation or weakness the paper does not report" in instructions
    assert sb.panels[0].dialogue[0].text == "Many translation systems now use attention."


# ============================================================================= round-3 verifier: repair round and odd shapes

BROKEN_STORYBOARD = json.dumps(storyboard_payload(3)).replace('"chapter": "problem",', '"chapter": "problem"', 1)  # missing comma


@pytest.mark.parametrize("reply", [
    pytest.param(BROKEN_STORYBOARD + "\n\nNote: on_image_labels is [] when a panel has no labels.", id="note-after"),
    pytest.param("Here is the script (empty lists are []):\n" + BROKEN_STORYBOARD, id="lead-in"),
    pytest.param("Here is the script (empty lists are []):\n" + json.dumps(storyboard_payload(3)).replace(
        '"Scene 2 in', '"Scene "2 in', 1), id="lead-in-stray-quote"),  # one unescaped quote: the object never closes
])
async def test_s8_malformed_payload_goes_to_the_repair_round_even_when_prose_mentions_an_empty_list(reply):
    with pytest.raises(ValueError):
        extract_json(reply, key="panels")  # a "[]" in the prose is not the payload
    engine = FakeEngine([reply, storyboard_payload(3)])
    sb = await StoryboarderV2(engine).write({}, make_cast(), (1, 40), "en-US")
    assert len(sb.panels) == 3 and len(engine.calls) == 2


async def test_s8_malformed_qa_verdict_with_an_empty_list_note_is_still_repaired():
    from engines.base import ImageContent
    from services.qa_judge import PanelQA

    from .fakes import TINY_PNG_B64, exact_verdict, make_sheets

    panels = [make_panel(1, dialogue=[("papi", "Attention is all you need.")])]
    good = exact_verdict(PanelQA(FakeEngine(), {}, "en-US")._expected(panels))
    bad = json.dumps(good).replace('"overall": 9', '"overall" 9') + "\n(stray_text is [] because nothing is wrong)"
    engine = FakeEngine([bad, good])
    result = await PanelQA(engine, make_sheets(), "en-US").judge(ImageContent.from_base64(TINY_PNG_B64), panels)
    assert len(engine.calls) == 2 and (result.text_exact, result.text_total) == (1, 1)


async def test_s8_truncated_bare_list_goes_to_the_repair_round():
    panels = storyboard_payload(4)["panels"]
    cut = json.dumps(panels)[: len(json.dumps(panels)) * 4 // 5]  # cut off at max_tokens inside panel 4
    for reply in (cut, "```json\n" + cut, "Here you go:\n" + cut):
        with pytest.raises(ValueError):
            extract_json(reply, key="panels")  # never one element of the list as if it were the payload
    engine = FakeEngine([cut, {"title": "T", "summary": "S", "panels": panels[:3]}])
    sb = await StoryboarderV2(engine).write({}, make_cast(), (1, 40), "en-US")
    assert [p.title for p in sb.panels] == ["Panel 1 title", "Panel 2 title", "Panel 3 title"] and len(engine.calls) == 2


async def test_s8_characters_sent_as_a_name_to_emotion_map_keep_every_character():
    assert PanelV2.from_dict({"characters": {"Papi": "curious", "kumo": "excited"}}, 1).characters == ["papi", "kumo"]
    sb = await write([{"n": 1, "chapter": "hook", "characters": {"papi": "curious", "kumo": "excited"},
                       "visual": "papi shows kumo a glowing board", "dialogue": [{"speaker": "papi", "text": "Look at this."}]}])
    p = sb.panels[0]
    assert p.characters == ["papi", "kumo"]
    assert "Characters in this panel: papi, kumo only." in build_single_prompt(p, make_cast(), "en-US", p.characters)


async def test_s8_an_empty_list_in_prose_never_replaces_a_parsed_object_without_the_key():
    brief = {k: v for k, v in BRIEF.items() if k != "key_idea"}  # the model left out "key_idea"
    for reply in ("```json\n" + json.dumps(brief) + "\n```\nNote: fun_facts is [] because the paper has none.",
                  json.dumps(brief) + "\n\n(fun_facts: [] since nothing is surprising)"):
        got = await StoryboarderV2(FakeEngine([reply])).analyze(PaperText("Attention Is All You Need", "body", []))
        assert got["mechanisms"] == BRIEF["mechanisms"] and got["evidence"] == BRIEF["evidence"]
    assert extract_json("[]\n\nNo line needs a fix.", key="problems") == {"problems": []}  # with no object, [] is the answer


# ============================================================================= integration: lone surrogates in model replies

def utf8(value) -> bytes:
    """What httpx sends for json=: UTF-8 of json.dumps(..., ensure_ascii=False). Raises on a lone surrogate."""
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize("reply", [
    pytest.param('{"panels": [{"title": "Wow \\ud83d", "dialogue": [{"speaker": "papi", "text": "Nice \\udc4d!"}]}], "\\ud83d": 1}', id="escape"),
    pytest.param('{"panels": [{"title": "Wow \ud83d", "dialogue": [{"speaker": "papi", "text": "Nice \udc4d!"}]}], "\ud83d": 1}', id="raw"),
    pytest.param('Here: [{"title": "Wow \\ud83d", "dialogue": [{"speaker": "papi", "text": "Nice \\udc4d!"}]}]', id="list-in-prose"),
])
def test_extract_json_replaces_lone_surrogates_so_the_text_can_be_sent_again(reply):
    data = extract_json(reply, key="panels")
    utf8(data)   # every later request that carries this text (Seedream prompt, judge, fit, translation) can be encoded
    p = data["panels"][0]
    assert (p["title"], p["dialogue"][0]["text"]) == ("Wow \ufffd", "Nice \ufffd!")
    assert extract_json('{"text": "ok \\ud83d\\ude00"}')["text"] == "ok \U0001F600"   # a complete pair is the emoji


async def test_a_broken_emoji_in_the_script_never_reaches_the_image_prompt():
    payload = storyboard_payload(2)
    payload["panels"][0]["dialogue"][0]["text"] += " \ud83d"   # half an emoji escape, decoded by json.loads
    sb = await StoryboarderV2(FakeEngine([payload])).write({}, make_cast(), (1, 40), "en-US")
    p = sb.panels[0]
    assert p.dialogue[0].text.endswith(" \ufffd")
    build_single_prompt(p, make_cast(), "en-US", p.characters).encode("utf-8")


async def test_the_repair_round_never_sends_a_lone_surrogate():
    broken = BROKEN_STORYBOARD.replace("Panel 1 title", "Panel 1 title \ud83d", 1)
    engine = FakeEngine([broken, storyboard_payload(3)])
    sb = await StoryboarderV2(engine).write({}, make_cast(), (1, 40), "en-US")
    assert len(sb.panels) == 3 and len(engine.calls) == 2
    repair = engine.calls[1][0][0].content
    assert "Panel 1 title \ufffd" in repair
    repair.encode("utf-8")


async def test_the_translated_script_is_fact_checked_in_its_own_language():
    """Independent audit: the English fact-check added qualifiers that the Japanese translation dropped again."""
    engine = FakeEngine([{"problems": [
        {"id": "1.n", "issue": "drops the paper's 'fine-tuning' qualifier", "corrected": "以前の微調整型モデルは片方向だった。"},
        {"id": "1.d0", "issue": "reply in the wrong language", "corrected": "Earlier fine-tuned models were one-directional."},
    ]}])
    sb = storyboard([make_panel(1, dialogue=[("papi", "昔は片方向だけ")], narration="以前の事前学習モデルは片方向だけだった。")])
    sb.language = "ja-JP"
    assert await StoryboarderV2(engine).factcheck(sb, PaperText("T", "We propose BERT.", []), "ja-JP") == 1
    assert sb.panels[0].narration == "以前の微調整型モデルは片方向だった。"
    assert sb.panels[0].dialogue[0].text == "昔は片方向だけ"            # an English correction is not put into a ja script
    instructions = engine.prompts[0].split("LINES:", 1)[0]
    assert "The lines are in Japanese" in instructions and "narration ≤ 34 characters" in instructions
    assert [f["id"] for f in sb.brief["factcheck_ja-JP"]] == ["1.n"]
