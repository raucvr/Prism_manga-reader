"""services.qa_judge: verdict parsing, retry decisions and failure handling."""

import pytest

from engines.base import ImageContent
from services.qa_judge import PanelQA, QAResult

from .fakes import TINY_PNG_B64, FakeEngine, exact_verdict, make_panel, make_sheets

PANEL = make_panel(1, characters=["papi", "kumo"],
                   dialogue=[("papi", "Attention is all you need."), ("kumo", "Even for long sentences?")],
                   narration="The Transformer reached 28.4 BLEU.", labels=["Query"])


def verdict(panels=None, overall=9) -> dict:
    return exact_verdict(PanelQA(FakeEngine(), {}, "en-US")._expected(panels or [PANEL]), overall)


async def judge(reply, panels=None, sheets=None):
    engine = FakeEngine([reply])
    qa = PanelQA(engine, make_sheets() if sheets is None else sheets, "en-US")
    result = await qa.judge(ImageContent.from_base64(TINY_PNG_B64), panels or [PANEL])
    return result, engine, qa


async def test_all_exact_verdict_passes():
    sheets = make_sheets()
    result, engine, qa = await judge(verdict(), sheets=sheets)
    assert (result.text_exact, result.text_total) == (3, 3)  # 2 bubbles + narration; labels are not counted
    assert result.characters_ok and not result.needs_retry and result.score == 9.0
    assert result.to_public() == {"score": 9.0, "text_exact": 3, "text_total": 3, "issues": [], "available": True}

    (messages, config), = engine.calls
    assert messages[0].images[:3] == list(sheets.values()) and messages[0].images[3].data == TINY_PNG_B64
    assert config.temperature == 0.0 and config.thinking is False
    assert qa.usage == {"prompt_tokens": 1000, "completion_tokens": 200}
    prompt = messages[0].content
    assert "image 1 = papi (brown dog professor)" in prompt
    assert "Image 4 is the generated page containing one panel" in prompt
    assert '"text": "Even for long sentences?"' in prompt


async def test_inexact_bubble_triggers_retry_with_targeted_feedback():
    v = verdict()
    v["panels"][0]["text_items"][1].update(exact=False, note="reads 'Even for long sentence?'")
    result, *_ = await judge(v)
    assert result.needs_retry and (result.text_exact, result.text_total) == (2, 3)
    assert result.issues == ["kumo's bubble must read exactly \"Even for long sentences?\" (reads 'Even for long sentence?')"]
    assert result.feedback() == result.issues[0]


@pytest.mark.parametrize("problem, issue", [
    ("wrong_species", "papi does not match the reference sheet looks like a cat"),
    ("missing", "kumo is missing from the panel"),
    ("extra", "remove the unexpected character: nezu"),
])
async def test_character_problems_trigger_retry(problem, issue):
    v = verdict()
    characters = v["panels"][0]["characters"]
    if problem == "wrong_species":
        characters[0].update(matches_reference=False, note="looks like a cat")
    elif problem == "missing":
        characters[1].update(present=False)
    else:
        v["panels"][0]["extra_characters"] = ["nezu"]
    result, *_ = await judge(v)
    assert not result.characters_ok and result.needs_retry
    assert result.text_exact == result.text_total == 3
    assert issue in result.issues


async def test_missing_label_alone_does_not_trigger_retry():
    v = verdict()
    v["panels"][0]["text_items"][3].update(found=False, exact=False, note="label missing")
    result, *_ = await judge(v)
    assert not result.needs_retry and result.issues == []
    assert (result.text_exact, result.text_total) == (3, 3)


async def test_low_overall_score_triggers_retry():
    result, *_ = await judge(verdict(overall=4))
    assert result.needs_retry and result.text_exact == 3 and result.characters_ok


@pytest.mark.parametrize("reply", [RuntimeError("vision model timed out"), "I cannot judge this image."])
async def test_judge_failure_never_blocks_the_panel(reply):
    result, *_ = await judge(reply)
    assert result.needs_retry is False and result.characters_ok
    assert len(result.issues) == 1 and result.issues[0].startswith("qa unavailable")


async def test_strip_verdict_counts_texts_of_both_panels():
    left = make_panel(1, characters=["papi"], dialogue=[("papi", "Left line.")])
    right = make_panel(2, characters=["kumo"], dialogue=[("kumo", "Right line.")], narration="Right narration.")
    result, engine, _ = await judge(verdict([left, right]), panels=[left, right], sheets={})
    assert (result.text_exact, result.text_total) == (3, 3) and not result.needs_retry
    prompt = engine.prompts[0]
    assert "Image 1 is the generated page containing two panels side by side" in prompt
    assert "Reference character sheets" not in prompt


def test_feedback_text():
    many = QAResult(score=3, text_exact=0, text_total=2, characters_ok=False, issues=[f"issue {i}" for i in range(7)])
    assert many.feedback() == "issue 0; issue 1; issue 2; issue 3"
    assert len(many.to_public()["issues"]) == 6
    assert QAResult(score=3, text_exact=1, text_total=2, characters_ok=True).feedback() == "text or characters did not match the script"
