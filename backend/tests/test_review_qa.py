"""Regression tests for the QA-judge review fixes S2, S6, S7 and S9 (services/qa_judge.py, attempt rank in services/jobs.py).

Plain tests lock in what the fixes got right; the round-2 strict xfails became plain regression tests once fixed in
round 3 (more round-3 cases are in test_qa_round3.py). All engines are fakes; jobs run in tmp_path.
"""

import copy
import json

import pytest
from PIL import Image

from engines.base import ImageContent
from services.qa_judge import PanelQA, _falsy, _is_noise, _truthy

from .fakes import EXPECTED_RE, PALETTE, TINY_PNG_B64, FakeEngine, exact_verdict, make_panel, make_sheets

PAPI, KUMO, NARRATION = "Attention is all you need.", "Even for long sentences?", "The Transformer reached 28.4 BLEU."
PANEL = make_panel(1, characters=["papi", "kumo"], dialogue=[("papi", PAPI), ("kumo", KUMO)], narration=NARRATION)


def qa_for(sheets=None, language="en-US", engine=None) -> PanelQA:
    return PanelQA(engine or FakeEngine(), make_sheets() if sheets is None else sheets, language)


def evaluate(verdict, panels=(PANEL,), sheets=None, language="en-US"):
    return qa_for(sheets, language)._evaluate(verdict, list(panels))


def clean_verdict(panels=(PANEL,), overall=9) -> dict:
    return exact_verdict(qa_for()._expected(list(panels)), overall)


def item(text, found=True, exact=True, **extra) -> dict:
    return {"text": text, "found": found, "exact": exact, **extra}


def ok_characters(*names) -> list[dict]:
    return [{"name": n, "present": True, "matches_reference": True, "note": ""} for n in names]


def one_panel(items, panel=PANEL, overall=9, **fields) -> dict:
    return {"panels": [{"panel_index": 1, "text_items": items, "characters": ok_characters(*panel.characters), **fields}],
            "overall": overall}


def with_field(field, value, verdict=None) -> dict:
    v = copy.deepcopy(verdict or clean_verdict())
    v["panels"][0][field] = value
    return v


def close_to(pixel, color, tolerance=45) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(pixel, color))


class ScriptedJudge:
    """Vision router for one-panel jobs: judgement k starts from an all-exact verdict scored ``overall[k]`` (default 9)
    and is then edited in place by ``tweaks[k]`` (``every`` applies to judgements without their own tweak)."""

    def __init__(self, tweaks=None, overall=None, every=None):
        self.tweaks, self.overall, self.every, self.calls = dict(tweaks or {}), dict(overall or {}), every, 0

    def __call__(self, prompt, messages, config):
        self.calls += 1
        verdict = exact_verdict(json.loads(EXPECTED_RE.search(prompt).group(1)), self.overall.get(self.calls, 9))
        tweak = self.tweaks.get(self.calls, self.every)
        if tweak is not None:
            tweak(verdict["panels"][0])
        return verdict


async def run_one_panel_job(job_env, judge: ScriptedJudge):
    job_env.paper.num_panels = 1          # panel 1: papi "Line for panel 1 please." + narration + label "Attention"
    job_env.vision.router = judge
    state = await job_env.run(qa=True)
    with Image.open(job_env.output_dir / state["job_id"] / "panel_01.jpg") as im:
        kept_color = im.convert("RGB").getpixel((im.width // 2, im.height // 2))
    return state, state["panels"][0], kept_color


# FakeImageEngine colours panel 1's k-th drawing PALETTE[k]: this tells which attempt's image was kept.
FIRST_ATTEMPT, SECOND_ATTEMPT = PALETTE[1], PALETTE[2]


# ============================================================================ S2: content-based matching of judge items

def test_s2_bubble_left_out_by_the_judge_fails_under_its_own_text():
    result = evaluate(one_panel([item(PAPI), item(NARRATION)]))        # kumo's bubble is not in the image
    assert result.needs_retry and (result.text_exact, result.text_total) == (2, 3)
    assert result.issues == [f'kumo\'s bubble must read exactly "{KUMO}" (not found in the image)']


def test_s2_items_listed_in_reading_order_are_matched_by_text():
    v = one_panel([item("The Transformer reached 28.5 BLEU.", exact=False, rendered="The Transformer reached 28.5 BLEU."),
                   item(PAPI), item(KUMO)])
    result = evaluate(v)
    assert result.needs_retry and (result.text_exact, result.text_total) == (2, 3)
    assert result.issues == [f'the narration box must read exactly "{NARRATION}" (image shows: The Transformer reached 28.5 BLEU.)']


def test_s2_items_that_only_carry_rendered_text_are_matched():
    v = one_panel([{"rendered": t, "found": True, "exact": True} for t in (NARRATION, KUMO, PAPI)])
    result = evaluate(v)
    assert not result.needs_retry and (result.text_exact, result.text_total) == (3, 3)


def test_s2_listed_bubble_keeps_its_own_item_when_a_label_is_similar():
    panel = make_panel(1, characters=["papi", "kumo"], dialogue=[("papi", PAPI), ("kumo", KUMO)], narration=NARRATION, labels=["Attention"])
    v = one_panel([item("Attention"), item(PAPI, found=False, exact=False), item(KUMO), item(NARRATION)], panel=panel)
    result = evaluate(v, [panel])
    assert result.needs_retry and result.text_exact == 2
    assert result.issues == [f'papi\'s bubble must read exactly "{PAPI}" (missing or altered)']


def test_s2_strip_verdict_is_matched_by_panel_index_not_list_position():
    left = make_panel(1, characters=["papi"], dialogue=[("papi", "Left line.")])
    right = make_panel(2, characters=["kumo"], dialogue=[("kumo", "Right line.")], narration="Right narration.")
    v = clean_verdict([left, right])
    v["panels"].reverse()                                   # the judge lists the right panel first ...
    v["panels"][0]["panel_index"], v["panels"][1]["panel_index"] = "2", "1"   # ... with string indices
    v["panels"][0]["text_items"][0].update(exact=False, rendered="Rihgt line.")
    result = evaluate(v, [left, right], sheets={})
    assert result.needs_retry and (result.text_exact, result.text_total) == (2, 3) and result.characters_ok
    assert result.issues == ['kumo\'s bubble must read exactly "Right line." (image shows: Rihgt line.)']


async def test_s2_job_retry_feedback_names_the_bubble_the_judge_left_out(job_env):
    def omit_bubble(panel_result):
        panel_result["text_items"] = [it for it in panel_result["text_items"] if not it["text"].startswith("Line for panel")]

    state, panel, kept = await run_one_panel_job(job_env, ScriptedJudge({1: omit_bubble}))
    assert panel["attempts"] == 2 and panel["qa"]["text_exact"] == panel["qa"]["text_total"] == 2
    assert 'papi\'s bubble must read exactly "Line for panel 1 please." (not found in the image)' in job_env.image.calls[1]["prompt"]
    assert close_to(kept, SECOND_ATTEMPT)


@pytest.mark.parametrize("language, dialogue, narration, label, reported, exact", [
    pytest.param("en-US", [("papi", PAPI), ("kumo", KUMO)], NARRATION, "Attention", [KUMO, NARRATION, "Attention"], 2, id="en"),
    pytest.param("zh-CN", [("papi", "注意力就是一切。")], "", "注意力", ["注意力"], 0, id="zh"),
])
def test_s2_missing_bubble_cannot_pass_by_borrowing_a_label_item(language, dialogue, narration, label, reported, exact):
    panel = make_panel(1, characters=["papi", "kumo"], dialogue=dialogue, narration=narration, labels=[label])
    result = evaluate(one_panel([item(t) for t in reported], panel=panel), [panel], language=language)
    assert result.needs_retry and result.text_exact == exact
    assert any(dialogue[0][1] in issue for issue in result.issues)


def test_s2_feedback_names_the_missing_line_not_a_similar_drawn_one():
    panel = make_panel(1, characters=["papi", "kumo"], dialogue=[("papi", "Look at the numbers."), ("kumo", "Look at the chart!")])
    result = evaluate(one_panel([item("Look at the chart!")], panel=panel), [panel])
    assert result.needs_retry and result.text_exact == 1
    assert result.issues == ['papi\'s bubble must read exactly "Look at the numbers." (not found in the image)']


def test_s2_text_less_items_do_not_force_a_false_retry():
    result = evaluate(one_panel([{"found": True, "exact": True}] * 3))
    assert not result.needs_retry
    assert not any("not found in the image" in issue for issue in result.issues)


def test_s2_strip_panel_missing_from_the_verdict_is_not_a_clean_pass():
    right = make_panel(2, characters=["nezu"], dialogue=[("nezu", "Prove it.")])
    v = {"panels": [{"panel_index": 2, "text_items": [item("Prove it.")], "characters": ok_characters("nezu")}], "overall": 9}
    result = evaluate(v, [PANEL, right])
    assert result.needs_retry
    assert any(PAPI in issue or "panel 1" in issue.lower() for issue in result.issues)


def test_s2_single_panel_verdict_with_a_wrong_panel_index_is_still_used():
    v = clean_verdict()
    v["panels"][0]["panel_index"] = 0
    v["panels"][0]["text_items"][1].update(exact=False, rendered="Even for long sentence?")
    result = evaluate(v)
    assert result.needs_retry and result.text_exact == 2


# ============================================================================ S6: score parsing and string booleans

@pytest.mark.parametrize("overall, score, retry", [
    (9, 9.0, False), ("9", 9.0, False), ("8 out of 10", 8.0, False), ("8/10", 8.0, False), ("8.5/10", 8.5, False),
    ("Score: 4", 4.0, True), ("4 (bubble cut off)", 4.0, True), (5.99, 5.99, True), (6, 6.0, False), ("0", 0.0, True),
])
def test_s6_score_strings_are_parsed(overall, score, retry):
    result = evaluate(clean_verdict(overall=overall))
    assert result.score == score and result.needs_retry is retry


@pytest.mark.parametrize("overall", [None, "N/A", "", "ten", "absent"])
def test_s6_missing_or_unreadable_score_never_forces_a_retry(overall):
    v = clean_verdict()
    if overall == "absent":
        v.pop("overall")
    else:
        v["overall"] = overall
    result = evaluate(v)
    assert not result.needs_retry and result.score == 0.0 and result.issues == []


@pytest.mark.parametrize("value, truthy, falsy", [
    (True, True, False), (False, False, True), ("true", True, False), ("false", False, True), ("FALSE", False, True),
    (" yes ", True, False), ("no", False, True), ("1", True, False), ("0", False, True), (1, True, False), (0, False, True),
    (None, False, False), ("", False, False), ("partially", False, False),
])
def test_s6_string_booleans(value, truthy, falsy):
    assert _truthy(value) is truthy and _falsy(value) is falsy


def test_s6_string_flags_on_text_items():
    v = clean_verdict()
    for it in v["panels"][0]["text_items"]:
        it.update(found="true", exact="false")
    failed = evaluate(v)
    assert failed.needs_retry and failed.text_exact == 0 and len(failed.issues) == 3

    for it in v["panels"][0]["text_items"]:
        it.update(found="True", exact="true", speaker_ok="true")
    for c in v["panels"][0]["characters"]:
        c.update(present="true", matches_reference="yes")
    passed = evaluate(v)
    assert not passed.needs_retry and passed.text_exact == 3 and passed.characters_ok and passed.layout_ok


@pytest.mark.parametrize("field, value, issue", [
    ("present", "false", "kumo is missing from the panel"),
    ("matches_reference", "no", "kumo does not match the reference sheet"),
])
def test_s6_string_false_character_flags_count(field, value, issue):
    v = clean_verdict()
    v["panels"][0]["characters"][1][field] = value
    result = evaluate(v)
    assert not result.characters_ok and result.needs_retry and result.issues == [issue]


async def test_s6_fenced_reply_with_string_flags_and_text_score_passes_through_judge():
    v = clean_verdict(overall="8 out of 10")
    for it in v["panels"][0]["text_items"]:
        it.update(found="true", exact="true", speaker_ok="true")
    reply = "Here is the verdict.\n```json\n" + json.dumps(v) + "\n```"
    result = await qa_for(engine=FakeEngine([reply])).judge(ImageContent.from_base64(TINY_PNG_B64), [PANEL])
    assert (result.score, result.text_exact, result.text_total, result.needs_retry) == (8.0, 3, 3, False)


async def test_s6_job_accepts_a_text_score_and_string_flags_in_one_attempt(job_env):
    def stringify(panel_result):
        for it in panel_result["text_items"]:
            it.update(found="true", exact="true")
        for c in panel_result["characters"]:
            c.update(present="true", matches_reference="true")

    state, panel, _ = await run_one_panel_job(job_env, ScriptedJudge(overall={1: "8 out of 10"}, every=stringify))
    assert state["status"] == "completed" and panel["attempts"] == 1 and state["cost"]["images_generated"] == 1
    assert panel["qa"] == {"score": 8.0, "text_exact": 2, "text_total": 2, "issues": [], "available": True}


# ============================================================================ S7: cast-name matching and the noise filter

@pytest.mark.parametrize("index, name", [(0, "Papi (dog professor)"), (1, "Kumo the cloud"), (0, "PAPI"), (0, "Papi's"), (0, "Professor Papi")])
def test_s7_expected_character_named_with_a_qualifier_is_recognised(index, name):
    v = clean_verdict()
    v["panels"][0]["characters"][index]["name"] = name
    result = evaluate(v)
    assert result.characters_ok and not result.needs_retry and result.issues == []
    v["panels"][0]["characters"][index]["present"] = False            # and it is still checked
    assert not evaluate(v).characters_ok


def test_s7_unexpected_cast_member_with_a_qualifier_is_flagged():
    v = clean_verdict()
    v["panels"][0]["characters"].append({"name": "Nezu (hedgehog)", "present": True, "matches_reference": True})
    result = evaluate(v)
    assert not result.characters_ok and result.issues == ["remove the unexpected character: nezu"]


def test_s7_character_expected_only_in_the_other_strip_panel_is_unexpected():
    left = make_panel(1, characters=["papi"], dialogue=[("papi", "Left line.")])
    right = make_panel(2, characters=["kumo"], dialogue=[("kumo", "Right line.")])
    v = clean_verdict([left, right])
    v["panels"][0]["characters"].append({"name": "Kumo (cloud)", "present": True, "matches_reference": True})
    result = evaluate(v, [left, right])
    assert not result.characters_ok and result.issues == ["remove the unexpected character: kumo"]


@pytest.mark.parametrize("extra, flagged", [
    ("Papi (dog professor)", False),           # the judge repeating an expected character
    ("None", False), ("n/a", False), ("", False),
    ("a small human child", True), ("Nezu the hedgehog", True), ("Two additional human child characters", True),
])
def test_s7_extra_characters(extra, flagged):
    result = evaluate(with_field("extra_characters", [extra]))
    assert result.characters_ok is (not flagged) and result.needs_retry is flagged


@pytest.mark.parametrize("note", [
    "All text is gibberish", "None of the bubbles have tails; text floats in the air", "All expected text is missing",
    "Gibberish unreadable text present on the open book Papi is holding",
])
def test_s7_real_problems_are_not_noise(note):
    assert not _is_noise(note)
    assert note in evaluate(with_field("issues", [note])).issues


@pytest.mark.parametrize("note", [   # the first three are verbatim notes from real judge runs
    "No critical issues; only minor acceptable addition of glasses to Papi which fits the professor context",
    "Speech bubble tail correctly points to Papi, no incorrect characters, no misspelled text, no cut off text",
    "No defects found: all expected text is exact, character matches reference perfectly, no extra content, all bubbles/labels are correctly placed",
    "No issues", "none", "",
])
def test_s7_purely_positive_or_empty_notes_are_dropped(note):
    assert evaluate(with_field("issues", [note])).issues == []


@pytest.mark.parametrize("extra", [
    "none of the cast, but a human child stands behind papi", "a human child standing next to Papi", "a second Papi on the right",
])
def test_s7_extra_that_only_mentions_an_expected_character_is_flagged(extra):
    result = evaluate(with_field("extra_characters", [extra]))
    assert not result.characters_ok and result.needs_retry


@pytest.mark.parametrize("extra", ["None found", "None visible", "No extra characters"])
def test_s7_placeholder_extra_characters_do_not_force_a_retry(extra):
    result = evaluate(with_field("extra_characters", [extra]))
    assert result.characters_ok and not result.needs_retry and result.issues == []


@pytest.mark.parametrize("note", [
    "Text matches, but the tail points at Kumo",
    "Speech bubble tail points correctly at Nezu, however the top text bubble has NO speech tail",
    "Papi is drawn correctly but appears twice",
])
def test_s7_mixed_notes_keep_the_problem(note):
    assert note in evaluate(with_field("issues", [note])).issues


@pytest.mark.parametrize("problem", ["missing", "off_model"])
def test_s7_sheetless_cast_characters_are_still_checked(problem):
    sheets = {}
    panel = make_panel(1, characters=["mochi", "pip"], dialogue=[("mochi", PAPI)])
    v = clean_verdict([panel])
    if problem == "missing":
        v["panels"][0]["characters"][1]["present"] = False
    else:
        v["panels"][0]["characters"][0].update(matches_reference=False, note="drawn as a dog")
    result = evaluate(v, [panel], sheets=sheets)
    assert not result.characters_ok and result.needs_retry


# ============================================================================ S9: stray/misspelled text, wrong tails, attempt rank

def test_s9_prompt_asks_for_stray_text_and_tails():
    prompt = qa_for()._prompt([PANEL])
    assert '"stray_text": []' in prompt and '"speaker_ok": true' in prompt and '"Transfomer"' in prompt


def test_s9_misspelled_stray_text_forces_a_retry_with_targeted_feedback():
    result = evaluate(with_field("stray_text", ["Transfomer"]))
    assert result.needs_retry and not result.layout_ok
    assert (result.text_exact, result.text_total, result.characters_ok, result.score) == (3, 3, True, 9.0)
    assert result.issues == ["remove or correct the stray text: Transfomer"] and result.feedback() == result.issues[0]


@pytest.mark.parametrize("entry", ["", "  ", "None", "none", "N/A", "-"])
def test_s9_empty_stray_text_entries_are_ignored(entry):
    result = evaluate(with_field("stray_text", [entry]))
    assert not result.needs_retry and result.layout_ok and result.issues == []


def test_s9_wrong_tail_forces_a_retry_but_narration_and_label_flags_are_ignored():
    panel = make_panel(1, characters=["papi", "kumo"], dialogue=[("papi", PAPI), ("kumo", KUMO)], narration=NARRATION, labels=["Transformer"])
    v = clean_verdict([panel])
    items = v["panels"][0]["text_items"]
    items[2]["speaker_ok"], items[3]["speaker_ok"] = False, "false"        # narration box, label
    assert evaluate(v, [panel]).layout_ok
    items[1]["speaker_ok"] = False                                         # kumo's bubble
    result = evaluate(v, [panel])
    assert result.needs_retry and not result.layout_ok and result.text_exact == 3
    assert result.issues == [f'the tail of kumo\'s bubble "{KUMO}" must point at kumo']


def add_stray(panel_result):
    panel_result["stray_text"] = ["Transfomer"]


async def test_s9_rank_layout_beats_score_and_the_retry_names_the_stray_text(job_env):
    state, panel, kept = await run_one_panel_job(job_env, ScriptedJudge({1: add_stray}, overall={1: 9, 2: 7}))
    assert panel["attempts"] == 2 and panel["qa"] == {"score": 7.0, "text_exact": 2, "text_total": 2, "issues": [], "available": True}
    assert close_to(kept, SECOND_ATTEMPT)
    assert "remove or correct the stray text: Transfomer" in job_env.image.calls[1]["prompt"]


async def test_s9_rank_exact_text_beats_clean_layout(job_env):
    def garble_bubble(panel_result):
        panel_result["text_items"][0].update(exact=False, rendered="Lne for panel 1 please.")

    state, panel, kept = await run_one_panel_job(job_env, ScriptedJudge({1: add_stray, 2: garble_bubble}))
    assert panel["attempts"] == 2 and panel["qa"]["text_exact"] == 2
    assert panel["qa"]["issues"] == ["remove or correct the stray text: Transfomer"]
    assert close_to(kept, FIRST_ATTEMPT)


async def test_s9_rank_fixed_tail_wins_a_score_tie(job_env):
    def wrong_tail(panel_result):
        panel_result["text_items"][0]["speaker_ok"] = False

    state, panel, kept = await run_one_panel_job(job_env, ScriptedJudge({1: wrong_tail}))
    assert panel["attempts"] == 2 and panel["qa"]["issues"] == [] and close_to(kept, SECOND_ATTEMPT)
    assert "the tail of papi's bubble" in job_env.image.calls[1]["prompt"]


@pytest.mark.parametrize("entry", ["Transfomer (correct spelling: Transformer)", "x-axis reads 'Transfomer', the other labels are correct"])
def test_s9_stray_text_with_a_positive_aside_still_forces_a_retry(entry):
    assert evaluate(with_field("stray_text", [entry])).needs_retry


@pytest.mark.parametrize("entry", ["No stray text", "None found", "none visible"])
def test_s9_placeholder_stray_text_does_not_force_a_retry(entry):
    result = evaluate(with_field("stray_text", [entry]))
    assert not result.needs_retry and result.layout_ok and result.issues == []


async def test_s9_placeholder_stray_text_does_not_double_the_image_cost(job_env):
    state, panel, _ = await run_one_panel_job(job_env, ScriptedJudge(every=lambda pr: pr.update(stray_text=["No stray text"])))
    assert panel["attempts"] == 1 and state["cost"]["images_generated"] == 1


def test_s9_stray_text_objects_count():
    assert evaluate(with_field("stray_text", [{"text": "Transfomer", "where": "chart x-axis"}])).needs_retry


def test_s9_typo_reported_only_in_issues_forces_a_retry():
    assert evaluate(with_field("issues", ["Chart x-axis label has a typo: 'Transfomer'"])).needs_retry
