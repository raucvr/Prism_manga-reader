"""Round-3 regression tests for services/qa_judge.py.

Covers: a usable panel verdict with empty text_items still applies every check (only an unusable verdict is
"qa unavailable"), string-valued list fields, ``QAResult.available`` (contract C1), the ``cast_names`` argument (C2),
the punctuation rules of P1, the JSON repair round of P4, and S2/S7/S9 cases beyond test_review_qa.py (the last section
holds the round-3 verifier's praise, placeholder and extra-character variants). All engines are fakes.
"""

import copy
import json

import pytest

from engines.base import ImageContent
from services.qa_judge import BLIND_PROMPT, PanelQA, QAResult, _same_text

from .fakes import EXPECTED_RE, TINY_PNG_B64, FakeEngine, exact_verdict, make_panel, make_sheets, sheetless_cast

PAPI, KUMO, NARRATION = "Attention is all you need.", "Even for long sentences?", "The Transformer reached 28.4 BLEU."
PANEL = make_panel(1, characters=["papi", "kumo"], dialogue=[("papi", PAPI), ("kumo", KUMO)], narration=NARRATION)
LEFT = make_panel(1, characters=["papi"], dialogue=[("papi", "Left line.")])
RIGHT = make_panel(2, characters=["kumo"], dialogue=[("kumo", "Right line.")], narration="Right narration.")
NOT_FOUND = [f'papi\'s bubble must read exactly "{PAPI}" (not found in the image)',
             f'kumo\'s bubble must read exactly "{KUMO}" (not found in the image)',
             f'the narration box must read exactly "{NARRATION}" (not found in the image)']
BROKEN = 'Verdict: {"panels": [{"panel_index": 1, text_items: []}], "overall": 9}'   # unquoted key: never parses


def evaluate(verdict, panels=(PANEL,), sheets=None, **kwargs):
    qa = PanelQA(FakeEngine(), make_sheets() if sheets is None else sheets, "en-US", **kwargs)
    return qa._evaluate(verdict, list(panels))


def clean(panels=(PANEL,), overall=9, **fields) -> dict:
    """An all-exact verdict; ``fields`` replace keys of every panel object."""
    v = exact_verdict(PanelQA(FakeEngine(), {}, "en-US")._expected(list(panels)), overall)
    for p in v["panels"]:
        p.update(copy.deepcopy(fields))
    return v


async def judge(*replies, panels=(PANEL,)):
    engine = FakeEngine(list(replies))
    qa = PanelQA(engine, make_sheets(), "en-US")
    result = await qa.judge(ImageContent.from_base64(TINY_PNG_B64), list(panels))
    return result, engine, qa


# ============================================================================ empty text_items in a usable verdict

def test_empty_text_items_with_other_problems_retries_instead_of_qa_unavailable():
    v = clean(overall=2, text_items=[], transcription=[], extra_characters=["a human child next to the desk"],
              issues=["No speech bubble or narration box was drawn"])
    v["panels"][0]["characters"][0]["present"] = False
    result = evaluate(v)
    assert result.available and result.needs_retry and not result.characters_ok
    assert (result.score, result.text_exact, result.text_total) == (2.0, 0, 3)
    assert result.issues == NOT_FOUND + ["papi is missing from the panel", "remove the unexpected character: a human child next to the desk",
                                         "No speech bubble or narration box was drawn"]


def test_empty_text_items_without_a_transcription_fail_every_line():
    result = evaluate(clean(text_items=[], transcription=[]))
    assert result.available and result.needs_retry and (result.text_exact, result.text_total) == (0, 3)
    assert result.issues == NOT_FOUND


def test_empty_text_items_count_only_the_lines_the_transcription_shows():
    result = evaluate(clean(text_items=[], transcription=[PAPI, "Even for long sentence?", NARRATION.rstrip(".")]))
    assert result.available and result.needs_retry and (result.text_exact, result.text_total) == (2, 3)
    assert result.issues == [f'kumo\'s bubble must read exactly "{KUMO}" (image shows: Even for long sentence?)']


def test_empty_text_items_with_a_full_transcription_pass():
    result = evaluate(clean(text_items=[]))          # exact_verdict transcribes every expected text
    assert result.available and not result.needs_retry and (result.text_exact, result.text_total) == (3, 3)


async def test_job_retries_a_panel_whose_verdict_lists_no_text_items(job_env):
    def router(prompt, messages, config):
        verdict = exact_verdict(json.loads(EXPECTED_RE.search(prompt).group(1)), 9)
        if len(job_env.vision.calls) == 1:
            verdict["panels"][0].update(text_items=[], transcription=[])
        return verdict

    job_env.paper.num_panels = 1
    job_env.vision.router = router
    state = await job_env.run(qa=True)
    panel = state["panels"][0]
    assert panel["attempts"] == 2 and panel["qa"]["text_exact"] == panel["qa"]["text_total"] == 2
    assert 'papi\'s bubble must read exactly "Line for panel 1 please." (not found in the image)' in job_env.image.calls[1]["prompt"]


# ============================================================================ C1: QAResult.available

@pytest.mark.parametrize("verdict", [
    {"overall": 9}, {"panels": [], "overall": 9}, {"panels": "none", "overall": 9}, {"panels": [{"panel_index": 1}], "overall": 3},
    {"panels": [{"panel_index": 1, "text_items": ["ok"], "issues": ["none"]}], "overall": 9},
], ids=["no-panels-key", "empty-panels", "panels-string", "empty-panel", "junk-panel"])
def test_verdict_without_usable_panel_data_is_unavailable(verdict):
    result = evaluate(verdict)
    assert (result.available, result.needs_retry, result.text_exact, result.text_total) == (False, False, 0, 0)
    assert result.issues == ["qa unavailable: judge returned no usable panel verdict"]
    assert result.to_public()["available"] is False


async def test_judge_exception_is_unavailable_and_never_retries():
    result, *_ = await judge(RuntimeError("vision model timed out"))
    assert not result.available and not result.needs_retry
    assert result.to_public() == {"score": 0.0, "text_exact": 0, "text_total": 0, "issues": ["qa unavailable: vision model timed out"],
                                  "available": False}


def test_available_is_true_by_default_and_for_a_partial_strip_verdict():
    assert QAResult(score=9, text_exact=1, text_total=1, characters_ok=True).to_public()["available"] is True
    v = clean(panels=(LEFT, RIGHT))
    del v["panels"][0]                                 # the judge describes only the right panel
    result = evaluate(v, [LEFT, RIGHT])
    assert result.available and result.needs_retry and (result.text_exact, result.text_total) == (2, 3)
    assert result.issues == ['papi\'s bubble must read exactly "Left line." (panel 1 not found in the image)']


# ============================================================================ string-valued list fields

@pytest.mark.parametrize("value", ["none", "None", "No stray text", "n/a"])
@pytest.mark.parametrize("field", ["stray_text", "extra_characters", "issues"])
def test_placeholder_strings_in_list_fields_are_ignored(field, value):
    result = evaluate(clean(**{field: value}))
    assert not result.needs_retry and result.layout_ok and result.characters_ok and result.issues == []


def test_string_valued_list_fields_count_once_not_per_letter():
    result = evaluate(clean(stray_text="Transfomer", extra_characters="a human child", issues="Kumo's tail points at Papi"))
    assert result.needs_retry and not result.layout_ok and not result.characters_ok
    assert result.issues == ["remove the unexpected character: a human child", "remove or correct the stray text: Transfomer",
                             "Kumo's tail points at Papi"]


def test_other_odd_field_types_are_ignored():
    result = evaluate(clean(characters="papi", transcription=PAPI, extra_characters={"name": "none"}, stray_text=7, issues=None))
    assert not result.needs_retry and result.text_exact == 3 and result.issues == []
    assert result.transcription == [PAPI]


async def test_string_none_in_stray_text_does_not_double_the_image_cost(job_env):
    def router(prompt, messages, config):
        verdict = exact_verdict(json.loads(EXPECTED_RE.search(prompt).group(1)), 9)
        for panel_result in verdict["panels"]:
            panel_result["stray_text"] = "none"
        return verdict

    job_env.paper.num_panels = 2
    job_env.vision.router = router
    state = await job_env.run(qa=True)
    assert state["status"] == "completed" and state["cost"]["images_generated"] == 2
    assert all(p["attempts"] == 1 and p["qa"]["issues"] == [] for p in state["panels"])


# ============================================================================ C2: explicit cast names

def test_explicit_cast_names_let_a_sheetless_cast_flag_an_unexpected_cast_member():
    cast, sheets = sheetless_cast(), {}
    panel = make_panel(1, characters=["mochi", "pip"], dialogue=[("mochi", PAPI)])
    v = clean(panels=(panel,))
    v["panels"][0]["characters"].append({"name": "Rook (grey crow)", "present": True, "matches_reference": True})
    result = evaluate(v, [panel], sheets=sheets, cast_names=cast.names())
    assert not result.characters_ok and result.needs_retry and result.issues == ["remove the unexpected character: rook"]


def test_sheetless_off_model_feedback_names_the_description_not_a_sheet():
    sheets = {}
    panel = make_panel(1, characters=["mochi", "pip"], dialogue=[("mochi", PAPI)])
    v = clean(panels=(panel,))
    v["panels"][0]["characters"][0].update(matches_reference=False, note="drawn as a dog")
    assert evaluate(v, [panel], sheets=sheets).issues == ["mochi does not match its description drawn as a dog"]


def test_cast_names_replace_the_sheet_names():
    assert PanelQA(FakeEngine(), make_sheets(), "en-US", cast_names=["papi", "kumo", "nezu", "hana"]).cast_names == ["papi", "kumo", "nezu", "hana"]
    assert PanelQA(FakeEngine(), make_sheets(), "en-US").cast_names == ["papi", "kumo", "nezu"]


# ============================================================================ P1: what counts as exact text

@pytest.mark.parametrize("expected, rendered, same", [
    (KUMO, "Even for long sentences？", True),
    ("注意力就是一切。", "注意力就是一切", True),
    ("注意力，就是一切。", "注意力,就是一切.", True),
    (PAPI, "Attention is all\nyou need", True),
    (KUMO, "Even for long sentences", False),
    (KUMO, "Even for long sentences!", False),
    ("真的吗？", "真的吗", False),
    (PAPI, "", False),
    (PAPI, None, False),
])
def test_p1_same_text_ignores_punctuation_width_and_a_final_period_but_not_question_marks(expected, rendered, same):
    assert _same_text(expected, rendered) is same


def test_p1_rendered_text_overrides_a_too_strict_exact_flag():
    v = clean()
    v["panels"][0]["text_items"][1].update(exact=False, rendered="Even for long sentences？")
    v["panels"][0]["text_items"][2].update(exact=False, rendered=NARRATION.rstrip("."))
    result = evaluate(v)
    assert not result.needs_retry and result.text_exact == 3 and result.issues == []
    v["panels"][0]["text_items"][1]["rendered"] = "Even for long sentences"       # the "?" is missing
    assert evaluate(v).needs_retry


def test_p1_prompt_states_the_exactness_rules():
    prompt = PanelQA(FakeEngine(), make_sheets(), "en-US")._prompt([PANEL])
    assert "full-width vs half-width punctuation and a missing final period" in prompt
    assert 'a missing or different "?" or "!" is NOT exact' in prompt


# ============================================================================ P4: one JSON repair round

async def test_p4_unparseable_verdict_gets_one_text_only_repair_round():
    result, engine, qa = await judge(BROKEN, clean())
    assert result.available and not result.needs_retry and (result.text_exact, result.text_total) == (3, 3)
    (first, _), (repair, config) = engine.calls
    assert first[0].images and repair[0].images == []
    assert repair[0].content.startswith("Return the following QA verdict as strictly valid JSON") and BROKEN in repair[0].content
    assert config.temperature == 0.0 and config.thinking is False
    assert qa.usage == {"prompt_tokens": 2000, "completion_tokens": 400}          # both calls are billed


async def test_p4_repair_that_still_fails_is_unavailable():
    result, engine, qa = await judge(BROKEN, "Sorry, I cannot produce JSON.")
    assert not result.available and not result.needs_retry and len(engine.calls) == 2
    assert len(result.issues) == 1 and result.issues[0].startswith("qa unavailable:")
    assert qa.usage == {"prompt_tokens": 2000, "completion_tokens": 400}


# ============================================================================ S2: matching edge cases

def test_s2_text_less_item_among_texted_items_falls_back_to_its_position():
    v = clean()
    v["panels"][0]["text_items"][0] = {"found": True, "exact": True}       # papi's item carries flags only
    result = evaluate(v)
    assert not result.needs_retry and result.text_exact == 3


def test_s2_zero_based_strip_indices_are_shifted():
    v = clean(panels=(LEFT, RIGHT))
    v["panels"][0]["panel_index"], v["panels"][1]["panel_index"] = 0, 1
    v["panels"][1]["text_items"][0].update(exact=False, rendered="Rihgt line.")
    result = evaluate(v, [LEFT, RIGHT])
    assert result.needs_retry and (result.text_exact, result.text_total) == (2, 3) and result.characters_ok
    assert result.issues == ['kumo\'s bubble must read exactly "Right line." (image shows: Rihgt line.)']


def test_s2_strip_numbered_with_storyboard_panel_numbers_is_matched_in_index_order():
    v = clean(panels=(LEFT, RIGHT))
    v["panels"][0]["panel_index"], v["panels"][1]["panel_index"] = 5, 6
    v["panels"].reverse()
    v["panels"][0]["text_items"][1].update(exact=False, rendered="Right narration")   # right panel, final period only
    result = evaluate(v, [LEFT, RIGHT])
    assert not result.needs_retry and (result.text_exact, result.text_total) == (3, 3) and result.characters_ok


# ============================================================================ S7: extra characters and notes

@pytest.mark.parametrize("extra, flagged", [
    ("Professor Papi", False), ("Kumo, the cloud creature", False), ("Papi and Kumo", False),
    ("Papi and Nezu", True), ("duplicate kumo in the background", True), ("a bird sitting on Kumo's head", True),
    ("Papi appears twice", True), ("Papi's double", True), ("None - only the expected characters", False),
    ("No unexpected characters; all characters match their reference sheets", False),
    ("No extra characters, but a human child appears behind Papi", True),
])
def test_s7_extra_character_entries(extra, flagged):
    result = evaluate(clean(extra_characters=[extra]))
    assert result.characters_ok is not flagged and result.needs_retry is flagged
    assert result.issues == ([f"remove the unexpected character: {extra}"] if flagged else [])


def test_s7_character_entry_is_about_the_first_name_it_mentions():
    v = clean()
    v["panels"][0]["characters"].append({"name": "Nezu (standing behind Papi)", "present": True, "matches_reference": True})
    result = evaluate(v)
    assert not result.characters_ok and result.issues == ["remove the unexpected character: nezu"]


def test_s7_note_about_a_missing_speech_tail_is_kept():
    note = "Text is correct; the top bubble has no speech tail"
    assert note in evaluate(clean(issues=[note])).issues


# ============================================================================ S9: stray text and typos in notes

@pytest.mark.parametrize("entry, shown", [
    ({"rendered": "Transfomer"}, "Transfomer"),
    ({"description": "garbled letters on the book"}, "garbled letters on the book"),
    ("Chart label 'Transfomer' (correct spelling 'Transformer')", "Chart label 'Transfomer' (correct spelling 'Transformer')"),
    ("None found; 'Transfomer' on the chart", "None found; 'Transfomer' on the chart"),
])
def test_s9_stray_text_entries_that_count(entry, shown):
    result = evaluate(clean(stray_text=[entry]))
    assert result.needs_retry and not result.layout_ok
    assert result.issues == [f"remove or correct the stray text: {shown}"]


@pytest.mark.parametrize("note", [   # the first is a real judge note without its "No critical issues" prefix
    "All text is exact, all characters match reference sheets, bubble tail is correctly attached to Papi, no cut-off or gibberish text, no unexpected characters.",
    "Text matches exactly, no misspelled, garbled or cut-off text",
])
def test_s9_praise_that_lists_absent_typos_is_dropped_and_never_retries(note):
    result = evaluate(clean(issues=[note]))
    assert not result.needs_retry and result.issues == []


def test_s9_typo_note_forces_a_retry_and_leads_the_free_form_notes():
    real = ("Chart x-axis label has a typo: 'Transfomer' (missing an 's' - correct spelling is Transformer), "
            "this is an unlisted extra text item with a spelling error")          # verbatim from job 20261021_173148_b7dc
    result = evaluate(clean(issues=["Papi's glasses are slightly thicker than on the sheet", real]))
    assert result.needs_retry and not result.layout_ok
    assert result.issues == [real, "Papi's glasses are slightly thicker than on the sheet"]


# ============================================================================ retry feedback

def test_strip_feedback_puts_text_problems_of_both_panels_first():
    v = clean(panels=(LEFT, RIGHT))
    left, right = v["panels"]
    left["characters"][0]["present"] = False
    left.update(extra_characters=["a human child", "a second Papi"], issues=["The background is too busy"])
    right["text_items"][0].update(exact=False, rendered="Rihgt line.")
    result = evaluate(v, [LEFT, RIGHT])
    assert result.feedback().split("; ")[0] == 'kumo\'s bubble must read exactly "Right line." (image shows: Rihgt line.)'


def test_identical_problems_reported_for_both_strip_panels_appear_once():
    result = evaluate(clean(panels=(LEFT, RIGHT), stray_text=["Transfomer"]), [LEFT, RIGHT])
    assert result.issues == ["remove or correct the stray text: Transfomer"]


def test_low_score_without_listed_problems_feeds_back_the_worst_problem():
    v = clean(overall=4)
    v["worst_problem"] = "Papi's face is distorted"
    result = evaluate(v)
    assert result.needs_retry and result.feedback() == "Papi's face is distorted"
    v["worst_problem"] = "None"
    assert evaluate(v).issues == []


# ============================================================================ verifier round 3: praise and placeholder variants

@pytest.mark.parametrize("note", [
    "Nothing is misspelled", "No text is garbled", "None of the text is garbled or misspelled", "Text is free from typos",
    "Typos: none", "Misspellings: none found", "There aren't any typos", "Text does not contain any gibberish",
])
def test_praise_that_negates_a_typo_never_retries(note):
    result = evaluate(clean(issues=[note]))
    assert not result.needs_retry and result.layout_ok and result.text_exact == 3


@pytest.mark.parametrize("note", [
    "Nothing is misspelled except the chart label 'Transfomer'", "Typos: 'Transfomer' on the chart",
    "The narration is not missing, but the label is misspelled",
])
def test_a_typo_next_to_negated_praise_still_retries(note):
    result = evaluate(clean(issues=[note]))
    assert result.needs_retry and not result.layout_ok and result.issues == [note]


async def test_job_keeps_a_perfect_first_attempt_whose_note_praises_the_spelling(job_env):
    def router(prompt, messages, config):
        first = len(job_env.vision.calls) == 1
        verdict = exact_verdict(json.loads(EXPECTED_RE.search(prompt).group(1)), 10 if first else 7)
        if first:
            verdict["panels"][0]["issues"] = ["Nothing is misspelled"]
        return verdict

    job_env.paper.num_panels = 1
    job_env.vision.router = router
    state = await job_env.run(qa=True)
    assert state["cost"]["images_generated"] == 1 and state["panels"][0]["qa"]["score"] == 10.0


@pytest.mark.parametrize("note", [
    "Kumo's leaves don't match the reference sheet", "Papi doesn't match his sheet: grey fur", "Kumo is drawn larger than expected",
])
def test_problem_notes_with_a_positive_word_stay_in_the_feedback(note):
    assert evaluate(clean(issues=[note])).issues == [note]


@pytest.mark.parametrize("extra", [
    "None besides Papi and Kumo", "No extra characters besides Papi and Kumo", "No other characters besides Papi",
    "none other than Papi and Kumo", "Only the expected characters (Papi, Kumo)",
])
def test_extra_entry_that_names_only_the_expected_cast_as_the_exception_is_a_placeholder(extra):
    result = evaluate(clean(extra_characters=[extra]))
    assert result.characters_ok and not result.needs_retry and result.issues == []


@pytest.mark.parametrize("extra", [
    "Papi's student (a human boy)", "Kumo's friend, a small bird", "a Papi-like dog", "Papi appears 2 times", "Kumo x2",
    "Two Kumo creatures", "Kumo and a human child", "None besides Papi, but a human child stands behind him",
])
def test_extra_entry_about_someone_else_than_the_expected_character_is_flagged(extra):
    result = evaluate(clean(extra_characters=[extra]))
    assert not result.characters_ok and result.needs_retry
    assert result.issues == [f"remove the unexpected character: {extra}"]


@pytest.mark.parametrize("extra", [
    "Kumo (light-blue cloud creature with two green leaves)", "Papi (panel 1)", "Papi (dog professor and mentor)", "Papi (Kumo's teacher)",
])
def test_the_expected_character_with_a_description_is_not_an_extra(extra):
    result = evaluate(clean(extra_characters=[extra]))
    assert result.characters_ok and not result.needs_retry and result.issues == []


def test_item_marked_not_found_fails_even_if_rendered_copies_the_expected_text():
    v = clean()
    v["panels"][0]["text_items"][0].update(found=False, exact=False, rendered=PAPI, note="bubble missing")
    result = evaluate(v)
    assert result.needs_retry and result.text_exact == 2
    assert result.issues == [f'papi\'s bubble must read exactly "{PAPI}" (bubble missing)']


# ============================================================================ integration: cast looks, lone surrogates

def test_cast_looks_describe_the_characters_without_a_sheet():
    """A sheet-less cast: the judge is told how the cast looks, so it does not report an expected character as an extra."""
    cast, sheets = sheetless_cast(), {}
    qa = PanelQA(FakeEngine(), sheets, "en-US", cast_names=cast.names(), cast_looks={m.name: m.look for m in cast.members})
    prompt = qa._prompt([make_panel(1, characters=["mochi", "pip"], dialogue=[("mochi", PAPI)])])
    looks = "; ".join(f"{m.name} = {m.look}" for m in cast.members)
    assert f"You are a strict QA judge for an educational manga. Character descriptions: {looks}. Image 1 is the generated page" in prompt


def test_cast_looks_only_describe_characters_that_have_no_sheet():
    looks = {"papi": "a dog", "kumo": "a cloud", "nezu": "a hedgehog."}
    with_looks = PanelQA(FakeEngine(), make_sheets(), "en-US", cast_looks=looks)._prompt([PANEL])
    assert with_looks == PanelQA(FakeEngine(), make_sheets(), "en-US")._prompt([PANEL])   # every member has a sheet
    partial = PanelQA(FakeEngine(), make_sheets(("papi", "kumo")), "en-US", cast_looks=looks)._prompt([PANEL])
    assert "Character descriptions: nezu = a hedgehog. Image 3 is the generated page" in partial


async def test_a_broken_emoji_in_the_verdict_never_reaches_the_retry_feedback():
    reply = json.dumps(clean(issues=["the tail points at papi \ud83d"]))   # sent as the escape "\ud83d"
    result, engine, qa = await judge(reply)
    assert "the tail points at papi \ufffd" in result.issues
    result.feedback().encode("utf-8")   # the next Seedream prompt carries it


async def test_the_judge_repair_round_never_sends_a_lone_surrogate():
    result, engine, qa = await judge(BROKEN.replace("Verdict:", "Verdict \ud83d:"), clean())
    assert result.available and len(engine.calls) == 2
    engine.calls[1][0][0].content.encode("utf-8")


# ============================================================================ wrong tails reported only in issues[], score scales

def test_a_wrong_tail_reported_only_in_issues_triggers_a_retry():
    """Real verdict (job 20261021_184517_9b53 p14): speaker_ok stayed true, but issues named the wrong tail."""
    note = "Speech bubble tail for line \"Even for long sentences?\" is incorrectly pointing at Papi, this line is supposed to be spoken by Kumo"
    result = evaluate(clean(issues=[note], overall=8))
    assert (result.needs_retry, result.layout_ok) == (True, False)
    assert note in result.feedback()


@pytest.mark.parametrize("note", ["Speech bubble tail correctly points to Papi, no incorrect characters, no misspelled text",
                                  "All tails point at the right speakers"])
def test_praise_about_tails_is_not_a_tail_error(note):
    result = evaluate(clean(issues=[note]))
    assert (result.needs_retry, result.layout_ok) == (False, True)


@pytest.mark.parametrize("overall, score", [("On a scale of 1-10: 8", 8.0), ("8 out of 10", 8.0), ("8/10", 8.0),
                                            ("7.5 (1 to 10 scale)", 7.5), (9, 9.0), ("scale 1–10, overall 4", 4.0)])
def test_the_score_scale_is_not_read_as_the_score(overall, score):
    result = evaluate(clean(overall=overall))
    assert result.score == score
    assert result.needs_retry == (score < 6)


# ============================================================================ blind transcription cross-check

JA = make_panel(1, characters=["papi"], dialogue=[("papi", "GLUEで過去の全手法を上回った")],
                narration="F1 93.2点 SQuAD1.1で人間平均を上回りました。", labels=["GLUE"])


def blind_router(texts, verdict=None):
    """Vision fake: the QA prompt gets ``verdict`` (default: all exact), the blind prompt gets ``texts``."""
    def route(prompt, messages, config):
        if BLIND_PROMPT[:60] in prompt:
            assert len(messages[0].images) == 1      # the panel only: sheets carry their own lettering
            return texts if isinstance(texts, (str, BaseException)) else {"texts": texts}
        return verdict if verdict is not None else exact_verdict(PanelQA(FakeEngine(), {}, "ja-JP")._expected([JA]))
    return route


async def blind_judge(texts, verdict=None, panels=(JA,), language="ja-JP"):
    engine = FakeEngine(router=blind_router(texts, verdict))
    qa = PanelQA(engine, make_sheets(), language, blind_check=True)
    return await qa.judge(ImageContent.from_base64(TINY_PNG_B64), list(panels)), engine, qa


async def test_a_wrong_kanji_the_judge_passed_is_caught_by_the_blind_reading():
    """Real case (BERT ja strip 9-10): the judge passed 上辿った as 上回った; read without the script it was reported as drawn."""
    result, engine, _ = await blind_judge(["F1 93.2点 SQuAD1.1で人間平均を上回りました。", "GLUE", "GLUEで過去の全手法を上辿った"])
    assert (result.needs_retry, result.text_exact, result.text_total) == (True, 1, 2)
    assert result.issues[0] == 'papi\'s bubble must read exactly "GLUEで過去の全手法を上回った" (image shows: GLUEで過去の全手法を上辿った)'
    assert len(engine.calls) == 2


async def test_the_blind_reading_ignores_line_breaks_punctuation_width_and_a_final_period():
    result, _, _ = await blind_judge(["F1 93.2点 SQuAD1.1で人間平均を\n上回りました", "ＧＬＵＥで過去の\n全手法を上回った。"])
    assert (result.needs_retry, result.text_exact, result.issues) == (False, 2, [])


async def test_a_dropped_character_is_caught():
    """Real case (ResNet zh panel 10): '这东西不止能用在分类任务上？' was drawn without 用 and passed as exact."""
    zh = make_panel(1, characters=["nezu"], dialogue=[("nezu", "这东西不止能用在分类任务上？")])
    verdict = exact_verdict(PanelQA(FakeEngine(), {}, "zh-CN")._expected([zh]))
    result, _, _ = await blind_judge(["这东西不止能\n在分类任务上?"], verdict=verdict, panels=(zh,), language="zh-CN")
    assert result.needs_retry and result.text_exact == 0
    assert "image shows: 这东西不止能 在分类任务上?" in result.issues[0]


@pytest.mark.parametrize("texts", ["not json at all", RuntimeError("vision timeout"), [], {"texts": "x"}])
async def test_a_failed_blind_reading_leaves_the_verdict_alone(texts):
    result, _, _ = await blind_judge(texts if not isinstance(texts, dict) else ["x"][:0])
    assert (result.needs_retry, result.text_exact, result.available) == (False, 2, True)


async def test_a_line_the_judge_already_failed_is_not_counted_twice():
    verdict = exact_verdict(PanelQA(FakeEngine(), {}, "ja-JP")._expected([JA]))
    verdict["panels"][0]["text_items"][0].update(exact=False, rendered="GLUEで過去の全手法を上辿った")
    result, _, _ = await blind_judge(["GLUEで過去の全手法を上辿った", "F1 93.2点 SQuAD1.1で人間平均を上回りました。"], verdict=verdict)
    assert (result.text_exact, result.text_total) == (1, 2)
    assert sum("GLUEで過去の全手法を上回った" in i for i in result.issues) == 1


async def test_labels_are_not_required_in_the_blind_reading_and_usage_is_counted():
    result, engine, qa = await blind_judge(["GLUEで過去の全手法を上回った", "F1 93.2点 SQuAD1.1で人間平均を上回りました。"])
    assert not result.needs_retry                       # the "GLUE" label is missing from the reading: labels never retry
    assert qa.usage == {"prompt_tokens": 2000, "completion_tokens": 400}


async def test_the_blind_reading_is_off_unless_asked_for():
    engine = FakeEngine(router=blind_router(["wrong"]))
    result = await PanelQA(engine, make_sheets(), "ja-JP").judge(ImageContent.from_base64(TINY_PNG_B64), [JA])
    assert not result.needs_retry and len(engine.calls) == 1


async def test_an_unavailable_verdict_is_not_turned_into_a_retry_by_the_blind_reading():
    result, _, _ = await blind_judge(["something else"], verdict="no json here")
    assert result.available is False and result.needs_retry is False


# ============================================================================ adopting a wording Seedream always draws

from services.qa_judge import adopt_wording, agreed_alternatives, wording_gate  # noqa: E402


def mismatch(expected, judge, kind="bubble", index=0):
    return {"panel": 1, "kind": kind, "index": index, "expected": expected, "judge": judge, "blind": None}


def test_both_readings_agreeing_on_a_synonym_make_a_candidate():
    """Real case (BERT ja p14): 莫大な was drawn as 巨大な in 10 of 10 renders; judge and blind reader both saw 巨大な,
    the reader splitting the bubble into its two drawn lines."""
    m = mismatch("大型版は莫大な計算資源が必要", "大型版は巨大な計算資源が必要")
    qa = QAResult(8, 1, 2, True, mismatches=[m], blind_texts=["大型版は巨大な", "計算資源が必要", "またこの論文では英語でのみ検証されています。"])
    assert agreed_alternatives(qa) == [m]


@pytest.mark.parametrize("judge, blind", [
    (None, ["大型版は巨大な計算資源が必要"]),                       # the judge passed the line: only one reader saw a change
    ("大型版は巨大な計算資源が必要", None),                          # no blind reading
    ("大型版は巨大な計算資源が必要", ["大型版は厄大な計算資源が必要"]),   # the readers disagree
    ("大型版は巨大な計算資源が必要", ["大型版は莫大な計算資源が必要"]),   # the blind reader sees the script's wording
])
def test_one_reading_alone_is_not_enough(judge, blind):
    qa = QAResult(8, 1, 2, True, mismatches=[mismatch("大型版は莫大な計算資源が必要", judge)], blind_texts=blind)
    assert agreed_alternatives(qa) == []


@pytest.mark.parametrize("expected, drawn", [
    ("莫大な計算量", "巨大な計算量"), ("大型版は莫大な計算資源が必要", "大型版は巨大な計算資源が必要"),
    ("この手法は画期的だ", "この手法は革新的だ"), ("海量的数据", "庞量的数据"),
])
def test_the_gate_lets_plain_ideograph_synonyms_through(expected, drawn):
    assert wording_gate(expected, drawn)


@pytest.mark.parametrize("expected, drawn", [
    ("RNNは並列化できない", "RNNは並列化できる"), ("BLEUで従来を上回った", "BLEUで従来を下回った"),
    ("学習データは約3万文", "学習データは約3千文"), ("速度は3倍になった", "速度は3割になった"), ("1回の学習は12時間", "1回の学習は12分間"),
    ("它不能并行计算", "它能并行计算了"), ("误差降低了一半", "误差提高了一半"), ("参数量达到1亿个", "参数量达到1万个"),
    ("注意力机制是核心", "注意机制是核心"), ("误差<5%，很稳定", "误差>5%，很稳定"), ("计算量为10¹⁹次", "计算量为10¹⁸次"),
    ("Error < 5% now!", "Error > 5% now!"), ("Cost grows as n² here", "Cost grows as n³ here"), ("全ての層で使う", "一部の層で使う"),
    ("Really?", "Really."), ("这东西不止能用在分类任务上？", "这东西不止能在分类任务上?"), ("速度提高了三倍", "速度提高了两倍"),
    ("タスク毎に全部モデル作り直し", "タスク毎に全部モデル作る直し"), ("双方向事前学習がNLPの常識を変えました。", "双方向事前学習がNLPの常識は変えました。"),
    ("精度が大幅に向上した", "精度が大幅に上昇した"), ("The model wins.", "The model won."),
])
def test_the_gate_stops_numbers_negation_direction_kana_and_english(expected, drawn):
    assert not wording_gate(expected, drawn)


@pytest.mark.parametrize("expected, drawn, adopted", [
    ("大型版は莫大な計算資源が必要", "大型版は巨大な\n計算資源が必要", "大型版は巨大な計算資源が必要"),
    ("这东西不止能用在分类任务上？", "这东西不止能用于分类任务上?", "这东西不止能用于分类任务上？"),   # the script's ？ stays
    ("F1 93.2点 SQuAD1.1で莫大な差", "F1 93.2点SQuAD1.1で巨大な差", "F1 93.2点 SQuAD1.1で巨大な差"),   # and its spaces
])
def test_adopt_wording_replaces_ideographs_only(expected, drawn, adopted):
    assert adopt_wording(expected, drawn) == adopted


async def test_the_blind_reading_is_attached_to_the_judges_mismatches():
    verdict = exact_verdict(PanelQA(FakeEngine(), {}, "ja-JP")._expected([JA]))
    verdict["panels"][0]["text_items"][0].update(exact=False, rendered="GLUEで過去の全手法を上まわった")
    result, _, _ = await blind_judge(["GLUEで過去の全手法を上まわった", "F1 93.2点 SQuAD1.1で人間平均を上回りました。"], verdict=verdict)
    assert result.mismatches == [{"panel": 1, "kind": "bubble", "index": 0, "expected": "GLUEで過去の全手法を上回った",
                                  "judge": "GLUEで過去の全手法を上まわった", "blind": "GLUEで過去の全手法を上まわった"}]


# ============================================================================ review of the solo changes: edge cases

@pytest.mark.parametrize("note", [
    "All bubble tails point where they should.", "Each tail points to the character it belongs to.",
    "None of the tails are wrong.", "No tail is misattributed.",
    "Kumo's visual attributes differ slightly from the reference sheet",
    "Speech bubble tail for the line points correctly at Nezu, however the top text bubble has NO speech tail",
])
def test_praise_negations_and_attributes_are_not_tail_errors(note):
    assert evaluate(clean(issues=[note])).layout_ok


@pytest.mark.parametrize("note", [
    "Speech bubble tail for line \"Even for long sentences?\" is incorrectly pointing at Papi",
    "The tail of Kumo's bubble points at Papi instead of Kumo",
    "Wrong tail: Papi's line is attributed to Kumo's bubble tail",
])
def test_a_wrong_tail_in_one_clause_is_caught(note):
    result = evaluate(clean(issues=[note]))
    assert (result.layout_ok, result.needs_retry) == (False, True)


@pytest.mark.parametrize("lines, drawn", [
    (["Self-attention compares every word pair."], ["Self-\nattention compares\nevery word pair."]),
    (["Pre-trained once, fine-tuned per task."], ["Pre-trained once, fine-", "tuned per task."]),
    (["No retraining—just better prompts."], ["No retraining—\njust better prompts."]),
    (["A transformer reads it all at once."], ["A trans-\nformer reads it all at once."]),
])
async def test_a_hyphen_or_dash_at_a_line_break_reads_both_ways(lines, drawn):
    panel = make_panel(1, characters=["papi"], dialogue=[("papi", lines[0])])
    verdict = exact_verdict(PanelQA(FakeEngine(), {}, "en-US")._expected([panel]))
    result, _, _ = await blind_judge(drawn, verdict=verdict, panels=(panel,), language="en-US")
    assert (result.needs_retry, result.text_exact) == (False, 1)


def test_a_dropped_trailing_ellipsis_is_a_text_error_but_a_dropped_period_is_not():
    assert not _same_text("Wait…", "Wait") and not _same_text("Wait...", "Wait")
    assert _same_text("Done.", "Done") and _same_text("完成了。", "完成了")


@pytest.mark.parametrize("overall, score", [("Score (1-10): 8", 8.0), ("scale of 10: 4", 4.0), ("8 (1-10)", 8.0)])
def test_more_score_scale_forms(overall, score):
    assert evaluate(clean(overall=overall)).score == score


def test_a_missing_score_is_not_a_low_score():
    result = evaluate(clean(overall="n/a"))
    assert (result.low_score, result.needs_retry) == (False, False)


async def test_a_blind_reading_that_saw_almost_nothing_is_ignored():
    result, _, _ = await blind_judge(["。"])
    assert (result.needs_retry, result.text_exact, result.blind_texts) == (False, 2, None)


async def test_the_feedback_quotes_the_whole_bubble_when_the_reader_splits_it():
    result, _, _ = await blind_judge(["F1 93.2点 SQuAD1.1で人間平均を上回りました。", "GLUEで過去の", "全手法を上辿った"])
    assert result.issues[0].endswith("(image shows: GLUEで過去の 全手法を上辿った)")


from services.qa_judge import same_facts  # noqa: E402


@pytest.mark.parametrize("a, b, same", [
    ("大きいBERTは事前学習に莫大な計算が必要", "大きいBERTは事前学習に膨大な計算が必要", True),
    ("参数量达到1亿个", "参数量达到1万个", False),
    ("速度提高了三倍", "速度提高了两倍", False),
    ("F1 93.2点で人間を上回った", "人間を上回ったF1 93.2点", True),           # order may change
    ("BERT beats GPT-1 by 7.7%", "BERT beats GPT by 7.7%", False),
    ("Cost grows as n² here", "Cost grows as n³ here", False),
    ("误差<5%", "误差>5%", False),
])
def test_same_facts(a, b, same):
    assert same_facts(a, b) is same


# ============================================================================ second review of the solo changes

@pytest.mark.parametrize("line", ["Use only attention.", "152层ResNet的Top-5误差仅4.49%", "It didn't train, sadly."])
def test_a_quoted_line_does_not_hide_a_wrong_tail(line):
    note = f'Speech bubble tail for line "{line}" is incorrectly pointing at Kumo, this line is supposed to be spoken by Papi'
    assert evaluate(clean(issues=[note])).layout_ok is False


@pytest.mark.parametrize("overall, score", [("1-10: 9", 9.0), ("[1-10] 9", 9.0), ("Score 1–10: 9", 9.0), ("1 to 10: 9", 9.0)])
def test_bare_score_ranges(overall, score):
    assert evaluate(clean(overall=overall)).score == score


async def test_a_word_hyphenated_across_joined_lines_reads_both_ways():
    panel = make_panel(1, characters=["papi"], dialogue=[("papi", "A transformer reads it all at once.")])
    verdict = exact_verdict(PanelQA(FakeEngine(), {}, "en-US")._expected([panel]))
    result, _, _ = await blind_judge(["A trans- former reads it all at once."], verdict=verdict, panels=(panel,), language="en-US")
    assert (result.needs_retry, result.text_exact) == (False, 1)


async def test_a_caption_copied_from_a_reference_sheet_is_stray_text():
    """The blind reader sees only the page, so 'Soft Spikes' there was really drawn (unlike a judge confusing images)."""
    result, _, _ = await blind_judge(["GLUEで過去の全手法を上回った", "F1 93.2点 SQuAD1.1で人間平均を上回りました。", "Soft Spikes"])
    assert (result.layout_ok, result.needs_retry) == (False, True)
    assert "remove the caption copied from the reference sheet: Soft Spikes" in result.feedback()


@pytest.mark.parametrize("expected, drawn", [
    ("左側しか見れないの？", "右側しか見れないの？"), ("网络越深越好", "网络越浅越好"), ("训练误差", "测试误差"),
    ("过拟合", "欠拟合"), ("可能会更好", "必能会更好"), ("原始输入", "原始输出"),
])
def test_the_gate_stops_direction_and_antonym_swaps(expected, drawn):
    assert not wording_gate(expected, drawn)


@pytest.mark.parametrize("a, b, same", [
    ("34层ResNet稳定优于18层ResNet", "18层ResNet稳定优于34层ResNet", False),   # numbers swapped
    ("SQuAD v1.1で+1.5%", "SQuAD v1.1で-1.5%", False),                       # sign flipped
    ("本当に速いの？", "本当に速いの。", False),                                # a question became a statement
    ("内訳：80%マスク、10%ランダム", "内訳：10%マスク、80%ランダム", False),
    ("大型版は莫大な計算資源が必要", "大型版は膨大な計算資源が必要", True),
])
def test_same_facts_keeps_number_order_signs_and_questions(a, b, same):
    assert same_facts(a, b) is same


async def test_a_wrong_label_counts_only_when_the_blind_reading_agrees():
    """QA used to ignore labels: 'original + new' drawn as 'original' shipped with a 9.0."""
    verdict = exact_verdict(PanelQA(FakeEngine(), {}, "ja-JP")._expected([JA]))
    verdict["panels"][0]["text_items"][2].update(exact=False, rendered="GLU")      # the "GLUE" label
    lines = ["GLUEで過去の全手法を上回った", "F1 93.2点 SQuAD1.1で人間平均を上回りました。"]
    wrong, _, _ = await blind_judge(lines + ["GLU"], verdict=verdict)
    assert (wrong.layout_ok, wrong.needs_retry) == (False, True) and 'the label must read exactly "GLUE"' in wrong.feedback()
    fine, _, _ = await blind_judge(lines + ["GLUE"], verdict=verdict)          # the blind reader sees the label right
    assert (fine.layout_ok, fine.needs_retry) == (True, False)
    alone = evaluate(verdict, panels=(JA,))                                    # no blind reading: labels still never retry
    assert alone.needs_retry is False
