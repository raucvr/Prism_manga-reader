"""services.storyboarder_v2, v3 story layer: abstract and its claims, story prompt, deterministic lint, fix rounds,
fact-check of board strings, the abstract-quiz gate, translation of boards, and v2 left as it was.

Fakes only; services.boards, services.evidence and services.abstract_quiz are used as they are."""

from __future__ import annotations

import json
import sys

import pytest

from services.paper_extractor import PaperText
from services.storyboarder_v2 import (
    BEAT_CHAPTER,
    _salvage_board,
    _story_board_issues,
    board_issues,
    _words_vs_chart,
    number_lines,
    scrub_unsupported,
    Bubble,
    Cast,
    CastMember,
    PanelV2,
    StoryboarderV2,
    StoryboardV2,
    brief_prompt_v3,
    coverage,
    find_abstract,
    fix_prompt,
    lint_v3,
    paper_examples,
    quote_in,
    split_sentences,
    story_prompt,
    story_range,
    translate_prompt,
    verify_claims,
    visible_text,
)

from .fakes import (
    BRIEF_V3,
    LONG_PAPER_TEXT,
    V3_CLAIMS,
    V3_EXAMPLE,
    V3_PAPER_TEXT,
    V3_RESULT,
    FakeEngine,
    PaperModel,
    fixed_panel,
    make_cast,
)


def v3_paper(text: str = V3_PAPER_TEXT) -> PaperText:
    return PaperText("Attention Is All You Need", text, ["Table 2: BLEU scores on the English-to-German newstest2014 test."])


def config_of(engine: FakeEngine, opening: str):
    return next(cfg for messages, cfg in engine.calls if messages[-1].content.startswith(opening))


def prompt_of(engine: FakeEngine, opening: str) -> str:
    return next(p for p in engine.prompts if opening in p)


def v3_panel(n: int, *, kind: str = "scene", beat: str = "how", dialogue=(("papi", "Attention looks at every word."),),
             narration: str = "", board=None, claims=(), characters=("papi", "kumo")) -> PanelV2:
    return PanelV2(n=n, chapter=BEAT_CHAPTER[beat], title=f"Panel {n}", characters=list(characters), scene="A study.",
                   visual="papi points.", dialogue=[Bubble(s, t) for s, t in dialogue], narration=narration, kind=kind,
                   board=board, beat=beat, claims=list(claims))


def v3_storyboard(panels, claims=V3_CLAIMS, brief=None) -> StoryboardV2:
    return StoryboardV2(title="T", summary="S", theme="kumomo", language="en-US", chapters=[], panels=panels, brief=dict(brief or {}),
                        version=3, abstract_claims=[dict(c) for c in claims])


# ----------------------------------------------------------------------------- abstract

def test_find_abstract_stops_at_footnotes_venue_and_first_section():
    body = "\n".join([
        "Attention Is All You Need", "Ashish Vaswani∗ Noam Shazeer∗", "Abstract",
        "The dominant sequence transduction models are based on complex recurrent or",
        "convolutional neural networks. Our model achieves 28.4 BLEU on the WMT 2014",
        "8 GPUs",   # a wrapped line that looks like a numbered heading but continues the sentence
        "and generalizes well.",
        "∗Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started",
        "the effort to evaluate this idea.",
        "31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.",
        "1 Introduction", "Recurrent neural networks ...",
    ])
    abstract = find_abstract(body)
    assert abstract.startswith("The dominant sequence transduction models") and abstract.endswith("and generalizes well.")
    assert "28.4 BLEU on the WMT 2014 8 GPUs" in abstract
    assert "Equal contribution" not in abstract and "Conference" not in abstract and "Introduction" not in abstract


@pytest.mark.parametrize("lines, expected", [
    (["Title", "Abstract—We propose a model. It works.", "1. Introduction", "Text."], "We propose a model. It works."),
    (["Title", "ABSTRACT We propose a model.", "I. INTRODUCTION"], "We propose a model."),
    (["Title", "A B S T R A C T", "We propose a model.", "Keywords: attention"], "We propose a model."),
    # a figure placed in the abstract's column: its heading ends the abstract
    (["Abstract", "We explore prompting.", "It helps.", "Standard Prompting", "Q: Roger has 5 tennis balls."], "We explore prompting. It helps."),
    (["Abstract", "We explore prompting.", "Example sentence shown in a figure: It is in this spirit ."], "We explore prompting."),
    (["Title", "Abstract reasoning is hard for models.", "1 Introduction"], ""),   # a sentence, not a heading
    (["Title", "1 Introduction", "No abstract here."], ""),
])
def test_find_abstract_variants(lines, expected):
    assert find_abstract("\n".join(lines)) == expected


def test_find_abstract_on_real_papers(attention_paper, cot_paper):
    attention = find_abstract(attention_paper.body)
    assert attention.startswith("The dominant sequence transduction models")
    assert attention.endswith("both with large and limited training data.") and "Equal contribution" not in attention
    assert "28.4 BLEU" in attention and "41.8" in attention
    cot = find_abstract(cot_paper.body)
    assert cot.startswith("We explore how generating a chain of thought") and cot.endswith("GPT-3 with a verifier.")
    assert "Standard Prompting" not in cot and "Roger" not in cot


def test_split_sentences_keeps_numbers_and_abbreviations():
    text = "We reach 28.4 BLEU, e.g. on WMT 2014. It costs 3.5 days. Fig. 3 shows heads."
    assert split_sentences(text) == ["We reach 28.4 BLEU, e.g. on WMT 2014.", "It costs 3.5 days.", "Fig. 3 shows heads."]


# ----------------------------------------------------------------------------- claims and examples

ABSTRACT = ("The dominant models are recurrent. We propose the Transformer, based solely on attention mechanisms. "
            "Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best "
            "results by over 2 BLEU. It also generalizes to English constituency parsing.")


def test_verify_claims_checks_quotes_reanchors_drops_and_covers_every_sentence():
    raw = [
        {"id": 7, "text": "28.4 BLEU on English-German", "quote": "achieves 28.4 BLEU on the WMT 2014 English-to-German translation task"},  # lost hyphen
        {"id": 0, "text": "Attention only", "quote": "We propose the Transformer, based solely on attention mechanisms"},
        {"id": 1, "text": "Works on parsing", "quote": "It also generalizes very well to English constituency parsing tasks"},  # paraphrased quote
        {"id": 2, "text": "Invented", "quote": "Our model wins every benchmark ever made by a huge margin"},                   # not in the abstract
        "The dominant models are recurrent.",                                                                                   # a bare string
    ]
    claims = verify_claims(raw, ABSTRACT, body="unused when there is an abstract")
    assert [c["id"] for c in claims] == list(range(len(claims)))
    assert [c["quote"] for c in claims] == [   # abstract order, not reply order
        "The dominant models are recurrent.",
        "We propose the Transformer, based solely on attention mechanisms",
        # the results sentence's "over 2 BLEU" was in no quote: the whole sentence becomes a claim of its own
        "Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results by over 2 BLEU.",
        "achieves 28.4 BLEU on the WMT 2014 English-to-German translation task",   # verified despite the lost hyphen
        "It also generalizes to English constituency parsing.",                    # paraphrased quote re-anchored on its sentence
    ]
    assert "Invented" not in [c["text"] for c in claims]    # dropped: nothing in the abstract supports it
    assert [c["text"] for c in claims][3:] == ["28.4 BLEU on English-German", "Works on parsing"]
    assert all(quote_in(c["quote"], ABSTRACT) for c in claims)


def test_verify_claims_without_an_abstract_checks_quotes_against_the_paper():
    claims = verify_claims([{"text": "a", "quote": "the big transformer model outperforms the best previously reported models"},
                            {"text": "b", "quote": "this sentence is nowhere in the paper at all"}], "", LONG_PAPER_TEXT)
    assert [c["text"] for c in claims] == ["a"]


def test_paper_examples_keeps_only_examples_found_in_the_paper():
    examples = paper_examples([
        {"text": "Example sentence shown in a figure: " + V3_EXAMPLE, "source": "Figure 3"},
        {"text": "A made-up example the paper never shows."}, V3_EXAMPLE, 42,
    ], V3_PAPER_TEXT)
    assert examples == [{"text": V3_EXAMPLE, "source": "Figure 3"}]


def test_story_range_keeps_v3_stories_at_16_to_24_panels_unless_asked_for_more():
    assert story_range((16, 32)) == (16, 24)
    assert story_range((12, 18)) == (16, 18)
    assert story_range((3, 6)) == (16, 16)
    assert story_range((28, 40)) == (28, 40)


# ----------------------------------------------------------------------------- prompts

def test_brief_prompt_v3_shows_the_abstract_and_asks_for_claims_examples_and_glossary():
    prompt = brief_prompt_v3(v3_paper(), "THE ABSTRACT TEXT.")
    assert prompt.startswith("Read the paper and its ABSTRACT and write a compact, faithful BRIEF")
    assert "ABSTRACT:\nTHE ABSTRACT TEXT." in prompt and V3_PAPER_TEXT in prompt
    for needle in ('"abstract_claims": [{"id": 0, "text"', '"quote"', "EVERY sentence", '"examples"', "Example sentence shown in a figure:",
                   '"glossary"', "higher or lower is better"):
        assert needle in prompt, needle
    assert '"points"' not in prompt   # v3 measures coverage on the abstract claims
    assert "no abstract heading was found" in brief_prompt_v3(v3_paper(), "")


def test_story_prompt_carries_claims_beats_boards_budgets_and_fidelity_rules():
    brief = {**BRIEF_V3, "examples": [{"text": V3_EXAMPLE, "source": "Figure 3"}]}
    prompt = story_prompt(brief, V3_CLAIMS, make_cast(), (16, 24), v3_paper())
    assert prompt.startswith("Turn this research paper into a SHORT MANGA STORY")
    assert json.dumps(V3_CLAIMS, ensure_ascii=False) in prompt and V3_PAPER_TEXT in prompt
    assert f"- {V3_EXAMPLE} (Figure 3)" in prompt and '"from_paper": true' in prompt
    for beat in ("hook", "problem", "idea", "how", "proof", "limits", "payoff", "recap"):
        assert f"{beat} (" in prompt, beat
    for kind in ("bar", "table", "flow", "compare", "links", "list"):
        assert f'{kind}: {{"kind": "{kind}"' in prompt, kind
    for needle in ("at most 3 speech bubbles of at most 16 words", "narration at most 20 words", "1-2 cast members",
                   "1-2 speech bubbles of at most 12 words", "narration at most 14 words", "16-24 panels",
                   "at most 44 words of bubbles and narration in total", '"baseline": true', '"open": true',
                   "claims (exactly 1 board panel", "bets something small", "No verdict words", "every claim must ALSO be taught",
                   '"paper_label"', "prior best", "single model or ensemble", "Never compute, round, convert",
                   "which direction is better", "Never overgeneralize", "Limitations only as the authors state them",
                   "nezu is a sharp reviewer", "kumo is the reader's stand-in", '"claims"'):
        assert needle in prompt, needle
    assert '"abstract_claims"' not in prompt.split("BRIEF (checked against the paper):", 1)[1].split("PAPER TEXT:", 1)[0]


def test_story_prompt_uses_the_casts_roles_and_labels_an_illustrative_example():
    cast = Cast(theme="custom", style="s", members=[CastMember("mochi", "mentor", "l", "v"), CastMember("pip", "student", "l", "v"),
                                                    CastMember("rook", "skeptic", "l", "v")])
    prompt = story_prompt({**BRIEF_V3, "examples": []}, V3_CLAIMS, cast, (14, 18), v3_paper())
    assert "pip is the reader's stand-in" in prompt and "rook is a sharp reviewer" in prompt and "mochi knows the paper" in prompt
    assert "The paper shows no example input" in prompt and '"from_paper": false' in prompt


# ----------------------------------------------------------------------------- lint

TABLE = {"kind": "table", "title": "WMT 2014 translation (BLEU)", "columns": ["Task", "BLEU"],
         "rows": [{"cells": ["English-German", "28.4"], "quote": "establishing a new state-of-the-art BLEU score of 28.4"},
                  {"cells": ["English-French", "41.0"], "quote": "our big model achieves a BLEU score of 41.0"}], "source": "Section 6"}
RECAP = {"kind": "list", "title": "What the paper showed", "items": ["Attention only", "28.4 BLEU on WMT 2014 English-German"]}
CLAIMS_WITH_RESULT = V3_CLAIMS + [{"id": 3, "text": "28.4 BLEU, over 2 BLEU better", "quote": V3_RESULT}]


def clean_panels() -> list[PanelV2]:
    return [
        v3_panel(1, beat="hook", narration="Translation turns one sentence into another.", claims=[0]),
        v3_panel(2, beat="idea", dialogue=[("papi", "Only attention: no recurrence, no convolutions.")], claims=[1, 2]),
        v3_panel(3, kind="board", beat="proof", board=TABLE, dialogue=[("nezu", "28.4, more than 2.0 above the rest?")], claims=[3],
                 characters=("nezu", "papi")),
        v3_panel(4, kind="board", beat="recap", board=RECAP, dialogue=[("nezu", "Fine. Convinced.")], claims=[0, 1, 2, 3],
                 characters=("nezu",)),
    ]


def test_lint_passes_a_clean_script():
    assert lint_v3(v3_storyboard(clean_panels(), CLAIMS_WITH_RESULT), V3_PAPER_TEXT) == {}


def test_lint_catches_an_unsupported_number_and_an_uncovered_claim():
    panels = clean_panels()
    panels[0].narration = "Recurrent models needed 99.9 hours to translate a sentence."
    for p in panels:
        p.claims = [c for c in p.claims if c != 2]
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    problems = lint_v3(sb, V3_PAPER_TEXT)
    assert any(x.startswith("evidence: narration:") and "99.9" in x for x in problems[1])
    homes = [n for n, found in problems.items() if any(x.startswith("claim 2 is not taught by any panel") for x in found)]
    assert len(homes) == 1 and sb.panels[homes[0] - 1].beat not in ("claims", "recap")   # a story panel, never the recap
    assert set(problems) == {1, homes[0]}


def test_lint_counts_a_claim_on_the_recap_or_claims_board_as_not_taught():
    panels = clean_panels()
    recap = next(p for p in panels if p.beat == "recap")
    for p in panels:
        p.claims = [c for c in p.claims if c != 2 or p is recap]
    if 2 not in recap.claims:
        recap.claims.append(2)
    problems = lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT), V3_PAPER_TEXT)
    assert any(x.startswith("claim 2 is not taught") for found in problems.values() for x in found)
    assert recap.n not in problems


def test_lint_catches_kind_mismatches_budgets_unsupported_board_values_and_missing_key_numbers():
    long_bubble = " ".join(["word"] * 17)
    bad_bar = {"kind": "bar", "title": "Made-up chart", "bars": [{"label": "A", "value": "77.7", "highlight": True}, {"label": "B", "value": "66.6"}]}
    panels = [
        v3_panel(1, beat="hook", dialogue=[("kumo", "a"), ("papi", "b"), ("kumo", "c"), ("papi", "d")], claims=[0, 1, 2]),
        v3_panel(2, kind="board", beat="proof", board=None, claims=[3]),
        v3_panel(3, kind="board", beat="proof", board=bad_bar, dialogue=[("nezu", long_bubble)], characters=("nezu", "papi", "kumo")),
        v3_panel(4, beat="how", board=RECAP, dialogue=[("papi", "x")]),
        v3_panel(5, kind="board", beat="recap", board={"kind": "list", "title": "Recap", "items": ["Attention only", "Fast"]},
                 dialogue=[], claims=[0, 1, 2, 3], characters=("nezu",)),
    ]
    problems = lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT), V3_PAPER_TEXT)
    assert problems[1] == ["4 speech bubbles: at most 3 on a scene panel"]
    assert problems[2][0].startswith("board panel without a board")
    assert any(x.startswith("evidence:") and "77.7" in x for x in problems[3])
    assert "3 cast members on a board panel: at most 2" in problems[3]
    assert "bubble 0 (nezu) has 17 words: at most 12 on a board panel" in problems[3]
    assert problems[4] == ['scene panel with a board: make it a board panel ("kind": "board") or remove the board']
    assert "board panel without a speech bubble: a cast member reacts to the board" in problems[5]
    # claim 3 is listed by panels 2 and 5; the recap (5) does not count, and the story panel (2) shows neither 28.4 nor "over 2"
    assert any(x.startswith("claim 3") and "28.4, 2" in x for x in problems[2])
    assert not any(x.startswith("claim 3") for x in problems[5])


def test_lint_allows_the_numbers_of_an_illustrative_running_example_only():
    panels = clean_panels()
    panels[0].narration = "Kumo bakes a layer cake with 12 recipe cards and waits 45 minutes."
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT, brief={"running_example": {"text": "Kumo bakes a 12 layer cake", "from_paper": False}})
    sb.running_example = "Kumo bakes a 12 layer cake"
    problems = lint_v3(sb, V3_PAPER_TEXT)
    assert list(problems) == [1] and not any("12" in x for x in problems[1]) and any("45" in x for x in problems[1])


def test_fix_prompt_lists_flagged_panels_with_problems_and_attaches_the_paper_when_numbers_are_involved():
    sb = v3_storyboard(clean_panels(), CLAIMS_WITH_RESULT)
    prompt = fix_prompt(sb, make_cast(), {3: ["evidence: rows.0.cells.1: value 28.4 is not supported"]}, paper_text=V3_PAPER_TEXT)
    assert prompt.startswith("Fix the flagged panels of this manga script")
    flagged = json.loads(prompt.split("FLAGGED PANELS:\n", 1)[1].split("\n\n", 1)[0])
    assert [f["panel"]["n"] for f in flagged] == [3] and flagged[0]["problems"] == ["evidence: rows.0.cells.1: value 28.4 is not supported"]
    assert flagged[0]["panel"]["board"]["rows"][0]["quote"] == TABLE["rows"][0]["quote"]
    assert prompt.endswith(V3_PAPER_TEXT) and '"kind": "table", "title": "WMT 2014 translation (BLEU)"}' in prompt   # outline: kind + title
    assert "PAPER TEXT:\n" not in fix_prompt(sb, make_cast(), {1: ["narration has 31 words: at most 30 on a scene panel"]})


# ----------------------------------------------------------------------------- build_v3

async def test_build_v3_produces_a_v3_storyboard_with_boards_claims_chapters_and_quiz():
    model = PaperModel()
    engine = FakeEngine(router=model)
    stages = []
    sb = await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "en-US", (16, 32), on_stage=lambda s, m: stages.append(s))

    assert model.seen == ["brief_v3", "briefcheck", "story", "factcheck", "quiz", "reader", "grade", "grade", "grade"]   # lint found nothing to fix
    assert list(dict.fromkeys(stages)) == ["analyzing", "storyboarding"]
    assert (sb.version, sb.language) == (3, "en-US") and [p.n for p in sb.panels] == list(range(1, 10))
    # the brief's three claims plus the abstract's results sentence, which no claim covered
    assert [c["id"] for c in sb.abstract_claims] == [0, 1, 2, 3] and sb.abstract_claims[3]["quote"] == V3_RESULT
    assert "abstract_claims" not in sb.brief and sb.brief["abstract"].endswith("by over 2 BLEU.")
    assert sb.brief["examples"] == [{"text": V3_EXAMPLE, "source": "Figure 3"}]
    assert sb.brief["running_example"]["from_paper"] is True and V3_EXAMPLE in sb.running_example
    boards = {p.n: p.board for p in sb.panels if p.kind == "board"}
    assert {n: b["kind"] for n, b in boards.items()} == {4: "flow", 6: "table", 9: "list"}
    assert boards[6]["rows"][0] == {"cells": ["English-German", "28.4"], "quote": "establishing a new state-of-the-art BLEU score of 28.4"}
    assert all(p.board is None for p in sb.panels if p.kind == "scene") and all(p.labels == [] for p in sb.panels)
    assert [(c.id, c.panel_numbers) for c in sb.chapters] == [("hook", [1]), ("problem", [2]), ("idea", [3]), ("how", [4]),
                                                             ("evidence", [5, 6]), ("payoff", [8]), ("limits", [7]), ("takeaway", [9])]
    assert [p.chapter for p in sb.panels] == [BEAT_CHAPTER[p.beat] for p in sb.panels]
    assert (sb.quiz["score"], sb.quiz["failed_claim_ids"], len(sb.quiz["questions"])) == (1.0, [], 8) and "usage" not in sb.quiz
    assert coverage(sb) == {"core": [4, 4], "important": [0, 0], "claims": [4, 4]} and sb.brief["lint"] == {}
    saved = sb.to_dict()
    assert (saved["version"], saved["running_example"], saved["quiz"]["score"]) == (3, sb.running_example, 1.0)
    assert saved["panels"][5]["kind"] == "board" and saved["panels"][5]["claims"] == [3]
    # thinking for the brief and the story; the quiz has its own settings
    assert config_of(engine, "Read the paper and its ABSTRACT").thinking is True
    assert config_of(engine, "Turn this research paper").thinking is True


async def test_build_v3_fix_round_repairs_an_unsupported_number_and_an_uncovered_claim():
    def story(payload):
        payload["panels"][1]["narration"] = "Recurrent models needed 99.9 hours for one long sentence."
        for p in payload["panels"]:
            p["claims"] = [c for c in p["claims"] if c != 2]
        return payload

    model = PaperModel(story=story)
    engine = FakeEngine(router=model)
    writer = StoryboarderV2(engine)
    sb = await writer.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))

    assert model.seen.count("fix") == 1
    fix = prompt_of(engine, "Fix the flagged panels")
    flagged = {f["panel"]["n"]: f["problems"] for f in json.loads(fix.split("FLAGGED PANELS:\n", 1)[1].split("\n\n", 1)[0])}
    homes = [n for n, found in flagged.items() if any(x.startswith("claim 2 is not taught") for x in found)]
    assert len(homes) == 1 and set(flagged) == {2, homes[0]} and homes[0] != 9   # a story panel, not the recap (9)
    assert any("99.9" in x for x in flagged[2])
    assert "PAPER TEXT:\n" in fix and config_of(engine, "Fix the flagged panels").thinking is False
    assert "99.9" not in sb.panels[1].narration and 2 in sb.panels[homes[0] - 1].claims
    assert coverage(sb)["claims"] == [4, 4] and sb.brief["lint"] == {}
    assert writer.usage["prompt_tokens"] > 0


async def test_build_v3_demotes_a_board_that_stays_invalid_after_two_fix_rounds():
    made_up = {"kind": "bar", "title": "Made-up speed chart", "highlight": True,
               "bars": [{"label": "Old", "value": "77.7"}, {"label": "New", "value": "66.6", "highlight": True}]}

    def story(payload):
        payload["panels"][5]["board"] = made_up
        return payload

    model = PaperModel(story=story, fix=lambda panel, problems: panel)   # a writer that never fixes anything
    sb = await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))

    assert model.seen.count("fix") == 2
    panel = sb.panels[5]
    assert (panel.kind, panel.board) == ("scene", None)
    assert "Made-up speed chart" not in panel.narration and panel.dialogue   # no caption for a chart nobody sees
    assert all(p.kind == "scene" or p.board for p in sb.panels)


async def test_build_v3_translates_board_strings_and_keeps_numbers_and_quotes():
    model = PaperModel()
    engine = FakeEngine(router=model)
    stages = []
    sb = await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "zh-CN", (16, 32), on_stage=lambda s, m: stages.append(s))

    assert model.seen[-2:] == ["translate", "factcheck"] and list(dict.fromkeys(stages))[-1] == "translating"
    assert sb.language == "zh-CN" and sb.version == 3
    table, recap = sb.panels[5].board, sb.panels[8].board
    assert table["title"] == "板书 2014" and table["columns"] == ["板书", "板书"]
    assert table["rows"][0] == {"cells": ["板书", "28.4"], "quote": "establishing a new state-of-the-art BLEU score of 28.4"}   # values and quotes stay
    assert recap["items"][-1] == "板书 28.4 2014"
    assert sb.panels[3].board["steps"] == ["板书"] * 4
    assert [c.title for c in sb.chapters][:2] == ["第1章", "第2章"]   # chapter titles translated, chapters from beats kept
    prompt = prompt_of(engine, "Translate this manga script")
    payload = json.loads(prompt.rsplit("\n\n", 1)[1])
    assert payload["panels"][5]["board"]["rows.0.cells.0"] == "English-German"
    assert "rows.0.cells.1" not in payload["panels"][5]["board"]   # numbers are not sent for translation
    assert "on a panel with a board, each bubble ≤ 22" in prompt and 'every string value of a panel\'s "board"' in prompt
    assert config_of(engine, "Translate this manga script").thinking is False


async def test_build_v3_quiz_gate_revises_failed_claims_and_stores_both_scores():
    model = PaperModel(quiz_misses=({2},))   # the first reader cannot answer the questions about claim 2
    engine = FakeEngine(router=model)
    sb = await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))

    assert model.seen.count("quiz") == 1 and model.seen.count("reader") == 2 and model.seen.count("revise") == 1
    assert sb.quiz["before"] == {"score": 0.75, "failed_claim_ids": [2]}
    assert (sb.quiz["score"], sb.quiz["failed_claim_ids"]) == (1.0, [])
    new = next(p for p in sb.panels if p.title == "Claim 2")
    assert (new.n, new.beat, new.chapter, new.claims) == (7, "proof", "evidence", [2])   # after the last proof panel
    assert [p.n for p in sb.panels] == list(range(1, 11))
    assert dict((c.id, c.panel_numbers) for c in sb.chapters)["evidence"] == [5, 6, 7]
    revise = prompt_of(engine, "A reader who never saw the paper read only the visible text")
    assert '"reader_answered": "not shown"' in revise and "PAPER TEXT:\n" in revise
    assert model.seen.count("factcheck") == 2   # the revised script is fact-checked again


async def test_build_v3_never_fails_on_the_quiz(monkeypatch):
    import services.abstract_quiz as aq

    async def broken(*_args, **_kwargs):
        raise RuntimeError("quiz exploded")

    monkeypatch.setattr(aq, "run_quiz", broken)
    sb = await StoryboarderV2(FakeEngine(router=PaperModel())).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert sb.quiz["score"] is None and "quiz exploded" in sb.quiz["error"] and len(sb.panels) == 9

    async def no_score(*_args, **_kwargs):   # run_quiz's own failure: no exception, an error in the result
        return {"score": None, "error": "ValueError: no JSON object in model reply", "usage": {}}

    monkeypatch.setattr(aq, "run_quiz", no_score)
    story = StoryboarderV2(FakeEngine(router=PaperModel()))
    sb = await story.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert sb.quiz["score"] is None and any(x["stage"] == "abstract quiz" and "no JSON" in x["error"] for x in story.skipped)

    calls = []

    async def once_broken(*args, **kwargs):   # an intermittent failure: the second run scores
        calls.append(1)
        return await no_score() if len(calls) == 1 else {"score": 1.0, "failed_claim_ids": [], "questions": [], "usage": {}}

    monkeypatch.setattr(aq, "run_quiz", once_broken)
    story = StoryboarderV2(FakeEngine(router=PaperModel()))
    sb = await story.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert len(calls) == 2 and sb.quiz["score"] == 1.0 and not story.skipped

    monkeypatch.setitem(sys.modules, "services.abstract_quiz", None)   # the module cannot be imported at all
    sb = await StoryboarderV2(FakeEngine(router=PaperModel())).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert sb.quiz == {"score": None, "failed_claim_ids": [], "questions": [], "error": "abstract quiz unavailable"}


async def test_build_v3_factcheck_corrects_board_strings_and_sends_whole_board_findings_to_a_fix_round():
    model = PaperModel(factcheck_problems=[
        {"id": "6.b.title", "issue": "the table shows test sets", "corrected": "WMT 2014 test sets (BLEU)"},
        {"id": "6.b.rows.0.cells.1", "issue": "tries to change a value", "corrected": "28.5"},
        {"id": "6.b", "issue": "mixes English-German and English-French in one table", "corrected": "keep only English-German"},
    ])
    engine = FakeEngine(router=model)
    sb = await StoryboarderV2(engine).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))

    assert sb.panels[5].board["title"] == "WMT 2014 test sets (BLEU)"
    assert sb.panels[5].board["rows"][0]["cells"][1] == "28.4"   # a string fix never changes a number
    fixes = {f["id"]: f for f in sb.brief["factcheck"]}
    assert fixes["6.b.title"]["before"] == "WMT 2014 translation (BLEU)" and "6.b.rows.0.cells.1" not in fixes
    check = prompt_of(engine, "You are a meticulous fact-checker")
    items = json.loads(check.split("LINES:\n", 1)[1].split("\n\nPAPER TEXT:", 1)[0])
    ids = {i["id"] for i in items}
    assert {"6.b", "6.b.title", "6.b.rows.0.cells.0", "9.b.items.3", "4.b.steps.0"} <= ids
    assert "mixes different metrics, units, datasets, splits or training setups" in check and '"prior best"' in check
    fix = prompt_of(engine, "Fix the flagged panels")
    assert "fact-check: mixes English-German and English-French in one table Fix: keep only English-German" in fix


async def test_fit_v3_uses_kind_budgets_and_refuses_rewrites_that_change_numbers():
    sb = v3_storyboard([
        v3_panel(1, dialogue=[("papi", " ".join(["word"] * 16) + " 28.4")]),                     # 17 words on a scene panel
        v3_panel(2, kind="board", board=RECAP, dialogue=[("nezu", " ".join(["word"] * 13))], characters=("nezu",)),
        v3_panel(3, dialogue=[("kumo", " ".join(["word"] * 16))]),                                # exactly at the limit
    ])
    fast = FakeEngine([{"items": [{"id": "1.d0", "text": "Short line without the number."}, {"id": "2.d0", "text": "Short board line."}]},
                       {"items": [{"id": "1.d0", "text": "Short line, still 28.4."}]}])
    assert await StoryboarderV2(FakeEngine(), fast_engine=fast).fit(sb, "en-US") == 2
    assert [b.text for p in sb.panels for b in p.dialogue][:2] == ["Short line, still 28.4.", "Short board line."]
    first = json.loads(fast.prompts[0].split("ITEMS:\n", 1)[1])
    assert [(i["id"], i["max"]) for i in first] == [("1.d0", 16), ("2.d0", 12)]


async def test_check_brief_corrects_claim_texts_and_never_drops_a_claim():
    brief = {"key_idea": "k", "abstract_claims": [{"id": 0, "text": "Wrong paraphrase.", "quote": "the abstract's words"},
                                                  {"id": 1, "text": "Unsupported.", "quote": "other abstract words"}]}
    engine = FakeEngine([{"problems": [{"id": "abstract_claims.0", "issue": "overstated", "corrected": "Right paraphrase."},
                                       {"id": "abstract_claims.1", "issue": "unsupported", "corrected": "DELETE"}]}])
    assert await StoryboarderV2(engine).check_brief(brief, PaperText("T", "body", [])) == 2
    assert [c["text"] for c in brief["abstract_claims"]] == ["Right paraphrase.", "other abstract words"]
    assert '"id": "abstract_claims.1"' in engine.prompts[0]


# ----------------------------------------------------------------------------- v2 untouched

async def test_v2_build_is_unaffected_by_the_v3_layer():
    model = PaperModel(num_panels=3)
    engine = FakeEngine(router=model)
    sb = await StoryboarderV2(engine).build(PaperText("T", "body", []), make_cast(), "zh-CN", (3, 6))
    assert model.seen == ["brief", "briefcheck", "storyboard", "factcheck", "translate", "factcheck"]
    assert sb.version == 2 and "version" not in sb.to_dict() and "abstract_claims" not in sb.to_dict()
    assert all(p.kind == "scene" and p.board is None and len(p.dialogue) <= 2 for p in sb.panels)
    assert set(coverage(sb)) == {"core", "important"}
    translate = prompt_of(engine, "Translate this manga script")
    assert "each bubble ≤ 22 characters" in translate and '"board"' not in translate
    check = prompt_of(engine, "You are a meticulous fact-checker")
    assert '"N.b"' not in check and "bubbles ≤ 14 words" in check
    sb2 = StoryboardV2(title="T", summary="S", theme="kumomo", language="en-US", chapters=[], panels=[v3_panel(1)])
    assert '"board"' not in translate_prompt(sb2, "zh-CN")


async def test_build_v3_without_an_abstract_heading_checks_claims_against_the_paper():
    text = "\n".join(line for line in LONG_PAPER_TEXT.splitlines() if line != "Abstract")
    engine = FakeEngine(router=PaperModel())
    sb = await StoryboarderV2(engine).build_v3(v3_paper(text), make_cast(), "en-US", (16, 32))
    assert "no abstract heading was found" in prompt_of(engine, "Read the paper and its ABSTRACT")
    assert [c["quote"] for c in sb.abstract_claims] == [c["quote"] for c in V3_CLAIMS] and sb.brief["abstract"] == ""
    assert sb.brief["examples"] == []   # LONG_PAPER_TEXT has no figure example
    assert sb.brief["running_example"]["from_paper"] is False   # the story claimed Figure 3, which this paper lacks
    assert all(p.claims == [c for c in p.claims if c in (0, 1, 2)] for p in sb.panels)


def test_running_example_from_paper_only_when_a_verified_example_shows_up():
    from services.storyboarder_v2 import _running_example
    brief = {"examples": [{"text": V3_EXAMPLE, "source": "Figure 3"}]}
    shown = [v3_panel(1, dialogue=[("kumo", f"Look: {V3_EXAMPLE}")])]
    assert _running_example({"text": "Kumo reads a sentence.", "from_paper": True}, brief, shown, V3_PAPER_TEXT) == \
        {"text": "Kumo reads a sentence.", "from_paper": True, "source": "Figure 3"}
    other = [v3_panel(1, dialogue=[("kumo", "I bake a cake in twelve steps.")])]
    assert _running_example({"text": "Kumo bakes a cake.", "from_paper": "true"}, brief, other, V3_PAPER_TEXT)["from_paper"] is False
    assert _running_example({"text": V3_EXAMPLE, "from_paper": "false"}, brief, shown, V3_PAPER_TEXT)["from_paper"] is False


# ----------------------------------------------------------------------------- review fixes

COST_PAPER = V3_PAPER_TEXT + "\nThe big model costs 2.3 · 10^19 FLOPs to train on 8 GPUs and won the 1st place."
COST_CLAIM = {"id": 0, "text": "Training costs 2.3 · 10^19 FLOPs on 8 GPUs and won 1st place.",
              "quote": "The big model costs 2.3 · 10^19 FLOPs to train on 8 GPUs and won the 1st place."}


@pytest.mark.parametrize("line, missing", [
    ("Training took 2.3 × 10¹⁹ FLOPs on eight GPUs and won first place.", []),   # superscripts, number words
    ("Training took 2.3e19 FLOPs on 8 GPUs, 1st place.", []),
    ("Training took 2.3 · 10^18 FLOPs on 8 GPUs, 1st place.", ["2.3 · 10^19"]),   # a different figure is not the claim's
    ("Training was cheap on eight GPUs and won first place.", ["2.3 · 10^19"]),
])
def test_lint_reads_a_claims_numbers_the_way_the_boards_do(line, missing):
    panels = [v3_panel(1, dialogue=[("papi", line)], claims=[0]),
              v3_panel(2, kind="board", beat="recap", board={"kind": "list", "title": "Recap", "items": ["Attention only", "Cheap training"]},
                       dialogue=[("nezu", "Fine.")], characters=("nezu",))]
    problems = lint_v3(v3_storyboard(panels, [COST_CLAIM]), COST_PAPER)
    needs = [x for found in problems.values() for x in found if x.startswith("claim 0")]
    assert [x.split("needs its number ")[1].split(" visible")[0] for x in needs] == ([", ".join(missing)] if missing else [])


async def test_build_v3_keeps_a_recap_board_whose_only_problems_the_renderer_copes_with():
    long_item = "28.4 BLEU on WMT 2014 English-German, which is more than 2.0 BLEU above the best earlier models"

    def story(payload):
        payload["panels"][8]["board"]["items"].append(long_item)        # 18 words: over the 14-word limit
        return payload

    model = PaperModel(story=story, fix=lambda panel, problems: panel)   # a writer that never fixes anything
    sb = await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))

    assert model.seen.count("fix") == 2
    recap = sb.panels[8]
    assert recap.kind == "board" and recap.board["items"][-1] == long_item and len(recap.board["items"]) == 5
    assert any("too long" in x for x in sb.brief["lint"]["9"])          # still reported, but the evidence stays drawn


async def test_build_v3_drops_only_the_unsupported_bar_of_a_chart():
    chart = {"kind": "bar", "title": "English-German BLEU", "unit": "BLEU", "bars": [
        {"label": "Transformer (big)", "value": "28.4", "quote": "establishing a new state-of-the-art BLEU score of 28.4", "highlight": True},
        {"label": "ConvS2S Ensemble", "value": "26.36", "quote": "ConvS2S Ensemble 26.36"},
        {"label": "Prior best", "value": "27.1"},                        # nothing in the paper supports it
        {"label": "ByteNet", "value": "23.75", "quote": "ByteNet 23.75"}]}

    def story(payload):
        payload["panels"][5]["board"] = chart
        return payload

    model = PaperModel(story=story, fix=lambda panel, problems: panel)
    sb = await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))

    board = sb.panels[5].board
    assert sb.panels[5].kind == "board" and [b["label"] for b in board["bars"]] == ["Transformer (big)", "ConvS2S Ensemble", "ByteNet"]
    assert "27.1" not in visible_text(sb.panels[5]) and board["bars"][0]["highlight"] is True
    assert sb.brief["lint"] == {}                                        # nothing unsupported is left


def test_a_demoted_board_never_moves_an_unsupported_number_into_the_narration():
    bad = {"kind": "bar", "title": "Transformer is 99.9 times faster",
           "bars": [{"label": "Old", "value": "77.7"}, {"label": "New", "value": "66.6", "highlight": True}]}
    sb = v3_storyboard([v3_panel(1, kind="board", beat="proof", board=bad, dialogue=[("nezu", "Show me.")], narration="Speed matters.")])
    StoryboarderV2(FakeEngine())._demote_boards(sb, V3_PAPER_TEXT)
    assert (sb.panels[0].kind, sb.panels[0].board, sb.panels[0].narration) == ("scene", None, "Speed matters.")


def test_salvage_shows_bars_of_different_units_as_a_table_and_follows_lower_is_better():
    mixed = {"kind": "bar", "title": "Quality and cost", "bars": [
        {"label": "BLEU", "value": "41.0", "quote": "our big model achieves a BLEU score of 41.0", "highlight": True},
        {"label": "Training time", "value": "3.5 days", "quote": "after training for 3.5 days on eight GPUs"}]}
    paper = "Top-5 error of the ensemble is 4.9% on the validation set.\nThe previous best top-5 error was 5.1% on it."
    lower = {"kind": "bar", "title": "Top-5 error (lower is better)", "higher_is_better": True, "bars": [
        {"label": "Ensemble", "value": "4.9%", "quote": "Top-5 error of the ensemble is 4.9%", "highlight": True},
        {"label": "Previous best", "value": "5.1%", "quote": "The previous best top-5 error was 5.1%"}]}
    sb = v3_storyboard([v3_panel(1, kind="board", beat="proof", board=mixed, dialogue=[("nezu", "Hm.")]),
                        v3_panel(2, kind="board", beat="proof", board=lower, dialogue=[("nezu", "Hm.")])])
    writer = StoryboarderV2(FakeEngine())
    writer._demote_boards(v3_storyboard(sb.panels[:1]), V3_PAPER_TEXT)
    writer._demote_boards(v3_storyboard(sb.panels[1:]), paper)
    table = sb.panels[0].board
    assert table["kind"] == "table" and [r["cells"] for r in table["rows"]] == [["BLEU", "41.0"], ["Training time", "3.5 days"]]
    assert table["rows"][1]["quote"] == "after training for 3.5 days on eight GPUs" and table["highlight_rows"] == [0]
    assert sb.panels[1].board["kind"] == "bar" and sb.panels[1].board["higher_is_better"] is False


async def test_build_v3_fix_round_keeps_a_valid_board_that_the_reply_left_out():
    def story(payload):
        for p in payload["panels"]:
            p["claims"] = [c for c in p["claims"] if c != 2]
        return payload

    def lazy_fix(panel, problems):   # fixes the claim but leaves out the board it did not need to touch
        out = fixed_panel(panel, problems)
        out.pop("board", None)
        return out

    model = PaperModel(story=story, fix=lazy_fix)
    sb = await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))

    recap = sb.panels[8]
    assert (recap.kind, recap.board["kind"]) == ("board", "list")   # the board the reply left out is kept
    assert any(2 in p.claims for p in sb.panels if p.beat not in ("claims", "recap"))
    assert model.seen.count("fix") == 1 and sb.brief["lint"] == {}


async def test_build_v3_reads_odd_reply_shapes():
    def story(payload):
        panels = payload["panels"]
        panels[0]["dialogue"] = [f"kumo: {panels[0]['dialogue'][0]['text']}"]          # "speaker: text" strings
        panels[1]["dialogue"] = [[b["speaker"], b["text"]] for b in panels[1]["dialogue"]]   # [speaker, text] pairs
        panels[3]["board"] = json.dumps(panels[3]["board"])                             # the board as a JSON string
        payload["panels"] = {str(p["n"]): {k: v for k, v in p.items() if k != "n"} for p in panels}   # keyed by number
        return payload

    sb = await StoryboarderV2(FakeEngine(router=PaperModel(story=story))).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert [p.n for p in sb.panels] == list(range(1, 10)) and sb.brief["lint"] == {}
    assert [(b.speaker, b.text[:20]) for b in sb.panels[0].dialogue] == [("kumo", "'The animal didn't c")]
    assert [b.speaker for b in sb.panels[1].dialogue] == ["papi", "kumo"]
    assert (sb.panels[3].kind, sb.panels[3].board["kind"]) == ("board", "flow")


async def test_v3_translation_must_keep_the_numbers_of_each_line():
    def storyboard():
        return v3_storyboard([v3_panel(1, dialogue=[("nezu", "28.4 on English-German. Fine.")], claims=[0]),
                              v3_panel(2, dialogue=[("papi", "It trains on 8 GPUs.")], claims=[1])])

    def reply(first: str, second: str = "它在八块 GPU 上训练。") -> dict:
        return {"title": "注意力", "panels": [{"n": 1, "title": "第1格", "dialogue": [{"speaker": "nezu", "text": first}]},
                                              {"n": 2, "title": "第2格", "dialogue": [{"speaker": "papi", "text": second}]}]}

    retry_ok = {"panels": [{"n": 1, "dialogue": [{"speaker": "nezu", "text": "英德翻译拿到 28.4。好吧。"}]}]}
    engine = FakeEngine([reply("英德翻译。好吧。"), retry_ok])          # the first reply drops 28.4
    sb = await StoryboarderV2(engine).translate(storyboard(), "zh-CN")
    assert [b.text for p in sb.panels for b in p.dialogue] == ["英德翻译拿到 28.4。好吧。", "它在八块 GPU 上训练。"]   # 8 → 八 is fine
    assert len(engine.calls) == 2 and '"28.4 on English-German. Fine."' in engine.prompts[1]

    engine = FakeEngine([reply("英德翻译。好吧。"), {"panels": [{"n": 1, "dialogue": [{"speaker": "nezu", "text": "还是没有数字。"}]}]}])
    sb = await StoryboarderV2(engine).translate(storyboard(), "zh-CN")
    assert sb.panels[0].dialogue[0].text == "还是没有数字。"              # twice: Chinese beats an English panel (logged)


def test_a_stray_section_number_is_neither_abstract_text_nor_a_claim():
    body = "\n".join(["Title", "Abstract", "We propose a model. It reaches 28.4 BLEU on the test set.", "1", "Introduction", "Text."])
    assert find_abstract(body) == "We propose a model. It reaches 28.4 BLEU on the test set."
    claims = verify_claims([], "We propose a model that works well. It reaches 28.4 BLEU. 1", body="")
    assert [c["quote"] for c in claims] == ["We propose a model that works well.", "It reaches 28.4 BLEU."]


def test_panel_replies_with_lists_dicts_and_loose_claim_ids_are_read():
    from services.storyboarder_v2 import _claim_ids, _panel_from_reply
    p = _panel_from_reply({"kind": "scene", "narration": ["Fact one.", "Fact two."], "dialogue": {"kumo": "Hi!", "papi": "Hello."},
                           "claims": ["Claim 3", {"id": 4}, 2.0, "C1", True, "Q5"]}, 1)
    assert p.narration == "Fact one. Fact two." and [(b.speaker, b.text) for b in p.dialogue] == [("kumo", "Hi!"), ("papi", "Hello.")]
    assert p.claims == [3, 4, 2, 1] and _claim_ids("-1") == []
    q = _panel_from_reply({"dialogue": ["papi: Note: it works", "Just a line.", "A speaker name that is far too long: text"]}, 2)
    assert [(b.speaker, b.text) for b in q.dialogue] == [("papi", "Note: it works"), ("", "Just a line."),
                                                         ("", "A speaker name that is far too long: text")]


async def test_build_v3_asks_once_more_when_the_story_reply_has_no_panels():
    calls = []

    def story(payload):
        calls.append(1)
        return {"title": "Attention, Please!", "panels": []} if len(calls) == 1 else payload

    model = PaperModel(story=story)
    sb = await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert model.seen.count("story") == 2 and len(sb.panels) == 9

    model = PaperModel(story=lambda payload: {"panels": {}})
    with pytest.raises(ValueError, match="no panels"):
        await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert model.seen.count("story") == 2


async def test_stray_reply_shapes_never_fail_translation_fact_check_or_fit():
    from services.storyboarder_v2 import _normalize_panels, extract_json
    sb = v3_storyboard([v3_panel(1, dialogue=[("nezu", "Attention looks at every word.")], narration="Fact one.")])
    engine = FakeEngine([{"title": ["not", "text"], "chapters": 1.5,
                          "panels": {"1": {"dialogue": [{"speaker": "nezu", "text": ["列表"]}], "narration": "事实一。"}}},
                         {"panels": {"1": {"dialogue": [{"speaker": "nezu", "text": "注意力会看每个词。"}], "narration": "事实一。"}}}])
    sb = await StoryboarderV2(engine).translate(sb, "zh-CN")
    assert sb.title == "T" and sb.panels[0].dialogue[0].text == "注意力会看每个词。"   # a list is no line: re-requested
    checker = FakeEngine([{"problems": {"id": "1.n", "issue": "x", "corrected": "事实二。"}}, {"problems": 0},
                          {"problems": [{"id": "1.d0", "issue": "x", "corrected": ["a", "b"]}]}])
    writer = StoryboarderV2(checker)
    assert await writer.factcheck(sb, v3_paper(), "zh-CN") == 1 and sb.panels[0].narration == "事实二。"
    assert await writer.factcheck(sb, v3_paper(), "zh-CN") == 0
    assert await writer.factcheck(sb, v3_paper(), "zh-CN") == 0 and sb.panels[0].dialogue[0].text == "注意力会看每个词。"
    long = v3_storyboard([v3_panel(1, dialogue=[("papi", " ".join(["word"] * 30))])])
    assert await StoryboarderV2(FakeEngine(), fast_engine=FakeEngine([{"items": 7}, {"items": "x"}])).fit(long, "en-US") == 0
    assert extract_json('{"a": NaN, "b": [Infinity, 1.5]}') == {"a": None, "b": [None, 1.5]}
    panels = [PanelV2.from_dict({"characters": ["kumo"], "dialogue": [{"speaker": "kumo", "text": "  "}, {"speaker": "papi", "text": "**"},
                                                                      {"speaker": "kumo", "text": "ok"}]}, 1)]
    _normalize_panels(panels, make_cast(), max_bubbles=None)
    assert [(b.speaker, b.text) for b in panels[0].dialogue] == [("kumo", "ok")]


def test_salvage_removes_tex_marks_but_keeps_quotes_verbatim():
    board = {"kind": "bar", "title": "**WMT 2014** English-German ($BLEU$)", "bars": [
        {"label": "$Transformer$ (big)", "value": "$28.4$", "quote": "establishing a new state-of-the-art BLEU score of 28.4", "highlight": True},
        {"label": "ByteNet", "value": "23.75", "quote": "ByteNet 23.75"}]}
    sb = v3_storyboard([v3_panel(1, kind="board", beat="proof", board=board, dialogue=[("nezu", "Hm.")])])
    StoryboarderV2(FakeEngine())._demote_boards(sb, V3_PAPER_TEXT)
    kept = sb.panels[0].board
    assert kept["title"] == "WMT 2014 English-German (BLEU)" and [(b["label"], b["value"]) for b in kept["bars"]] == \
        [("Transformer (big)", "28.4"), ("ByteNet", "23.75")]
    assert kept["bars"][0]["quote"] == "establishing a new state-of-the-art BLEU score of 28.4"


def test_lint_flags_props_that_carry_writing_and_verdict_words():
    panels = clean_panels()
    panels[0].visual = "nezu leans forward, pen hovering over her clipboard, a wall clock behind her."
    panels[1].dialogue[0].text = "So the paper proved it works perfectly."
    problems = lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT), V3_PAPER_TEXT)
    assert any("props that carry writing (clipboard, pen hovering, wall clock)" in x for x in problems[1])
    assert any(x.startswith("verdict words perfectly, proved") for x in problems[2])
    panels[1].dialogue[0].text = "Its example reads 'The Law will never be perfect'."   # the paper's own words are no verdict
    assert not any(x.startswith("verdict words") for x in lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT), V3_PAPER_TEXT).get(2, []))


def test_lint_wants_the_running_example_in_the_hook_and_the_payoff():
    panels = clean_panels() + [v3_panel(5, beat="payoff", dialogue=[("kumo", "Now I get it!")])]
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    sb.running_example = f'Kumo reads one sentence: "{V3_EXAMPLE}"'
    problems = lint_v3(sb, V3_PAPER_TEXT)
    assert any(x.startswith("the hook does not show the running example") for x in problems[1])
    assert any(x.startswith("the payoff does not show the running example") and "solved" in x for x in problems[5])
    panels[0].narration = "The animal didn't cross the street: who was too tired?"
    panels[4].dialogue[0].text = "It looks at the animal: the animal was tired, not the street!"
    assert not any("running example" in x for found in lint_v3(sb, V3_PAPER_TEXT).values() for x in found)


def test_lint_wants_named_bars_one_highlight_and_honest_links():
    chart = {"kind": "bar", "title": "GLUE", "source": "Table 1", "bars": [
        {"label": "Prior SOTA", "paper_label": "OpenAI GPT", "value": "72.8", "baseline": True},
        {"label": "BERT base", "value": "79.6", "highlight": True}, {"label": "BERT large", "value": "80.5", "highlight": True}]}
    found = _story_board_issues(chart)
    assert any(x.startswith('bar "Prior SOTA" names no system') and '"OpenAI GPT"' in x for x in found)
    assert any(x.startswith("2 highlighted bars") for x in found)
    named = {**chart, "bars": [{**chart["bars"][0], "label": "OpenAI GPT"}, chart["bars"][2]]}
    assert _story_board_issues(named) == ["only 2 bars: show the field too: add 1-3 other systems from the same table column "
                                          "(same metric, dataset and setting), each with its quote, when the paper lists them"]
    assert _story_board_issues({**named, "bars": named["bars"] + [{"label": "ELMo", "value": "70.0"}]}) == []
    assert _story_board_issues({"kind": "bar", "bars": [{"label": "MoE prior best", "paper_label": "MoE", "value": "40.56"}]}) == []
    links = {"kind": "links", "title": "Actual attention", "note": "Exact weights reproduced from Figure 3", "tokens": ["a", "b"], "from": 0, "weights": [0, 1]}
    assert any(x.startswith("links board claims exact or measured weights") for x in _story_board_issues(links))


def test_number_lines_quote_the_paper_lines_that_hold_the_abstracts_numbers():
    text = number_lines(CLAIMS_WITH_RESULT, V3_PAPER_TEXT)
    assert text.startswith("28.4: ") and '"Transformer (big) 28.4"' in text
    assert "2014" not in text.split(":", 1)[0]                       # years are no key numbers
    assert number_lines([], V3_PAPER_TEXT) == ""


def test_words_and_chart_must_show_the_same_comparison():
    chart = {"kind": "bar", "title": "GLUE", "source": "Table 1", "bars": [
        {"label": "OpenAI GPT", "value": "75.1", "baseline": True}, {"label": "BERT LARGE", "value": "82.1", "highlight": True}]}
    panel = v3_panel(3, kind="board", beat="proof", board=chart, dialogue=[("nezu", "A huge 7.7 point gap.")], characters=("nezu",))
    assert _words_vs_chart(panel) and "7.7" in _words_vs_chart(panel)[0]
    panel.dialogue[0].text = "A 7.0 point gap over OpenAI GPT."              # the drawn gap
    assert _words_vs_chart(panel) == []
    panel.dialogue[0].text = "82.1 points on average."                     # a drawn value
    assert _words_vs_chart(panel) == []


def test_scrub_removes_only_the_sentence_with_a_number_the_paper_lacks():
    panels = clean_panels()
    panels[1].dialogue[0].text = "Only attention. Performance drops 10.7 F1 without it."
    panels[1].narration = "It dropped 99.9 points."
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    removed = scrub_unsupported(sb, V3_PAPER_TEXT)
    assert [b.text for b in sb.panels[1].dialogue] == ["Only attention."] and sb.panels[1].narration == ""
    assert len(removed) == 2 and all(x.startswith("panel 2:") for x in removed)


def test_lint_catches_mixed_setups_and_a_weak_baseline():
    parsing = {"kind": "bar", "title": "Parsing F1", "bars": [
        {"label": "Dyer et al. WSJ only", "value": "91.7", "baseline": True}, {"label": "Transformer WSJ only", "value": "91.3"},
        {"label": "Transformer semi-supervised", "value": "92.7", "highlight": True}]}
    assert any(x.startswith("the chart mixes training data (semi-supervised, wsj only)") for x in _story_board_issues(parsing))
    bleu = {"kind": "bar", "title": "EN-DE BLEU", "bars": [
        {"label": "GNMT Ensemble", "value": "26.30", "baseline": True}, {"label": "ConvS2S Ensemble", "value": "26.36"},
        {"label": "Transformer base", "value": "27.3"}, {"label": "Transformer big", "value": "28.4", "highlight": True}]}
    weak = [x for x in _story_board_issues(bleu) if x.startswith('the baseline "GNMT Ensemble" is not the strongest')]
    assert weak and "ConvS2S Ensemble" in weak[0] and "Transformer base" not in weak[0]   # the paper's own variant is no rival
    bleu["bars"][0]["baseline"], bleu["bars"][1]["baseline"] = False, True
    assert not any("not the strongest" in x for x in _story_board_issues(bleu))


def test_an_unsupported_bar_gets_the_paper_lines_that_name_it():
    paper = ("Figure 2 legend\nPrior best\n  60  55  57\nTable 1: GSM8K solve rate\n"
             "Prior best  N/A (finetuning) 55^(a)  57.4^(b)\nChain of thought 56.9 (+39.0) 79.0 (+9.6)\nPrior best numbers are from Cobbe et al.")
    chart = {"kind": "bar", "title": "GSM8K", "source": "Figure 2", "bars": [
        {"label": "Prior best", "value": "55", "quote": "Prior best 55", "baseline": True},
        {"label": "PaLM 540B chain of thought", "value": "57", "quote": "chain-of-thought prompting 57", "highlight": True}]}
    found = [x for x in board_issues(chart, paper) if x.startswith("evidence: bar")]
    assert '"Prior best  N/A (finetuning) 55^(a)  57.4^(b)"' in found[0].split("Paper lines that name it: ", 1)[1].split(" | ")[0]
    assert found[1].split("Paper lines that name it: ", 1)[1].startswith('"Chain of thought 56.9 (+39.0)')   # the table's 56.9 for the figure's 57


def test_lint_flags_word_cards_and_figure_shaped_props():
    panels = clean_panels()
    panels[0].visual = "Papi points to word cards arranged in a line, next to a toy robot."
    problems = lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT), V3_PAPER_TEXT)[1]
    assert any("props that carry writing (word cards)" in x for x in problems)
    assert any("shaped like a person or creature (toy robot)" in x or "(robot)" in x for x in problems)


def test_salvage_snaps_figure_read_values_to_the_papers_rows():
    paper = ("Figure 2 legend\nPrior best\n  60  55  57\nTable 1\nPrior best  N/A (finetuning) 55^(a)  57.4^(b)\n"
             "Chain of thought 56.9 (+39.0) 79.0 (+9.6)\nPaLM 540B  Standard  17.9  69.4")
    chart = {"kind": "bar", "title": "GSM8K", "source": "Figure 2", "bars": [
        {"label": "PaLM 540B standard prompting", "value": "18", "quote": "x"},
        {"label": "Finetuned GPT-3 + verifier", "paper_label": "Prior best", "value": "55", "quote": "Prior best 55", "baseline": True},
        {"label": "PaLM 540B chain of thought", "value": "57", "quote": "chain-of-thought prompting 57", "highlight": True}]}
    snapped = _salvage_board(chart, paper)
    assert [(b["value"], b["quote"].split()[0]) for b in snapped["bars"]] == [("17.9", "PaLM"), ("55", "Prior"), ("56.9", "Chain")]
    far = {**chart, "bars": [{**chart["bars"][0], "value": "25"}, chart["bars"][2]]}   # 25 is no reading of 17.9
    assert _salvage_board(far, paper) is None           # one bar left: no chart (a one-bar chart compares nothing)


async def test_a_demoted_board_takes_its_caption_with_it():
    broken = {"kind": "links", "title": "Attention weights for the verb making", "tokens": ["laws", "making", "difficult"],
              "from": 1, "weights": [0, 0, 0]}

    def story(payload):
        payload["panels"][3]["board"] = broken
        payload["panels"][3]["narration"] = "Attention weights for the verb making."
        return payload

    model = PaperModel(story=story, fix=lambda panel, problems: panel)
    sb = await StoryboarderV2(FakeEngine(router=model)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    panel = sb.panels[3]
    assert (panel.kind, panel.board, panel.narration) == ("scene", None, "") and panel.dialogue


def test_a_near_value_is_never_snapped_from_a_prose_sentence():
    paper = ("the accuracy of GPT-3 on SST-2 ranges from near chance (54.3%) to near SOTA (93.4%)\n"
             "Prior best  N/A (finetuning) 55^(a)  57.4^(b)\nChain of thought 56.9 (+39.0) 79.0 (+9.6)")
    chart = {"kind": "bar", "title": "GSM8K", "bars": [
        {"label": "GPT-3 + verifier", "value": "55", "quote": "x", "baseline": True},
        {"label": "PaLM 540B chain of thought", "value": "57", "quote": "y", "highlight": True}]}
    assert _salvage_board(chart, paper) is None           # 54.3 is not the prior best: the GPT-3 bar goes, one bar is no chart


async def test_build_v3_asks_again_when_the_story_reply_cannot_be_read():
    model = PaperModel()
    broken = {"n": 0}

    def router(prompt, messages, config):
        user = messages[-1].content
        if user.startswith("Turn this research paper") and broken["n"] == 0:
            broken["n"] += 1
            return '{"panels": [{"n": 1, "kind": "scene" "beat": "hook"}]}'   # a missing comma
        if user.startswith("Return the following as strictly valid JSON") and broken["n"] == 1:
            broken["n"] += 1
            return '{"panels": [ still broken'
        return model(prompt, messages, config)

    sb = await StoryboarderV2(FakeEngine(router=router)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert broken["n"] == 2 and len(sb.panels) >= 9           # the second story call was read


async def test_quiz_revision_places_new_panels_numbered_past_the_script():
    """A revision reply that numbers its new panel after the last one ("n": 99) instead of placing it ("after") is still
    inserted, before the recap."""
    seen = {"revise": 0}
    model = PaperModel(quiz_misses=[{3}])

    def router(prompt, messages, config):
        user = messages[-1].content
        if user.startswith("A reader who never saw the paper read only the visible text"):
            seen["revise"] += 1
            return {"panels": [{"n": 99, "kind": "scene", "beat": "proof", "title": "New", "characters": ["papi"], "scene": "a lab",
                                "visual": "papi points", "dialogue": [{"speaker": "papi", "text": "Our model achieves 28.4 BLEU."}],
                                "narration": "", "emotions": {}, "board": None, "claims": [3]}]}
        return model(prompt, messages, config)

    sb = await StoryboarderV2(FakeEngine(router=router)).build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert seen["revise"] == 1 and sb.quiz.get("before") is not None          # the revision applied and was re-tested
    assert any(b.text == "Our model achieves 28.4 BLEU." for p in sb.panels for b in p.dialogue)
    assert sb.panels[-1].beat == "recap"


async def test_a_quiz_that_fails_after_the_revision_is_recorded(monkeypatch):
    import services.abstract_quiz as aq
    original, calls = aq.run_quiz, []

    async def flaky(*args, **kwargs):   # the first quiz runs, the one after the revision cannot
        calls.append(1)
        if len(calls) == 1:
            return await original(*args, **kwargs)
        return {"score": None, "error": "ValueError: no JSON object in model reply", "usage": {}}

    monkeypatch.setattr(aq, "run_quiz", flaky)
    story = StoryboarderV2(FakeEngine(router=PaperModel(quiz_misses=[{3}])))
    sb = await story.build_v3(v3_paper(), make_cast(), "en-US", (16, 32))
    assert len(calls) == 2 and sb.quiz.get("revised") and "quiz after the revision failed" in sb.quiz["error"]
    assert sb.quiz["score"] is not None                          # the first score stays, marked as before the revision
    assert any(x["stage"] == "abstract quiz after the revision" and "no JSON" in x["error"] for x in story.skipped)
