"""Lint rules from the reader reports: no "every claim was verified", no drawing notes on boards, no chart talk on a
scene panel, and a comparison claim drawn as a comparison."""

from __future__ import annotations

from services.storyboarder_v2 import lint_v3

from .fakes import V3_PAPER_TEXT
from .test_story_v3 import CLAIMS_WITH_RESULT, clean_panels, v3_panel, v3_storyboard


def problems_of(panels):
    return lint_v3(v3_storyboard(panels, CLAIMS_WITH_RESULT), V3_PAPER_TEXT)


def test_the_clean_script_still_passes():
    assert problems_of(clean_panels()) == {}


def test_certifying_the_whole_paper_is_flagged_in_text_and_on_boards():
    panels = clean_panels()
    panels[3].dialogue[0].text = "Every claim was verified."
    panels[3].board = {**panels[3].board, "subtitle": "All claims checked against experimental results"}
    found = problems_of(panels)[4]
    assert any("never that every claim was verified" in x and x.startswith('"all claims checked"') for x in found)


def test_a_board_that_describes_its_drawing_is_flagged():
    panels = clean_panels()
    panels[3].board = {**panels[3].board, "subtitle": "Empty checkboxes: we will verify each"}
    assert any("describe the drawing" in x for x in problems_of(panels)[4])


def test_chart_talk_on_a_scene_panel_is_flagged():
    panels = clean_panels()
    panels[1].dialogue[0].text = "This chart shows attention only."
    assert any("scene panel with no chart" in x for x in problems_of(panels)[2])


def test_a_comparison_claim_needs_a_drawn_comparison():
    panels = clean_panels()
    # the results claim (28.4 BLEU, over 2 BLEU better) taught on a scene panel only: its number is there, its comparison is not
    panels[2] = v3_panel(3, beat="proof", dialogue=[("nezu", "28.4 BLEU, over 2.0 above the rest?")], claims=[3],
                         characters=("nezu", "papi"))
    found = problems_of(panels)
    assert any("is a comparison, but no panel that teaches it draws one" in x for x in found.get(3, []))


def test_the_recap_ticks_only_lines_whose_numbers_the_story_shows():
    from services.storyboarder_v2 import _honest_recap
    panels = clean_panels()                        # panel 3's table shows 28.4; the recap lists it and an unshown 41.8
    panels[3].board = {"kind": "list", "title": "What the paper showed", "open": False,
                       "items": ["Attention only", "28.4 BLEU on WMT 2014 English-German", "41.8 BLEU on English-French"]}
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [0, 1]
    panels[3].board["items"] = panels[3].board["items"][:2]
    _honest_recap(sb)
    assert "checked" not in sb.panels[3].board       # every line shown: plain ticks
    # a number a teaching panel's list board showed counts as shown (Hivemind's "26,000 queries" list), and so does one a
    # character said (Driver Misconduct's 1.60%: readers had seen it in a bubble)
    panels[3].board["items"] = ["Attention only", "28.4 BLEU on WMT 2014 English-German", "26,000 real user queries"]
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [0, 1]
    panels[1].kind, panels[1].board = "board", {"kind": "list", "title": "The dataset", "items": ["26,000 real user queries"]}
    _honest_recap(sb)
    assert "checked" not in sb.panels[3].board
    # a title that certifies every line goes when a line only gets a bullet
    panels[3].board = {**panels[3].board, "title": "Verified Results", "subtitle": "Verified results from this work",
                       "items": ["28.4 BLEU", "41.8 BLEU on English-French"]}
    _honest_recap(sb, retitle=False)   # a relayout of a translated script keeps its titles
    assert sb.panels[3].board["checked"] == [0] and sb.panels[3].board["title"] == "Verified Results"
    _honest_recap(sb)
    assert sb.panels[3].board["checked"] == [0] and sb.panels[3].board["title"] == "What the paper showed"
    assert "subtitle" not in sb.panels[3].board
    panels[1].dialogue[0].text = "And 41.8 on English-French."
    _honest_recap(sb)
    assert "checked" not in sb.panels[3].board


def test_a_chart_in_a_scene_picture_is_flagged():
    panels = clean_panels()
    panels[1].scene, panels[1].visual = "Kumo looks at the bar chart", "Nezu traces the rising bar heights"
    found = problems_of(panels).get(2, [])
    assert any("shows a chart (bar chart, bar heights)" in x for x in found)
    panels[1].scene, panels[1].visual = "A bar at night", "Kumo orders a juice"   # a bar is not a chart
    assert not any("shows a chart" in x for x in problems_of(panels).get(2, []))


def test_the_gap_a_chart_draws_counts_as_shown_in_the_recap():
    from services.storyboarder_v2 import _honest_recap
    panels = clean_panels()
    panels[2].board = {"kind": "bar", "title": "WMT 2014 English-German (BLEU)",
                       "bars": [{"label": "ConvS2S ensemble", "value": "26.36", "baseline": True},
                                {"label": "Transformer (big)", "value": "28.4", "highlight": True}]}
    panels[3].board = {"kind": "list", "title": "What the paper showed",
                       "items": ["Attention only", "28.4 BLEU, +2.04 over the best ensemble"]}   # the badge draws +2.04
    sb = v3_storyboard(panels, CLAIMS_WITH_RESULT)
    _honest_recap(sb)
    assert "checked" not in sb.panels[3].board
