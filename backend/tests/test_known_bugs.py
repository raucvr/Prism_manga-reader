"""Regression tests for defects found (and fixed) in the v2 backend on 2026-10-21.

Each test describes the correct behaviour and fails today for the reason given in its marker. When the
owner fixes a bug the test XPASSes, which ``strict=True`` turns into a failure: delete the marker then,
and the test stays as a regression test. Run ``pytest --runxfail tests/test_known_bugs.py`` to see the
raw failures.
"""

import asyncio

import pytest

import services.jobs as jobs
from config_loader import ProviderConfig
from engines.base import ImageContent
from engines.byteplus import BytePlusEngine
from services import engine_select
from services.qa_judge import PanelQA
from services.storyboarder_v2 import PanelV2, StoryboarderV2

from .fakes import LONG_PAPER_TEXT, TINY_PNG_B64, FakeConfig, FakeEngine, make_cast, make_panel, make_sheets


def bug(reason, **kwargs):
    return pytest.mark.xfail(strict=True, reason=reason, **kwargs)


async def judge(reply):
    panel = make_panel(1, characters=["papi", "kumo"], dialogue=[("papi", "Attention is all you need."), ("kumo", "Even for long sentences?")],
                       narration="The Transformer reached 28.4 BLEU.")
    return await PanelQA(FakeEngine([reply]), make_sheets(), "en-US").judge(ImageContent.from_base64(TINY_PNG_B64), [panel])


# ----------------------------------------------------------------------------- paper extraction

def test_cot_body_stops_before_references(cot_paper):
    assert cot_paper.stats["references_cut_at_line"] is not None
    assert "arXiv preprint" not in cot_paper.body


# ----------------------------------------------------------------------------- storyboard

def test_null_fields_do_not_become_the_string_none():
    panel = PanelV2.from_dict({"title": None, "scene": None, "visual": None, "chapter": None}, 1)
    assert (panel.title, panel.scene, panel.visual, panel.chapter) == ("", "", "", "main")


async def test_speaker_of_a_dropped_bubble_is_not_added_to_the_panel():
    reply = {"panels": [{"n": 1, "chapter": "hook", "characters": ["papi"], "dialogue": [
        {"speaker": "papi", "text": "One."}, {"speaker": "kumo", "text": "Two."}, {"speaker": "nezu", "text": "Three."}]}]}
    sb = await StoryboarderV2(FakeEngine([reply])).write({}, make_cast(), (1, 3), "en-US")
    assert sb.panels[0].characters == ["papi", "kumo"]


# ----------------------------------------------------------------------------- QA judge

@pytest.mark.parametrize("reply", ['{"overall": "8/10"}', '{"panels": [{"text_items": ["ok"]}], "overall": 9}'])
async def test_oddly_shaped_verdict_does_not_raise(reply):
    result = await judge(reply)
    assert result.needs_retry is False


async def test_verdict_that_checks_no_text_is_not_a_clean_pass():
    result = await judge({"overall": 9})
    assert result.needs_retry or result.issues


# ----------------------------------------------------------------------------- jobs

async def test_oddly_shaped_verdict_does_not_fail_the_job(job_env):
    job_env.paper.num_panels = 1
    job_env.vision.router = lambda prompt, messages, config: {"overall": "8/10"}
    state = await job_env.run(qa=True)
    assert state["status"] == "completed"


async def test_retry_that_fixes_the_text_wins_a_score_tie(job_env):
    job_env.paper.num_panels = 1
    job_env.judge.fail_first_for, job_env.judge.fail_overall = {1}, 9
    state = await job_env.run(qa=True)
    panel = state["panels"][0]
    assert panel["attempts"] == 2
    assert panel["qa"]["text_exact"] == panel["qa"]["text_total"]


async def test_cancel_while_storyboarding_ends_cancelled(job_env):
    holder = {}

    async def cancel_now():
        await job_env.manager.cancel(holder["job_id"])

    job_env.paper.on_storyboard = cancel_now
    holder["job_id"] = job_env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)["job_id"]
    await asyncio.wait_for(job_env.manager.jobs[holder["job_id"]].rt.task, timeout=10)
    assert job_env.image.calls == []
    assert job_env.manager.get(holder["job_id"])["status"] == "cancelled"


async def test_job_files_live_in_the_managers_output_dir(job_env, tmp_path):
    mine = tmp_path / "mine"  # job_env points the module-level OUTPUT_DIR at tmp_path/"output"
    manager = jobs.JobManager(output_dir=mine)
    state = manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)
    await asyncio.wait_for(manager.jobs[state["job_id"]].rt.task, timeout=10)
    assert (mine / state["job_id"] / "job.json").is_file()


async def test_user_supplied_title_is_kept(job_env):
    state = await job_env.run(title="My Reading Group Notes", qa=False)
    assert state["title"] == "My Reading Group Notes"


async def test_text_accepted_by_create_is_not_rejected_later(job_env):
    text = "Self-attention relates every token to every other token. " * 5  # 284 characters after strip()
    try:
        state = job_env.manager.create(pdf_bytes=None, text=text, qa=False)
    except ValueError:
        return  # rejecting it up front would be a valid fix
    await asyncio.wait_for(job_env.manager.jobs[state["job_id"]].rt.task, timeout=10)
    assert job_env.manager.get(state["job_id"])["status"] == "completed"


# ----------------------------------------------------------------------------- engine selection

def test_legacy_openrouter_config_still_builds_engines(monkeypatch):
    core = {"default_model": "google/gemini-3-pro-image-preview", "image_model": "google/gemini-3-pro-image-preview"}
    openrouter = ProviderConfig(name="openrouter", enabled=True, api_key="test-key-not-real", base_url="https://openrouter.example.invalid/api/v1",
                                models=["google/gemini-3-pro-image-preview"])
    monkeypatch.setattr(engine_select, "get_config", lambda: FakeConfig(core, {"openrouter": openrouter}))
    monkeypatch.setattr(engine_select, "create_engine", lambda provider, model=None, **kwargs: FakeEngine())
    engines = engine_select.build_engines()
    assert not any(isinstance(e, BytePlusEngine) for e in (engines.text, engines.fast_text, engines.vision, engines.image))
