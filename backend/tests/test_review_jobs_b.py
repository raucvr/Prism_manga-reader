"""Regression tests for the 2026-10-22 job-state fixes J5, J6, J7, J8 and J11 (services/jobs.py).

J5  a failed redraw keeps the panel's previous good image
J6  the job status after a redraw reflects the panels (not always "completed")
J7  cancelling during setup ends the job "cancelled" and stops the LLM calls
J8  every task builds its own engines and closes them; a redraw never reuses closed ones
J11 cancel settles panels that were "rendering"/"qa" before anything is persisted

Engines are fakes and the network is blocked (see conftest.py). Tests marked ``xfail(strict=True)`` describe
the correct behaviour for a scenario that the current code still gets wrong. When it is fixed the test
XPASSes, ``strict`` turns that into a failure, and the marker should then be removed.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time

import pytest
from PIL import Image

import services.jobs as jobs
from engines.base import ImageContent, ImageResponse
from services.engine_select import Engines
from services.paper_extractor import extract_text_only

from .fakes import LONG_PAPER_TEXT, FakeEngine, FakeImageEngine, b64

ACTIVE = ("rendering", "qa")
PANEL_H = 1536   # a 64x96 fake panel resized to the 1024 px wide long image
GAP = 24


# ----------------------------------------------------------------------------- helpers

async def until(predicate, timeout: float = 5.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.001)


def task_of(env, job_id):
    return env.manager.jobs[job_id].rt.task


async def finish(env, job_id, manager=None):
    manager = manager or env.manager
    await asyncio.wait_for(manager.jobs[job_id].rt.task, timeout=10)


def on_disk(env, job_id) -> dict:
    return json.loads((env.output_dir / job_id / "job.json").read_text(encoding="utf-8"))


def statuses(state: dict) -> list[str]:
    return [p["status"] for p in state["panels"]]


def combined_height(env, job_id) -> int:
    with Image.open(env.output_dir / job_id / "combined.jpg") as im:
        return im.height


def hold_panel(env, n: int, gate: asyncio.Event) -> None:
    """Make the fake Seedream block (inside the call) on every prompt for panel ``n`` until ``gate`` is set."""
    async def hook(prompt):
        if f"Line for panel {n} " in prompt:
            await gate.wait()
    env.image.hook = hook


def hold_judge(env, gate: asyncio.Event) -> None:
    """Make the fake vision judge block until ``gate`` is set."""
    judge = env.judge

    def router(prompt, messages, config):
        async def held():
            await gate.wait()
            return judge(prompt, messages, config)
        return held()
    env.vision.router = router


async def cancelled_after_first_panel(env, num_panels: int = 4) -> str:
    """A job stopped while panel 1 was drawn: panels -> done, pending, pending, ..."""
    env.settings["concurrency"] = 1
    env.paper.num_panels = num_panels
    job_id = env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)["job_id"]

    async def cancel_on_first_image(prompt):
        if len(env.image.calls) == 1:
            await env.manager.cancel(job_id)

    env.image.hook = cancel_on_first_image
    await finish(env, job_id)
    env.image.hook = None
    assert statuses(env.manager.get(job_id)) == ["done"] + ["pending"] * (num_panels - 1)
    return job_id


async def interrupted_by_restart(env) -> tuple[str, object]:
    """A job whose process died mid-run (job.json still says 'rendering'), reloaded by a fresh manager."""
    job_id = await cancelled_after_first_panel(env)
    path = env.output_dir / job_id / "job.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["status"], state["stage_message"] = "rendering", "Drawing panel 2 of 4"
    path.write_text(json.dumps(state), encoding="utf-8")
    fresh = jobs.JobManager(output_dir=env.output_dir)
    loaded = fresh.get(job_id)
    assert (loaded["status"], loaded["error"]) == ("failed", "Server restarted while the job was running")
    return job_id, fresh


async def drain_other_tasks(timeout: float = 5.0) -> None:
    """Let orphaned worker tasks finish so nothing is left pending when the test's loop closes."""
    me = asyncio.current_task()
    others = [t for t in asyncio.all_tasks() if t is not me and not t.done()]
    if others:
        await asyncio.wait(others, timeout=timeout)


# ============================================================================= J5: failed redraw keeps the good panel

async def test_j5_failed_redraw_keeps_the_previous_image(job_env):
    state = await job_env.run(qa=True)
    job_id, job_dir = state["job_id"], job_env.output_dir / state["job_id"]
    before = state["panels"][1]
    old_bytes = (job_dir / "panel_02.jpg").read_bytes()

    job_env.image.fail_panels = {2}
    assert await job_env.manager.regenerate_panel(job_id, 2) == {"status": "queued"}
    await finish(job_env, job_id)

    after = job_env.manager.get(job_id)
    p2 = after["panels"][1]
    assert {k: p2[k] for k in ("status", "image_url", "width", "height", "qa")} == \
           {k: before[k] for k in ("status", "image_url", "width", "height", "qa")}
    assert p2["error"].startswith("redraw failed: RuntimeError: Seedream exploded")
    assert p2["attempts"] == before["attempts"] + job_env.settings["max_attempts"]  # cumulative (frontend cache key)
    assert (job_dir / "panel_02.jpg").read_bytes() == old_bytes
    assert after["status"] == "completed"
    assert combined_height(job_env, job_id) == 3 * PANEL_H + 2 * GAP  # the kept panel is still in the long image
    disk_p2 = on_disk(job_env, job_id)["panels"][1]
    assert (disk_p2["status"], disk_p2["image_url"]) == ("done", before["image_url"])
    assert disk_p2["error"].startswith("redraw failed:")


async def test_j5_failed_redraw_keeps_both_halves_of_a_strip(job_env):
    job_env.paper.num_panels = 4
    state = await job_env.run(layout="strip", qa=True)
    job_id = state["job_id"]
    fields = ("status", "image_url", "width", "height", "qa", "strip_index", "strip_width", "strip_height")
    before = [{k: p[k] for k in fields} for p in state["panels"][:2]]

    job_env.image.fail_panels = {1}  # the strip prompt mentions panels 1 and 2
    await job_env.manager.regenerate_panel(job_id, 2)
    await finish(job_env, job_id)

    after = job_env.manager.get(job_id)
    assert [{k: p[k] for k in fields} for p in after["panels"][:2]] == before
    assert all(p["error"].startswith("redraw failed:") for p in after["panels"][:2])
    assert combined_height(job_env, job_id) == 4 * 1024 + 3 * GAP


@pytest.mark.parametrize("reply, error", [
    ("garbage", "redraw failed: UnidentifiedImageError"),
    ("url", "redraw failed: ValueError: engine returned an image URL"),
])
async def test_j5_undecodable_redraw_keeps_the_previous_image(job_env, monkeypatch, reply, error):
    state = await job_env.run(qa=False)
    job_id = state["job_id"]
    before = state["panels"][1]
    real = job_env.image.generate_image

    async def seedream(prompt, config=None, reference_images=None):
        if "Line for panel 2 " not in prompt:
            return await real(prompt, config, reference_images)
        if reply == "garbage":
            image = ImageContent.from_base64(b64(b"<html>429 Too Many Requests</html>"), "image/jpeg")
        else:
            image = ImageContent.from_url("https://cdn.example.invalid/panel.jpeg", "image/jpeg")
        response = ImageResponse(images=[image], model="fake-seedream", prompt=prompt)
        response.usage = {}
        return response

    monkeypatch.setattr(job_env.image, "generate_image", seedream)
    await job_env.manager.regenerate_panel(job_id, 2)
    await finish(job_env, job_id)

    p2 = job_env.manager.get(job_id)["panels"][1]
    assert (p2["status"], p2["image_url"], p2["width"]) == ("done", before["image_url"], before["width"])
    assert p2["error"].startswith(error)


async def test_j5_failed_first_draw_of_a_pending_panel_is_marked_failed(job_env):
    job_id = await cancelled_after_first_panel(job_env)
    job_env.image.fail_panels = {2}
    await job_env.manager.regenerate_panel(job_id, 2)
    await finish(job_env, job_id)

    p2 = job_env.manager.get(job_id)["panels"][1]
    assert (p2["status"], p2["image_url"]) == ("failed", None)
    assert p2["error"] == "RuntimeError: Seedream exploded for panel(s) [2]"  # nothing to keep, so no "redraw failed"


async def test_j5_failed_redraw_after_restart_then_a_successful_one(job_env):
    state = await job_env.run(qa=False)
    job_id, job_dir = state["job_id"], job_env.output_dir / state["job_id"]
    old_bytes = (job_dir / "panel_03.jpg").read_bytes()
    fresh = jobs.JobManager(output_dir=job_env.output_dir)  # the snapshot comes from job.json

    job_env.image.fail_panels = {3}
    await fresh.regenerate_panel(job_id, 3)
    await finish(job_env, job_id, fresh)
    p3 = fresh.get(job_id)["panels"][2]
    assert (p3["status"], p3["image_url"]) == ("done", state["panels"][2]["image_url"])
    assert p3["error"].startswith("redraw failed:")
    assert (job_dir / "panel_03.jpg").read_bytes() == old_bytes

    job_env.image.fail_panels = set()
    await fresh.regenerate_panel(job_id, 3)
    await finish(job_env, job_id, fresh)
    p3 = fresh.get(job_id)["panels"][2]
    assert p3["status"] == "done" and "error" not in p3  # the redraw-failed note is cleared
    assert (job_dir / "panel_03.jpg").read_bytes() != old_bytes


async def test_j5_redraw_failing_outside_the_attempt_loop_keeps_the_previous_image(job_env):
    state = await job_env.run(qa=False)
    job_id = state["job_id"]
    before = state["panels"][1]
    real_persist = job_env.manager._persist
    armed = {"on": True}

    def persist(job):
        if armed["on"] and any(p["status"] == "rendering" for p in job.state["panels"]):
            armed["on"] = False
            raise OSError(28, "No space left on device")  # one transient write failure as the redraw starts
        real_persist(job)

    job_env.manager._persist = persist
    await job_env.manager.regenerate_panel(job_id, 2)
    await finish(job_env, job_id)

    p2 = job_env.manager.get(job_id)["panels"][1]
    assert (p2["status"], p2["image_url"]) == ("done", before["image_url"])
    assert combined_height(job_env, job_id) == 3 * PANEL_H + 2 * GAP


# ============================================================================= J6: job status after a redraw

async def test_j6_drawing_one_pending_panel_keeps_a_cancelled_job_cancelled(job_env):
    job_id = await cancelled_after_first_panel(job_env, num_panels=4)

    await job_env.manager.regenerate_panel(job_id, 2)
    await finish(job_env, job_id)
    state = job_env.manager.get(job_id)
    assert state["status"] == "cancelled"
    assert statuses(state) == ["done", "done", "pending", "pending"]
    assert "2 panel(s) still not drawn" in state["stage_message"]
    assert on_disk(job_env, job_id)["status"] == "cancelled"
    assert state["combined_url"] and combined_height(job_env, job_id) == 2 * PANEL_H + GAP

    for n in (3, 4):
        await job_env.manager.regenerate_panel(job_id, n)
        await finish(job_env, job_id)
    state = job_env.manager.get(job_id)
    assert (state["status"], state["stage_message"]) == ("completed", "Done")  # nothing left to draw
    assert statuses(state) == ["done"] * 4


async def test_j6_completed_job_with_a_failed_panel_stays_completed(job_env):
    job_env.image.fail_panels = {3}
    state = await job_env.run(qa=False)
    job_id = state["job_id"]
    assert statuses(state) == ["done", "done", "failed"]

    await job_env.manager.regenerate_panel(job_id, 1)
    await finish(job_env, job_id)
    state = job_env.manager.get(job_id)
    assert (state["status"], state["stage_message"]) == ("completed", "Done with 1 failed panel(s): [3]")


async def test_j6_job_failed_by_a_restart_stays_failed_while_panels_remain(job_env):
    job_id, fresh = await interrupted_by_restart(job_env)

    await fresh.regenerate_panel(job_id, 2)
    await finish(job_env, job_id, fresh)
    state = fresh.get(job_id)
    assert state["status"] == "failed"
    assert statuses(state) == ["done", "done", "pending", "pending"]
    assert "2 panel(s) still not drawn" in state["stage_message"]


async def test_j6_redraw_that_cannot_start_restores_the_previous_status(job_env):
    state = await job_env.run(qa=False)
    job_id = state["job_id"]
    (job_env.output_dir / job_id / "storyboard.json").unlink()
    fresh = jobs.JobManager(output_dir=job_env.output_dir)  # must rebuild the storyboard from disk -> fails

    await fresh.regenerate_panel(job_id, 2)
    await finish(job_env, job_id, fresh)
    after = fresh.get(job_id)
    assert after["status"] == "completed"
    assert after["stage_message"].startswith("Redraw of panel 2 failed: FileNotFoundError")
    assert statuses(after) == ["done"] * 3
    assert on_disk(job_env, job_id)["status"] == "completed"
    assert not fresh.jobs[job_id].running


async def test_j6_redraw_whose_stitching_fails_restores_the_previous_status(job_env, monkeypatch):
    job_id = await cancelled_after_first_panel(job_env, num_panels=3)

    def broken_assemble(job, *args, **kwargs):
        raise OSError("disk unplugged")

    monkeypatch.setattr(job_env.manager, "_assemble", broken_assemble)
    await job_env.manager.regenerate_panel(job_id, 2)
    await finish(job_env, job_id)
    after = job_env.manager.get(job_id)
    assert after["status"] == "cancelled"
    assert not any(s in ACTIVE for s in statuses(after))


async def test_j6_stopping_a_redraw_of_a_completed_job_leaves_it_completed(job_env):
    state = await job_env.run(qa=False)
    job_id = state["job_id"]
    gate = asyncio.Event()
    hold_panel(job_env, 2, gate)
    try:
        await job_env.manager.regenerate_panel(job_id, 2)
        await until(lambda: len(job_env.image.calls) == 4, what="the redraw call")
        await job_env.manager.cancel(job_id)  # the Stop button shown while the redraw runs
        await finish(job_env, job_id)
    finally:
        gate.set()
    assert job_env.manager.get(job_id)["status"] == "completed"


async def test_j6_job_that_stays_failed_after_a_redraw_keeps_an_error(job_env):
    job_id, fresh = await interrupted_by_restart(job_env)
    await fresh.regenerate_panel(job_id, 2)
    await finish(job_env, job_id, fresh)
    state = fresh.get(job_id)
    assert state["status"] == "failed"
    assert state["error"]


# ============================================================================= J7: cancel during setup

STAGES = [
    ("Read the paper and write a compact, faithful BRIEF", "en-US", "analyzing"),
    ("Write the manga script", "en-US", "storyboarding"),
    ("You are a meticulous fact-checker", "en-US", "storyboarding"),
    ("Translate this manga script", "zh-CN", "translating"),
]


@pytest.mark.parametrize("prefix, language, stage", STAGES, ids=["brief", "script", "factcheck", "translate"])
async def test_j7_cancel_during_an_llm_stage_stops_the_job(job_env, prefix, language, stage):
    persisted = []
    real_persist = job_env.manager._persist

    def persist(job):
        persisted.append(job.state["status"])
        real_persist(job)

    job_env.manager._persist = persist
    gate, entered = asyncio.Event(), asyncio.Event()
    paper = job_env.paper

    def router(prompt, messages, config):
        if not messages[-1].content.startswith(prefix):
            return paper(prompt, messages, config)

        async def held():
            entered.set()
            await gate.wait()
            reply = paper(prompt, messages, config)
            return await reply if inspect.isawaitable(reply) else reply
        return held()

    job_env.text.router = router
    job_id = job_env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, language=language, qa=True)["job_id"]
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert job_env.manager.get(job_id)["status"] == stage
        calls_at_cancel = len(job_env.text.calls)
        assert (await job_env.manager.cancel(job_id))["status"] == "cancelled"  # what POST /cancel does
        await finish(job_env, job_id)
    finally:
        gate.set()
    for _ in range(5):
        await asyncio.sleep(0)

    state = job_env.manager.get(job_id)
    assert (state["status"], state["stage_message"]) == ("cancelled", "Cancelled")
    assert len(job_env.text.calls) == calls_at_cancel  # no LLM call after Stop
    assert job_env.image.calls == []
    after_cancel = persisted[persisted.index("cancelled"):]
    assert set(after_cancel) == {"cancelled"}  # never flipped back to a running stage
    assert on_disk(job_env, job_id)["status"] == "cancelled"
    assert (job_env.text.closed, job_env.vision.closed, job_env.image.closed) == (1, 1, 1)
    assert job_env.manager.jobs[job_id].rt.engines is None


async def test_j7_cancel_before_the_job_starts(job_env):
    job_id = job_env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)["job_id"]
    assert (await job_env.manager.cancel(job_id))["status"] == "cancelled"
    task = task_of(job_env, job_id)
    await asyncio.wait({task}, timeout=5)
    assert task.done() and not job_env.manager.jobs[job_id].running
    assert job_env.manager.get(job_id)["status"] == "cancelled"
    assert on_disk(job_env, job_id)["status"] == "cancelled"
    assert job_env.text.calls == [] and job_env.image.calls == []


async def test_j7_cancel_while_the_pdf_is_being_extracted(job_env, monkeypatch):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow_extract(pdf_bytes, max_chars=120_000, filename=""):
        started.set()
        release.wait(5)
        finished.set()
        return extract_text_only(LONG_PAPER_TEXT)

    monkeypatch.setattr(jobs, "extract_paper", slow_extract)
    job_id = job_env.manager.create(pdf_bytes=b"%PDF-1.4 not parsed", text=None, filename="paper.pdf", qa=False)["job_id"]
    try:
        await until(started.is_set, what="extraction to start")
        await job_env.manager.cancel(job_id)
        await finish(job_env, job_id)
        assert job_env.manager.get(job_id)["status"] == "cancelled"
    finally:
        release.set()
    await until(finished.is_set, what="the extraction thread")
    for _ in range(5):
        await asyncio.sleep(0)
    assert job_env.manager.get(job_id)["status"] == "cancelled"
    assert job_env.text.calls == []  # the finished extraction did not resume the pipeline


# ============================================================================= J8: engines per task

@pytest.fixture
def engine_sets(job_env, monkeypatch):
    """Make every build_engines() call return a NEW engine set; record any call made to a closed engine."""
    built: list[Engines] = []
    used_after_close: list[str] = []
    hooks = {"image": None}

    class Text(FakeEngine):
        async def generate_text(self, messages, config=None):
            if self.closed:
                used_after_close.append(f"set{self.set_no}:{self.model}")
            return await super().generate_text(messages, config)

    class Seedream(FakeImageEngine):
        async def generate_image(self, prompt, config=None, reference_images=None):
            if self.closed:
                used_after_close.append(f"set{self.set_no}:image")
            return await super().generate_image(prompt, config, reference_images)

    def build():
        n = len(built)
        text = Text(router=job_env.paper, model="fake-seed-text")
        vision = Text(router=job_env.judge, usage={"prompt_tokens": 2000, "completion_tokens": 100}, model="fake-seed-vision")
        image = Seedream(hook=lambda prompt: hooks["image"](prompt) if hooks["image"] else None)
        for engine in (text, vision, image):
            engine.set_no = n
        engines = Engines(text=text, fast_text=text, vision=vision, image=image)
        built.append(engines)
        return engines

    monkeypatch.setattr(jobs, "build_engines", build)
    return built, used_after_close, hooks


def assert_released(job) -> None:
    assert (job.rt.engines, job.rt.renderer, job.rt.qa) == (None, None, None)


def closed_counts(engines: Engines) -> tuple[int, int, int]:
    return engines.text.closed, engines.vision.closed, engines.image.closed


async def test_j8_every_redraw_builds_fresh_engines_and_closes_them(job_env, engine_sets):
    built, used_after_close, _ = engine_sets
    state = await job_env.run(qa=True)
    job_id = state["job_id"]
    job = job_env.manager.jobs[job_id]
    assert len(built) == 1 and closed_counts(built[0]) == (1, 1, 1)
    assert_released(job)

    for n in (2, 3):  # in-process redraws after the run closed its engines
        await job_env.manager.regenerate_panel(job_id, n)
        await finish(job_env, job_id)
        assert_released(job)
    fresh = jobs.JobManager(output_dir=job_env.output_dir)  # a redraw after a restart
    await fresh.regenerate_panel(job_id, 1)
    await finish(job_env, job_id, fresh)
    assert_released(fresh.jobs[job_id])

    assert len(built) == 4
    assert [closed_counts(e) for e in built] == [(1, 1, 1)] * 4  # each set closed exactly once
    assert used_after_close == []
    assert [len(e.image.calls) for e in built] == [3, 1, 1, 1]  # each redraw drew with its own (new) renderer
    assert [len(e.vision.calls) for e in built] == [3, 1, 1, 1]
    assert [p["status"] for p in fresh.get(job_id)["panels"]] == ["done"] * 3


async def test_j8_redraw_after_a_cancel_uses_new_engines(job_env, engine_sets):
    built, used_after_close, hooks = engine_sets
    job_env.settings["concurrency"] = 1
    job_id = job_env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)["job_id"]

    async def cancel_on_first_image(prompt):
        if len(built[0].image.calls) == 1:
            await job_env.manager.cancel(job_id)

    hooks["image"] = cancel_on_first_image
    await finish(job_env, job_id)
    hooks["image"] = None
    assert job_env.manager.get(job_id)["status"] == "cancelled"
    assert closed_counts(built[0]) == (1, 1, 1)

    await job_env.manager.regenerate_panel(job_id, 2)
    await finish(job_env, job_id)
    assert len(built) == 2 and len(built[1].image.calls) == 1 and len(built[0].image.calls) == 1
    assert closed_counts(built[1]) == (1, 1, 1)
    assert used_after_close == []
    assert_released(job_env.manager.jobs[job_id])


async def test_j8_engines_are_closed_when_setup_fails(job_env, engine_sets):
    built, _, _ = engine_sets
    paper = job_env.paper

    def model_down(prompt, messages, config):
        if messages[-1].content.startswith("Write the manga script"):
            raise RuntimeError("model down")
        return paper(prompt, messages, config)

    job_env.paper = model_down  # picked up by the engine-set builder
    state = await job_env.run(qa=False)
    assert state["status"] == "failed" and "model down" in state["error"]
    assert len(built) == 1 and closed_counts(built[0]) == (1, 1, 1)
    assert_released(job_env.manager.jobs[state["job_id"]])


async def test_j8_engines_are_closed_when_a_redraw_cannot_build_its_renderer(job_env, engine_sets, monkeypatch):
    built, _, _ = engine_sets
    state = await job_env.run(qa=False)
    job_id = state["job_id"]

    def broken_build_renderers(job, settings):
        raise RuntimeError("bad panel size")

    monkeypatch.setattr(job_env.manager, "_build_renderers", broken_build_renderers)
    await job_env.manager.regenerate_panel(job_id, 2)
    await finish(job_env, job_id)
    after = job_env.manager.get(job_id)
    assert after["status"] == "completed" and after["stage_message"].startswith("Redraw of panel 2 failed")
    assert len(built) == 2 and closed_counts(built[1]) == (1, 1, 1)
    assert_released(job_env.manager.jobs[job_id])


# ============================================================================= J11: cancel settles active panels

async def test_j11_cancel_settles_rendering_and_qa_panels_before_persisting(job_env):
    job_env.settings["concurrency"] = 2
    job_env.paper.num_panels = 4
    image_gate, judge_gate = asyncio.Event(), asyncio.Event()
    hold_panel(job_env, 2, image_gate)
    hold_judge(job_env, judge_gate)
    job_id = job_env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=True)["job_id"]
    job = job_env.manager.jobs[job_id]
    try:
        await until(lambda: statuses(job.state) == ["qa", "rendering", "pending", "pending"], what="panel 1 in qa, 2 rendering")
        await job_env.manager.cancel(job_id)
        disk = on_disk(job_env, job_id)  # written by cancel() itself, before the task saw the cancellation
        assert disk["status"] == "cancelled"
        assert [p["status"] for p in disk["panels"]] == ["pending"] * 4  # nothing had an image yet
        await finish(job_env, job_id)
    finally:
        image_gate.set()
        judge_gate.set()
    for _ in range(5):
        await asyncio.sleep(0)
    state = job_env.manager.get(job_id)
    assert state["status"] == "cancelled" and statuses(state) == ["pending"] * 4
    assert [p["status"] for p in on_disk(job_env, job_id)["panels"]] == ["pending"] * 4
    assert len(job_env.image.calls) == 2  # panels 3 and 4 were never started


async def test_j11_cancel_during_a_redraw_puts_back_the_previous_image(job_env):
    state = await job_env.run(qa=True)
    job_id, job_dir = state["job_id"], job_env.output_dir / state["job_id"]
    before = state["panels"][1]
    old_bytes = (job_dir / "panel_02.jpg").read_bytes()
    gate = asyncio.Event()
    hold_judge(job_env, gate)
    job = job_env.manager.jobs[job_id]
    try:
        await job_env.manager.regenerate_panel(job_id, 2)
        await until(lambda: job.state["panels"][1]["status"] == "qa", what="the redraw to reach QA")
        await job_env.manager.cancel(job_id)
        disk_p2 = on_disk(job_env, job_id)["panels"][1]
        assert (disk_p2["status"], disk_p2["image_url"], disk_p2["width"]) == ("done", before["image_url"], before["width"])
        await finish(job_env, job_id)
    finally:
        gate.set()
    p2 = job_env.manager.get(job_id)["panels"][1]
    assert {k: p2[k] for k in ("status", "image_url", "width", "height", "qa")} == \
           {k: before[k] for k in ("status", "image_url", "width", "height", "qa")}
    assert (job_dir / "panel_02.jpg").read_bytes() == old_bytes
    assert not any(s in ACTIVE for s in statuses(job_env.manager.get(job_id)))
    assert not job.running  # the job can be redrawn again


async def test_j11_cancel_settles_both_halves_of_a_strip(job_env):
    job_env.settings["concurrency"] = 1
    job_env.paper.num_panels = 4
    gate = asyncio.Event()
    hold_panel(job_env, 1, gate)
    job_id = job_env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, layout="strip", qa=False)["job_id"]
    job = job_env.manager.jobs[job_id]
    try:
        await until(lambda: statuses(job.state)[:2] == ["rendering", "rendering"], what="the first strip")
        await job_env.manager.cancel(job_id)
        assert [p["status"] for p in on_disk(job_env, job_id)["panels"]] == ["pending"] * 4
        await finish(job_env, job_id)
    finally:
        gate.set()
    assert statuses(job_env.manager.get(job_id)) == ["pending"] * 4


async def test_j11_job_that_fails_mid_render_leaves_no_panel_drawing(job_env):
    job_env.settings["concurrency"] = 2
    job_env.paper.num_panels = 2
    job_env.image.fail_panels = {1}
    gate = asyncio.Event()
    hold_panel(job_env, 2, gate)
    real_persist = job_env.manager._persist

    def persist(job):
        panels = job.state["panels"]
        if job.state["status"] == "rendering" and panels and panels[0]["status"] == "failed":
            raise OSError(28, "No space left on device")  # the disk fills up while panel 1's failure is recorded
        real_persist(job)

    job_env.manager._persist = persist
    job_id = job_env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)["job_id"]
    try:
        await finish(job_env, job_id)
        assert any("Line for panel 2 " in c["prompt"] for c in job_env.image.calls)  # panel 2 was being drawn
        disk = on_disk(job_env, job_id)
        assert disk["status"] == "failed"
        assert [p["n"] for p in disk["panels"] if p["status"] in ACTIVE] == []
        assert [p["n"] for p in job_env.manager.get(job_id)["panels"] if p["status"] in ACTIVE] == []
    finally:
        gate.set()
        await drain_other_tasks()
