"""Regression tests for the round-3 job fixes (services/jobs.py, services/cast.py, tools/job_report.py).

C1-C5 are the round-3 cross-group contracts:
C1 an attempt whose QA verdict is unavailable never beats a checked attempt
C2 the judge gets the cast names (sheet-less themes still get character checks)
C3 ``redraw_panels`` and a "Redrawing panel N" stage while a user redraw runs
C4 a failed redraw keeps the picture and says why in the panel's ``error``
C5 ``render_attempts`` per render; ``attempts`` stays cumulative and changes only with a new picture
plus: Stop while stitching, atomic image writes, unreadable panel files, shared reference sheets, lone surrogates in
model replies, the content-filter feedback (P3) and the job report's retry count.
Every engine is a fake (conftest.job_env) and conftest blocks the network.
"""

import asyncio
import errno
import json
import os
import re
import threading
from pathlib import Path

import pytest
from fastapi.responses import JSONResponse
from PIL import Image

import services.cast as cast
import services.jobs as jobs
from engines.base import ImageContent
from services.qa_judge import BLIND_PROMPT, QAResult
from tools import job_report

from .fakes import LONG_PAPER_TEXT, LOOKS, PALETTE, image_bytes, make_cast, make_panel

PANEL_H, GAP = 1536, 24   # a 64x96 fake panel in the 1024 px wide long image


# ----------------------------------------------------------------------------- helpers

def panel(state: dict, n: int) -> dict:
    return next(p for p in state["panels"] if p["n"] == n)


def on_disk(env, job_id: str) -> dict:
    return json.loads((env.output_dir / job_id / "job.json").read_text(encoding="utf-8"))


def close_to(pixel, color, tolerance=45) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(pixel, color))


def colour_of(path: Path):
    with Image.open(path) as im:
        return im.convert("RGB").getpixel((im.width // 2, im.height // 2))


async def until(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.002)


async def finish(manager, job_id: str) -> None:
    await asyncio.wait_for(manager.jobs[job_id].rt.task, timeout=10)


async def redraw(env, job_id: str, n: int) -> dict:
    assert await env.manager.regenerate_panel(job_id, n) == {"status": "queued"}
    await finish(env.manager, job_id)
    return env.manager.get(job_id)


async def stopped_after_first_panel(env, num_panels: int = 4) -> str:
    """A job stopped while panel 1 was drawn: panels -> done, pending, pending, ..."""
    env.settings["concurrency"] = 1
    env.paper.num_panels = num_panels
    job_id = env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)["job_id"]

    async def stop_on_first_image(prompt):
        if len(env.image.calls) == 1:
            await env.manager.cancel(job_id)

    env.image.hook = stop_on_first_image
    await finish(env.manager, job_id)
    env.image.hook = None
    assert [p["status"] for p in env.manager.get(job_id)["panels"]] == ["done"] + ["pending"] * (num_panels - 1)
    return job_id


def persist_job(out: Path, job_id: str, **changes) -> Path:
    state = jobs._new_state(job_id, title="Persisted", theme="kumomo", language="en-US", layout="single", qa=True,
                            panel_range="auto", fast=False, source="text")
    state.pop("_started")
    state.update(changes)
    (out / job_id).mkdir(parents=True, exist_ok=True)
    (out / job_id / "job.json").write_text(json.dumps(state), encoding="utf-8")
    return out / job_id


def verdict(score: float, exact: int, total: int, *, retry: bool = False, available: bool = True, issues=()) -> QAResult:
    return QAResult(score=score, text_exact=exact, text_total=total, characters_ok=True, issues=list(issues), needs_retry=retry,
                    available=available)


def scripted_qa(monkeypatch, verdicts, seen=None) -> None:
    """Replace PanelQA with a judge that answers ``verdicts`` in order; ``seen`` records the constructor arguments."""
    queue = list(verdicts)

    class ScriptedQA:
        def __init__(self, vision_engine, sheets, language, retry_threshold=6.0, cast_names=None, cast_looks=None, blind_check=False):
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
            if seen is not None:
                seen.append({"sheets": list(sheets), "cast_names": cast_names, "cast_looks": cast_looks, "blind_check": blind_check})

        async def judge(self, image, panels):
            return queue.pop(0)

    monkeypatch.setattr(jobs, "PanelQA", ScriptedQA)


def gate_assembly(manager):
    """Hold _assemble (it runs in a worker thread) until ``release`` is set; returns (entered, release, finished)."""
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real = manager._assemble

    def assemble(job, *args, **kwargs):
        entered.set()
        release.wait(5)
        try:
            return real(job, *args, **kwargs)
        finally:
            finished.set()

    manager._assemble = assemble
    return entered, release, finished


# ----------------------------------------------------------------------------- C1 / S9: which attempt ships

async def test_c1_an_unchecked_retry_never_replaces_a_checked_attempt(job_env, monkeypatch):
    job_env.paper.num_panels = 1
    scripted_qa(monkeypatch, [verdict(8, 1, 2, retry=True, issues=["papi's bubble reads 'Lne for panel 1'"]),
                              verdict(0, 0, 0, available=False, issues=["qa unavailable: Expecting property name"])])
    state = await job_env.run(qa=True)

    p1 = panel(state, 1)
    assert len(job_env.image.calls) == 2   # the first verdict asked for a retry
    assert (p1["status"], p1["qa"]["score"], p1["qa"]["text_exact"]) == ("done", 8, 1)   # the checked attempt ships
    assert close_to(colour_of(job_env.output_dir / state["job_id"] / "panel_01.jpg"), PALETTE[1])   # attempt 2 drew PALETTE[2]


async def test_s9_more_exact_lines_win_when_no_attempt_is_fully_exact(job_env, monkeypatch):
    job_env.paper.num_panels = 1
    scripted_qa(monkeypatch, [verdict(7, 2, 3, retry=True), verdict(7, 1, 3, retry=True)])
    state = await job_env.run(qa=True)

    assert len(job_env.image.calls) == 2
    assert panel(state, 1)["qa"]["text_exact"] == 2   # same score: the attempt with more exact lines is kept
    assert close_to(colour_of(job_env.output_dir / state["job_id"] / "panel_01.jpg"), PALETTE[1])


# ----------------------------------------------------------------------------- C2: cast names for the judge

async def test_c2_the_judge_gets_the_cast_names(job_env, monkeypatch):
    seen = []
    scripted_qa(monkeypatch, [verdict(9, 2, 2)] * 3, seen)
    state = await job_env.run(qa=True)
    assert state["status"] == "completed"
    assert seen and all(s["cast_names"] == ["papi", "kumo", "nezu"] for s in seen)


async def test_c2_a_theme_without_sheets_still_names_its_cast(job_env, monkeypatch):
    monkeypatch.setattr(jobs, "build_cast", lambda theme: (make_cast(), {}))
    seen = []
    scripted_qa(monkeypatch, [verdict(9, 2, 2)] * 3, seen)
    await job_env.run(qa=True)
    assert seen and all((s["sheets"], s["cast_names"]) == ([], ["papi", "kumo", "nezu"]) for s in seen)


async def test_the_judge_gets_the_looks_of_the_cast(job_env, monkeypatch):
    """Integration follow-up of C2: a theme without sheets is described to the judge, so it can recognise the cast."""
    monkeypatch.setattr(jobs, "build_cast", lambda theme: (make_cast(), {}))
    seen = []
    scripted_qa(monkeypatch, [verdict(9, 2, 2)] * 4, seen)
    state = await job_env.run(qa=True)
    await redraw(job_env, state["job_id"], 2)   # a redraw rebuilds the judge (_ensure_runtime)
    assert len(seen) == 2 and all(s["cast_looks"] == LOOKS for s in seen)


# ----------------------------------------------------------------------------- C3: a redraw says what it redraws

@pytest.mark.parametrize("layout, n, redrawn, message", [("single", 2, [2], "Redrawing panel 2"),
                                                         ("strip", 2, [1, 2], "Redrawing panels 1-2")])
async def test_c3_a_redraw_names_the_panels_it_redraws(job_env, layout, n, redrawn, message):
    job_env.paper.num_panels = 4
    state = await job_env.run(layout=layout, qa=False)
    job_id = state["job_id"]
    assert "redraw_panels" not in state
    seen = []
    job_env.image.hook = lambda prompt: seen.append((job_env.manager.get(job_id), on_disk(job_env, job_id)))

    after = await redraw(job_env, job_id, n)
    ((during, disk),) = seen
    assert during["status"] == "rendering" and during["redraw_panels"] == redrawn == disk["redraw_panels"]
    assert during["stage_message"] == message
    assert after["status"] == "completed"
    assert "redraw_panels" not in after and "redraw_panels" not in on_disk(job_env, job_id)


async def test_c3_a_redraw_retry_still_reads_as_a_redraw(job_env):
    state = await job_env.run(qa=True)
    job_id = state["job_id"]
    job_env.judge.fail_first_for = {2}
    job_env.judge.judged.clear()
    messages = []
    job_env.image.hook = lambda prompt: messages.append(job_env.manager.get(job_id)["stage_message"])
    await redraw(job_env, job_id, 2)
    assert messages == ["Redrawing panel 2", "Redrawing panel 2 (retry 2)"]


@pytest.mark.parametrize("layout, redrawn", [("single", [2]), ("strip", [1, 2])])
async def test_c3_the_job_list_names_the_panels_a_redraw_draws(job_env, layout, redrawn):
    """The home page's recent-jobs list reads "Redrawing panel N", not "Drawing x/y", while a user redraw runs."""
    job_env.paper.num_panels = 4
    state = await job_env.run(layout=layout, qa=False)
    job_id = state["job_id"]
    assert "redraw_panels" not in job_env.manager.list()[0]
    listed = []
    job_env.image.hook = lambda prompt: listed.append(job_env.manager.list()[0])

    await redraw(job_env, job_id, 2)
    ((during,),) = [listed]
    assert (during["job_id"], during["status"], during["redraw_panels"]) == (job_id, "rendering", redrawn)
    assert "redraw_panels" not in job_env.manager.list()[0]


@pytest.mark.parametrize("status", ["completed", "rendering"])
def test_c3_a_restart_drops_a_stale_redraw_marker(job_env, status):
    out, job_id = job_env.output_dir, "20261022_110000_c3c3"
    url = f"/output/{job_id}/panel_01.jpg"
    job_dir = persist_job(out, job_id, status=status, redraw_panels=[1],
                          panels=[dict(jobs._panel_state(make_panel(1)), status="done", image_url=url, width=4, height=6, attempts=1)])
    (job_dir / "panel_01.jpg").write_bytes(image_bytes(4, 6, fmt="JPEG"))

    state = jobs.JobManager(output_dir=out).get(job_id)   # a fresh process
    assert "redraw_panels" not in state and "redraw_panels" not in json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    assert state["status"] in jobs.TERMINAL and panel(state, 1)["status"] == "done"


@pytest.mark.parametrize("pending, status", [(False, "completed"), (True, "cancelled")])
def test_a_restart_during_a_redraw_without_the_saved_status_never_fails_the_job(job_env, pending, status):
    """job.json written during a redraw by code that did not save the pre-redraw status: the redraw marker alone
    must not turn a drawn manga into 'failed' (which offers a paid full re-run)."""
    out, job_id = job_env.output_dir, "20261022_110000_c3d4"
    url = f"/output/{job_id}/panel_01.jpg"
    panels = [dict(jobs._panel_state(make_panel(1)), status="done", image_url=url, width=4, height=6, attempts=1),
              dict(jobs._panel_state(make_panel(2)), status="rendering", image_url=url, width=4, height=6, attempts=1),
              dict(jobs._panel_state(make_panel(3)), status="pending" if pending else "done", image_url=None if pending else url,
                   width=None if pending else 4, height=None if pending else 6, attempts=0 if pending else 1)]
    job_dir = persist_job(out, job_id, status="rendering", stage_message="Redrawing panel 2", redraw_panels=[2], panels=panels)
    (job_dir / "panel_01.jpg").write_bytes(image_bytes(4, 6, fmt="JPEG"))

    state = jobs.JobManager(output_dir=out).get(job_id)   # a fresh process
    assert (state["status"], state["error"], state["stage_message"]) == (status, None, "Redraw interrupted by a server restart")
    assert [p["status"] for p in state["panels"]] == ["done", "done", "pending" if pending else "done"]
    assert "redraw_panels" not in state and on_disk(job_env, job_id)["status"] == status


@pytest.mark.parametrize("previous", ["completed", "failed"])
async def test_a_restart_during_a_redraw_puts_the_job_back_as_it_was(job_env, tmp_path, previous):
    state = await job_env.run(qa=False)
    job_id = state["job_id"]
    error = "OSError: [Errno 28] No space left on device" if previous == "failed" else None
    if previous == "failed":   # e.g. a job whose run failed after drawing every panel
        job_env.manager.jobs[job_id].state.update(status="failed", error=error)
    crashed = tmp_path / "crashed"

    def die_mid_redraw(prompt):   # the process stops here: job.json and the images stay as they are on disk
        import shutil
        shutil.copytree(job_env.output_dir / job_id, crashed / job_id)

    job_env.image.hook = die_mid_redraw
    await redraw(job_env, job_id, 2)

    restarted = jobs.JobManager(output_dir=crashed).get(job_id)
    assert (restarted["status"], restarted["error"]) == (previous, error)
    assert [p["status"] for p in restarted["panels"]] == ["done"] * 3 and "redraw_panels" not in restarted
    assert not [k for k in json.loads((crashed / job_id / "job.json").read_text(encoding="utf-8")) if "redraw" in k]


# ----------------------------------------------------------------------------- C4: a failed redraw says why

@pytest.mark.parametrize("layout, n, redrawn", [("single", 2, [2]), ("strip", 3, [3, 4])])
async def test_c4_a_redraw_that_cannot_start_reports_the_failure_on_its_panels(job_env, monkeypatch, layout, n, redrawn):
    job_env.paper.num_panels = 4
    state = await job_env.run(layout=layout, qa=False)
    job_id = state["job_id"]
    kept = ("status", "image_url", "width", "height", "attempts")
    working_engines = jobs.build_engines

    def no_api_key():
        raise ValueError("ModelArk API key is empty")

    monkeypatch.setattr(jobs, "build_engines", no_api_key)
    after = await redraw(job_env, job_id, n)
    assert after["status"] == "completed"
    for m in redrawn:
        p = panel(after, m)
        assert p["error"] == "redraw failed: ValueError: ModelArk API key is empty"
        assert {k: p[k] for k in kept} == {k: panel(state, m)[k] for k in kept}   # the picture is kept
        assert panel(on_disk(job_env, job_id), m)["error"] == p["error"]
    assert not any("error" in p for p in after["panels"] if p["n"] not in redrawn)

    monkeypatch.setattr(jobs, "build_engines", working_engines)
    after = await redraw(job_env, job_id, n)
    assert all("error" not in panel(after, m) and panel(after, m)["status"] == "done" for m in redrawn)   # cleared


async def test_c4_a_repeated_setup_failure_still_moves_updated_at(job_env, monkeypatch):
    """The reader settles a "Try again" that fails before drawing with the same error by updated_at changing:
    it has millisecond resolution, so a retry within the same second is still seen."""
    state = await job_env.run(qa=False)
    job_id = state["job_id"]

    def no_api_key():
        raise ValueError("ModelArk API key is empty")

    monkeypatch.setattr(jobs, "build_engines", no_api_key)
    first = await redraw(job_env, job_id, 2)
    await asyncio.sleep(0.003)
    second = await redraw(job_env, job_id, 2)
    assert panel(first, 2)["error"] == panel(second, 2)["error"] == "redraw failed: ValueError: ModelArk API key is empty"
    assert first["status"] == second["status"] == "completed"
    assert second["updated_at"] != first["updated_at"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d", second["updated_at"])


async def test_c4_a_failed_draw_on_a_stopped_job_is_named_in_the_stage(job_env):
    job_id = await stopped_after_first_panel(job_env)
    job_env.image.fail_panels = {2}
    after = await redraw(job_env, job_id, 2)
    assert after["status"] == "cancelled"
    assert after["stage_message"] == "Redraw of panel 2 failed; 2 panel(s) still not drawn"
    assert panel(after, 2)["status"] == "failed"


# ----------------------------------------------------------------------------- C5: attempts per render

async def test_c5_render_attempts_count_the_latest_render_while_attempts_accumulate(job_env):
    job_env.judge.fail_first_for = {1}
    state = await job_env.run(qa=True)
    job_id = state["job_id"]
    assert [(p["attempts"], p["render_attempts"]) for p in state["panels"]] == [(2, 2), (1, 1), (1, 1)]

    after = await redraw(job_env, job_id, 1)   # passes QA at once this time
    assert (panel(after, 1)["attempts"], panel(after, 1)["render_attempts"]) == (3, 1)

    job_env.image.fail_panels = {2}
    after = await redraw(job_env, job_id, 2)   # every attempt fails: the old picture stays
    assert (panel(after, 2)["attempts"], panel(after, 2)["render_attempts"]) == (3, 2)
    assert panel(on_disk(job_env, job_id), 1)["render_attempts"] == 1


# ----------------------------------------------------------------------------- Stop while stitching

async def test_stop_while_stitching_leaves_memory_and_job_json_in_agreement(job_env):
    manager = job_env.manager
    entered, release, finished = gate_assembly(manager)
    job_id = manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)["job_id"]
    try:
        await until(entered.is_set)
        assert manager.get(job_id)["status"] == "assembling"
        assert (await manager.cancel(job_id))["status"] == "cancelled"
        await finish(manager, job_id)
    finally:
        release.set()
    await until(finished.is_set)   # the orphaned stitching thread has finished as well

    state, disk = manager.jobs[job_id].state, on_disk(job_env, job_id)
    assert (state["status"], disk["status"]) == ("cancelled", "cancelled")
    assert state["combined_url"] is None and disk["combined_url"] is None


async def test_stop_while_stitching_a_redraw_keeps_the_previous_long_image_url(job_env):
    state = await job_env.run(qa=False)
    job_id, manager = state["job_id"], job_env.manager
    entered, release, finished = gate_assembly(manager)
    assert await manager.regenerate_panel(job_id, 2) == {"status": "queued"}
    try:
        await until(entered.is_set)
        await manager.cancel(job_id)
        await finish(manager, job_id)
    finally:
        release.set()
    await until(finished.is_set)

    live, disk = manager.jobs[job_id].state, on_disk(job_env, job_id)
    assert live["combined_url"] == disk["combined_url"] == state["combined_url"]
    assert (live["status"], disk["status"]) == ("completed", "completed")   # a stopped redraw puts the job back


async def test_stop_while_stitching_a_redraw_leaves_the_long_image_file_unchanged(job_env):
    state = await job_env.run(qa=False)
    job_id, manager = state["job_id"], job_env.manager
    combined = job_env.output_dir / job_id / "combined.jpg"
    old_bytes = combined.read_bytes()
    entered, release, finished = gate_assembly(manager)
    assert await manager.regenerate_panel(job_id, 2) == {"status": "queued"}
    try:
        await until(entered.is_set)
        await manager.cancel(job_id)
        await finish(manager, job_id)
    finally:
        release.set()
    await until(finished.is_set)   # the orphaned stitching thread has finished as well

    # the long image keeps its key, so its file must not change either: a browser that cached the key would show stale bytes
    assert manager.get(job_id)["combined_url"] == state["combined_url"]
    assert combined.read_bytes() == old_bytes and not list(combined.parent.glob("*.tmp"))


async def test_an_image_key_never_serves_two_different_files(job_env):
    """image_url?v=<attempts> and combined_url are cache keys: sampled at every loop step, each maps to one file content."""
    job_env.paper.num_panels = 4
    state = await job_env.run(qa=True)
    job_id, job_dir = state["job_id"], job_env.output_dir / state["job_id"]
    contents: dict[str, set] = {}
    sampling = True

    def sample():
        s = job_env.manager.get(job_id)
        keys = [(f"{p['image_url']}?v={p['attempts']}", Path(p["image_url"]).name) for p in s["panels"] if p.get("image_url")]
        keys.append((s["combined_url"], "combined.jpg"))
        for key, name in keys:
            contents.setdefault(key, set()).add((job_dir / name).read_bytes())

    async def sampler():
        while sampling:
            sample()
            await asyncio.sleep(0)

    task = asyncio.create_task(sampler())
    job_env.judge.fail_first_for = {3}
    job_env.judge.judged.clear()
    for n in (2, 3, 2):
        await redraw(job_env, job_id, n)
    sampling = False
    await task
    sample()
    assert len(contents) > 5 and all(len(c) == 1 for c in contents.values())


async def test_stopping_a_redraw_before_it_started_keeps_the_job_status(job_env):
    state = await job_env.run(qa=False)
    job_id, manager = state["job_id"], job_env.manager
    assert await manager.regenerate_panel(job_id, 2) == {"status": "queued"}
    task = manager.jobs[job_id].rt.task
    stopped = await manager.cancel(job_id)   # the redraw task has not run a single step yet
    await asyncio.wait({task}, timeout=5)

    assert task.cancelled() and stopped["status"] == "completed"
    assert (manager.get(job_id)["status"], on_disk(job_env, job_id)["status"]) == ("completed", "completed")
    assert len(job_env.image.calls) == 3


async def test_stopping_the_redraw_of_a_failed_panel_puts_it_back_as_it_was(job_env):
    job_env.image.fail_panels = {3}
    state = await job_env.run(qa=False)
    job_id, before = state["job_id"], panel(state, 3)
    assert (state["status"], before["status"]) == ("completed", "failed")
    job_env.image.fail_panels = set()
    calls, gate = len(job_env.image.calls), asyncio.Event()

    async def hold(prompt):
        await gate.wait()

    job_env.image.hook = hold
    try:
        assert await job_env.manager.regenerate_panel(job_id, 3) == {"status": "queued"}
        await until(lambda: len(job_env.image.calls) == calls + 1)
        await job_env.manager.cancel(job_id)
        await finish(job_env.manager, job_id)
    finally:
        gate.set()

    after = job_env.manager.get(job_id)
    p3 = panel(after, 3)
    assert (p3["status"], p3["image_url"], p3["error"]) == ("failed", None, before["error"])
    assert after["status"] == "completed" and "pending" not in [p["status"] for p in after["panels"]]
    assert panel(on_disk(job_env, job_id), 3)["status"] == "failed"


# ----------------------------------------------------------------------------- files on disk

async def test_a_partial_image_write_never_destroys_the_previous_picture(job_env, monkeypatch):
    state = await job_env.run(qa=False)
    job_id, job_dir = state["job_id"], job_env.output_dir / state["job_id"]
    before = panel(state, 2)
    old_bytes = (job_dir / "panel_02.jpg").read_bytes()
    real_write_bytes = Path.write_bytes

    def write_bytes(self, data):
        if self.name.startswith("panel_02"):
            real_write_bytes(self, data[: len(data) // 2])   # the disk fills up halfway through the file
            raise OSError(errno.ENOSPC, "No space left on device", str(self))
        return real_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", write_bytes)
    after = await redraw(job_env, job_id, 2)

    p2 = panel(after, 2)
    assert (job_dir / "panel_02.jpg").read_bytes() == old_bytes
    assert (p2["status"], p2["image_url"], p2["attempts"]) == ("done", before["image_url"], before["attempts"])
    assert p2["error"].startswith("redraw failed: OSError")
    assert after["status"] == "completed" and not list(job_dir.glob("*.tmp"))
    with Image.open(job_dir / "combined.jpg") as combined:
        assert combined.size == (1024, 3 * PANEL_H + 2 * GAP)


async def test_an_unreadable_panel_file_is_left_out_of_the_long_image(job_env):
    state = await job_env.run(qa=False)
    job_id, job_dir = state["job_id"], job_env.output_dir / state["job_id"]
    (job_dir / "panel_03.jpg").write_bytes(b"\xff\xd8\xff\xe0 no longer a picture")

    after = await redraw(job_env, job_id, 1)
    assert (after["status"], after["stage_message"]) == ("completed", "Done")
    assert after["combined_url"] != state["combined_url"]
    with Image.open(job_dir / "combined.jpg") as combined:
        assert combined.size == (1024, 2 * PANEL_H + GAP)   # panels 1 and 2


# ----------------------------------------------------------------------------- J9: memory per job

def test_j9_reference_sheets_are_read_once_and_shared_until_a_file_changes(tmp_path, monkeypatch):
    sheet_dir = tmp_path / "sheets"
    sheet_dir.mkdir()
    sheets = []
    for order, name in enumerate(("papi", "kumo", "nezu"), start=1):
        path = sheet_dir / f"{order}. {name}.png"
        path.write_bytes(image_bytes(4, 4, PALETTE[order]))
        sheets.append((order, name, path))
    monkeypatch.setattr(cast, "load_reference_sheets", lambda image_dir=None: list(sheets))

    _, first = cast.build_cast("kumomo")
    _, second = cast.build_cast("kumomo")
    assert first is not second and all(first[name] is second[name] for name in first)   # one copy per process

    papi = sheets[0][2]
    papi.write_bytes(image_bytes(8, 8, PALETTE[5]))   # the sheet is edited on disk
    stat = papi.stat()
    os.utime(papi, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    _, third = cast.build_cast("kumomo")
    assert third["papi"] is not first["papi"] and third["papi"].data == ImageContent.from_file(papi).data
    assert third["kumo"] is first["kumo"]


async def test_j9_a_finished_job_releases_its_cast_sheets_and_style_anchor(job_env):
    job_env.settings["style_anchor"] = True
    state = await job_env.run(qa=False)
    rt = job_env.manager.jobs[state["job_id"]].rt
    assert (rt.cast, rt.sheets, rt.anchor, rt.pdf_bytes, rt.text) == (None, {}, None, None, None)

    after = await redraw(job_env, state["job_id"], 2)   # a redraw rebuilds what it needs
    assert after["status"] == "completed" and panel(after, 2)["attempts"] == 2
    assert (rt.cast, rt.sheets, rt.anchor) == (None, {}, None)


# ----------------------------------------------------------------------------- lone surrogates in model replies

def assert_valid_utf8_json(path: Path):
    return json.loads(path.read_bytes().decode("utf-8"))   # strict: raises on surrogates or broken bytes


async def test_a_lone_surrogate_from_the_judge_cannot_break_job_json_or_the_api(job_env, monkeypatch):
    job_env.paper.num_panels = 1
    scripted_qa(monkeypatch, [verdict(9, 2, 2, issues=["the tail points at kumo \ud83d"])])   # half an emoji escape
    state = await job_env.run(qa=True)
    job_id = state["job_id"]

    assert state["status"] == "completed" and panel(state, 1)["status"] == "done"
    assert panel(state, 1)["qa"]["issues"] == ["the tail points at kumo \ufffd"]
    disk = assert_valid_utf8_json(job_env.output_dir / job_id / "job.json")
    assert panel(disk, 1)["qa"]["issues"] == ["the tail points at kumo \ufffd"]
    json.loads(JSONResponse(job_env.manager.get(job_id)).body)   # what GET /api/jobs/{id} sends
    json.loads(JSONResponse({"jobs": job_env.manager.list()}).body)


async def test_a_lone_surrogate_in_the_script_cannot_break_the_job_files(job_env):
    paper = job_env.paper

    def router(prompt, messages, config):
        reply = paper(prompt, messages, config)
        if not messages[-1].content.startswith("Write the manga script"):
            return reply

        async def with_broken_emoji():
            payload = await reply
            payload["title"] = "Attention, Please! \ud83d"
            payload["panels"][0]["dialogue"][0]["text"] += " \udc4d"
            return payload
        return with_broken_emoji()

    job_env.text.router = router
    state = await job_env.run(qa=False)
    job_dir = job_env.output_dir / state["job_id"]

    assert state["status"] == "completed" and all(p["status"] == "done" for p in state["panels"])
    assert state["title"] == "Attention, Please! \ufffd"
    for name in ("job.json", "storyboard.json"):
        assert_valid_utf8_json(job_dir / name)
    lines = (job_dir / "prompts.jsonl").read_bytes().decode("utf-8").splitlines()
    assert len(lines) == 3 and all(json.loads(line)["prompt"] for line in lines)
    json.loads(JSONResponse(job_env.manager.get(state["job_id"])).body)


def test_the_job_api_serves_a_job_json_holding_a_lone_surrogate_escape(job_env, monkeypatch):
    from fastapi.testclient import TestClient

    import main
    import routes.jobs as job_routes

    out, job_id = job_env.output_dir, "20261022_120000_d83d"
    persist_job(out, job_id, status="completed", title="Attention \ud83d")   # json.dumps keeps it as a \ud83d escape
    manager = jobs.JobManager(output_dir=out)
    monkeypatch.setattr(job_routes, "get_job_manager", lambda: manager)
    client = TestClient(main.app)   # no context manager: the lifespan (watchers, real engines) never starts
    try:
        one, listed = client.get(f"/api/jobs/{job_id}"), client.get("/api/jobs")
    finally:
        client.close()
    assert one.status_code == 200 and one.json()["title"] == "Attention \ufffd"
    assert listed.status_code == 200 and listed.json()["jobs"][0]["title"] == "Attention \ufffd"


# ----------------------------------------------------------------------------- P3: content-filter feedback

@pytest.mark.parametrize("error", [
    "OutputImageSensitiveContentDetected.PolicyViolation: the generated image may infringe copyright",
    "InputTextSensitiveContentDetected: the request may contain sensitive information",
])
async def test_p3_a_content_filter_rejection_steers_the_next_attempt(job_env, error):
    job_env.paper.num_panels = 1
    real = job_env.image.generate_image
    prompts = []

    async def seedream(prompt, config=None, reference_images=None):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise RuntimeError(error)
        return await real(prompt, config, reference_images)

    job_env.image.generate_image = seedream
    state = await job_env.run(qa=False)

    first, second = prompts
    assert "content filter" not in first
    assert "blocked by the content filter" in second
    assert "draw only the original characters from the reference sheets" in second
    assert "no logos, brands, celebrities or existing cartoon characters" in second
    assert (panel(state, 1)["status"], panel(state, 1)["attempts"]) == ("done", 2)


async def test_p3_other_failures_retry_without_content_filter_advice(job_env):
    job_env.paper.num_panels = 1
    real = job_env.image.generate_image
    prompts = []

    async def seedream(prompt, config=None, reference_images=None):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise RuntimeError("ReadTimeout: Seedream did not answer within 180 s")
        return await real(prompt, config, reference_images)

    job_env.image.generate_image = seedream
    state = await job_env.run(qa=False)

    assert len(prompts) == 2 and prompts[0] == prompts[1]   # the same prompt again, no feedback block
    assert "previous attempt" not in prompts[1]
    assert panel(state, 1)["status"] == "done"


# ----------------------------------------------------------------------------- tools/job_report.py

def report_rows(text: str) -> dict[int, list[str]]:
    return {int(line.split()[0]): line.split() for line in text.splitlines() if line[:3].strip().isdigit()}


async def test_job_report_counts_qa_retries_apart_from_user_redraws(job_env):
    job_env.judge.fail_first_for = {3}
    state = await job_env.run(qa=True)
    job_id = state["job_id"]
    for n in (1, 2):   # the user redraws two panels that had passed QA at once
        await redraw(job_env, job_id, n)

    text = job_report.report(job_env.output_dir / job_id)
    rows = report_rows(text)
    assert [rows[n][3] for n in (1, 2, 3)] == ["1", "1", "2"]   # "att": attempts of the latest render
    assert "retried: 1 " in text and "redrawn by the user: 2 " in text


def test_job_report_reads_job_json_written_before_render_attempts_and_available(tmp_path):
    job_dir = tmp_path / "20261021_173148_b7dc"
    job_dir.mkdir()
    old = {"job_id": job_dir.name, "status": "completed", "stage_message": "Done", "cost": {"images_generated": 3}, "panels": [
        {"n": 1, "chapter": "hook", "status": "done", "attempts": 2, "title": "One",
         "qa": {"score": 8.0, "text_exact": 2, "text_total": 2, "issues": []}},
        {"n": 2, "chapter": "hook", "status": "done", "attempts": 1, "render_attempts": 1, "title": "Two",
         "qa": {"score": 0.0, "text_exact": 0, "text_total": 0, "issues": ["qa unavailable: timeout"], "available": False}},
    ]}
    (job_dir / "job.json").write_text(json.dumps(old), encoding="utf-8")

    text = job_report.report(job_dir)
    rows = report_rows(text)
    assert (rows[1][3], rows[1][4]) == ("2", "8.0")   # no render_attempts: falls back to attempts
    assert (rows[2][3], rows[2][4]) == ("1", "n/a")   # an unavailable verdict has no score
    assert "retried: 1 " in text and "redrawn by the user: 0 " in text


# ----------------------------------------------------------------------------- the long image never fails a drawn manga

async def test_a_long_image_taller_than_jpeg_allows_is_narrowed(job_env, monkeypatch):
    """JPEG stops at 65,535 px: a 43+ panel manga used to fail at stitching with every panel drawn and paid for."""
    monkeypatch.setattr(jobs, "JPEG_MAX_SIDE", 2000)
    state = await job_env.run(qa=False)
    assert (state["status"], state["stage_message"]) == ("completed", "Done")
    with Image.open(job_env.output_dir / state["job_id"] / "combined.jpg") as combined:
        assert combined.height <= 2000 and combined.width < 1024


async def test_a_stitching_failure_completes_the_manga_without_the_long_image(job_env, monkeypatch):
    def broken(self, job, *args, **kwargs):
        raise OSError("broken data stream when writing image file")

    monkeypatch.setattr(jobs.JobManager, "_assemble", broken)
    state = await job_env.run(qa=False)
    assert state["status"] == "completed" and state["combined_url"] is None
    assert "the long image could not be built: OSError" in state["stage_message"]
    assert [p["status"] for p in state["panels"]] == ["done"] * len(state["panels"])


# ----------------------------------------------------------------------------- blind transcription in a real run

async def test_a_line_misdrawn_but_passed_by_the_judge_is_redrawn_when_the_blind_reading_disagrees(job_env):
    """The judge (shown the script) passes every panel; the script-free reading of panel 2's first image disagrees."""
    job_env.settings["qa_blind_check"] = True
    judge, blind_calls = job_env.vision.router, []

    def router(prompt, messages, config):
        if BLIND_PROMPT[:60] not in prompt:
            return judge(prompt, messages, config)
        blind_calls.append(prompt)
        lines = [t for n in range(1, 4) for t in (f"Fact number {n} from the paper.", f"Line for panel {n} please.")]
        if len(blind_calls) == 2:                     # one image comes back with a misspelt line
            return {"texts": [t.replace("Line", "Lne") for t in lines]}
        return {"texts": lines}

    job_env.vision.router = router
    state = await job_env.run()
    assert state["status"] == "completed"
    assert len(job_env.image.calls) == 4               # 3 panels + 1 redraw
    assert sorted(p["render_attempts"] for p in state["panels"]) == [1, 1, 2]
    assert all(p["qa"]["text_exact"] == p["qa"]["text_total"] for p in state["panels"])
    assert len(blind_calls) == 4


# ----------------------------------------------------------------------------- adopting the drawn wording

ALT = "第1格的台句"   # a same-length ideograph change of the fake zh line 第1格的台词


def zh_routers(job_env, *, accept=True, verdicts=None):
    """Vision: panel 1's judge verdicts come from ``verdicts`` (attempt -> (rendered or None, overall)); the blind reader
    reads what that verdict says is drawn. Text: the wording check answers ``accept``."""
    judge, paper = job_env.vision.router, job_env.text.router
    verdicts = verdicts or {1: (ALT, 9)}
    state = {"qa": 0, "drawn": "第1格的台词", "checks": []}

    def vision(prompt, messages, config):
        if BLIND_PROMPT[:60] in prompt:
            lines = [t for n in (1, 2, 3) for t in (f"论文中的第{n}个事实。", state["drawn"] if n == 1 else f"第{n}格的台词")]
            return {"texts": lines}
        verdict = judge(prompt, messages, config)
        if "第1格的台词" in prompt:
            state["qa"] += 1
            rendered, overall = verdicts.get(state["qa"], (None, 9))
            state["drawn"] = rendered or "第1格的台词"
            if rendered:
                verdict["panels"][0]["text_items"][0].update(exact=False, rendered=rendered)
            verdict["overall"] = overall
        return verdict

    def text(prompt, messages, config):
        if messages[-1].content.startswith("You check the lettering"):
            state["checks"].append(messages[-1].content)
            return accept if isinstance(accept, (str, BaseException)) else {"accept": accept, "reason": "plain synonym"}
        return paper(prompt, messages, config)

    job_env.vision.router, job_env.text.router = vision, text
    job_env.settings["qa_blind_check"] = True
    return state


async def test_a_wording_both_readers_agree_on_is_adopted_when_its_attempt_is_published(job_env):
    routing = zh_routers(job_env)
    state = await job_env.run(language="zh-CN")
    assert state["status"] == "completed" and len(job_env.image.calls) == 3      # adopted instead of redrawn
    p1 = panel(state, 1)
    assert p1["dialogue"][0]["text"] == ALT and p1["qa"]["text_exact"] == p1["qa"]["text_total"]
    assert state["text_adjustments"] == [{"panel": 1, "from": "第1格的台词", "to": ALT}]
    saved = json.loads((job_env.output_dir / state["job_id"] / "storyboard.json").read_text(encoding="utf-8"))
    assert saved["panels"][0]["dialogue"][0]["text"] == ALT
    assert "第1格的台词" in routing["checks"][0] and ALT in routing["checks"][0] and "Simplified Chinese" in routing["checks"][0]


@pytest.mark.parametrize("accept", [False, "no json", RuntimeError("text model down")])
async def test_a_rejected_or_unchecked_wording_is_redrawn_and_never_enters_the_script(job_env, accept):
    zh_routers(job_env, accept=accept)
    state = await job_env.run(language="zh-CN")
    assert len(job_env.image.calls) == 4                                       # panel 1 redrawn
    assert panel(state, 1)["dialogue"][0]["text"] == "第1格的台词" and "text_adjustments" not in state


async def test_wording_adopted_on_an_attempt_that_is_not_published_stays_out_of_the_script(job_env):
    """Attempt 1 draws the script line (score 5, retried); attempt 2 draws the synonym (adopted, score 4): attempt 1 wins
    the ranking, so its picture and the script's own line are published together."""
    zh_routers(job_env, verdicts={1: (None, 5), 2: (ALT, 4)})
    state = await job_env.run(language="zh-CN")
    p1 = panel(state, 1)
    assert p1["render_attempts"] == 2 and p1["qa"]["score"] == 5
    assert p1["dialogue"][0]["text"] == "第1格的台词" and "text_adjustments" not in state
    saved = json.loads((job_env.output_dir / state["job_id"] / "storyboard.json").read_text(encoding="utf-8"))
    assert saved["panels"][0]["dialogue"][0]["text"] == "第1格的台词"


async def test_a_failed_storyboard_write_keeps_the_script_and_publishes_the_verdict_as_judged(job_env, monkeypatch):
    zh_routers(job_env)
    real = jobs.JobManager._write_atomic

    def write_atomic(path, data):
        if path.name == "storyboard.json" and ALT in data.decode("utf-8"):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(path, data)

    monkeypatch.setattr(jobs.JobManager, "_write_atomic", staticmethod(write_atomic))
    state = await job_env.run(language="zh-CN")
    p1 = panel(state, 1)
    assert p1["dialogue"][0]["text"] == "第1格的台词" and "text_adjustments" not in state
    assert p1["qa"]["text_exact"] < p1["qa"]["text_total"]                     # the picture shows the other wording


async def test_p3_without_reference_sheets_the_retry_asks_for_original_mascots(job_env, monkeypatch):
    """A sheet-less theme has no sheets to point at: the retry must not promise any (it would contradict itself)."""
    monkeypatch.setattr(jobs, "build_cast", lambda theme: (make_cast(), {}))
    job_env.paper.num_panels = 1
    real, prompts = job_env.image.generate_image, []

    async def seedream(prompt, config=None, reference_images=None):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise RuntimeError("OutputImageSensitiveContentDetected.PolicyViolation: may infringe copyright")
        return await real(prompt, config, reference_images)

    job_env.image.generate_image = seedream
    await job_env.run(qa=False)
    assert "draw simple original mascots that follow the descriptions above" in prompts[1]
    assert "reference sheets" not in prompts[1].split("fix them:")[1]
    assert "Never copy the captions on the reference sheets" not in prompts[1]


# ----------------------------------------------------------------------------- rewriting a line lettered wrong twice

def rewrite_routers(job_env, rewrite, accept=True):
    """Panel 1's line 第1格的台词 is lettered as 第1格的台问 every time; any other line passes. The text model answers the
    rewrite request with ``rewrite``."""
    judge, paper = job_env.vision.router, job_env.text.router
    seen = {"rewrites": [], "qa_prompts": []}

    def vision(prompt, messages, config):
        verdict = judge(prompt, messages, config)
        if "第1格的台词" in prompt:
            seen["qa_prompts"].append(prompt)
            verdict["panels"][0]["text_items"][0].update(exact=False, rendered="第1格的台问")
        return verdict

    def text(prompt, messages, config):
        if messages[-1].content.startswith("An image AI letters the lines"):
            seen["rewrites"].append(messages[-1].content)
            return {"text": rewrite}
        if messages[-1].content.startswith("You check the lettering"):
            seen.setdefault("checks", []).append(messages[-1].content)
            return {"accept": accept, "reason": "same meaning"}
        return paper(prompt, messages, config)

    job_env.vision.router, job_env.text.router = vision, text
    job_env.settings["max_attempts"] = 3
    return seen


async def test_a_line_lettered_wrong_twice_is_rewritten_for_the_last_attempt_and_published(job_env):
    seen = rewrite_routers(job_env, "第1格的台语")   # 台 is a counter (meaning-critical), so a rewrite must keep it
    state = await job_env.run(language="zh-CN")
    prompts = [c["prompt"] for c in job_env.image.calls if "第1格" in c["prompt"]]
    assert len(prompts) == 3 and "第1格的台词" in prompts[1] and "第1格的台语" in prompts[2]
    assert "第1格的台词" not in prompts[2]                       # neither the line nor the retry feedback quotes it
    assert len(seen["rewrites"]) == 1 and "第1格的台问" in seen["rewrites"][0]
    p1 = panel(state, 1)
    assert p1["dialogue"][0]["text"] == "第1格的台语" and p1["qa"]["text_exact"] == p1["qa"]["text_total"]
    assert state["text_adjustments"] == [{"panel": 1, "from": "第1格的台词", "to": "第1格的台语"}]
    saved = json.loads((job_env.output_dir / state["job_id"] / "storyboard.json").read_text(encoding="utf-8"))
    assert saved["panels"][0]["dialogue"][0]["text"] == "第1格的台语"


@pytest.mark.parametrize("rewrite", ["第2格的对话", "", "Panel one line", "第1格的对话，以及很多很多很多额外的话"])
async def test_a_rewrite_that_changes_a_number_drops_the_script_or_runs_long_is_refused(job_env, rewrite):
    rewrite_routers(job_env, rewrite)
    state = await job_env.run(language="zh-CN")
    prompts = [c["prompt"] for c in job_env.image.calls if "第1格" in c["prompt"]]
    assert len(prompts) == 3 and all("第1格的台词" in p for p in prompts)
    assert panel(state, 1)["dialogue"][0]["text"] == "第1格的台词" and "text_adjustments" not in state


@pytest.mark.parametrize("rewrite", ["第1格的对话", "第1格的台语"])
async def test_a_rewrite_is_used_only_after_the_text_model_confirms_its_meaning(job_env, rewrite):
    """第1格的对话 drops the counter 台 (meaning-critical); 第1格的台语 passes the guards but the checker says no."""
    seen = rewrite_routers(job_env, rewrite, accept=False)
    state = await job_env.run(language="zh-CN")
    prompts = [c["prompt"] for c in job_env.image.calls if "第1格" in c["prompt"]]
    assert all(rewrite not in p.split("fix them:")[0] for p in prompts)
    assert panel(state, 1)["dialogue"][0]["text"] == "第1格的台词" and "text_adjustments" not in state
    assert len(seen.get("checks", [])) == (1 if rewrite == "第1格的台语" else 0)


async def test_random_misdrawings_are_retried_but_never_rewritten(job_env):
    """A rewrite is for a word Seedream cannot letter: two different glitches are left to the retry."""
    judge = job_env.vision.router            # the plain judge: everything exact unless changed below
    seen = rewrite_routers(job_env, "第1格的台语")
    count = {"n": 0}

    def vision(prompt, messages, config):
        verdict = judge(prompt, messages, config)
        if "第1格的台词" in prompt:
            count["n"] += 1
            if count["n"] <= 2:   # two different glitches, then a clean attempt
                verdict["panels"][0]["text_items"][0].update(exact=False, rendered=["第1格的台问", "第1格的抬词"][count["n"] - 1])
        return verdict

    job_env.vision.router = vision
    state = await job_env.run(language="zh-CN")
    assert seen["rewrites"] == [] and count["n"] == 3
    assert panel(state, 1)["render_attempts"] == 3 and panel(state, 1)["qa"]["text_exact"] == panel(state, 1)["qa"]["text_total"]


async def test_english_lines_are_never_rewritten(job_env):
    judge = job_env.vision.router
    seen = rewrite_routers(job_env, "Line for panel 1, please!")

    def vision(prompt, messages, config):
        verdict = judge(prompt, messages, config)
        if "Line for panel 1 please." in prompt:
            verdict["panels"][0]["text_items"][0].update(exact=False, rendered="Lne for panel 1 please.")
        return verdict

    job_env.vision.router = vision
    await job_env.run(language="en-US")
    assert seen["rewrites"] == [] and len([c for c in job_env.image.calls if "panel 1" in c["prompt"]]) == 3


# ----------------------------------------------------------------------------- second review: state after rewrites/adoptions

async def test_after_a_rewrite_no_feedback_quotes_the_old_line_but_a_wrong_tail_still_applies(job_env):
    seen = rewrite_routers(job_env, "第1格的台语")
    failing = job_env.vision.router

    def vision(prompt, messages, config):
        verdict = failing(prompt, messages, config)
        if "第1格的台词" in prompt:
            p = verdict["panels"][0]
            p["text_items"][0]["speaker_ok"] = False
            p["issues"] = ["Speech bubble text has an incorrect character '问': '第1格的台问' instead of expected '第1格的台词'"]
        return verdict

    job_env.vision.router = vision
    await job_env.run(language="zh-CN")
    third = [c["prompt"] for c in job_env.image.calls if "第1格" in c["prompt"]][2]
    assert "第1格的台词" not in third and "台问" not in third
    assert "the tail of papi's bubble \"第1格的台语\" must point at papi" in third
    assert seen["rewrites"]


async def test_a_failed_image_write_after_an_adoption_leaves_the_script_untouched(job_env, monkeypatch):
    zh_routers(job_env)
    real = jobs.JobManager._write_atomic

    def write_atomic(path, data):
        if path.name == "panel_01.jpg":
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(path, data)

    monkeypatch.setattr(jobs.JobManager, "_write_atomic", staticmethod(write_atomic))
    state = await job_env.run(language="zh-CN")
    p1 = panel(state, 1)
    assert p1["status"] == "failed" and p1["dialogue"][0]["text"] == "第1格的台词" and "text_adjustments" not in state
    saved = json.loads((job_env.output_dir / state["job_id"] / "storyboard.json").read_text(encoding="utf-8"))
    assert saved["panels"][0]["dialogue"][0]["text"] == "第1格的台词"


async def test_a_rewrite_the_script_cannot_take_is_published_as_not_exact(job_env, monkeypatch):
    rewrite_routers(job_env, "第1格的台语")
    plain = job_env.vision.router

    def vision(prompt, messages, config):   # the rewritten line is lettered correctly on attempt 3
        if "第1格的台语" in prompt:
            return exact_after(prompt, messages, config)
        return plain(prompt, messages, config)

    exact_after = job_env.judge
    job_env.vision.router = vision
    real = jobs.JobManager._write_atomic

    def write_atomic(path, data):
        if path.name == "storyboard.json" and "第1格的台语" in data.decode("utf-8"):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(path, data)

    monkeypatch.setattr(jobs.JobManager, "_write_atomic", staticmethod(write_atomic))
    state = await job_env.run(language="zh-CN")
    p1 = panel(state, 1)
    assert p1["dialogue"][0]["text"] == "第1格的台词" and p1["qa"]["text_exact"] < p1["qa"]["text_total"]
    assert p1["qa"]["issues"][0] == 'the picture letters "第1格的台语" but the script still reads "第1格的台词"'


async def test_an_adopted_wording_is_lettered_by_the_following_attempts(job_env):
    """Attempt 1's 第1格的台句 is adopted but the page scores 5, so it is redrawn; attempt 2 letters the adopted line."""
    zh_routers(job_env, verdicts={1: (ALT, 5), 2: (None, 9)})
    state = await job_env.run(language="zh-CN")
    prompts = [c["prompt"] for c in job_env.image.calls if "第1格" in c["prompt"]]
    assert len(prompts) == 2 and f'"{ALT}"' in prompts[1]
    assert panel(state, 1)["dialogue"][0]["text"] == ALT and state["text_adjustments"][0]["to"] == ALT


async def test_on_a_tie_the_attempt_drawn_with_the_script_wins_over_a_failed_rewrite(job_env):
    """The original line is misdrawn twice, the rewrite is misdrawn too, every attempt scores the same: the script stays."""
    seen = rewrite_routers(job_env, "第1格的台语")
    failing = job_env.vision.router

    def vision(prompt, messages, config):
        verdict = failing(prompt, messages, config)
        if "第1格的台语" in prompt:
            verdict["panels"][0]["text_items"][0].update(exact=False, rendered="第1格的抬语")
        return verdict

    job_env.vision.router = vision
    state = await job_env.run(language="zh-CN")
    assert seen["rewrites"] and panel(state, 1)["render_attempts"] == 3
    assert panel(state, 1)["dialogue"][0]["text"] == "第1格的台词" and "text_adjustments" not in state


async def test_every_publish_resyncs_the_readers_text_with_the_script(job_env):
    """job.json written before storyboard.json caught up (e.g. a restart between the two writes) heals on the next draw."""
    state = await job_env.run(language="zh-CN", qa=False)
    job = job_env.manager.jobs[state["job_id"]]
    panel(job.state, 1)["dialogue"][0]["text"] = "stale text"
    after = await redraw(job_env, state["job_id"], 1)
    assert panel(after, 1)["dialogue"][0]["text"] == "第1格的台词"


async def test_the_same_character_failing_twice_triggers_a_rewrite_even_when_drawn_differently(job_env):
    """BERT zh: 掩 came out as 密 in one attempt and as 掎 in the next; the word is what Seedream cannot letter."""
    seen = rewrite_routers(job_env, "第1格的台语")
    failing = job_env.vision.router
    count = {"n": 0}

    def vision(prompt, messages, config):
        verdict = failing(prompt, messages, config)
        if "第1格的台词" in prompt:
            count["n"] += 1
            verdict["panels"][0]["text_items"][0].update(exact=False, rendered=["第1格的台问", "第1格的台同"][min(count["n"], 2) - 1])
        return verdict

    job_env.vision.router = vision
    await job_env.run(language="zh-CN")
    assert len(seen["rewrites"]) == 1 and "Characters it keeps getting wrong: 词" in seen["rewrites"][0]


def test_failing_chars():
    from services.jobs import _failing_chars
    assert _failing_chars("这15%里：80%掩码", "这15%里：80%密码") == {"掩"}
    assert _failing_chars("掩码训练", "") == set("掩码训练")
    assert _failing_chars("Use only attention.", "Use only attention") == set()
