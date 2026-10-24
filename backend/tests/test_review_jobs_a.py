"""Regression tests for the job-state review fixes J1-J4 (services/jobs.py and the frontend cache-key contract).

J1  a regenerated panel must not keep showing the old image: the reader's cache key is ``image_url`` + ``?v=<attempts>``
    (frontend/src/lib/job-utils.ts), and ``combined_url`` gets a fresh ``?v=`` whenever combined.jpg is rebuilt
J2  "Regenerate" after "Stop" must not resurrect the cancelled run
J3  one broken worker may only fail its own panels: never the whole job, never leaving sibling workers running
J4  after a server restart, panels that were rendering / in QA must not stay "in progress" forever

Tests marked ``xfail(strict=True)`` describe the correct behaviour of a defect that is still present; once it is fixed
they XPASS, which strict mode reports as a failure: delete the marker then and keep the test as a regression test.
Every engine is a fake (conftest.job_env) and conftest blocks the network.
"""

import asyncio
import errno
import itertools
import json
import random
import re
import time as _real_time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import services.jobs as jobs
from engines.base import ImageContent, ImageResponse

from .fakes import LONG_PAPER_TEXT, b64, image_bytes, make_panel

WORKER = "JobManager._render_all.<locals>.worker"
ACTIVE = ("rendering", "qa")


# ----------------------------------------------------------------------------- helpers

def frontend_image_key(state: dict, n: int):
    """Exact mirror of ``buildImageVersions`` + ``versionedUrl`` in frontend/src/lib/job-utils.ts: the URL the reader
    puts in ``<img src>`` for panel ``n`` (strip partners share one image, so the version is the max over them).
    If the frontend changes how it derives the cache key, update this helper with it."""
    panels = state["panels"]
    url = next(p for p in panels if p["n"] == n).get("image_url")
    if not url:
        return None
    version = max((p.get("attempts") or 0) for p in panels if p.get("image_url") == url)
    if not version or url.startswith(("data:", "blob:")):
        return url
    return f"{url}{'&' if '?' in url else '?'}v={version}"


def panel(state: dict, n: int) -> dict:
    return next(p for p in state["panels"] if p["n"] == n)


def file_of(job_dir: Path, url: str) -> Path:
    return job_dir / url.split("?", 1)[0].rsplit("/", 1)[-1]


def live_workers() -> list:
    return [t for t in asyncio.all_tasks() if not t.done() and getattr(t.get_coro(), "__qualname__", "") == WORKER]


async def drain(tasks) -> None:
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def stable(state: dict) -> dict:
    """Public state without the fields that legitimately change on every persist."""
    out = dict(state, updated_at=None)
    out["cost"] = dict(state["cost"], elapsed_seconds=None)
    return out


async def wait_for(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0.001)


def on_disk(job_dir: Path) -> dict:
    return json.loads((job_dir / "job.json").read_text(encoding="utf-8"))


def serve_image_for(job_env, n: int, image: ImageContent) -> None:
    """From now on the fake Seedream answers every prompt for panel ``n`` with ``image``."""
    real = job_env.image.generate_image

    async def generate_image(prompt, config=None, reference_images=None):
        if f"Line for panel {n} " not in prompt:
            return await real(prompt, config, reference_images)
        job_env.image.calls.append({"prompt": prompt, "config": config, "refs": list(reference_images or [])})
        await asyncio.sleep(0)
        response = ImageResponse(images=[image], model="fake-seedream", prompt=prompt)
        response.usage, response.size = {"output_tokens": 6144}, "1024x1536"
        return response

    job_env.image.generate_image = generate_image


def truncated_jpeg(width: int = 256, height: int = 384) -> bytes:
    """A JPEG whose headers are intact but whose scan data is cut short (Image.open/verify accept it, load() fails)."""
    buf = BytesIO()
    Image.frombytes("RGB", (width, height), random.Random(7).randbytes(width * height * 3)).save(buf, "JPEG", quality=85)
    data = buf.getvalue()
    return data[: int(len(data) * 0.9)]


async def slow_seedream(prompt) -> None:
    await asyncio.sleep(0.02)  # real renders take a while, so sibling workers overlap


def persist_job(out: Path, job_id: str, **changes) -> Path:
    state = jobs._new_state(job_id, title="Persisted", theme="kumomo", language="en-US", layout=changes.pop("layout", "single"),
                            qa=True, panel_range="auto", fast=False, source="text")
    state.pop("_started")
    state.update(changes)
    (out / job_id).mkdir(parents=True, exist_ok=True)
    (out / job_id / "job.json").write_text(json.dumps(state), encoding="utf-8")
    return out / job_id


async def regenerate_and_poll(job_env, job_id: str, n: int):
    """Regenerate panel ``n``; return (the public states a GET returned while Seedream was drawing it, the final state)."""
    polls = []

    async def poll(prompt):
        polls.append(job_env.manager.get(job_id))

    job_env.image.hook = poll
    assert await job_env.manager.regenerate_panel(job_id, n) == {"status": "queued"}
    await asyncio.wait_for(job_env.manager.jobs[job_id].rt.task, timeout=10)
    job_env.image.hook = None
    return polls, job_env.manager.get(job_id)


# ----------------------------------------------------------------------------- J1: cache keys after a redraw

async def test_j1_attempts_accumulate_so_the_panel_cache_key_changes_after_each_redraw(job_env):
    state = await job_env.run(qa=True)
    job_id = state["job_id"]
    assert [p["attempts"] for p in state["panels"]] == [1, 1, 1]
    keys = [frontend_image_key(state, 2)]

    _, after = await regenerate_and_poll(job_env, job_id, 2)
    assert [p["attempts"] for p in after["panels"]] == [1, 2, 1]  # cumulative, not reset to 1
    keys.append(frontend_image_key(after, 2))

    job_env.judge.fail_first_for = {2}
    job_env.judge.judged.clear()
    _, after = await regenerate_and_poll(job_env, job_id, 2)  # QA rejects the first try: two more attempts
    assert [p["attempts"] for p in after["panels"]] == [1, 4, 1]
    keys.append(frontend_image_key(after, 2))

    assert len(set(keys)) == 3
    assert frontend_image_key(after, 1) == frontend_image_key(state, 1)  # untouched panels keep their key (stay cached)
    assert frontend_image_key(after, 3) == frontend_image_key(state, 3)


async def test_j1_strip_partners_share_the_new_cache_key(job_env):
    job_env.paper.num_panels = 4
    state = await job_env.run(layout="strip", qa=False)
    _, after = await regenerate_and_poll(job_env, state["job_id"], 2)

    assert [p["attempts"] for p in after["panels"]] == [2, 2, 1, 1]
    assert frontend_image_key(after, 1) == frontend_image_key(after, 2) != frontend_image_key(state, 2)
    assert frontend_image_key(after, 3) == frontend_image_key(state, 3)


class SteppingClock:
    """Stands in for the ``time`` module inside services.jobs: ``time()`` advances one second per call (the combined
    image version has one-second resolution); everything else is the real module."""

    def __init__(self, start: int = 1_800_000_000):
        self._ticks = itertools.count(start)

    def time(self) -> float:
        return float(next(self._ticks))

    def time_ns(self) -> int:
        return int(self.time() * 1_000_000_000)

    def __getattr__(self, name):
        return getattr(_real_time, name)


async def test_j1_combined_url_gets_a_new_version_whenever_it_is_reassembled(job_env, monkeypatch):
    monkeypatch.setattr(jobs, "time", SteppingClock())
    state = await job_env.run(qa=False)
    job_id = state["job_id"]
    combined = job_env.output_dir / job_id / "combined.jpg"
    first, first_bytes = state["combined_url"], combined.read_bytes()
    assert re.fullmatch(rf"/output/{job_id}/combined\.jpg\?v=\d+", first)

    _, after = await regenerate_and_poll(job_env, job_id, 2)
    assert re.fullmatch(rf"/output/{job_id}/combined\.jpg\?v=\d+", after["combined_url"])
    assert after["combined_url"] != first
    assert combined.read_bytes() != first_bytes
    assert on_disk(combined.parent)["combined_url"] == after["combined_url"]  # survives a restart


@pytest.mark.parametrize("layout, qa, qa_rejects_first", [("single", False, False), ("single", True, True), ("strip", False, False)],
                         ids=["single", "single-qa-retry", "strip"])
async def test_j1_cache_key_polled_during_a_redraw_is_not_reused_for_the_new_image(job_env, layout, qa, qa_rejects_first):
    job_env.paper.num_panels = 4
    state = await job_env.run(layout=layout, qa=qa)
    job_id = state["job_id"]
    job_dir = job_env.output_dir / job_id
    old_bytes = file_of(job_dir, panel(state, 2)["image_url"]).read_bytes()
    if qa_rejects_first:
        job_env.judge.fail_first_for = {2}
        job_env.judge.judged.clear()

    polls, after = await regenerate_and_poll(job_env, job_id, 2)
    if not polls or file_of(job_dir, panel(after, 2)["image_url"]).read_bytes() == old_bytes:
        raise RuntimeError("scenario broken: no poll during the redraw, or the redraw produced no new image")

    # Every poll above was answered while the old file was still on disk (the new one is written at the very end),
    # so whatever key the reader loaded then now holds the old picture in the browser's cache.
    keys_holding_old_bytes = {frontend_image_key(p, 2) for p in polls}
    assert frontend_image_key(after, 2) not in keys_holding_old_bytes


# ----------------------------------------------------------------------------- J2: regenerate after Stop

async def test_j2_stop_ends_the_run_and_a_later_redraw_does_not_resurrect_it(job_env):
    job_env.settings["concurrency"] = 1
    job_env.paper.num_panels = 5
    never = asyncio.Event()

    async def hang_on_panel_2(prompt):  # Seedream never answers: only a real task cancellation ends this call
        if "Line for panel 2" in prompt:
            await never.wait()

    job_env.image.hook = hang_on_panel_2
    job_id = job_env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)["job_id"]
    job = job_env.manager.jobs[job_id]
    await wait_for(lambda: len(job_env.image.calls) == 2)
    run_task = job.rt.task

    assert (await job_env.manager.cancel(job_id))["status"] == "cancelled"
    if not run_task.done():  # while the cancelled run unwinds, a redraw is refused (HTTP 409), not started next to it
        with pytest.raises(ValueError, match="still running"):
            await job_env.manager.regenerate_panel(job_id, 1)
    await asyncio.wait_for(run_task, timeout=5)  # ends even though the hung Seedream call never returned
    assert not job.running and live_workers() == []
    stopped = job_env.manager.get(job_id)
    assert stopped["status"] == "cancelled"
    assert [p["status"] for p in stopped["panels"]] == ["done", "pending", "pending", "pending", "pending"]
    assert len(job_env.image.calls) == 2 and job_env.image.closed >= 1  # the run went through its cleanup

    job_env.image.hook = None
    assert await job_env.manager.regenerate_panel(job_id, 1) == {"status": "queued"}
    assert job.rt.task is not run_task
    await asyncio.wait_for(job.rt.task, timeout=10)
    await asyncio.sleep(0.05)  # a resurrected run would now go on drawing panels 2-5

    final = job_env.manager.get(job_id)
    redraws = job_env.image.calls[2:]
    assert len(redraws) == 1 and "Line for panel 1 " in redraws[0]["prompt"]
    assert [p["status"] for p in final["panels"]] == ["done", "pending", "pending", "pending", "pending"]
    assert panel(final, 1)["attempts"] > panel(stopped, 1)["attempts"]
    assert [p["attempts"] for p in final["panels"][1:]] == [p["attempts"] for p in stopped["panels"][1:]]
    assert final["status"] == "cancelled" and "still not drawn" in final["stage_message"]
    assert on_disk(job_env.output_dir / job_id)["status"] == "cancelled"


async def test_j2_a_second_redraw_is_refused_while_the_first_one_runs(job_env):
    state = await job_env.run(qa=False)
    job_id = state["job_id"]
    assert await job_env.manager.regenerate_panel(job_id, 2) == {"status": "queued"}
    first = job_env.manager.jobs[job_id].rt.task
    with pytest.raises(ValueError, match="still running"):
        await job_env.manager.regenerate_panel(job_id, 3)
    assert job_env.manager.jobs[job_id].rt.task is first  # not replaced by a second, concurrent _regenerate
    await asyncio.wait_for(first, timeout=10)

    final = job_env.manager.get(job_id)
    assert (final["status"], [p["attempts"] for p in final["panels"]]) == ("completed", [1, 2, 1])
    assert len(job_env.image.calls) == 4


async def test_j2_stop_during_a_redraw_keeps_the_old_image_and_a_new_redraw_works(job_env):
    state = await job_env.run(qa=False)
    job_id = state["job_id"]
    job_dir = job_env.output_dir / job_id
    old_url, old_bytes = panel(state, 2)["image_url"], (job_dir / "panel_02.jpg").read_bytes()
    never = asyncio.Event()

    async def hang(prompt):
        await never.wait()

    job_env.image.hook = hang
    assert await job_env.manager.regenerate_panel(job_id, 2) == {"status": "queued"}
    task = job_env.manager.jobs[job_id].rt.task
    await wait_for(lambda: len(job_env.image.calls) == 4)
    assert (await job_env.manager.cancel(job_id))["status"] == "cancelled"
    await asyncio.wait_for(task, timeout=5)

    stopped = job_env.manager.get(job_id)
    assert stopped["status"] in jobs.TERMINAL
    assert (panel(stopped, 2)["status"], panel(stopped, 2)["image_url"]) == ("done", old_url)
    assert not any(p["status"] in ACTIVE for p in stopped["panels"])
    assert (job_dir / "panel_02.jpg").read_bytes() == old_bytes

    job_env.image.hook = None
    assert await job_env.manager.regenerate_panel(job_id, 2) == {"status": "queued"}
    await asyncio.wait_for(job_env.manager.jobs[job_id].rt.task, timeout=10)
    final = job_env.manager.get(job_id)
    assert (final["status"], final["stage_message"]) == ("completed", "Done")
    assert (job_dir / "panel_02.jpg").read_bytes() != old_bytes and len(job_env.image.calls) == 5


def test_j2_regenerate_route_answers_409_while_a_cancelled_run_is_still_unwinding(job_env, monkeypatch):
    from fastapi.testclient import TestClient

    import main
    import routes.jobs as job_routes

    job_id = "20261022_101500_b2b2"
    persist_job(job_env.output_dir, job_id, status="completed", panels=[dict(jobs._panel_state(make_panel(1)), status="done")])
    manager = job_env.manager
    monkeypatch.setattr(job_routes, "get_job_manager", lambda: manager)
    job = manager._load(job_id)
    job.state["status"] = "cancelled"  # Stop was pressed: the status is terminal already...
    job.rt.task = SimpleNamespace(done=lambda: False, cancel=lambda: None)  # ...but the run task has not finished yet

    client = TestClient(main.app)  # no context manager: the lifespan (watchers, real engines) never starts
    try:
        response = client.post(f"/api/jobs/{job_id}/panels/1/regenerate")
    finally:
        client.close()
    assert response.status_code == 409 and response.json()["detail"] == "Job is still running"


# ----------------------------------------------------------------------------- J3: one broken worker

async def test_j3_an_undecodable_image_fails_only_its_own_panel(job_env):
    job_env.settings["concurrency"] = 2
    job_env.paper.num_panels = 6
    job_env.image.hook = slow_seedream
    serve_image_for(job_env, 1, ImageContent.from_base64(b64(b"<html>502 Bad Gateway</html>"), "image/jpeg"))

    job_id = job_env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)["job_id"]
    await asyncio.wait_for(job_env.manager.jobs[job_id].rt.task, timeout=10)
    final, calls, leftovers = job_env.manager.get(job_id), len(job_env.image.calls), live_workers()
    await asyncio.sleep(0.1)

    assert leftovers == [] and len(job_env.image.calls) == calls  # nothing keeps drawing after the run returned
    assert final["status"] == "completed" and "[1]" in final["stage_message"]
    p1 = panel(final, 1)
    assert (p1["status"], p1["image_url"], p1["attempts"]) == ("failed", None, 2)  # retried, then failed alone
    assert "UnidentifiedImageError" in p1["error"]
    assert [p["status"] for p in final["panels"][1:]] == ["done"] * 5
    with Image.open(job_env.output_dir / job_id / "combined.jpg") as combined:
        assert combined.size == (1024, 5 * 1536 + 4 * 24)


async def test_j3_an_image_url_instead_of_inline_data_is_rejected_per_attempt(job_env):
    serve_image_for(job_env, 2, ImageContent.from_url("https://cdn.example.invalid/panel.jpg", "image/jpeg"))
    state = await job_env.run(qa=False)

    assert state["status"] == "completed"
    p2 = panel(state, 2)
    assert (p2["status"], p2["attempts"]) == ("failed", 2) and "image URL" in p2["error"]
    assert [p["status"] for p in state["panels"]] == ["done", "failed", "done"]


async def test_j3_a_disk_error_saving_one_panel_fails_only_that_panel(job_env, monkeypatch):
    real_write_bytes = Path.write_bytes

    def write_bytes(self, data):
        if self.name.startswith("panel_02"):
            raise OSError(errno.ENOSPC, "No space left on device", str(self))
        return real_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", write_bytes)
    job_env.settings["concurrency"] = 3
    job_env.image.hook = slow_seedream
    state = await job_env.run(qa=False)

    assert state["status"] == "completed" and live_workers() == []
    p2 = panel(state, 2)
    assert p2["status"] == "failed" and "No space left on device" in p2["error"]
    assert [p["status"] for p in state["panels"]] == ["done", "failed", "done"]
    with Image.open(job_env.output_dir / state["job_id"] / "combined.jpg") as combined:
        assert combined.size == (1024, 2 * 1536 + 24)


async def test_j3_stop_leaves_no_worker_running(job_env):
    job_env.settings["concurrency"] = 2
    job_env.paper.num_panels = 4
    started, interrupted = [], []
    never = asyncio.Event()

    async def hang(prompt):
        started.append(prompt)
        try:
            await never.wait()
        except asyncio.CancelledError:
            interrupted.append(prompt)
            raise

    job_env.image.hook = hang
    job_id = job_env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)["job_id"]
    await wait_for(lambda: len(started) == 2)
    await job_env.manager.cancel(job_id)
    await asyncio.wait_for(job_env.manager.jobs[job_id].rt.task, timeout=5)

    assert live_workers() == [] and len(interrupted) == 2  # both in-flight renders were cancelled, not abandoned
    final = job_env.manager.get(job_id)
    assert final["status"] == "cancelled" and [p["status"] for p in final["panels"]] == ["pending"] * 4


async def test_j3_a_truncated_jpeg_fails_only_its_panel_not_the_whole_job(job_env):
    bad = truncated_jpeg()
    with Image.open(BytesIO(bad)) as im:  # precondition: the header parses, the pixel data does not
        im.verify()
    try:
        with Image.open(BytesIO(bad)) as im:
            im.load()
    except OSError:
        pass
    else:
        raise RuntimeError("fixture broken: the truncated JPEG decodes completely")
    serve_image_for(job_env, 2, ImageContent.from_base64(b64(bad), "image/jpeg"))

    state = await job_env.run(qa=False)
    assert state["status"] == "completed", (state["status"], state["error"])
    assert [p["status"] for p in state["panels"]] == ["done", "failed", "done"]
    with Image.open(job_env.output_dir / state["job_id"] / "combined.jpg") as combined:
        assert combined.size == (1024, 2 * 1536 + 24)


async def test_j3_an_exception_escaping_a_worker_does_not_leave_its_siblings_running(job_env, monkeypatch):
    job_env.settings["concurrency"] = 2
    job_env.paper.num_panels = 6
    job_env.image.hook = slow_seedream
    real_write_bytes, real_persist = Path.write_bytes, jobs.JobManager._persist
    fault = {"armed": False, "fired": False}

    def write_bytes(self, data):  # an I/O error saving panel 2 ...
        if self.name.startswith("panel_02"):
            fault["armed"] = True
            raise OSError(errno.EIO, "Input/output error", str(self))
        return real_write_bytes(self, data)

    def persist(self, job):  # ... and again on the very next job.json write (the worker's error handler)
        if fault["armed"] and not fault["fired"]:
            fault["fired"] = True
            raise OSError(errno.EIO, "Input/output error", "job.json.tmp")
        return real_persist(self, job)

    monkeypatch.setattr(Path, "write_bytes", write_bytes)
    monkeypatch.setattr(jobs.JobManager, "_persist", persist)
    job_id = job_env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)["job_id"]
    await asyncio.wait_for(job_env.manager.jobs[job_id].rt.task, timeout=10)
    at_return, leftovers = job_env.manager.get(job_id), live_workers()
    await asyncio.sleep(0.3)
    later = job_env.manager.get(job_id)
    await drain(leftovers)  # clean up before asserting so nothing leaks into other tests
    if not fault["fired"]:
        raise RuntimeError("scenario broken: the injected I/O errors never happened")

    assert leftovers == []
    assert [p["n"] for p in at_return["panels"] if p["status"] in ACTIVE] == []
    assert stable(later) == stable(at_return)  # nothing changes after the run has returned
    assert not any("render_panel" in (p.get("error") or "") for p in later["panels"])


# ----------------------------------------------------------------------------- J4: restart while panels are in progress

def test_j4_restart_settles_panels_that_were_rendering_or_in_qa(job_env):
    out, job_id = job_env.output_dir, "20261022_101010_a4a4"
    url = lambda n: f"/output/{job_id}/panel_{n:02d}.jpg"  # noqa: E731
    drawn = {"width": 64, "height": 96, "attempts": 1}
    job_dir = persist_job(out, job_id, status="rendering", panels=[
        dict(jobs._panel_state(make_panel(1)), status="done", image_url=url(1), **drawn),
        dict(jobs._panel_state(make_panel(2)), status="rendering", image_url=url(2), **dict(drawn, attempts=2)),  # redraw of a drawn panel
        dict(jobs._panel_state(make_panel(3)), status="rendering", attempts=1),  # first render
        dict(jobs._panel_state(make_panel(4)), status="qa", attempts=1),  # first render, being judged
        dict(jobs._panel_state(make_panel(5)), status="qa", image_url=url(5), **drawn),  # its file is gone
        dict(jobs._panel_state(make_panel(6)), status="pending"),
    ])
    for n in (1, 2):
        (job_dir / f"panel_{n:02d}.jpg").write_bytes(image_bytes(4, 6, fmt="JPEG"))
    expected = ["done", "done", "failed", "failed", "failed", "pending"]

    state = jobs.JobManager(output_dir=out).get(job_id)  # a fresh process: nothing in memory
    assert state["status"] in jobs.TERMINAL
    assert [p["status"] for p in state["panels"]] == expected
    assert panel(state, 2)["image_url"] == url(2) and "error" not in panel(state, 2)  # the previous image is kept
    assert [panel(state, n)["error"] for n in (3, 4, 5)] == ["interrupted by a server restart"] * 3
    assert [p["status"] for p in on_disk(job_dir)["panels"]] == expected  # persisted, not only patched in memory
    again = jobs.JobManager(output_dir=out).get(job_id)
    assert [p["status"] for p in again["panels"]] == expected and again["status"] == state["status"]


async def test_j4_restart_mid_render_then_draw_the_interrupted_panel(job_env):
    job_env.settings["concurrency"] = 1
    frozen = asyncio.Event()  # the "dead" process never gets past panel 2

    async def freeze_on_panel_2(prompt):
        if "Line for panel 2" in prompt:
            await frozen.wait()

    job_env.image.hook = freeze_on_panel_2
    job_id = job_env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=True)["job_id"]
    dead = job_env.manager.jobs[job_id].rt.task
    job_dir = job_env.output_dir / job_id
    try:
        await wait_for(lambda: len(job_env.image.calls) == 2)
        assert [p["status"] for p in on_disk(job_dir)["panels"]] == ["done", "rendering", "pending"]

        restarted = jobs.JobManager(output_dir=job_env.output_dir)
        state = restarted.get(job_id)
        assert state["status"] in jobs.TERMINAL
        assert [p["status"] for p in state["panels"]] == ["done", "failed", "pending"]
        assert panel(state, 2)["error"] == "interrupted by a server restart"

        job_env.image.hook = None
        assert await restarted.regenerate_panel(job_id, 2) == {"status": "queued"}
        await asyncio.wait_for(restarted.jobs[job_id].rt.task, timeout=10)
        after = restarted.get(job_id)
        assert [p["status"] for p in after["panels"]] == ["done", "done", "pending"]
        assert "error" not in panel(after, 2)
    finally:
        await drain([dead])


@pytest.mark.parametrize("layout, sharing", [("single", [2]), ("strip", [1, 2])])
async def test_j4_restart_during_a_redraw_keeps_the_previous_image(job_env, layout, sharing):
    job_env.paper.num_panels = 4
    state = await job_env.run(layout=layout, qa=True)
    job_id = state["job_id"]
    job_dir = job_env.output_dir / job_id
    before = {n: panel(state, n) for n in sharing}
    old_bytes = file_of(job_dir, before[2]["image_url"]).read_bytes()
    stuck = asyncio.Event()
    job_env.vision.router = lambda prompt, messages, config: stuck.wait()  # the judge never answers

    assert await job_env.manager.regenerate_panel(job_id, 2) == {"status": "queued"}
    dead = job_env.manager.jobs[job_id].rt.task
    try:
        await wait_for(lambda: all(panel(on_disk(job_dir), n)["status"] == "qa" for n in sharing))
        restarted = jobs.JobManager(output_dir=job_env.output_dir).get(job_id)
        assert restarted["status"] in jobs.TERMINAL
        assert not any(p["status"] in ACTIVE for p in restarted["panels"])
        for n in sharing:
            p = panel(restarted, n)
            assert (p["status"], p["image_url"]) == ("done", before[n]["image_url"]) and "error" not in p
        assert file_of(job_dir, before[2]["image_url"]).read_bytes() == old_bytes
        assert restarted["combined_url"] == state["combined_url"]
    finally:
        await drain([dead])
