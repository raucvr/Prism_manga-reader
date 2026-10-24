"""Regression tests for the 2026-10-22 review fixes J9, J10, W2, W3 and W8 (job memory, upload limit, reader contract).

Plain tests lock in what the fixes got right. Tests marked ``xfail(strict=True)`` describe the correct behaviour for
the parts that are still broken: once the owner fixes the code they XPASS, which ``strict`` turns into a failure,
so delete the marker then and the test stays as a regression test.

W2/W3/W8 are partly frontend behaviour (Reader.tsx ``jobStopped = isTerminal(job.status)``, PanelFrame.tsx button and
spinner logic, useJob.ts extra polls). There is no frontend test runner, so those parts were verified by reading the
code; the tests here pin the backend state the reader derives its decisions from.
"""

import asyncio
import gc
import json
import random
import threading
import tracemalloc

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import UploadFile

import main
import routes.jobs as job_routes
import services.cast as cast
import services.jobs as jobs
from engines.base import ImageContent

from .fakes import LONG_PAPER_TEXT, make_panel

ACTIVE = {"rendering", "qa"}  # panel statuses the reader shows with a spinner


def bug(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


# ----------------------------------------------------------------------------- helpers

def statuses(state) -> list[str]:
    return [p["status"] for p in state["panels"]]


def on_disk(output_dir, job_id) -> dict:
    return json.loads((output_dir / job_id / "job.json").read_text(encoding="utf-8"))


async def wait_task(manager, job_id, timeout=10):
    await asyncio.wait_for(manager.jobs[job_id].rt.task, timeout=timeout)


async def until(predicate, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.002)


def text_pdf(paragraphs: int = 12) -> bytes:
    """A small real PDF with enough extractable text for a job."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    body = "Attention Is All You Need\n\n" + "\n".join(
        f"Paragraph {i}: the Transformer relies entirely on self-attention to relate every token to every other token."
        for i in range(paragraphs))
    page.insert_textbox(pymupdf.Rect(36, 36, 560, 800), body, fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


def persist_state(out, job_id, status, panels):
    state = jobs._new_state(job_id, title="Persisted", theme="kumomo", language="en-US", layout="single", qa=True,
                            panel_range="auto", fast=False, source="text")
    state.pop("_started")
    state.update(status=status, panels=panels)
    (out / job_id).mkdir(parents=True, exist_ok=True)
    (out / job_id / "job.json").write_text(json.dumps(state), encoding="utf-8")


async def stopped_job(job_env, num_panels: int = 4):
    """A job the user stopped while panel 1 was being drawn. Returns (job_id, public state seen mid-run)."""
    job_env.settings["concurrency"] = 1
    job_env.paper.num_panels = num_panels
    manager = job_env.manager
    job_id = manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)["job_id"]
    seen = {}

    async def stop_on_first_image(prompt):
        if len(job_env.image.calls) == 1:
            seen["run"] = manager.get(job_id)
            await manager.cancel(job_id)

    job_env.image.hook = stop_on_first_image
    await wait_task(manager, job_id)
    job_env.image.hook = None
    state = manager.get(job_id)
    assert (state["status"], statuses(state)) == ("cancelled", ["done"] + ["pending"] * (num_panels - 1))
    return job_id, seen["run"]


# ----------------------------------------------------------------------------- J9: memory kept per job

async def test_j9_finished_job_keeps_no_source_text_engines_or_best_images(job_env):
    state = await job_env.run(qa=True)
    rt = job_env.manager.jobs[state["job_id"]].rt
    assert state["status"] == "completed"
    assert not hasattr(rt, "best_images")
    assert (rt.pdf_bytes, rt.text) == (None, None)
    assert (rt.engines, rt.renderer, rt.qa) == (None, None, None)


async def test_j9_pdf_bytes_are_released_before_the_first_panel_is_drawn(job_env):
    pdf = text_pdf()
    held_while_drawing = []

    def hook(prompt):
        (job,) = job_env.manager.jobs.values()
        held_while_drawing.append(job.rt.pdf_bytes)

    job_env.image.hook = hook
    state = await job_env.run(pdf_bytes=pdf, text=None, filename="paper.pdf", qa=False)
    assert state["status"] == "completed"
    assert held_while_drawing == [None, None, None]
    assert job_env.manager.jobs[state["job_id"]].rt.pdf_bytes is None
    assert (job_env.output_dir / state["job_id"] / "source.pdf").read_bytes() == pdf  # still available on disk


async def test_j9_failed_extraction_releases_the_pdf_bytes(job_env):
    corrupt = b"%PDF-1.4\n" + b"\x00not really a pdf" * 20_000  # passes the route's %PDF check; PyMuPDF and pdfplumber reject it
    state = await job_env.run(pdf_bytes=corrupt, text=None, filename="broken.pdf", qa=False)
    assert state["status"] == "failed"
    assert job_env.manager.jobs[state["job_id"]].rt.pdf_bytes is None


async def test_j9_stop_while_reading_the_pdf_releases_the_pdf_bytes(job_env, monkeypatch):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    real_extract = jobs.extract_paper

    def slow_extract(*args, **kwargs):
        started.set()
        release.wait(5)
        try:
            return real_extract(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(jobs, "extract_paper", slow_extract)
    manager = job_env.manager
    job_id = manager.create(pdf_bytes=text_pdf(), text=None, filename="paper.pdf", qa=False)["job_id"]
    try:
        await until(started.is_set)
        assert (await manager.cancel(job_id))["status"] == "cancelled"
        await wait_task(manager, job_id)
        assert manager.get(job_id)["status"] == "cancelled"
        assert manager.jobs[job_id].rt.pdf_bytes is None
    finally:
        release.set()
        if started.is_set():
            await until(finished.is_set)


def _noise_sheets(directory, size: int = 400):
    rng = random.Random(9)
    directory.mkdir()
    sheets = []
    for order, name in enumerate(("papi", "kumo", "nezu"), start=1):
        path = directory / f"{order}. {name}.png"
        Image.frombytes("RGB", (size, size), rng.randbytes(size * size * 3)).save(path)  # incompressible, ~480 KB
        sheets.append((order, name, path))
    return sheets


async def test_j9_finished_jobs_do_not_each_keep_a_copy_of_the_reference_sheets(job_env, monkeypatch, tmp_path):
    sheets = _noise_sheets(tmp_path / "sheets")
    monkeypatch.setattr(cast, "load_reference_sheets", lambda image_dir=None: list(sheets))
    monkeypatch.setattr(jobs, "build_cast", cast.build_cast)  # the real one (job_env installs a tiny fake)
    one_set = sum(len(ImageContent.from_file(path).data) for _, _, path in sheets)

    async def one_job():
        state = await job_env.run(qa=False)
        assert state["status"] == "completed"
        for engine in (job_env.text, job_env.vision, job_env.image):
            engine.calls.clear()  # the fakes record prompts and refs; production engines don't

    def large_live_blocks() -> int:
        gc.collect()
        return sum(trace.size for trace in tracemalloc.take_snapshot().traces if trace.size >= 256 * 1024)

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        await one_job()  # warm-up
        before = large_live_blocks()
        for _ in range(3):
            await one_job()
        retained = large_live_blocks() - before
    finally:
        if not was_tracing:
            tracemalloc.stop()
    assert retained < one_set, f"3 finished jobs keep {retained / 1e6:.1f} MB of large buffers (one set of sheets = {one_set / 1e6:.1f} MB)"


# ----------------------------------------------------------------------------- J10: upload size limit

class UploadRecorder(jobs.JobManager):
    """Real persistence over tmp_path; ``create`` only records its kwargs (no pipeline task)."""

    def __init__(self, output_dir):
        super().__init__(output_dir=output_dir)
        self.created = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return {"job_id": "20261022_090000_c0de", "status": "queued"}


LIMIT = 4096  # routes.jobs.MAX_PDF_BYTES during these tests


@pytest.fixture
def uploads(monkeypatch, tmp_path):
    out = tmp_path / "output"
    monkeypatch.setattr(jobs, "OUTPUT_DIR", out)
    manager = UploadRecorder(out)
    monkeypatch.setattr(job_routes, "get_job_manager", lambda: manager)
    monkeypatch.setattr(job_routes, "MAX_PDF_BYTES", LIMIT)
    reads = []
    real_read = UploadFile.read

    async def recording_read(self, size=-1):
        data = await real_read(self, size)
        reads.append((size, len(data)))
        return data

    monkeypatch.setattr(UploadFile, "read", recording_read)
    client = TestClient(main.app)  # not a context manager: the lifespan (watchers, engines) never starts
    yield client, manager, reads
    client.close()


def pdf_of(size: int) -> bytes:
    return b"%PDF-1.4\n" + b"0" * (size - 9)


def test_j10_upload_of_exactly_the_limit_is_accepted_intact(uploads):
    client, manager, _ = uploads
    payload = pdf_of(LIMIT)
    response = client.post("/api/jobs", files={"file": ("paper.pdf", payload, "application/pdf")})
    assert response.status_code == 202
    assert manager.created[-1]["pdf_bytes"] == payload


@pytest.mark.parametrize("size", [LIMIT + 1, 50 * LIMIT])
def test_j10_oversized_upload_is_rejected_after_reading_at_most_limit_plus_one_byte(uploads, size):
    client, manager, reads = uploads
    response = client.post("/api/jobs", files={"file": ("big.pdf", pdf_of(size), "application/pdf")})
    assert response.status_code == 413
    assert manager.created == []
    # At most one bounded read (none if the request is refused even earlier): the rest of the upload is never
    # pulled into memory. Before the fix this was a single unbounded file.read() of the whole upload.
    assert len(reads) <= 1 and all(got <= LIMIT + 1 for _, got in reads), reads


async def _post_streamed(body: bytes, content_type: str, chunk: int = 64 * 1024):
    """Call the ASGI app directly and count how many body chunks it pulls before it starts answering."""
    chunks = [body[i:i + chunk] for i in range(0, len(body), chunk)]
    progress = {"pulled": 0, "pulled_at_response": None, "status": None}
    finished = asyncio.Event()

    async def receive():
        if progress["pulled"] < len(chunks):
            i = progress["pulled"]
            progress["pulled"] += 1
            return {"type": "http.request", "body": chunks[i], "more_body": i + 1 < len(chunks)}
        await finished.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            progress["status"], progress["pulled_at_response"] = message["status"], progress["pulled"]
        elif message["type"] == "http.response.body" and not message.get("more_body"):
            finished.set()

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/api/jobs", "raw_path": b"/api/jobs", "query_string": b"", "root_path": "",
        "headers": [(b"host", b"testserver"), (b"content-type", content_type.encode()),
                    (b"content-length", str(len(body)).encode())],
        "client": ("127.0.0.1", 50000), "server": ("testserver", 80),
    }
    await asyncio.wait_for(main.app(scope, receive, send), timeout=10)
    return progress, len(chunks)


async def test_j10_oversized_upload_is_refused_before_the_whole_body_is_received(uploads):
    _, manager, _ = uploads
    boundary = "prismreview"
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"big.pdf\"\r\n"
            "Content-Type: application/pdf\r\n\r\n").encode()
    body = head + pdf_of(2 * 1024 * 1024) + f"\r\n--{boundary}--\r\n".encode()
    progress, total = await _post_streamed(body, f"multipart/form-data; boundary={boundary}")
    assert progress["status"] == 413 and manager.created == []
    assert progress["pulled_at_response"] < total, f"pulled {progress['pulled_at_response']}/{total} body chunks before answering"


# ----------------------------------------------------------------------------- W2: "Draw this panel" on a stopped job

async def test_w2_drawing_one_panel_of_a_stopped_job_leaves_it_stopped(job_env):
    manager = job_env.manager
    job_id, _ = await stopped_job(job_env)

    assert await manager.regenerate_panel(job_id, 3) == {"status": "queued"}
    await wait_task(manager, job_id)
    state = manager.get(job_id)
    assert state["status"] == "cancelled"  # terminal again: the reader shows 'Not drawn' + a button, not a spinner
    assert statuses(state) == ["done", "pending", "done", "pending"]
    assert "2 panel(s) still not drawn" in state["stage_message"]
    assert on_disk(job_env.output_dir, job_id)["status"] == "cancelled"

    for n in (2, 4):  # drawing the remaining panels one by one finishes the manga
        await manager.regenerate_panel(job_id, n)
        await wait_task(manager, job_id)
    final = manager.get(job_id)
    assert (final["status"], final["stage_message"], statuses(final)) == ("completed", "Done", ["done"] * 4)


async def test_w2_drawing_a_panel_after_a_restart_keeps_the_job_failed(job_env):
    job_id, _ = await stopped_job(job_env)
    path = job_env.output_dir / job_id / "job.json"  # pretend the process died mid-run
    state = json.loads(path.read_text(encoding="utf-8"))
    state["status"] = "rendering"
    path.write_text(json.dumps(state), encoding="utf-8")

    fresh = jobs.JobManager(output_dir=job_env.output_dir)
    assert fresh.get(job_id)["status"] == "failed"
    await fresh.regenerate_panel(job_id, 2)
    await wait_task(fresh, job_id)
    after = fresh.get(job_id)
    assert (after["status"], statuses(after)) == ("failed", ["done", "done", "pending", "pending"])
    assert "2 panel(s) still not drawn" in after["stage_message"]


VOLATILE = {"stage_message", "updated_at", "progress", "cost", "panels", "combined_url", "error"}


async def test_w2_a_redraw_of_a_stopped_job_does_not_look_like_a_resumed_run(job_env):
    manager = job_env.manager
    job_id, during_run = await stopped_job(job_env)
    seen = {}
    job_env.image.hook = lambda prompt: seen.setdefault("redraw", manager.get(job_id))
    await manager.regenerate_panel(job_id, 3)
    await wait_task(manager, job_id)

    during_redraw = seen["redraw"]
    assert statuses(during_redraw) == ["done", "pending", "rendering", "pending"]
    # Correct: some job-level field tells the reader that panels 2 and 4 will not be drawn: a terminal status kept
    # during the redraw, a distinct status, or a field naming the panels being redrawn.
    signal = {k for k in set(during_run) | set(during_redraw) if k not in VOLATILE and during_run.get(k) != during_redraw.get(k)}
    assert during_redraw["status"] in jobs.TERMINAL or signal, (during_redraw["status"], during_redraw["stage_message"])


# ----------------------------------------------------------------------------- W3: in-flight panels after restart / Stop

def test_w3_restart_settles_panels_that_were_being_drawn(job_env):
    out, job_id = job_env.output_dir, "20261022_070000_aaaa"
    (out / job_id).mkdir(parents=True)
    (out / job_id / "panel_02.jpg").write_bytes(b"previous picture")
    persist_state(out, job_id, "rendering", [
        dict(jobs._panel_state(make_panel(1)), status="done", image_url=f"/output/{job_id}/panel_01.jpg", width=64, height=96, attempts=1),
        dict(jobs._panel_state(make_panel(2)), status="qa", image_url=f"/output/{job_id}/panel_02.jpg", width=64, height=96, attempts=2),
        dict(jobs._panel_state(make_panel(3)), status="rendering", attempts=1),
        dict(jobs._panel_state(make_panel(4)), status="rendering", image_url=f"/output/{job_id}/panel_04.jpg", width=64, height=96, attempts=2),
        dict(jobs._panel_state(make_panel(5)), status="pending"),
    ])

    state = jobs.JobManager(output_dir=out).get(job_id)  # a fresh process
    assert (state["status"], state["stage_message"]) == ("failed", "Interrupted by a server restart")
    assert statuses(state) == ["done", "done", "failed", "failed", "pending"]
    restart = "interrupted by a server restart"
    assert [p.get("error") for p in state["panels"]] == [None, None, restart, restart, None]
    assert state["panels"][1]["image_url"] == f"/output/{job_id}/panel_02.jpg"  # the redraw's previous picture is kept
    assert statuses(on_disk(out, job_id)) == ["done", "done", "failed", "failed", "pending"]  # persisted at once


async def test_w3_stop_settles_drawing_and_checking_panels_at_once(job_env):
    job_env.settings["concurrency"] = 2
    manager, judge = job_env.manager, job_env.judge
    render_gate, qa_gate = asyncio.Event(), asyncio.Event()

    async def hold_panel_2(prompt):
        if "Line for panel 2" in prompt:
            await render_gate.wait()

    def held_judge(prompt, messages, config):
        async def verdict():
            await qa_gate.wait()
            return judge(prompt, messages, config)
        return verdict()

    job_env.image.hook = hold_panel_2
    job_env.vision.router = held_judge
    job_id = manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=True)["job_id"]
    try:
        await until(lambda: statuses(manager.get(job_id))[:2] == ["qa", "rendering"])
        stopped = await manager.cancel(job_id)  # like the HTTP handler: from outside the pipeline task
        assert stopped["status"] == "cancelled"
        assert not ACTIVE & set(statuses(stopped))  # the reader's next poll already sees no spinner
        assert not ACTIVE & set(statuses(on_disk(job_env.output_dir, job_id)))
        await wait_task(manager, job_id)
    finally:
        render_gate.set()
        qa_gate.set()
    final = manager.get(job_id)
    assert final["status"] == "cancelled" and not ACTIVE & set(statuses(final))
    assert statuses(final)[2] == "pending"
    assert len(job_env.image.calls) == 2  # nothing else was drawn after Stop


async def test_w3_stopping_a_redraw_keeps_the_previous_picture(job_env):
    manager = job_env.manager
    state = await job_env.run(qa=False)
    job_id = state["job_id"]
    before = state["panels"][1]
    picture = (job_env.output_dir / job_id / "panel_02.jpg").read_bytes()
    gate = asyncio.Event()

    async def hold(prompt):
        await gate.wait()

    job_env.image.hook = hold
    try:
        await manager.regenerate_panel(job_id, 2)
        await until(lambda: manager.get(job_id)["panels"][1]["status"] == "rendering")
        stopped = await manager.cancel(job_id)
        assert stopped["panels"][1]["status"] == "done"
        await wait_task(manager, job_id)
    finally:
        gate.set()
    after = manager.get(job_id)["panels"][1]
    assert (after["status"], after["image_url"]) == ("done", before["image_url"])
    assert (job_env.output_dir / job_id / "panel_02.jpg").read_bytes() == picture


def test_w3_a_fresh_process_never_reports_panels_in_flight(job_env):
    out = job_env.output_dir
    ids = {"20261022_070000_bbbb": "failed", "20261022_070000_cccc": "cancelled"}
    for job_id, status in ids.items():
        persist_state(out, job_id, status, [
            dict(jobs._panel_state(make_panel(1)), status="done", image_url=f"/output/{job_id}/panel_01.jpg", width=64, height=96),
            dict(jobs._panel_state(make_panel(2)), status="rendering", attempts=1),
            dict(jobs._panel_state(make_panel(3)), status="qa", attempts=1),
        ])
    fresh = jobs.JobManager(output_dir=out)  # no pipeline task can be running in a new process
    for job_id in ids:
        assert not ACTIVE & set(statuses(fresh.get(job_id))), job_id


async def test_w3_a_failed_run_leaves_no_panel_drawing(job_env, monkeypatch):
    job_env.settings.update(concurrency=2, max_attempts=1)
    job_env.paper.num_panels = 2
    manager = job_env.manager
    gate = asyncio.Event()

    async def hold_panel_2(prompt):
        if "Line for panel 2" in prompt:
            await gate.wait()

    job_env.image.hook = hold_panel_2
    real_group, real_update = manager._render_group, manager._update_progress
    raised = []

    async def render_group(job, group, settings):
        if group[0].n == 1:  # panel 1 breaks outside the attempt loop while panel 2 is being drawn...
            while statuses(job.state)[1] != "rendering":
                await asyncio.sleep(0.001)
            raise RuntimeError("unexpected failure")
        return await real_group(job, group, settings)

    def update_progress(job):  # ...and job.json cannot be written when the worker records that failure
        if statuses(job.state)[0] == "failed" and not raised:
            raised.append(True)
            raise OSError(28, "No space left on device")
        return real_update(job)

    monkeypatch.setattr(manager, "_render_group", render_group)
    monkeypatch.setattr(manager, "_update_progress", update_progress)
    job_id = manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)["job_id"]
    try:
        await wait_task(manager, job_id)
        state = manager.get(job_id)
        assert state["status"] == "failed" and "No space left" in state["error"]
        assert not ACTIVE & set(statuses(state))
        assert not ACTIVE & set(statuses(on_disk(job_env.output_dir, job_id)))
    finally:
        gate.set()
        await asyncio.sleep(0.05)  # let an orphan worker (current code) finish before the loop closes


# ----------------------------------------------------------------------------- W8: redraw buttons only when the backend accepts

async def test_w8_redraws_are_refused_exactly_while_a_task_runs(job_env):
    """PanelFrame shows Regenerate / 'Draw this panel' only when isTerminal(job.status); this pins the backend side:
    while the pipeline runs the status is non-terminal and a redraw is a 409, once it has wound down the status is
    terminal and a redraw is accepted, and a redraw in progress refuses a second one."""
    job_env.settings["concurrency"] = 1
    job_env.paper.num_panels = 3
    manager = job_env.manager
    gate = asyncio.Event()

    async def hold(prompt):
        await gate.wait()

    job_env.image.hook = hold
    job_id = manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)["job_id"]
    try:
        await until(lambda: len(job_env.image.calls) == 1)
        assert manager.get(job_id)["status"] not in jobs.TERMINAL  # reader: no redraw buttons...
        with pytest.raises(ValueError, match="still running"):     # ...because the backend refuses
            await manager.regenerate_panel(job_id, 1)

        await manager.cancel(job_id)
        await wait_task(manager, job_id)
        assert manager.get(job_id)["status"] in jobs.TERMINAL      # reader: buttons shown...
        assert await manager.regenerate_panel(job_id, 2) == {"status": "queued"}  # ...and accepted

        await until(lambda: len(job_env.image.calls) == 2)         # the redraw is being drawn
        with pytest.raises(ValueError, match="still running"):     # a second redraw would be a 409
            await manager.regenerate_panel(job_id, 3)
    finally:
        gate.set()
    await wait_task(manager, job_id)
    assert manager.get(job_id)["status"] in jobs.TERMINAL
    assert await manager.regenerate_panel(job_id, 3) == {"status": "queued"}
    await wait_task(manager, job_id)
    assert statuses(manager.get(job_id)) == ["pending", "done", "done"]
