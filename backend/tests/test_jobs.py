"""services.jobs: the v2 job pipeline end to end with fake engines; all files go to tmp_path."""

import asyncio
import json
import re

import pytest
from PIL import Image

import services.jobs as jobs

from .fakes import ATTENTION_PDF, LONG_PAPER_TEXT, PALETTE, make_panel


def close_to(pixel, color, tolerance=45) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(pixel, color))


def persist_state(out, job_id, **changes):
    state = jobs._new_state(job_id, title=changes.pop("title", "A paper"), theme="kumomo", language="en-US", layout="single",
                            qa=True, panel_range="auto", fast=False, source="text")
    state.pop("_started")
    state.update(changes)
    (out / job_id).mkdir(parents=True)
    (out / job_id / "job.json").write_text(json.dumps(state), encoding="utf-8")


async def test_text_job_runs_end_to_end(job_env):
    state = await job_env.run(text=LONG_PAPER_TEXT, language="en-US", qa=True)
    job_id = state["job_id"]
    job_dir = job_env.output_dir / job_id

    assert (state["status"], state["stage_message"], state["error"]) == ("completed", "Done", None)
    assert state["progress"] == {"done": 3, "total": 3, "percent": 100}
    assert (state["title"], state["summary"]) == ("Attention, Please!", "Attention replaces recurrence.")
    assert [c["id"] for c in state["chapters"]] == ["hook", "problem", "idea"]
    for n, panel in enumerate(state["panels"], start=1):
        assert (panel["n"], panel["status"], panel["attempts"], panel["render_attempts"]) == (n, "done", 1, 1)
        assert panel["image_url"] == f"/output/{job_id}/panel_{n:02d}.jpg"
        assert (panel["width"], panel["height"]) == (64, 96)
        assert panel["qa"] == {"score": 9.0, "text_exact": 2, "text_total": 2, "issues": [], "available": True}
        assert "error" not in panel
    for name in ["panel_01.jpg", "panel_02.jpg", "panel_03.jpg", "storyboard.json", "job.json", "combined.jpg", "prompts.jsonl", "paper.txt"]:
        assert (job_dir / name).is_file(), name
    assert state["combined_url"].startswith(f"/output/{job_id}/combined.jpg")
    assert state["storyboard_url"] == f"/output/{job_id}/storyboard.json"

    cost = state["cost"]
    assert cost["images_generated"] == 3 and cost["image_tokens"] == 3 * 6144
    assert cost["text_prompt_tokens"] == len(job_env.text.calls) * 1000 + len(job_env.vision.calls) * 2000 > 0
    assert cost["text_completion_tokens"] == len(job_env.text.calls) * 200 + len(job_env.vision.calls) * 100 > 0

    on_disk = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "completed" and "_started" not in on_disk
    assert [p["n"] for p in json.loads((job_dir / "storyboard.json").read_text(encoding="utf-8"))["panels"]] == [1, 2, 3]
    assert len((job_dir / "prompts.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert "Layer normalization" not in (job_dir / "paper.txt").read_text(encoding="utf-8")  # references cut
    with Image.open(job_dir / "combined.jpg") as combined:
        assert combined.size == (1024, 3 * 1536 + 2 * 24)

    assert job_env.paper.seen[:3] == ["brief", "briefcheck", "storyboard"] and "translate" not in job_env.paper.seen
    assert len(job_env.image.calls) == 3 and len(job_env.vision.calls) == 3
    assert [len(call["refs"]) for call in job_env.image.calls] == [2, 2, 2]  # only the sheets of the panel's characters
    assert (job_env.text.closed, job_env.vision.closed, job_env.image.closed) == (1, 1, 1)


async def test_a_failed_panel_does_not_fail_the_job(job_env):
    job_env.image.fail_panels = {2}
    state = await job_env.run(qa=False)
    p1, p2, p3 = state["panels"]

    assert state["status"] == "completed"
    assert "1 failed panel" in state["stage_message"] and "[2]" in state["stage_message"]
    assert (p2["status"], p2["image_url"], p2["attempts"]) == ("failed", None, 2)
    assert "Seedream exploded" in p2["error"]
    assert p1["status"] == p3["status"] == "done" and "error" not in p1
    assert state["progress"]["percent"] == 100 and state["cost"]["images_generated"] == 2
    assert len([c for c in job_env.image.calls if "Line for panel 2" in c["prompt"]]) == 2  # max_attempts

    job_dir = job_env.output_dir / state["job_id"]
    assert not (job_dir / "panel_02.jpg").exists()
    with Image.open(job_dir / "combined.jpg") as combined:
        assert combined.size == (1024, 2 * 1536 + 24)
        top, bottom = combined.getpixel((512, 700)), combined.getpixel((512, 1536 + 24 + 700))
    assert close_to(top, PALETTE[1]) and close_to(bottom, PALETTE[3])  # panels 1 and 3, in order


async def test_qa_failure_triggers_one_targeted_retry(job_env):
    job_env.judge.fail_first_for = {1}
    state = await job_env.run(qa=True)
    p1, p2, p3 = state["panels"]

    assert (p1["attempts"], p2["attempts"], p3["attempts"]) == (2, 1, 1)
    assert state["cost"]["images_generated"] == 4 and len(job_env.vision.calls) == 4
    assert p1["status"] == "done" and p1["qa"]["text_exact"] == p1["qa"]["text_total"] == 2  # the better attempt is kept
    first, second = [c["prompt"] for c in job_env.image.calls if "Line for panel 1" in c["prompt"]]
    assert "previous attempt" not in first
    assert 'fix them: papi\'s bubble must read exactly "Line for panel 1 please."' in second


async def test_strip_layout_pairs_panels_into_shared_images(job_env):
    job_env.paper.num_panels = 5
    state = await job_env.run(layout="strip", qa=True)
    job_id, panels = state["job_id"], state["panels"]

    assert state["status"] == "completed" and state["layout"] == "strip"
    assert [p["image_url"] for p in panels] == (
        [f"/output/{job_id}/strip_01-02.jpg"] * 2 + [f"/output/{job_id}/strip_03-04.jpg"] * 2 + [f"/output/{job_id}/panel_05.jpg"])
    assert [p["strip_index"] for p in panels] == [0, 1, 0, 1, None]
    assert all((p["strip_width"], p["strip_height"], p["width"], p["height"]) == (128, 64, 64, 64) for p in panels[:4])
    assert (panels[4]["width"], panels[4]["height"]) == (64, 96)
    assert panels[0]["qa"]["text_total"] == 4  # bubble + narration of both panels in the strip

    assert [(c["config"].width, c["config"].height) for c in job_env.image.calls] == [(2048, 1024), (2048, 1024), (1024, 1536)]
    first = job_env.image.calls[0]["prompt"]
    assert "PANEL 1 (left):" in first and "Line for panel 1" in first and "Line for panel 2" in first
    assert state["cost"]["images_generated"] == 3
    with Image.open(job_env.output_dir / job_id / "combined.jpg") as combined:
        assert combined.size == (1024, 4 * 1024 + 1536 + 4 * 24)  # four cropped halves + the single tail panel


async def test_cancel_during_rendering_stops_the_remaining_panels(job_env):
    job_env.settings["concurrency"] = 1
    job_env.paper.num_panels = 4
    state = job_env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, qa=False)
    job_id = state["job_id"]

    async def cancel_on_first_image(prompt):
        if len(job_env.image.calls) == 1:
            assert (await job_env.manager.cancel(job_id))["status"] == "cancelled"

    job_env.image.hook = cancel_on_first_image
    await asyncio.wait_for(job_env.manager.jobs[job_id].rt.task, timeout=10)

    final = job_env.manager.get(job_id)
    assert (final["status"], final["stage_message"]) == ("cancelled", "Cancelled")
    assert len(job_env.image.calls) == 1
    assert [p["status"] for p in final["panels"]] == ["done", "pending", "pending", "pending"]
    job_dir = job_env.output_dir / job_id
    assert final["combined_url"] is None and not (job_dir / "combined.jpg").exists()
    assert json.loads((job_dir / "job.json").read_text(encoding="utf-8"))["status"] == "cancelled"
    assert job_env.image.closed == 1  # engines are closed on the way out


async def test_chinese_job_translates_the_script_before_rendering(job_env):
    state = await job_env.run(language="zh-CN", qa=True)
    assert (state["status"], state["language"], state["title"]) == ("completed", "zh-CN", "注意力就是一切")
    assert job_env.paper.seen[-2:] == ["translate", "factcheck"]   # the translation is fact-checked again
    assert state["panels"][0]["dialogue"][0]["text"] == "第1格的台词"
    assert [c["title"] for c in state["chapters"]] == ["第1章", "第2章", "第3章"]
    prompt = job_env.image.calls[0]["prompt"]
    assert '"第1格的台词"' in prompt and "Simplified Chinese" in prompt


async def test_pdf_job_extracts_the_paper_first(job_env):
    if not ATTENTION_PDF.exists():
        pytest.skip("attention_is_all_you_need.pdf not downloaded — see tests/fixtures/README.md")
    job_env.paper.num_panels = 2
    pdf = ATTENTION_PDF.read_bytes()
    state = await job_env.run(pdf_bytes=pdf, text=None, filename="attention.pdf", qa=False)
    job_dir = job_env.output_dir / state["job_id"]
    assert state["status"] == "completed" and state["params"]["source"] == "attention.pdf"
    assert (job_dir / "source.pdf").read_bytes() == pdf
    assert "Attention Is All You Need" in (job_dir / "paper.txt").read_text(encoding="utf-8")
    brief_prompt = job_env.text.prompts[0]
    assert "Figure 1: The Transformer - model architecture." in brief_prompt and "Scaled Dot-Product Attention" in brief_prompt


async def test_create_validates_text_and_normalises_options(job_env):
    with pytest.raises(ValueError, match="500 characters"):
        job_env.manager.create(pdf_bytes=None, text="too short")
    state = job_env.manager.create(pdf_bytes=None, text=LONG_PAPER_TEXT, theme="vaporwave", language="fr-FR", layout="grid", panel_range="huge")
    job_id = state["job_id"]
    assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{4}", job_id)
    assert (state["status"], state["theme"], state["language"], state["layout"]) == ("queued", "kumomo", "en-US", "single")
    assert state["params"] == {"qa": True, "panel_range": "auto", "fast": False, "source": "text", "title_given": False}
    assert json.loads((job_env.output_dir / job_id / "job.json").read_text(encoding="utf-8"))["status"] == "queued"  # persisted at once
    await asyncio.wait_for(job_env.manager.jobs[job_id].rt.task, timeout=10)
    assert job_env.manager.get(job_id)["status"] == "completed"


def test_load_and_list_persisted_jobs(job_env):
    out = job_env.output_dir
    persist_state(out, "20261020_101010_aaaa", status="completed", title="Old")
    persist_state(out, "20261021_090000_bbbb", status="rendering", title="Interrupted")
    (out / "0_not_a_job").mkdir()
    (out / "0_not_a_job" / "job.json").write_text("{}", encoding="utf-8")
    (out / "20261019_000000_cccc").mkdir()  # no job.json

    manager = jobs.JobManager(output_dir=out)  # a fresh process: nothing in memory
    listed = manager.list()
    assert [j["job_id"] for j in listed] == ["20261021_090000_bbbb", "20261020_101010_aaaa"]
    assert set(listed[0]) == {"job_id", "status", "title", "created_at", "progress", "language", "theme"}
    interrupted = manager.get("20261021_090000_bbbb")
    assert interrupted["status"] == "failed" and interrupted["error"] == "Server restarted while the job was running"
    assert "_started" not in interrupted
    assert manager.get("20261020_101010_aaaa")["status"] == "completed"
    assert [j["job_id"] for j in manager.list(limit=1)] == ["20261021_090000_bbbb"]
    for bad in ["../../etc/passwd", "0_not_a_job", "20261021_090000_ffff", ""]:
        assert manager.get(bad) is None


async def test_regenerate_panel_after_restart_rerenders_only_that_panel(job_env):
    state = await job_env.run(qa=True)
    job_id = state["job_id"]
    job_dir = job_env.output_dir / job_id
    before = {name: (job_dir / name).read_bytes() for name in ("panel_01.jpg", "panel_02.jpg", "panel_03.jpg", "combined.jpg")}
    calls_before = len(job_env.image.calls)

    fresh = jobs.JobManager(output_dir=job_env.output_dir)  # state, storyboard and engines rebuilt from disk
    assert await fresh.regenerate_panel(job_id, 2) == {"status": "queued"}
    await asyncio.wait_for(fresh.jobs[job_id].rt.task, timeout=10)

    new_calls = job_env.image.calls[calls_before:]
    assert len(new_calls) == 1 and "Line for panel 2" in new_calls[0]["prompt"]
    after = fresh.get(job_id)
    assert (after["status"], after["stage_message"]) == ("completed", "Done")
    assert [p["status"] for p in after["panels"]] == ["done"] * 3 and after["cost"]["images_generated"] == 4
    assert (job_dir / "panel_02.jpg").read_bytes() != before["panel_02.jpg"]
    assert (job_dir / "panel_01.jpg").read_bytes() == before["panel_01.jpg"]
    assert (job_dir / "panel_03.jpg").read_bytes() == before["panel_03.jpg"]
    assert (job_dir / "combined.jpg").read_bytes() != before["combined.jpg"]  # re-assembled
    assert len((job_dir / "prompts.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    assert json.loads((job_dir / "job.json").read_text(encoding="utf-8"))["status"] == "completed"


async def test_regenerate_rejects_running_jobs_and_unknown_panels(job_env):
    state = await job_env.run(qa=False)
    job_id = state["job_id"]
    with pytest.raises(ValueError, match="No panel 9"):
        await job_env.manager.regenerate_panel(job_id, 9)
    assert await job_env.manager.regenerate_panel("20261021_000000_dead", 1) is None
    job_env.manager.jobs[job_id].state["status"] = "rendering"
    with pytest.raises(ValueError, match="still running"):
        await job_env.manager.regenerate_panel(job_id, 1)


def test_public_hides_private_fields_and_empty_panel_errors(job_env):
    state = jobs._new_state("20261021_120000_abcd", title="T", theme="kumomo", language="en-US", layout="single",
                            qa=True, panel_range="auto", fast=False, source="text")
    state["panels"] = [dict(jobs._panel_state(make_panel(1)), status="done"),
                       dict(jobs._panel_state(make_panel(2)), status="failed", error="RuntimeError: boom")]
    job = jobs.Job(state)

    public = job_env.manager.public(job)
    assert "_started" not in public and "_started" in state
    assert "error" not in public["panels"][0] and public["panels"][1]["error"] == "RuntimeError: boom"
    assert public["cost"]["elapsed_seconds"] >= 0
    public["cost"]["images_generated"] = 99
    assert state["cost"]["images_generated"] == 0  # a copy, not the live state

    state["status"], state["cost"]["elapsed_seconds"] = "completed", 12.5
    assert job_env.manager.public(job)["cost"]["elapsed_seconds"] == 12.5
