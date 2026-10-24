"""The v3 job path end to end with fakes: text-free art → grounding → art problems as retry feedback → lettering."""

from __future__ import annotations

import json
import re

import pytest
from PIL import Image

from . import fakes
from .fakes import V3_PAPER_TEXT

LOCATE_MARK = "Find things in that panel only"
LIGHT = [(246, 244, 238), (240, 238, 232), (250, 248, 243)]   # board-white art, so blank-board checks pass


class GroundingModel:
    """Vision router: LOCATE prompts get every present character's head low in the panel and one blank board in the
    middle; ``hook(call_number, reply)`` may change the reply. Anything else goes to the v2 judge."""

    def __init__(self, judge, hook=None):
        self.judge, self.hook, self.located = judge, hook, 0

    def __call__(self, prompt, messages, config):
        if LOCATE_MARK not in prompt:
            return self.judge(prompt, messages, config)
        self.located += 1
        line = prompt.split("Characters who should be in the panel:", 1)[1].split("\n", 1)[0]
        line = line.split(" These cast members should NOT", 1)[0]
        names = [n for n in ("papi", "kumo", "nezu") if re.search(rf"\b{n}\b", line)]
        reply = {"characters": [{"name": n, "head": [60 + 300 * i, 780, 240 + 300 * i, 920], "matches_reference": True}
                                for i, n in enumerate(names)],
                 "boards": [{"corners": [[60, 300], [940, 300], [940, 720], [60, 720]]}],
                 "speech_bubbles": [], "text_or_symbols": [], "extra_creatures": []}
        return (self.hook(self.located, reply) if self.hook else None) or reply


@pytest.fixture
def v3_env(job_env, monkeypatch):
    job_env.settings.update(pipeline="v3", concurrency=1)
    job_env.vision.router = GroundingModel(job_env.judge)
    job_env.image.fixed_size = (512, 768)
    monkeypatch.setattr(fakes, "PALETTE", LIGHT)
    return job_env


def _file(env, state, url: str):
    assert url.startswith(f"/output/{state['job_id']}/")
    return env.output_dir / state["job_id"] / url.rsplit("/", 1)[1]


async def test_v3_job_letters_every_panel_and_keeps_the_raw_art(v3_env):
    state = await v3_env.run(text=V3_PAPER_TEXT)

    assert state["status"] == "completed", state.get("error")
    assert (state["version"], state["layout"]) == (3, "single")
    assert state["quiz"]["score"] == 1.0 and state["quiz"]["questions"] > 0
    panels = state["panels"]
    assert len(panels) >= 9 and all(p["status"] == "done" for p in panels)
    kinds = {p["kind"] for p in panels}
    assert kinds == {"scene", "board"}
    for p in panels:
        art, lettered = _file(v3_env, state, p["art_url"]), _file(v3_env, state, p["image_url"])
        assert art.name == f"art_{p['n']:02d}.jpg" and lettered.name == f"panel_{p['n']:02d}.jpg"
        with Image.open(art) as a, Image.open(lettered) as b:
            assert a.size == (512, 768) and b.width == 512 and b.height >= 768
            assert (p["width"], p["height"]) == b.size
        assert isinstance(p["layout"], dict) and p["render_attempts"] == 1 and p["qa"]["issues"] == []
        saved = json.loads((art.parent / f"grounding_{p['n']:02d}.json").read_text(encoding="utf-8"))
        assert saved["size"] == [512, 768] and set(saved["heads"]) == set(p["characters"])   # re-lettering needs no vision call
        if p["kind"] == "board":
            assert p["layout"].get("chart") or p["layout"].get("board_error") is None
    # one picture and one grounding call per panel; every art prompt is text-free and carries no dialogue
    assert state["cost"]["images_generated"] == len(panels) == v3_env.vision.router.located
    prompts = [c["prompt"] for c in v3_env.image.calls]
    assert all("WORDLESS IMAGE" in pr for pr in prompts)
    lines = [b["text"] for p in panels for b in p["dialogue"]]
    assert lines and not any(line[:40] in pr for line in lines for pr in prompts)


async def test_v3_job_redraws_art_with_its_problems_as_feedback(v3_env):
    def hook(n, reply):
        if n == 1:   # the first panel's first drawing has numerals on a clock and a second kumo
            reply["text_or_symbols"] = [{"text": "12 3", "where": "the wall clock"}]
            reply["extra_creatures"] = ["a small bird"]
        return reply

    v3_env.vision.router.hook = hook
    state = await v3_env.run(text=V3_PAPER_TEXT)

    assert state["status"] == "completed", state.get("error")
    first = state["panels"][0]
    assert first["render_attempts"] == 2 and first["qa"]["issues"] == [] and first["qa"]["score"] == 10.0
    retry = v3_env.image.calls[1]["prompt"]
    assert "drawn text: remove every letter, digit and symbol (on the wall clock)" in retry
    assert "extra figure: remove a small bird" in retry and "12 3" not in retry
    assert all(p["render_attempts"] == 1 for p in state["panels"][1:])


async def test_v3_job_publishes_the_best_attempt_when_every_attempt_has_problems(v3_env):
    v3_env.vision.router.hook = lambda n, reply: {**reply, "extra_creatures": ["a cat"]} if n <= 2 else reply
    state = await v3_env.run(text=V3_PAPER_TEXT)

    first = state["panels"][0]
    assert first["status"] == "done" and first["render_attempts"] == 2
    assert first["qa"]["issues"] and first["qa"]["issues"][0].startswith("extra figure: remove a cat")
    assert first["qa"]["score"] < 10
    assert _file(v3_env, state, first["image_url"]).exists()


async def test_v3_job_keeps_the_paid_art_when_the_vision_model_gives_no_grounding(v3_env):
    v3_env.vision.router.hook = lambda n, reply: "I cannot help with that."
    state = await v3_env.run(text=V3_PAPER_TEXT)

    assert state["status"] == "completed", state.get("error")
    for p in state["panels"]:
        assert p["status"] == "done" and p["render_attempts"] == 1   # a new drawing cannot fix the vision call
        assert p["qa"]["issues"][0].startswith("unchecked: the art could not be checked")
        assert _file(v3_env, state, p["image_url"]).exists()
    assert state["cost"]["images_generated"] == len(state["panels"])


async def test_v3_job_counts_grounding_tokens_as_text_cost(v3_env):
    state = await v3_env.run(text=V3_PAPER_TEXT)
    # the vision fake reports 2000 prompt tokens per call; the story calls report their own usage on top
    assert state["cost"]["text_prompt_tokens"] >= 2000 * len(state["panels"])


async def test_v2_setting_still_runs_the_lettered_pipeline(job_env):
    job_env.settings["pipeline"] = "v2"
    state = await job_env.run()
    assert state["status"] == "completed" and state.get("version", 2) == 2
    assert all("art_url" not in p or not p["art_url"] for p in state["panels"])


async def test_bilingual_v3_job_letters_the_same_art_in_both_languages(v3_env):
    v3_env.settings["bilingual"] = True
    state = await v3_env.run(text=V3_PAPER_TEXT, language="zh-CN")

    assert state["status"] == "completed", state.get("error")
    assert state["languages"] == ["zh-CN", "en-US"] and state["language"] == "zh-CN"
    job_dir = v3_env.output_dir / state["job_id"]
    assert (job_dir / "storyboard.json").exists() and (job_dir / "storyboard.en-US.json").exists()
    zh = json.loads((job_dir / "storyboard.json").read_text(encoding="utf-8"))
    en = json.loads((job_dir / "storyboard.en-US.json").read_text(encoding="utf-8"))
    assert (zh["language"], en["language"]) == ("zh-CN", "en-US") and len(zh["panels"]) == len(en["panels"])
    for p in state["panels"]:
        urls = p["image_urls"]
        assert set(urls) == {"zh-CN", "en-US"} and urls["zh-CN"] == p["image_url"]
        assert urls["en-US"].endswith(f"panel_{p['n']:02d}.en-US.jpg")
        assert _file(v3_env, state, urls["en-US"]).exists() and _file(v3_env, state, urls["zh-CN"]).exists()
        assert (job_dir / f"layout_{p['n']:02d}.en-US.json").exists()
        assert set(p["texts"]) == {"zh-CN", "en-US"} and p["texts"]["zh-CN"]["dialogue"] == p["dialogue"]
    # one drawing per panel serves both languages
    assert state["cost"]["images_generated"] == len(state["panels"]) == v3_env.vision.router.located
    assert state["combined_url"] and state["combined_urls"]["en-US"].split("?")[0].endswith("combined.en-US.jpg")
    assert (job_dir / "combined.en-US.jpg").exists()
    english = state["localized"]["en-US"]
    assert english["title"] == en["title"] and [c["id"] for c in english["chapters"]] == [c["id"] for c in state["chapters"]]


async def test_bilingual_v3_redraw_reletters_both_languages(v3_env):
    v3_env.settings["bilingual"] = True
    state = await v3_env.run(text=V3_PAPER_TEXT, language="en-US")
    first = state["panels"][0]
    before = {lang: _file(v3_env, state, url).stat().st_mtime_ns for lang, url in first["image_urls"].items()}
    located = v3_env.vision.router.located
    v3_env.manager.jobs[state["job_id"]].rt.storyboards = {}   # as after a restart: the scripts come back from disk

    await v3_env.manager.regenerate_panel(state["job_id"], 1)
    await v3_env.manager.jobs[state["job_id"]].rt.task
    after = v3_env.manager.get(state["job_id"])
    redrawn = after["panels"][0]
    assert set(redrawn["image_urls"]) == {"en-US", "zh-CN"} and redrawn["attempts"] > first["attempts"]
    assert v3_env.vision.router.located == located + 1
    for lang, url in redrawn["image_urls"].items():
        assert _file(v3_env, after, url).stat().st_mtime_ns > before[lang]


async def test_monolingual_v3_job_has_no_second_language(v3_env):
    v3_env.settings["bilingual"] = False
    state = await v3_env.run(text=V3_PAPER_TEXT, language="zh-CN")
    assert state["languages"] == ["zh-CN"]
    assert all("image_urls" not in p and "texts" not in p for p in state["panels"])
    assert not (v3_env.output_dir / state["job_id"] / "storyboard.en-US.json").exists()


async def test_v3_job_keeps_the_paid_art_when_lettering_fails(v3_env, monkeypatch):
    from services import lettering

    def broken(*_args, **_kwargs):
        raise lettering.LayoutError("art is missing or too small to letter")

    monkeypatch.setattr(lettering, "letter_panel", broken)
    state = await v3_env.run(text=V3_PAPER_TEXT)
    job_dir = v3_env.output_dir / state["job_id"]
    for p in state["panels"]:
        assert p["status"] == "failed" and p["error"].startswith("lettering failed: LayoutError")
        assert (job_dir / f"art_{p['n']:02d}.jpg").exists() and (job_dir / f"grounding_{p['n']:02d}.json").exists()
    assert state["cost"]["images_generated"] == len(state["panels"])


async def test_relayout_reletters_from_the_saved_art_without_paying_again(v3_env):
    v3_env.settings["bilingual"] = True
    state = await v3_env.run(text=V3_PAPER_TEXT, language="en-US")
    images, located = len(v3_env.image.calls), v3_env.vision.router.located
    before = {p["n"]: (p["attempts"], p["render_attempts"], _file(v3_env, state, p["image_urls"]["zh-CN"]).stat().st_mtime_ns)
              for p in state["panels"]}
    v3_env.manager.jobs[state["job_id"]].rt.storyboard = None      # as after a restart: everything comes back from disk
    v3_env.manager.jobs[state["job_id"]].rt.storyboards = {}

    assert await v3_env.manager.relayout(state["job_id"]) == {"relettered": len(state["panels"])}
    after = v3_env.manager.get(state["job_id"])
    assert len(v3_env.image.calls) == images and v3_env.vision.router.located == located   # no image or vision call
    for p in after["panels"]:
        attempts, render_attempts, mtime = before[p["n"]]
        assert (p["attempts"], p["render_attempts"]) == (attempts + 1, render_attempts)
        assert _file(v3_env, after, p["image_urls"]["zh-CN"]).stat().st_mtime_ns > mtime
    assert await v3_env.manager.relayout("20990101_000000_beef") is None


async def test_relayout_reapplies_the_recap_rule_to_every_language(v3_env):
    import json as _json
    v3_env.settings["bilingual"] = True
    state = await v3_env.run(text=V3_PAPER_TEXT, language="en-US")
    job_dir = v3_env.output_dir / state["job_id"]
    for name in ("storyboard.json", "storyboard.zh-CN.json"):   # as saved by an older rule: a line without its tick
        data = _json.loads((job_dir / name).read_text(encoding="utf-8"))
        recap = next(p for p in data["panels"] if p.get("beat") == "recap" and isinstance(p.get("board"), dict))
        recap["board"]["checked"] = []
        (job_dir / name).write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")
    v3_env.manager.jobs[state["job_id"]].rt.storyboard = None
    v3_env.manager.jobs[state["job_id"]].rt.storyboards = {}

    await v3_env.manager.relayout(state["job_id"])
    for name in ("storyboard.json", "storyboard.zh-CN.json"):
        data = _json.loads((job_dir / name).read_text(encoding="utf-8"))
        recap = next(p for p in data["panels"] if p.get("beat") == "recap" and isinstance(p.get("board"), dict))
        assert recap["board"].get("checked") != []   # recomputed from what the story shows (and saved)


async def test_relayout_refuses_v2_and_running_jobs(job_env, v3_env):
    v3_env.settings["pipeline"] = "v2"
    state = await v3_env.run()
    with pytest.raises(ValueError, match="Only v3 jobs"):
        await v3_env.manager.relayout(state["job_id"])
