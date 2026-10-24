"""Shared fixtures for the Prism v2 backend suite.

Safety: every engine in this suite is a fake, and the autouse ``_no_network`` fixture additionally
blocks httpx's real transports and outbound sockets, so no test can reach ModelArk by accident.
All job output goes to ``tmp_path`` (``services.jobs.OUTPUT_DIR`` is patched as well, because
``Job.dir`` reads the module global).
"""

from __future__ import annotations

import asyncio
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from .fakes import (  # noqa: E402
    ATTENTION_PDF,
    COT_PDF,
    LONG_PAPER_TEXT,
    FakeEngine,
    FakeImageEngine,
    JudgeModel,
    PaperModel,
    make_cast,
    make_sheets,
)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise RuntimeError("Network access is disabled in the test suite")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)

    def guard(real):
        def connect(sock, address):
            if sock.family == getattr(socket, "AF_UNIX", None):
                return real(sock, address)
            raise RuntimeError(f"Network access is disabled in the test suite (connect to {address!r})")
        return connect

    monkeypatch.setattr(socket.socket, "connect", guard(socket.socket.connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(socket.socket.connect_ex))


@pytest.fixture(autouse=True)
def _no_preflight(monkeypatch):
    """Creating a job first checks the API key and the models (JobManager.preflight): a fresh clone has no key, and the
    suite never talks to ModelArk. The preflight has its own tests (test_model_roles, test_routes)."""
    import services.jobs as jobs_module

    async def skip(self):
        return None

    monkeypatch.setattr(jobs_module.JobManager, "preflight", skip)


# ----------------------------------------------------------------------------- real papers (parsed once)

def _require_pdf(path):
    if not path.exists():
        pytest.skip(f"{path.name} not downloaded — see tests/fixtures/README.md")
    return path


@pytest.fixture(scope="session")
def attention_paper():
    from services.paper_extractor import extract_paper
    return extract_paper(_require_pdf(ATTENTION_PDF))


@pytest.fixture(scope="session")
def cot_paper():
    from services.paper_extractor import extract_paper
    return extract_paper(_require_pdf(COT_PDF).read_bytes(), filename=COT_PDF.name)


# ----------------------------------------------------------------------------- job manager with fake engines

JOB_SETTINGS = {
    "layout": "single",
    "panel_width": 1024,
    "panel_height": 1536,
    "strip_width": 2048,
    "strip_height": 1024,
    "concurrency": 3,
    "qa_enabled": True,
    "max_attempts": 2,
    "min_panels": 3,
    "max_panels": 6,
    "style_anchor": False,
    "fast": False,
    "output_format": "jpeg",
}


@dataclass
class JobEnv:
    manager: object
    text: FakeEngine
    vision: FakeEngine
    image: FakeImageEngine
    paper: PaperModel
    judge: JudgeModel
    settings: dict
    output_dir: Path

    async def run(self, **kwargs) -> dict:
        """``create()`` a job (default: the synthetic text paper), wait for its task, return the public state."""
        params = {"pdf_bytes": None, "text": LONG_PAPER_TEXT, "language": "en-US", "qa": True}
        params.update(kwargs)
        state = self.manager.create(**params)
        await asyncio.wait_for(self.manager.jobs[state["job_id"]].rt.task, timeout=10)
        return self.manager.get(state["job_id"])


@pytest.fixture
def job_env(monkeypatch, tmp_path) -> JobEnv:
    import services.jobs as jobs
    from services.engine_select import Engines

    out = tmp_path / "output"
    settings = dict(JOB_SETTINGS)
    paper, judge = PaperModel(num_panels=3), JudgeModel()
    text = FakeEngine(router=paper, model="fake-seed-text")
    vision = FakeEngine(router=judge, usage={"prompt_tokens": 2000, "completion_tokens": 100}, model="fake-seed-vision")
    image = FakeImageEngine()

    monkeypatch.setattr(jobs, "OUTPUT_DIR", out)
    monkeypatch.setattr(jobs, "_settings", lambda: dict(settings))
    monkeypatch.setattr(jobs, "build_engines", lambda: Engines(text=text, fast_text=text, vision=vision, image=image))
    monkeypatch.setattr(jobs, "build_cast", lambda theme: (make_cast(), make_sheets()))
    return JobEnv(manager=jobs.JobManager(output_dir=out), text=text, vision=vision, image=image, paper=paper,
                  judge=judge, settings=settings, output_dir=out)
