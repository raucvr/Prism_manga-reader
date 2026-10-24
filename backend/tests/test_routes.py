"""HTTP layer: /api/jobs validation and status codes, plus /api/config (TestClient without lifespan)."""

import json

import pytest
from fastapi.testclient import TestClient

import main
import routes.jobs as job_routes
import services.jobs as jobs

from .fakes import LONG_PAPER_TEXT, make_panel

PDF_BYTES = b"%PDF-1.4\n% minimal test payload\n"


class RecordingManager(jobs.JobManager):
    """Real get/list/cancel/regenerate over tmp_path; ``create`` only records its kwargs (no pipeline task)."""

    def __init__(self, output_dir):
        super().__init__(output_dir=output_dir)
        self.created = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return {"job_id": "20261021_120000_abcd", "status": "queued"}

    async def preflight(self):   # the model checks are tested on their own (test_preflight_*)
        self.preflights = getattr(self, "preflights", 0) + 1


@pytest.fixture
def api(monkeypatch, tmp_path):
    out = tmp_path / "output"
    monkeypatch.setattr(jobs, "OUTPUT_DIR", out)
    manager = RecordingManager(out)
    monkeypatch.setattr(job_routes, "get_job_manager", lambda: manager)
    client = TestClient(main.app)  # not used as a context manager: the lifespan (watchers, engines) never starts
    yield client, manager
    client.close()


def persisted_job(manager, job_id, status="completed"):
    state = jobs._new_state(job_id, title="Persisted", theme="kumomo", language="en-US", layout="single", qa=True,
                            panel_range="auto", fast=False, source="text")
    state.pop("_started")
    state.update(status=status, panels=[dict(jobs._panel_state(make_panel(1)), status="done")])
    (manager.output_dir / job_id).mkdir(parents=True)
    (manager.output_dir / job_id / "job.json").write_text(json.dumps(state), encoding="utf-8")


def test_create_job_from_text_returns_202_and_forwards_defaults(api):
    client, manager = api
    response = client.post("/api/jobs", data={"text": LONG_PAPER_TEXT})
    assert response.status_code == 202
    assert response.json() == {"job_id": "20261021_120000_abcd", "status": "queued"}
    (kwargs,) = manager.created
    assert kwargs == {"pdf_bytes": None, "text": LONG_PAPER_TEXT, "filename": "", "title": "", "theme": "kumomo", "language": "en-US",
                      "layout": None, "qa": None, "panel_range": "auto", "fast": None}


def test_create_job_from_pdf_upload_forwards_every_field(api):
    client, manager = api
    response = client.post("/api/jobs", files={"file": ("paper.pdf", PDF_BYTES, "application/pdf")},
                           data={"title": "  My Title  ", "theme": "kumomo", "language": "zh-CN", "layout": "strip",
                                 "qa": "true", "panel_range": "short", "fast": "false"})
    assert response.status_code == 202
    kwargs = manager.created[-1]
    assert (kwargs["pdf_bytes"], kwargs["filename"], kwargs["text"]) == (PDF_BYTES, "paper.pdf", None)
    assert (kwargs["title"], kwargs["theme"], kwargs["language"], kwargs["layout"], kwargs["panel_range"]) == (
        "My Title", "kumomo", "zh-CN", "strip", "short")
    assert (kwargs["qa"], kwargs["fast"]) == (True, False)


@pytest.mark.parametrize("raw, expected", [("true", True), ("false", False), ("1", True), ("off", False), ("", None)])
def test_create_job_parses_boolean_flags(api, raw, expected):
    client, manager = api
    assert client.post("/api/jobs", data={"text": LONG_PAPER_TEXT, "qa": raw, "fast": raw}).status_code == 202
    assert (manager.created[-1]["qa"], manager.created[-1]["fast"]) == (expected, expected)


@pytest.mark.parametrize("field, value", [("language", "fr-FR"), ("layout", "grid"), ("theme", "vaporwave"), ("theme", "chiikawa"),
                                          ("theme", "ghibli"), ("panel_range", "huge")])
def test_create_job_rejects_invalid_choices_with_422(api, field, value):
    client, manager = api
    response = client.post("/api/jobs", data={"text": LONG_PAPER_TEXT, field: value})
    assert response.status_code == 422
    assert response.json()["detail"].startswith(f"{field} must be one of")
    assert manager.created == []


def test_create_job_requires_a_pdf_or_enough_text(api):
    client, manager = api
    for data in ({"theme": "kumomo"}, {"text": "x" * 199}):
        response = client.post("/api/jobs", data=data)
        assert response.status_code == 400 and "500 characters" in response.json()["detail"]
    assert manager.created == []


def test_create_job_rejects_non_pdf_uploads(api, monkeypatch):
    client, manager = api
    response = client.post("/api/jobs", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400 and response.json()["detail"] == "Only PDF files are supported"
    response = client.post("/api/jobs", files={"file": ("paper.pdf", b"<html>not a pdf</html>", "application/pdf")})
    assert response.status_code == 400 and response.json()["detail"] == "File is not a valid PDF"
    monkeypatch.setattr(job_routes, "MAX_PDF_BYTES", 10)
    assert client.post("/api/jobs", files={"file": ("big.pdf", PDF_BYTES, "application/pdf")}).status_code == 413
    assert manager.created == []


def test_unknown_jobs_return_404(api):
    client, _ = api
    for job_id in ("unknown", "20261021_000000_dead"):
        assert client.get(f"/api/jobs/{job_id}").status_code == 404
        assert client.post(f"/api/jobs/{job_id}/panels/1/regenerate").status_code == 404
        response = client.post(f"/api/jobs/{job_id}/cancel")
        assert response.status_code == 404 and response.json() == {"detail": "Job not found"}


def test_regenerate_conflicts_return_409(api):
    client, manager = api
    persisted_job(manager, "20261021_101500_beef", status="completed")
    response = client.post("/api/jobs/20261021_101500_beef/panels/99/regenerate")
    assert response.status_code == 409 and response.json()["detail"] == "No panel 99"
    manager.jobs["20261021_101500_beef"].state["status"] = "rendering"
    response = client.post("/api/jobs/20261021_101500_beef/panels/1/regenerate")
    assert response.status_code == 409 and response.json()["detail"] == "Job is still running"


def test_get_list_and_cancel_a_persisted_job(api):
    client, manager = api
    persisted_job(manager, "20261021_101500_beef", status="completed")
    response = client.get("/api/jobs/20261021_101500_beef")
    assert response.status_code == 200
    body = response.json()
    assert (body["job_id"], body["status"], body["title"]) == ("20261021_101500_beef", "completed", "Persisted")
    assert "_started" not in body and "error" not in body["panels"][0]
    assert [j["job_id"] for j in client.get("/api/jobs").json()["jobs"]] == ["20261021_101500_beef"]
    assert client.post("/api/jobs/20261021_101500_beef/cancel").json() == {"status": "completed"}  # terminal: unchanged


def test_config_endpoint_still_works(api):
    client, _ = api
    response = client.get("/api/config")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"default_model", "fallback_text_model", "providers", "manga_settings", "output_settings"}
    assert all(set(p) == {"name", "enabled", "has_api_key", "models"} and isinstance(p["has_api_key"], bool) for p in body["providers"])
    assert {"default_style", "default_character"} <= set(body["manga_settings"])


def test_create_job_answers_400_with_what_to_do_when_the_account_cannot_serve(api, monkeypatch):
    """A model that is not activated (or no balance) is said before the job exists, with the fix; nothing is created."""
    from engines.byteplus import AccountError
    client, manager = api

    async def refuse():
        raise AccountError(404, json.dumps({"error": {"code": "ModelNotOpen", "message": "not open"}}), "seed-2-0-pro-260328")

    monkeypatch.setattr(manager, "preflight", refuse)
    response = client.post("/api/jobs", data={"text": LONG_PAPER_TEXT})
    assert response.status_code == 400 and "seed-2-0-pro-260328" in response.json()["detail"]
    assert "activate" in response.json()["detail"] and manager.created == []

    async def no_key():
        raise ValueError("The BytePlus ModelArk API key is not set")

    monkeypatch.setattr(manager, "preflight", no_key)
    response = client.post("/api/jobs", data={"text": LONG_PAPER_TEXT})
    assert response.status_code == 400 and "API key is not set" in response.json()["detail"]
