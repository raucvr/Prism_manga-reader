"""Round 3 regression tests for W6 (config hot reload) and J10 (upload size limit before the body is read).

Config tests build private ``ConfigManager`` instances over ``tmp_path``; the real ``config/`` directory and the
module singleton are never touched. Upload tests drive ``main.app`` directly over ASGI with job creation recorded,
so no pipeline task starts and nothing reaches ModelArk (conftest's ``_no_network`` also blocks real transports).
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient
from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileMovedEvent

import config_loader
import main
import routes.config as config_routes
import routes.jobs as job_routes
import services.jobs as jobs
from config_loader import ConfigChangeHandler, ConfigManager, as_int

# ============================================================================= W6: config hot reload

GOOD_YAML = """\
providers:
  byteplus:
    enabled: true
    api_key: "test-key-not-real"
manga_settings:
  concurrency: 3
"""

FAKE_SECRET = "sk-test-not-a-real-key-7f3a"
# What a reader sees in the middle of a save, on a line that holds a key
HALF_WRITTEN_YAML = f'providers:\n  byteplus: {{enabled: true, api_key: "{FAKE_SECRET}", [\n'


@pytest.fixture
def config_file(monkeypatch, tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    path = cfg_dir / "api_config.yaml"
    monkeypatch.setattr(config_loader, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(config_loader, "API_CONFIG_PATH", path)
    return path


def load_manager(path, text: str) -> ConfigManager:
    """A fresh ConfigManager over ``path``; ``__new__`` is bypassed so the module singleton stays untouched."""
    path.write_text(text, encoding="utf-8")
    manager = object.__new__(ConfigManager)
    manager._initialized = False
    manager.__init__()
    return manager


def test_w6_reload_reports_whether_the_new_file_was_applied(config_file):
    manager = load_manager(config_file, GOOD_YAML)
    config_file.write_text(GOOD_YAML.replace("concurrency: 3", "concurrency: 4"), encoding="utf-8")
    assert manager.reload() is True and manager.manga_settings.concurrency == 4
    config_file.write_text(HALF_WRITTEN_YAML, encoding="utf-8")
    assert manager.reload() is False and manager.manga_settings.concurrency == 4


@pytest.mark.parametrize("text", ["{}\n", "# being edited\n", "\n\n"], ids=["empty-mapping", "comments-only", "blank-lines"])
def test_w6_a_config_with_no_settings_on_hot_reload_keeps_the_previous_one(config_file, text):
    manager = load_manager(config_file, GOOD_YAML)
    config_file.write_text(text, encoding="utf-8")
    applied = manager.reload()
    assert manager.get_provider("byteplus") is not None and manager.get_provider("byteplus").enabled
    assert manager.manga_settings.concurrency == 3 and applied is False


def test_w6_the_watchdog_handler_never_lets_an_exception_escape(tmp_path):
    def broken_reload():
        raise RuntimeError("reload bug")

    # Raising here would end watchdog's observer thread, and with it every later hot reload.
    ConfigChangeHandler(broken_reload).on_modified(FileModifiedEvent(str(tmp_path / "api_config.yaml")))


@pytest.mark.parametrize("raw", [10**400, float("-inf"), "-1e999", "inf"])
def test_w6_as_int_falls_back_for_numbers_too_large_for_an_int(raw):
    assert as_int(raw, 5) == 5


def test_w6_a_huge_yaml_integer_does_not_crash_startup(config_file):
    manager = load_manager(config_file, GOOD_YAML.replace("concurrency: 3", "concurrency: 1" + "0" * 400))
    assert manager.manga_settings.concurrency == 5


@pytest.mark.parametrize("core", ["core_engine:\n", "core_engine: byteplus\n"], ids=["emptied", "scalar"])
def test_w6_a_core_engine_that_is_not_a_mapping_reads_as_absent(config_file, core):
    names = ("default_model", "fallback_text_model", "image_model", "text_model", "fast_text_model", "vision_model")
    expected = {name: getattr(load_manager(config_file, GOOD_YAML), name) for name in names}
    # every model line commented out leaves `core_engine:` null; the startup log and GET /api/config read these
    manager = load_manager(config_file, core + GOOD_YAML)
    assert {name: getattr(manager, name) for name in names} == expected


def test_w6_reload_endpoint_fails_without_echoing_the_file_then_succeeds_once_fixed(config_file, monkeypatch):
    manager = load_manager(config_file, GOOD_YAML)

    async def no_reset():
        return None

    monkeypatch.setattr(config_routes, "get_config", lambda: manager)
    monkeypatch.setattr(config_routes, "reset_client", no_reset)
    client = TestClient(main.app)  # no lifespan: no watcher, no engines
    try:
        config_file.write_text(HALF_WRITTEN_YAML, encoding="utf-8")
        failed = client.post("/api/config/reload")
        config_file.write_text(GOOD_YAML.replace("concurrency: 3", "concurrency: 6"), encoding="utf-8")
        fixed = client.post("/api/config/reload")
    finally:
        client.close()
    assert failed.status_code == 500 and "previous configuration" in failed.json()["detail"]
    assert FAKE_SECRET not in failed.text  # parser messages can quote the file; the API never does
    assert fixed.status_code == 200 and fixed.json()["status"] == "success"
    assert manager.manga_settings.concurrency == 6


TEMPLATE_YAML = """\
providers:
  byteplus:
    enabled: true
    api_key: "${PRISM_ROUND3_UNSET_KEY}"
manga_settings:
  concurrency: 9
"""


@pytest.fixture
def template(config_file, monkeypatch):
    """api_config.yaml.example next to the config, with a placeholder key like the repo's template."""
    monkeypatch.delenv("PRISM_ROUND3_UNSET_KEY", raising=False)
    path = config_file.with_name("api_config.yaml.example")
    path.write_text(TEMPLATE_YAML, encoding="utf-8")
    return path


def test_w6_deleting_the_file_while_running_keeps_its_config_not_the_template(config_file, template):
    manager = load_manager(config_file, GOOD_YAML)
    config_file.unlink()  # rm, or a save that first renames the file away
    # watchdog on macOS reports the rm as a modification of the now missing file
    ConfigChangeHandler(manager.reload).dispatch(FileModifiedEvent(str(config_file)))
    assert manager.get_provider("byteplus").api_key == "test-key-not-real" and manager.manga_settings.concurrency == 3
    assert manager.reload() is False


def test_w6_startup_without_the_file_still_uses_the_template_and_reloads_it(config_file, template):
    manager = object.__new__(ConfigManager)
    manager._initialized = False
    manager.__init__()
    assert manager.manga_settings.concurrency == 9
    template.write_text(TEMPLATE_YAML.replace("concurrency: 9", "concurrency: 8"), encoding="utf-8")
    assert manager.reload() is True and manager.manga_settings.concurrency == 8
    config_file.write_text(GOOD_YAML, encoding="utf-8")  # the user creates the real file
    assert manager.reload() is True and manager.get_provider("byteplus").api_key == "test-key-not-real"


def test_w6_an_atomic_save_renamed_over_the_file_is_applied(config_file):
    manager = load_manager(config_file, GOOD_YAML)
    tmp = config_file.with_name(".api_config.yaml.tmp")
    tmp.write_text(GOOD_YAML.replace("concurrency: 3", "concurrency: 7"), encoding="utf-8")
    os.replace(tmp, config_file)
    # what watchdog delivers for an atomic save (temp file renamed over the config): a move, no modified event
    ConfigChangeHandler(manager.reload).dispatch(FileMovedEvent(str(tmp), str(config_file)))
    assert manager.manga_settings.concurrency == 7


def test_w6_an_atomic_save_from_another_directory_is_applied(config_file, tmp_path):
    manager = load_manager(config_file, GOOD_YAML)
    tmp = tmp_path / "TemporaryItems" / "api_config.yaml"
    tmp.parent.mkdir()
    tmp.write_text(GOOD_YAML.replace("concurrency: 3", "concurrency: 6"), encoding="utf-8")
    os.replace(tmp, config_file)
    # renamed in from outside the watched directory, the save reaches watchdog as a creation only
    ConfigChangeHandler(manager.reload).dispatch(FileCreatedEvent(str(config_file)))
    assert manager.manga_settings.concurrency == 6


def test_w6_moving_the_file_away_does_not_reload(config_file):
    calls = []
    ConfigChangeHandler(lambda: calls.append(1)).dispatch(FileMovedEvent(str(config_file), str(config_file) + "~"))
    assert calls == []


# ============================================================================= J10: upload size limit

LIMIT = 4096  # routes.jobs.MAX_PDF_BYTES during these tests
CHUNK = 64 * 1024
BOUNDARY = "prismround3"


class UploadRecorder(jobs.JobManager):
    """Real persistence over tmp_path; ``create`` only records its kwargs (no pipeline task)."""

    def __init__(self, output_dir):
        super().__init__(output_dir=output_dir)
        self.created = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return {"job_id": "20261022_100000_c0de", "status": "queued"}


@pytest.fixture
def recorder(monkeypatch, tmp_path):
    out = tmp_path / "output"
    monkeypatch.setattr(jobs, "OUTPUT_DIR", out)
    manager = UploadRecorder(out)
    monkeypatch.setattr(job_routes, "get_job_manager", lambda: manager)
    monkeypatch.setattr(job_routes, "MAX_PDF_BYTES", LIMIT)
    return manager


def multipart_pdf(size: int) -> bytes:
    head = (f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"big.pdf\"\r\n"
            "Content-Type: application/pdf\r\n\r\n").encode()
    return head + b"%PDF-1.4\n" + b"0" * (size - 9) + f"\r\n--{BOUNDARY}--\r\n".encode()


async def post_streamed(body: bytes, *, content_length: bool = True, extra_headers=()) -> tuple[dict, int]:
    """POST /api/jobs straight into the ASGI app; record how many body chunks it pulled before answering."""
    chunks = [body[i:i + CHUNK] for i in range(0, len(body), CHUNK)]
    out = {"pulled": 0, "pulled_at_response": None, "status": None, "headers": {}, "body": b""}
    finished = asyncio.Event()

    async def receive():
        if out["pulled"] < len(chunks):
            i = out["pulled"]
            out["pulled"] += 1
            return {"type": "http.request", "body": chunks[i], "more_body": i + 1 < len(chunks)}
        await finished.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"], out["pulled_at_response"] = message["status"], out["pulled"]
            out["headers"] = dict(message.get("headers") or [])
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")
            if not message.get("more_body"):
                finished.set()

    headers = [(b"host", b"testserver"), (b"content-type", f"multipart/form-data; boundary={BOUNDARY}".encode()),
               *extra_headers]
    if content_length:
        headers.append((b"content-length", str(len(body)).encode()))
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/api/jobs", "raw_path": b"/api/jobs", "query_string": b"", "root_path": "", "headers": headers,
        "client": ("127.0.0.1", 50000), "server": ("testserver", 80),
    }
    await asyncio.wait_for(main.app(scope, receive, send), timeout=10)
    return out, len(chunks)


async def test_j10_a_chunked_upload_without_content_length_is_cut_off_at_the_limit(recorder):
    out, total = await post_streamed(multipart_pdf(2 * 1024 * 1024), content_length=False)
    assert out["status"] == 413 and recorder.created == []
    # Refused once the byte count passed the limit plus the multipart slack (within 256 KB), not at the 2 MB end.
    assert out["pulled_at_response"] <= 4 < total
    assert json.loads(out["body"])["detail"].startswith("PDF larger than")


async def test_j10_the_early_413_is_json_with_cors_headers_and_reads_no_body(recorder):
    origin = b"http://localhost:3000"
    out, _ = await post_streamed(multipart_pdf(2 * 1024 * 1024), extra_headers=[(b"origin", origin)])
    assert (out["status"], out["pulled_at_response"], recorder.created) == (413, 0, [])
    # The browser can read the answer (and show its detail) only with the CORS header.
    assert out["headers"].get(b"access-control-allow-origin") == origin
    assert json.loads(out["body"])["detail"].startswith("PDF larger than")


# ----------------------------------------------------------------------------- legacy PDF routes share the upload limit

@pytest.mark.parametrize("path", ["/api/manga/upload-pdf", "/api/manga/generate/pdf"])
def test_legacy_pdf_routes_refuse_an_oversized_upload(path, monkeypatch):
    import routes.jobs as job_routes
    import routes.manga as manga_routes
    from fastapi.testclient import TestClient
    import main

    import routes.generation as generation
    monkeypatch.setitem(main.app.dependency_overrides, generation.require_generate_api, lambda: None)   # the paid legacy routes are off by default
    monkeypatch.setattr(job_routes, "MAX_PDF_BYTES", 1000)
    monkeypatch.setattr(manga_routes, "MAX_PDF_BYTES", 1000)
    body = b"%PDF-1.4\n" + b"0" * 5000
    with TestClient(main.app) as client:
        early = client.post(path, files={"file": ("big.pdf", body, "application/pdf")})
        assert early.status_code == 413          # the middleware: Content-Length is over the limit plus slack
    monkeypatch.setattr(main, "MULTIPART_SLACK", 10_000)
    with TestClient(main.app) as client:
        late = client.post(path, files={"file": ("big.pdf", body, "application/pdf")})
        assert late.status_code == 413           # the route's bounded read
