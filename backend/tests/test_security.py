"""Cross-site request refusal (main.SameOriginWrites) and the /api/generate switch (server.enable_generate_api).

Any web page can make its visitor's browser POST a plain form to Prism on localhost, and without the guard that
starts a paid job. Here job creation is only recorded, the generate routes get a fake model client, and conftest's
autouse ``_no_network`` blocks real transports, so nothing reaches ModelArk. TestClient is never used as a context
manager, so the lifespan (config watcher, real engines) never starts; config tests use private ``ConfigManager``
instances over ``tmp_path``.
"""

from __future__ import annotations

import asyncio

import pytest
import yaml
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import config_loader
import main
import routes.generation as generation
import routes.jobs as job_routes
import services.jobs as jobs
from config_loader import ConfigManager
from engines.base import TextResponse

from .fakes import LONG_PAPER_TEXT

FRONTEND, FRONTEND_IP = main.local_origins(main.FRONTEND_PORT)
BACKEND, BACKEND_IP = main.local_origins(main.BACKEND_PORT)
EVIL = "https://evil.example"
PDF_BYTES = b"%PDF-1.4\n% minimal test payload\n"
REFUSED = "Refused: this request came from another site"


class RecordingManager(jobs.JobManager):
    """Real get/list/cancel over tmp_path; ``create`` only records its kwargs (no pipeline task)."""

    def __init__(self, output_dir):
        super().__init__(output_dir=output_dir)
        self.created = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return {"job_id": "20261023_120000_5afe", "status": "queued"}


@pytest.fixture
def api(monkeypatch, tmp_path):
    out = tmp_path / "output"
    monkeypatch.setattr(jobs, "OUTPUT_DIR", out)
    manager = RecordingManager(out)
    monkeypatch.setattr(job_routes, "get_job_manager", lambda: manager)
    client = TestClient(main.app)
    yield client, manager
    client.close()


def post_form(client, headers, *, multipart=True):
    """What a hidden, auto-submitting <form method="post"> sends: a multipart PDF upload, or urlencoded text."""
    if multipart:
        return client.post("/api/jobs", files={"file": ("paper.pdf", PDF_BYTES, "application/pdf")}, headers=headers)
    return client.post("/api/jobs", data={"text": LONG_PAPER_TEXT}, headers=headers)


# ============================================================================= cross-site writes

CROSS_SITE = {
    "other-site": {"Origin": EVIL, "Sec-Fetch-Site": "cross-site"},
    "no-fetch-metadata": {"Origin": EVIL},                                      # browsers without Sec-Fetch-*
    "opaque-origin": {"Origin": "null", "Sec-Fetch-Site": "cross-site"},       # sandboxed iframe, data: or file: page
    "fetch-metadata-only": {"Sec-Fetch-Site": "cross-site"},
    "other-local-app": {"Origin": "http://localhost:5173", "Sec-Fetch-Site": "same-site"},
    "lan-address": {"Origin": f"http://192.168.1.20:{main.FRONTEND_PORT}", "Sec-Fetch-Site": "same-origin"},
    "dns-rebinding": {"Origin": f"http://rebind.evil.example:{main.FRONTEND_PORT}", "Sec-Fetch-Site": "same-origin"},
    "lookalike-host": {"Origin": f"{FRONTEND}.evil.example", "Sec-Fetch-Site": "cross-site"},
}

OWN_PAGES = {
    "ui-through-next-proxy": {"Origin": FRONTEND, "Sec-Fetch-Site": "same-origin"},
    "ui-on-127.0.0.1": {"Origin": FRONTEND_IP, "Sec-Fetch-Site": "same-origin"},
    "ui-calling-backend-port": {"Origin": FRONTEND, "Sec-Fetch-Site": "same-site"},
    "api-docs-page": {"Origin": BACKEND_IP, "Sec-Fetch-Site": "same-origin"},
    "curl-or-script": {},
}


@pytest.mark.parametrize("multipart", [True, False], ids=["multipart", "urlencoded"])
@pytest.mark.parametrize("headers", CROSS_SITE.values(), ids=CROSS_SITE.keys())
def test_a_form_post_from_another_site_cannot_start_a_paid_job(api, headers, multipart):
    client, manager = api
    response = post_form(client, headers, multipart=multipart)
    assert response.status_code == 403 and manager.created == []
    assert response.json()["detail"].startswith(REFUSED) and FRONTEND in response.json()["detail"]
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("headers", OWN_PAGES.values(), ids=OWN_PAGES.keys())
def test_prisms_own_pages_and_non_browser_clients_still_create_jobs(api, headers):
    client, manager = api
    response = post_form(client, headers)
    assert response.status_code == 202 and len(manager.created) == 1


@pytest.mark.parametrize("method, path", [
    ("POST", "/api/jobs/20261023_120000_5afe/cancel"),
    ("POST", "/api/jobs/20261023_120000_5afe/panels/1/regenerate"),
    ("PUT", "/api/jobs"), ("PATCH", "/api/jobs"), ("DELETE", "/api/jobs/20261023_120000_5afe"),
])
def test_every_state_changing_method_is_checked_before_routing(api, method, path):
    client, _ = api
    assert client.request(method, path, headers={"Origin": EVIL}).status_code == 403  # not the route's 404 / 405
    assert client.request(method, path, headers={"Origin": FRONTEND}).status_code in (404, 405)


def test_reads_pass_and_cors_still_hides_them_from_other_sites(api):
    client, _ = api
    foreign = client.get("/api/jobs", headers={"Origin": EVIL, "Sec-Fetch-Site": "cross-site"})
    assert foreign.status_code == 200 and "access-control-allow-origin" not in foreign.headers
    assert client.get("/api/jobs", headers={"Origin": FRONTEND}).headers["access-control-allow-origin"] == FRONTEND


async def test_a_cross_site_upload_is_refused_before_its_body_is_read(api):
    pulled, sent = [], []

    async def receive():
        pulled.append(1)
        return {"type": "http.request", "body": b"0" * 65536, "more_body": True}

    async def send(message):
        sent.append(message)

    # 50 MB declared: under the upload limit, so only the guard can answer without reading the body
    headers = [(b"host", b"127.0.0.1:8000"), (b"origin", EVIL.encode()), (b"content-length", b"50000000"),
               (b"content-type", b"multipart/form-data; boundary=prism")]
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/api/jobs", "raw_path": b"/api/jobs", "query_string": b"", "root_path": "", "headers": headers,
        "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8000),
    }
    await asyncio.wait_for(main.app(scope, receive, send), timeout=10)
    assert sent[0]["status"] == 403 and pulled == []


@pytest.mark.parametrize("environ, ports", [
    ({}, (3000, 8000)),
    ({"FRONTEND_PORT": "3005", "BACKEND_PORT": "8005"}, (3005, 8005)),
    ({"FRONTEND_PORT": "", "BACKEND_PORT": "not-a-port"}, (3000, 8000)),
])
def test_the_allowed_origins_follow_the_ports_the_start_scripts_export(environ, ports):
    assert main.app_ports(environ) == ports
    assert main.local_origins(*ports) == [f"http://localhost:{ports[0]}", f"http://127.0.0.1:{ports[0]}",
                                          f"http://localhost:{ports[1]}", f"http://127.0.0.1:{ports[1]}"]
    assert main.APP_ORIGINS == frozenset(main.local_origins(*main.app_ports()))


def test_the_guard_checks_against_the_origins_it_is_given():
    async def endpoint(scope, receive, send):
        await JSONResponse({"ok": True})(scope, receive, send)

    client = TestClient(main.SameOriginWrites(endpoint, origins={"http://localhost:3005"}))
    assert client.post("/", headers={"Origin": "http://localhost:3005"}).status_code == 200
    assert client.post("/", headers={"Origin": "http://localhost:3000"}).status_code == 403
    assert client.post("/", headers={"Origin": "http://LOCALHOST:3005", "Sec-Fetch-Site": "same-origin"}).status_code == 200


# ============================================================================= /api/generate is off by default

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


class FakeClient:
    """Stands in for engines.ModelClient: records the prompts and answers "pong"."""

    def __init__(self):
        self.prompts = []

    async def generate_text(self, prompt, system_prompt=None, images=None, config=None):
        self.prompts.append(prompt)
        return TextResponse(content="pong", model="fake-seed", usage={"prompt_tokens": 1, "completion_tokens": 1})


@pytest.fixture
def generate_api(monkeypatch, config_file):
    """POST to /api/generate/* with the given config text; the model client and the text engine are fakes."""
    fake = FakeClient()

    async def get_client():
        return fake

    def get_text_engine():
        raise AssertionError("the streaming route must not build an engine while the switch is off")

    monkeypatch.setattr(generation, "get_client", get_client)
    monkeypatch.setattr(generation, "get_text_engine", get_text_engine)
    client = TestClient(main.app)

    def use(text: str) -> ConfigManager:
        manager = load_manager(config_file, text)
        monkeypatch.setattr(generation, "get_config", lambda: manager)
        return manager

    yield client, fake, use
    client.close()


GENERATE_ON = "server:\n  enable_generate_api: true\n"


@pytest.mark.parametrize("text, enabled", [
    ("core_engine: {}\n", False),
    ("server:\n", False),
    ("server: on\n", False),                                   # not a mapping
    ("server:\n  enable_generate_api: false\n", False),
    ("server:\n  enable_generate_api: \"yes\"\n", True),
    (GENERATE_ON, True),
])
def test_the_generate_switch_is_off_unless_the_config_turns_it_on(config_file, text, enabled):
    assert load_manager(config_file, text).generate_api_enabled is enabled


def test_the_template_ships_with_the_generate_switch_off():
    template = config_loader.PROJECT_ROOT / "config" / "api_config.yaml.example"
    assert yaml.safe_load(template.read_text(encoding="utf-8"))["server"]["enable_generate_api"] is False


@pytest.mark.parametrize("path", ["/api/generate/text", "/api/generate/text/stream", "/api/generate/image",
                                  "/api/generate/multimodal"])
def test_the_generate_routes_answer_404_while_the_switch_is_off(generate_api, path):
    client, fake, use = generate_api
    use("core_engine: {}\n")
    response = client.post(path, json={"prompt": "hi"})
    assert response.status_code == 404 and "server.enable_generate_api" in response.json()["detail"]
    assert fake.prompts == []


def test_turning_the_generate_switch_on_and_off_applies_without_a_restart(generate_api, config_file):
    client, fake, use = generate_api
    manager = use(GENERATE_ON)
    response = client.post("/api/generate/text", json={"prompt": "ping"})
    assert response.status_code == 200 and response.json()["content"] == "pong" and fake.prompts == ["ping"]

    config_file.write_text("server:\n  enable_generate_api: false\n", encoding="utf-8")
    assert manager.reload() is True                            # what the config watcher does on save
    assert client.post("/api/generate/text", json={"prompt": "ping"}).status_code == 404
    assert fake.prompts == ["ping"]


@pytest.mark.parametrize("headers, allowed", [
    ({}, True),                                                                  # TestClient's own host
    ({"host": "localhost:8000"}, True), ({"host": "127.0.0.1:3000"}, True), ({"host": "[::1]:8000"}, True),
    ({"host": "rebind.evil.example:8000"}, False),                               # DNS rebinding straight to the backend
    ({"host": "127.0.0.1:8000", "x-forwarded-host": "rebind.evil.example:3000"}, False),   # ...or through the Next.js proxy
    ({"host": "127.0.0.1:8000", "x-forwarded-host": "localhost:3000"}, True),
])
def test_reads_answer_only_to_this_machine(api, headers, allowed):
    client, _manager = api
    for path in ("/api/jobs", "/output/missing.json"):
        response = client.get(path, headers=headers)
        assert (response.status_code != 403) == allowed, (path, headers, response.status_code)
        if not allowed:
            assert response.json()["detail"].startswith("Refused: Prism answers only to localhost")


def test_host_name_strips_the_port():
    assert [main.host_name(h) for h in ("localhost:3000", "127.0.0.1", "[::1]:8000", "Example.COM:80, other")] == [
        "localhost", "127.0.0.1", "[::1]", "example.com"]
