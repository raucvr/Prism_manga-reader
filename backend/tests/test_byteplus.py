"""engines.byteplus: size clamping, payload building and HTTP retry logic (``_post`` is always faked)."""

import json
from types import SimpleNamespace

import httpx
import pytest

from engines import byteplus
from engines.base import GenerationConfig, ImageContent, ImageGenerationConfig, Message
from engines.byteplus import (
    SEEDREAM_MAX_PIXELS,
    SEEDREAM_MIN_PIXELS,
    BytePlusAPIError,
    BytePlusEngine,
    normalize_size,
)

from .fakes import TINY_PNG_B64

FAKE_KEY = "test-key-not-real"


@pytest.fixture
def engine():
    return BytePlusEngine(api_key=FAKE_KEY)


@pytest.fixture
def seedream_calls(engine, monkeypatch):
    calls = []

    async def fake_post(path, payload):
        calls.append((path, payload))
        return {"data": [{"b64_json": TINY_PNG_B64, "size": "1024x1536", "output_format": "jpeg"}], "usage": {"output_tokens": 6144}}

    monkeypatch.setattr(engine, "_post", fake_post)
    return calls


# ----------------------------------------------------------------------------- sizes

@pytest.mark.parametrize("width, height", [(1024, 1536), (2048, 1024)])
def test_normalize_size_keeps_supported_sizes(width, height):
    assert normalize_size(width, height) == f"{width}x{height}"


@pytest.mark.parametrize("width, height", [(800, 800), (3000, 3000), (16, 16), (100, 9000), (5000, 1200), (961, 961)])
def test_normalize_size_clamps_into_the_seedream_pixel_range(width, height):
    w, h = map(int, normalize_size(width, height).split("x"))
    assert SEEDREAM_MIN_PIXELS <= w * h <= SEEDREAM_MAX_PIXELS
    assert w % 16 == 0 and h % 16 == 0
    if 1 / 16 <= width / height <= 16:
        assert abs((w / h) / (width / height) - 1) < 0.2  # aspect ratio roughly kept
    else:
        assert 1 / 16 <= w / h <= 16  # extreme ratios are clamped into Seedream's allowed range
    if width * height < SEEDREAM_MIN_PIXELS:
        assert w * h >= 921_600
    if width * height > SEEDREAM_MAX_PIXELS:
        assert w * h <= 4_624_220


@pytest.mark.parametrize("key", ["", "   ", "${ARK_API_KEY}", None])
def test_engine_requires_an_api_key(key):
    with pytest.raises(ValueError, match="API key"):
        BytePlusEngine(api_key=key)


# ----------------------------------------------------------------------------- text

def test_text_payload_defaults_to_thinking_disabled(engine):
    payload = engine._text_payload([Message.system("sys"), Message.user("hi")], None)
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["model"] == byteplus.DEFAULT_TEXT_MODEL
    assert payload["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert (payload["temperature"], payload["max_tokens"], payload["top_p"]) == (0.7, 4096, 0.95)
    assert "response_format" not in payload and "stop" not in payload


def test_text_payload_passes_overrides(engine):
    config = GenerationConfig(temperature=0.2, max_tokens=8000, stop_sequences=["END"])
    config.thinking = True
    config.model = "dola-seed-2-1-turbo-260628"
    config.response_format = {"type": "json_schema", "json_schema": {"name": "out", "schema": {"type": "object"}, "strict": True}}
    payload = engine._text_payload([Message.user("hi")], config)
    assert payload["thinking"] == {"type": "enabled"} and payload["model"] == "dola-seed-2-1-turbo-260628"
    assert payload["response_format"] == config.response_format and payload["stop"] == ["END"]
    # an engine-level default applies when the call leaves thinking unset
    assert BytePlusEngine(api_key=FAKE_KEY, thinking=True)._text_payload([Message.user("x")], GenerationConfig())["thinking"] == {"type": "enabled"}


def test_vision_messages_become_image_url_parts(engine):
    message = Message.user("Judge this.", images=[ImageContent.from_base64("QUJD", "image/png"), ImageContent.from_url("https://example.invalid/a.jpg")])
    assert engine._build_messages([message]) == [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        {"type": "image_url", "image_url": {"url": "https://example.invalid/a.jpg"}},
        {"type": "text", "text": "Judge this."},
    ]}]


async def test_generate_text_maps_the_response_and_usage(engine, monkeypatch):
    seen = []

    async def fake_post(path, payload):
        seen.append((path, payload))
        return {"model": "seed-2-0-pro-260328", "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18, "completion_tokens_details": {"reasoning_tokens": 3}}}

    monkeypatch.setattr(engine, "_post", fake_post)
    response = await engine.generate_text([Message.user("hi")])
    assert seen[0][0] == "/chat/completions"
    assert (response.content, response.finish_reason, response.model) == ('{"ok": true}', "stop", "seed-2-0-pro-260328")
    assert response.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18, "reasoning_tokens": 3, "cached_tokens": 0}
    # the engine counts what each model spent (the job report prices models apart); cached prompt tokens are read too
    async def cached_post(path, payload):
        return {"model": "x", "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 60}}}

    monkeypatch.setattr(engine, "_post", cached_post)
    await engine.generate_text([Message.user("again")])
    assert engine.tokens == {engine.model: {"prompt_tokens": 111, "completion_tokens": 12, "cached_tokens": 60}}


# ----------------------------------------------------------------------------- images

async def test_generate_image_payload_with_two_references(engine, seedream_calls):
    refs = [ImageContent.from_base64("QUFB", "image/png"), ImageContent.from_base64("QkJC", "image/png")]
    response = await engine.generate_image("draw papi", ImageGenerationConfig(width=1024, height=1536), reference_images=refs)
    (path, payload), = seedream_calls
    assert path == "/images/generations"
    assert payload == {
        "model": byteplus.DEFAULT_IMAGE_MODEL, "prompt": "draw papi", "size": "1024x1536", "response_format": "b64_json",
        "watermark": False, "output_format": "jpeg", "image": ["data:image/png;base64,QUFB", "data:image/png;base64,QkJC"],
    }
    assert response.images[0].data == TINY_PNG_B64 and response.images[0].mime_type == "image/jpeg"
    assert response.usage == {"output_tokens": 6144} and response.size == "1024x1536"


async def test_generate_image_payload_with_one_reference_in_fast_mode(engine, seedream_calls):
    config = ImageGenerationConfig(width=2048, height=1024)
    config.extra = {"fast": True, "output_format": "png", "watermark": False}
    await engine.generate_image("strip", config, reference_images=[ImageContent.from_url("https://example.invalid/sheet.png")])
    payload = seedream_calls[0][1]
    assert payload["image"] == "https://example.invalid/sheet.png"  # a single reference is sent as a string
    assert payload["optimize_prompt_options"] == {"mode": "fast"}
    assert (payload["size"], payload["output_format"], payload["watermark"], payload["response_format"]) == ("2048x1024", "png", False, "b64_json")


async def test_generate_image_without_refs_explicit_size_and_reference_cap(engine, seedream_calls):
    config = ImageGenerationConfig(width=800, height=800)
    config.extra = {"size": "2K"}
    await engine.generate_image("no refs", config)
    payload = seedream_calls[0][1]
    assert payload["size"] == "2K" and "image" not in payload and "optimize_prompt_options" not in payload

    await engine.generate_image("many refs", reference_images=[ImageContent.from_url(f"https://example.invalid/{i}.png") for i in range(12)])
    assert len(seedream_calls[1][1]["image"]) == byteplus.SEEDREAM_MAX_REFERENCE_IMAGES == 10


async def test_generate_image_surfaces_api_errors(engine, monkeypatch):
    async def error_reply(path, payload):
        return {"error": {"code": "InvalidParameter", "message": "bad size"}}

    monkeypatch.setattr(engine, "_post", error_reply)
    with pytest.raises(RuntimeError, match="Seedream error"):
        await engine.generate_image("x")

    async def empty_reply(path, payload):
        return {"data": []}

    monkeypatch.setattr(engine, "_post", empty_reply)
    with pytest.raises(ValueError, match="no image"):
        await engine.generate_image("x")


# ----------------------------------------------------------------------------- HTTP (mock transport only)

async def test_post_retries_transient_errors_and_fails_fast_on_client_errors(engine, monkeypatch):
    statuses, requests, sleeps = [429, 503, 200], [], []

    def handler(request):
        requests.append(request)
        status = statuses.pop(0)
        return httpx.Response(status, json={"ok": True} if status == 200 else {"error": "busy"})

    async def no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(byteplus, "asyncio", SimpleNamespace(sleep=no_sleep))
    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    assert await engine._post("/chat/completions", {"x": 1}) == {"ok": True}
    assert len(requests) == 3 and len(sleeps) == 2
    assert 3 <= sleeps[0] <= 5 and 6 <= sleeps[1] <= 8  # exponential backoff with jitter
    assert str(requests[0].url) == f"{byteplus.DEFAULT_BASE_URL}/chat/completions"
    assert json.loads(requests[0].content) == {"x": 1}

    statuses[:] = [400]
    with pytest.raises(BytePlusAPIError) as excinfo:
        await engine._post("/chat/completions", {})
    assert excinfo.value.status == 400 and len(requests) == 4 and len(sleeps) == 2  # no retry
    await engine.close()


async def test_an_unpatched_engine_cannot_reach_the_network():
    engine = BytePlusEngine(api_key=FAKE_KEY, max_retries=0)
    with pytest.raises(RuntimeError, match="Network access is disabled"):
        await engine.generate_text([Message.user("hi")])
    await engine.close()


async def test_long_calls_stream_and_assemble_the_reply(engine, monkeypatch):
    """A thinking call (or one allowed a long reply) is streamed; the chunks become one reply with the usage of the last
    chunk; a stream that breaks off is retried from the start."""
    from engines.base import GenerationConfig
    sse = "\n".join([
        'data: {"model": "deepseek-v4-pro-ga-260813", "choices": [{"delta": {"role": "assistant", "reasoning_content": "hmm"}}]}',
        'data: {"choices": [{"delta": {"content": "{\\"ok\\""}}]}',
        'data: {"choices": [{"delta": {"content": ": true}"}, "finish_reason": "stop"}]}',
        'data: {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 9, "prompt_tokens_details": {"cached_tokens": 30},'
        ' "completion_tokens_details": {"reasoning_tokens": 4}}}',
        "data: [DONE]", ""])
    calls, sleeps = [], []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    async def no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(byteplus, "asyncio", SimpleNamespace(sleep=no_sleep))
    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    config = GenerationConfig(max_tokens=24000)
    config.thinking = True
    response = await engine.generate_text([Message.user("think")], config)
    assert (response.content, response.finish_reason, response.model) == ('{"ok": true}', "stop", "deepseek-v4-pro-ga-260813")
    assert response.usage["cached_tokens"] == 30 and response.usage["reasoning_tokens"] == 4
    assert calls[1]["stream"] is True and calls[1]["stream_options"] == {"include_usage": True} and len(sleeps) == 1
    await engine.close()


def test_reasoning_effort_goes_with_thinking_only(engine):
    from engines.base import GenerationConfig
    config = GenerationConfig()
    config.thinking = True
    engine.reasoning_effort = "low"
    assert engine._text_payload([Message.user("x")], config)["reasoning_effort"] == "low"
    config.reasoning_effort = "high"                                   # a call may ask for more
    assert engine._text_payload([Message.user("x")], config)["reasoning_effort"] == "high"
    config.thinking = False                                           # no thinking: the API refuses an effort
    assert "reasoning_effort" not in engine._text_payload([Message.user("x")], config)


@pytest.mark.parametrize("status,code,account", [
    (404, "ModelNotOpen", True), (403, "AccountOverdueError", True), (429, "QuotaExceeded", True), (429, "SetLimitExceeded", True),
    (401, "AuthenticationError", True), (403, "AccessDenied", True), (429, "RateLimitExceeded", False),
    (429, "ModelAccountTpmRateLimitExceeded", False), (503, "ServerOverloaded", False), (400, "InvalidParameter", False)])
async def test_account_errors_are_raised_at_once_with_what_to_do(engine, monkeypatch, status, code, account):
    """A refused key, an inactive model, an overdue account, a used-up quota or a spending limit never get better by
    waiting: one request, then an AccountError that says what to do. Rate limits are still retried."""
    requests, sleeps = [], []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"code": code, "message": f"{code} message"}})

    async def no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(byteplus, "asyncio", SimpleNamespace(sleep=no_sleep))
    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(BytePlusAPIError) as excinfo:
        await engine._post("/images/generations", {"model": "dola-seedream-5-0-pro-260628"})
    assert isinstance(excinfo.value, byteplus.AccountError) is account
    if account:
        assert len(requests) == 1 and excinfo.value.code == code
        assert "ModelArk" in str(excinfo.value) and " / " in excinfo.value.hint   # English and Chinese
        if code == "ModelNotOpen":
            assert "dola-seedream-5-0-pro-260628" in excinfo.value.hint and "activate" in excinfo.value.hint
    else:
        assert len(requests) == (engine.max_retries + 1 if status in byteplus.RETRYABLE_STATUS else 1)
    await engine.close()


async def test_a_content_filter_refusal_is_an_error_not_an_empty_answer(engine, monkeypatch):
    async def filtered(path, payload):
        return {"model": payload["model"], "choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

    monkeypatch.setattr(engine, "_post", filtered)
    with pytest.raises(byteplus.ContentFilteredError) as excinfo:
        await engine.generate_text([Message.user("a paper that trips the filter")])
    assert excinfo.value.code == "ContentFilter" and not getattr(excinfo.value, "account", False)


async def test_an_ssl_error_mid_reply_is_retried(engine, monkeypatch):
    import ssl
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise ssl.SSLError("[SSL] record layer failure")
        return httpx.Response(200, json={"ok": True})

    async def no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(byteplus, "asyncio", SimpleNamespace(sleep=no_sleep))
    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await engine._post("/chat/completions", {"x": 1}) == {"ok": True}
    assert len(calls) == 2 and len(sleeps) == 1
    await engine.close()
