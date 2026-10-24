"""services.engine_select: provider/model specs -> engines (construction only, no calls)."""

import pytest

from config_loader import ProviderConfig
from engines.byteplus import DEFAULT_IMAGE_MODEL, DEFAULT_TEXT_MODEL, BytePlusEngine
from services import engine_select

from .fakes import FakeConfig

BYTEPLUS = ProviderConfig(name="byteplus", enabled=True, api_key="test-key-not-real", base_url="https://ark.example.invalid/api/v3")


def test_split_model_spec():
    assert engine_select._split("byteplus/seed-2-0-pro-260328") == ("byteplus", "seed-2-0-pro-260328")
    assert engine_select._split("openrouter/google/gemini-3-pro-image-preview") == ("openrouter", "google/gemini-3-pro-image-preview")
    assert engine_select._split("seed-2-0-pro-260328") == (None, "seed-2-0-pro-260328")


async def test_build_engines_creates_one_engine_per_spec(monkeypatch):
    core = {"text_model": "byteplus/seed-2-0-pro-260328", "fast_text_model": "byteplus/dola-seed-2-1-turbo-260628",
            "vision_model": "byteplus/seed-2-0-pro-260328", "image_model": "byteplus/dola-seedream-5-0-pro-260628"}
    monkeypatch.setattr(engine_select, "get_config", lambda: FakeConfig(core, {"byteplus": BYTEPLUS}))
    engines = engine_select.build_engines()

    assert all(isinstance(e, BytePlusEngine) for e in (engines.text, engines.fast_text, engines.vision, engines.image))
    assert engines.text is engines.vision and engines.text is not engines.fast_text
    assert (engines.text.model, engines.fast_text.model) == ("seed-2-0-pro-260328", "dola-seed-2-1-turbo-260628")
    assert engines.image.image_model == "dola-seedream-5-0-pro-260628"
    assert engines.image.base_url == "https://ark.example.invalid/api/v3"

    closed = []
    for e in (engines.text, engines.fast_text, engines.image):
        async def fake_close(e=e):
            closed.append(e)
        monkeypatch.setattr(e, "close", fake_close)
    await engines.close()
    assert closed == [engines.text, engines.fast_text, engines.image]  # the shared text/vision engine closes once


def test_build_engines_defaults_and_disabled_provider(monkeypatch):
    monkeypatch.setattr(engine_select, "get_config", lambda: FakeConfig({}, {"byteplus": BYTEPLUS}))
    engines = engine_select.build_engines()
    assert engines.text.model == DEFAULT_TEXT_MODEL and engines.image.image_model == DEFAULT_IMAGE_MODEL

    disabled = ProviderConfig(name="byteplus", enabled=False, api_key="test-key-not-real")
    monkeypatch.setattr(engine_select, "get_config", lambda: FakeConfig({}, {"byteplus": disabled}))
    with pytest.raises(RuntimeError, match="not enabled"):
        engine_select.build_engines()
