"""
Engine selection for the v2 pipeline.

``core_engine.{text_model, fast_text_model, vision_model, image_model}`` are written as ``provider/model``, and so are the
optional ``check_model`` (fact-check, brief check, abstract quiz) and ``translate_model`` (translation), which default to
``text_model``: a second model family can check the writer's work.
A ModelArk spec may end in ``?effort=low`` (or medium / high): the reasoning effort of that role's thinking calls
(DeepSeek V4 Pro reasons for minutes at its default, high).
``byteplus`` builds a :class:`BytePlusEngine`; other providers go through the legacy factory in ``engines``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config_loader import get_config
from engines import create_engine, BaseEngine
from engines.base import GenerationConfig, Message
from engines.byteplus import BytePlusEngine, DEFAULT_BASE_URL, DEFAULT_IMAGE_MODEL, DEFAULT_TEXT_MODEL


@dataclass
class Engines:
    text: BaseEngine                       # the writer: brief, story, fix rounds, quiz revisions
    fast_text: BaseEngine                  # line fitting, the quiz's naive reader
    vision: BaseEngine                     # reads the art: grounding, panel QA
    image: BaseEngine
    check: Optional[BaseEngine] = None     # fact-check, brief check, quiz questions and grades (default: text)
    translate: Optional[BaseEngine] = None  # translation (default: text)

    def __post_init__(self):
        self.check = self.check or self.text
        self.translate = self.translate or self.text

    def _unique(self) -> list[BaseEngine]:
        return list({id(e): e for e in (self.text, self.fast_text, self.vision, self.image, self.check, self.translate)}.values())

    def token_usage(self) -> dict[str, dict[str, int]]:
        """Tokens spent so far per model id, over every engine ({model: {"prompt_tokens", "completion_tokens",
        "cached_tokens"}}); engines that do not count tokens add nothing."""
        out: dict[str, dict[str, int]] = {}
        for e in self._unique():
            for model, used in (getattr(e, "tokens", None) or {}).items():
                mine = out.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0})
                for key in mine:
                    mine[key] += int(used.get(key, 0) or 0)
        return out

    async def close(self) -> None:
        for e in self._unique():
            await e.close()


# account errors that concern one model only: the role falls back to the writer's model (an account-wide error, such as
# a refused key or an overdue account, would stop the writer too and is raised)
FALLBACK_CODES = ("ModelNotOpen", "InvalidEndpointOrModel", "QuotaExceeded", "AccessDenied", "OperationDenied")


class FallbackEngine(BaseEngine):
    """A role's own model (the checker, the translator) with the writer's model behind it. A call the role's model cannot
    take, because its content filter refused the text (DeepSeek V4 Pro refuses some ordinary paper text) or the model is
    not activated on this account, is made on the writer's model instead; each reason is kept in ``fallbacks`` for the
    job's quality record, and a model that is not activated is not asked again. So the default configuration works on
    an account that activated only the Seed models, and is better on one that also activated DeepSeek."""

    def __init__(self, primary: BaseEngine, fallback: BaseEngine):
        super().__init__(getattr(primary, "api_key", ""), getattr(primary, "base_url", "") or "", getattr(primary, "model", ""))
        self.primary, self.fallback = primary, fallback
        self.fallbacks: list[str] = []
        self._inactive = False

    @property
    def name(self) -> str:
        return self.primary.name

    @property
    def supports_image_generation(self) -> bool:
        return False

    @property
    def supports_vision(self) -> bool:
        return self.primary.supports_vision

    @property
    def tokens(self) -> dict:
        return getattr(self.primary, "tokens", {})   # the fallback's tokens are counted on the writer's own engine

    async def generate_text(self, messages, config=None):
        if not self._inactive:
            try:
                return await self.primary.generate_text(messages, config)
            except Exception as e:  # noqa: BLE001
                code = str(getattr(e, "code", "") or "")
                filtered = code == "ContentFilter"
                inactive = bool(getattr(e, "account", False)) and code.startswith(FALLBACK_CODES)
                if not (filtered or inactive):
                    raise
                self._inactive = inactive
                note = (f"{self.model}: {'its content filter refused the text' if filtered else code}; "
                        f"{getattr(self.fallback, 'model', 'the writer model')} took over")
                if note not in self.fallbacks:
                    self.fallbacks.append(note)
                    print(f"[engine_select] {note}")
        return await self.fallback.generate_text(messages, config)

    async def generate_text_stream(self, messages, config=None):
        async for chunk in self.primary.generate_text_stream(messages, config):
            yield chunk

    async def close(self) -> None:
        await self.primary.close()


KNOWN_PROVIDERS = ("byteplus", "openrouter", "google_genai")
MODELARK_PREFIXES = ("seed", "dola-", "doubao", "deepseek", "kimi", "glm", "skylark", "gpt-oss", "ep-")


def _split(spec: str) -> tuple[Optional[str], str]:
    """``provider/model`` → (provider, model) for known providers; otherwise (None, spec) — e.g. OpenRouter ids like google/gemini-x."""
    if spec and "/" in spec:
        provider, model = spec.split("/", 1)
        if provider in KNOWN_PROVIDERS:
            return provider, model
    return None, spec


def _byteplus(model: str, image_model: Optional[str] = None, effort: Optional[str] = None) -> BytePlusEngine:
    config = get_config()
    provider = config.get_provider("byteplus")
    if provider is None or not provider.enabled:
        raise RuntimeError("Provider 'byteplus' is not enabled in config/api_config.yaml")
    return BytePlusEngine(api_key=provider.api_key, base_url=provider.base_url or DEFAULT_BASE_URL,
                          model=model, image_model=image_model or DEFAULT_IMAGE_MODEL, reasoning_effort=effort)


def _options(spec: str) -> tuple[str, dict]:
    """``provider/model?effort=low`` → (``provider/model``, {"effort": "low"})."""
    base, _, query = str(spec or "").partition("?")
    options = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
    return base, {k.strip(): v.strip() for k, v in options.items()}


def _engine_for(spec: str, *, image: bool = False) -> BaseEngine:
    spec, options = _options(spec)
    provider, model = _split(spec)
    config = get_config()
    inferred = provider is None
    if provider is None:
        # legacy spec without a known provider prefix: route to whichever provider is enabled
        def enabled(name: str) -> bool:
            pc = config.get_provider(name)
            return bool(pc and pc.enabled)
        if enabled("openrouter"):
            return create_engine("openrouter", model=model)
        if enabled("byteplus"):
            provider = "byteplus"
        elif enabled("google_genai"):
            return create_engine("google_genai", model=model.split("/")[-1])
        else:
            raise RuntimeError(f"No enabled provider can serve model '{spec}'")
    if provider == "byteplus":
        if "/" in model or (inferred and not model.startswith(MODELARK_PREFIXES)):
            # e.g. a legacy google/gemini-x or nano-banana-pro id that ModelArk cannot serve
            fallback = DEFAULT_IMAGE_MODEL if image else DEFAULT_TEXT_MODEL
            print(f"[engine_select] WARNING: '{spec}' is not a ModelArk model id; using {fallback}")
            model = fallback
        return _byteplus(DEFAULT_TEXT_MODEL if image else model, image_model=model if image else None,
                         effort=None if image else options.get("effort"))
    return create_engine(provider, model=model)


def build_engines(overrides: Optional[dict] = None) -> Engines:
    """The engines of ``core_engine``; ``overrides`` replaces some of its specs ({"text_model": "byteplus/..."}) for one
    run (tools/dry_run.py compares writers this way)."""
    config = get_config()
    raw = {**(config.get_raw_config().get("core_engine", {}) or {}), **{k: v for k, v in (overrides or {}).items() if v}}
    text_spec = raw.get("text_model") or raw.get("default_model") or f"byteplus/{DEFAULT_TEXT_MODEL}"
    fast_spec = raw.get("fast_text_model") or text_spec
    vision_spec = raw.get("vision_model") or text_spec
    image_spec = raw.get("image_model") or f"byteplus/{DEFAULT_IMAGE_MODEL}"
    check_spec = raw.get("check_model") or text_spec
    translate_spec = raw.get("translate_model") or text_spec

    cache: dict[str, BaseEngine] = {}

    def get(spec: str, image: bool = False) -> BaseEngine:
        key = ("img:" if image else "txt:") + spec
        if key not in cache:
            cache[key] = _engine_for(spec, image=image)
        return cache[key]

    def role(spec: str) -> BaseEngine:
        """The checker's or translator's engine: the writer's own when it is the same model, else with the writer behind."""
        if spec == text_spec:
            return get(text_spec)
        key = "role:" + spec
        if key not in cache:
            cache[key] = FallbackEngine(get(spec), get(text_spec))
        return cache[key]

    return Engines(text=get(text_spec), fast_text=get(fast_spec), vision=get(vision_spec), image=get(image_spec, image=True),
                   check=role(check_spec), translate=role(translate_spec))


_ANSWERING: set[str] = set()   # models that answered a preflight in this process


async def preflight(engines: Engines) -> None:
    """One 1-token call to every ModelArk text and vision model not yet seen answering in this process, before a job is
    created: a model that is not activated, or an account without balance, surfaces before the script is paid for.
    Raises the engine's AccountError; any other error passes (a job retries a flaky network on its own). The image
    model cannot be probed for free: a job stops at its first account error instead."""
    for engine in {id(e): e for e in (engines.text, engines.fast_text, engines.vision, engines.check, engines.translate)}.values():
        model = getattr(engine, "model", "")
        if not isinstance(engine, BytePlusEngine) or model in _ANSWERING:
            continue
        config = GenerationConfig(temperature=0.0, max_tokens=1)
        config.thinking = False
        try:
            await engine.generate_text([Message.user("Reply OK.")], config)
            _ANSWERING.add(model)
        except Exception as e:  # noqa: BLE001
            if getattr(e, "account", False):
                raise
