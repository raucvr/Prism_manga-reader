"""
BytePlus ModelArk Engine
- Text / vision: OpenAI-compatible ``/chat/completions`` (Seed 2.x and DeepSeek V4 models, image inputs as data URIs,
  ``thinking`` on/off, ``response_format`` incl. json_schema on models that support it); long calls (thinking, or a large
  ``max_tokens``) are streamed, so a slow model that is still writing is never mistaken for a dead connection
- Images: ``/images/generations`` with Seedream 5.0 Pro (one image per call, 2–10 reference images,
  explicit ``WxH`` sizes, no negative prompt, no temperature)

Docs: https://ai.byteplus.com/ark/region:ap-southeast-1/docs/image-generation-api.md
"""

import asyncio
import json
import random
import ssl
from typing import Optional, AsyncIterator

import httpx

from .base import (
    BaseEngine,
    Message,
    ImageContent,
    GenerationConfig,
    ImageGenerationConfig,
    TextResponse,
    ImageResponse,
)


DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
DEFAULT_TEXT_MODEL = "seed-2-0-pro-260328"
DEFAULT_IMAGE_MODEL = "dola-seedream-5-0-pro-260628"

# Seedream 5.0 Pro / Flash explicit-size limits (image generation scenario)
SEEDREAM_MIN_PIXELS = 1280 * 720            # 921,600
SEEDREAM_MAX_PIXELS = 4_624_220             # 2048x2048x1.1025
SEEDREAM_MAX_REFERENCE_IMAGES = 10

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
STREAM_FROM_TOKENS = 8000   # replies allowed this long (or with thinking) are streamed: DeepSeek V4 Pro thinks for minutes


class BytePlusAPIError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"ModelArk HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body
        self.code, self.detail = _error_code(body)


def _error_code(body: str) -> tuple[str, str]:
    """(code, message) of a ModelArk error body ({"error": {"code": "ModelNotOpen", "message": ...}}); ("", "") if none."""
    try:
        err = json.loads(body).get("error") or {}
    except (ValueError, AttributeError):
        return "", ""
    return (str(err.get("code") or ""), str(err.get("message") or "")) if isinstance(err, dict) else ("", "")


# what the user must do, per ModelArk error code (https://ai.byteplus.com/ark/region:ap-southeast-1/docs/error-code.md)
ACCOUNT_HINTS = {
    "AuthenticationError": ("The ModelArk API key was refused: check providers.byteplus.api_key in config/api_config.yaml "
                            "(or ARK_API_KEY) and restart the backend.",
                            "ModelArk 拒绝了 API key：请检查 config/api_config.yaml 中的 providers.byteplus.api_key（或环境变量 ARK_API_KEY），然后重启后端。"),
    "ModelNotOpen": ("The model {model} is not activated for this account: activate it in the ModelArk console "
                     "(Activation management), then try again.",
                     "模型 {model} 尚未开通：请在 ModelArk 控制台的开通管理中开通后重试。"),
    "InvalidEndpointOrModel": ("The model {model} does not exist for this account or region: check core_engine in config/api_config.yaml.",
                               "当前账号或地域没有模型 {model}：请检查 config/api_config.yaml 中的 core_engine。"),
    "AccountOverdueError": ("The BytePlus account is overdue: top up in the BytePlus Billing Center, then try again.",
                            "BytePlus 账户已欠费：请在费用中心充值后重试。"),
    "QuotaExceeded": ("The free quota of {model} is used up: activate its paid service in the ModelArk console.",
                      "模型 {model} 的免费额度已用完：请在 ModelArk 控制台开通付费服务。"),
    "SetLimitExceeded": ("A limit set on the account stopped the request (Safe Experience Mode or an inference limit): raise or "
                         "turn it off in the ModelArk console.",
                         "账户上设置的限额（安全体验模式或推理限额）拦下了请求：请在 ModelArk 控制台调高或关闭。"),
    "AccessDenied": ("This API key may not use {model}: check the key's permissions in the ModelArk console.",
                     "当前 API key 无权使用 {model}：请在 ModelArk 控制台检查权限。"),
    "OperationDenied": ("ModelArk refused the request for {model} (the service is not enabled or not allowed for this account): "
                        "check the account in the ModelArk console.",
                        "ModelArk 拒绝了对 {model} 的请求（该服务未开通或账号不允许）：请在 ModelArk 控制台检查账号。"),
}
RATE_CODES = ("RateLimitExceeded", "ServerOverloaded", "RequestBurstTooFast", "ModelAccountTpmRateLimitExceeded",
              "ModelAccountRpmRateLimitExceeded")


class AccountError(BytePlusAPIError):
    """The account cannot serve the request (a refused key, a model not activated, an overdue account, a used-up quota, a
    spending limit): waiting and retrying never help, so it is raised at once and stops the job, with what to do."""

    account = True   # best-effort steps (fact-check, quiz, translation fixes) re-raise errors that carry this flag

    def __init__(self, status: int, body: str, model: str = ""):
        super().__init__(status, body)
        self.model = model
        key = next((k for k in ACCOUNT_HINTS if self.code.startswith(k)), "AuthenticationError" if status == 401 else "")
        en, zh = ACCOUNT_HINTS.get(key, ("ModelArk refused the request: {detail}", "ModelArk 拒绝了请求：{detail}"))
        fill = {"model": model or "the model", "detail": self.detail or body[:200]}
        self.hint = f"{en.format(**fill)} / {zh.format(**fill)}"

    def __str__(self) -> str:
        return f"{self.hint} (ModelArk {self.code or self.status})"


class ContentFilteredError(BytePlusAPIError):
    """The provider's content filter refused the request and answered nothing (HTTP 200, finish_reason
    "content_filter", 0 tokens): DeepSeek V4 Pro does this for some ordinary paper text (CoT's CommonsenseQA examples)
    that Seed and DeepSeek V4 Flash read fine. Never an empty answer to repair: the caller retries on another model."""

    def __init__(self, model: str):
        super().__init__(200, json.dumps({"error": {"code": "ContentFilter", "message": f"{model} refused the request (content filter)"}}))
        self.model = model


def _api_error(status: int, body: str, model: str = "") -> BytePlusAPIError:
    """The error for a ModelArk reply: an AccountError when the account must change something first."""
    code, _detail = _error_code(body)
    account = (status == 401 or any(code.startswith(k) for k in ACCOUNT_HINTS)) and not any(code.startswith(r) for r in RATE_CODES)
    return AccountError(status, body, model) if account else BytePlusAPIError(status, body)


def normalize_size(width: int, height: int) -> str:
    """Fit a requested WxH into Seedream's limits: total pixels in [921,600, 4,624,220], aspect ratio in [1/16, 16],
    both sides multiples of 16. Sizes that already satisfy the limits are kept unchanged (e.g. 1024x1536, 2048x1024)."""
    width, height = max(int(width), 16), max(int(height), 16)
    ratio = min(max(width / height, 1 / 16), 16)
    pixels = width * height
    if not (SEEDREAM_MIN_PIXELS <= pixels <= SEEDREAM_MAX_PIXELS) or ratio != width / height:
        target = min(max(pixels, SEEDREAM_MIN_PIXELS * 1.01), SEEDREAM_MAX_PIXELS * 0.99)
        height = (target / ratio) ** 0.5
        width = height * ratio
    w = max(16, int(round(width / 16)) * 16)
    h = max(16, int(round(height / 16)) * 16)
    for _ in range(512):            # nudge in 16 px steps until every limit holds
        p = w * h
        if p > SEEDREAM_MAX_PIXELS:
            if w >= h:
                w -= 16
            else:
                h -= 16
        elif p < SEEDREAM_MIN_PIXELS:
            if w <= h:
                w += 16
            else:
                h += 16
        elif w > 16 * h:
            h += 16
        elif h > 16 * w:
            w += 16
        else:
            break
    return f"{w}x{h}"


class BytePlusEngine(BaseEngine):
    """ModelArk engine: Seed 2.x for text/vision, Seedream 5.0 Pro for images."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_TEXT_MODEL,
        image_model: str = DEFAULT_IMAGE_MODEL,
        thinking: bool = False,
        max_retries: int = 6,
        reasoning_effort: Optional[str] = None,
    ):
        if not api_key or api_key.strip() == "" or api_key.startswith("${"):
            raise ValueError(
                "The BytePlus ModelArk API key is not set: put it in config/api_config.yaml (providers.byteplus.api_key) "
                "or the ARK_API_KEY environment variable, then restart the backend. / BytePlus ModelArk API key 未设置："
                "请在 config/api_config.yaml 的 providers.byteplus.api_key 中设置，或设置环境变量 ARK_API_KEY，然后重启后端。"
            )
        super().__init__(api_key, base_url or DEFAULT_BASE_URL, model)
        self.image_model = image_model or DEFAULT_IMAGE_MODEL
        self.default_thinking = thinking
        self.reasoning_effort = reasoning_effort   # how long a thinking call may reason (DeepSeek V4 Pro: low | high, default high)
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self.tokens: dict[str, dict[str, int]] = {}   # model → prompt / completion / cached tokens spent through this engine

    # ------------------------------------------------------------------ meta
    @property
    def name(self) -> str:
        return "BytePlus ModelArk"

    @property
    def supports_image_generation(self) -> bool:
        return bool(self.image_model)

    @property
    def supports_vision(self) -> bool:
        return True

    # --------------------------------------------------------------- client
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(600.0, connect=20.0, read=420.0),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
        return self._client

    async def _post(self, path: str, payload: dict) -> dict:
        """POST with exponential backoff on transient errors (429 / 5xx / network)."""
        client = await self._get_client()
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(url, json=payload)
                if response.status_code < 400:
                    return response.json()
                err = _api_error(response.status_code, response.text, str(payload.get("model") or ""))
                if isinstance(err, AccountError) or response.status_code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    raise err
                last_error = err
            except (httpx.TimeoutException, httpx.TransportError, ssl.SSLError) as e:   # "record layer failure" mid-read too
                if attempt == self.max_retries:
                    raise RuntimeError(f"ModelArk {path} unreachable after {attempt + 1} attempts: {type(e).__name__} {e}") from e
                last_error = f"{type(e).__name__} {e}".strip()
            delay = min(3 * 2 ** attempt + random.uniform(0, 2), 45)
            print(f"[BytePlus] {path} attempt {attempt + 1} failed ({last_error}); retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
        raise RuntimeError(f"unreachable: {last_error}")

    async def _post_stream(self, path: str, payload: dict) -> dict:
        """``_post`` with "stream": true, assembled into the non-streaming reply shape (content, finish_reason, usage).
        The read timeout then counts the gap between chunks, not the whole reply: DeepSeek V4 Pro can think for longer
        than any sane whole-reply timeout. Transient errors, including a stream that breaks off, are retried from the
        start with the same backoff as ``_post``."""
        client = await self._get_client()
        url = f"{self.base_url}{path}"
        body = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                async with client.stream("POST", url, json=body) as response:
                    if response.status_code < 400:
                        return await self._assemble(response, payload.get("model", self.model))
                    err = _api_error(response.status_code, (await response.aread()).decode(errors="replace"), str(payload.get("model") or ""))
                if isinstance(err, AccountError) or response.status_code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    raise err
                last_error = err
            except (httpx.TimeoutException, httpx.TransportError, ssl.SSLError) as e:   # "record layer failure" mid-read too
                if attempt == self.max_retries:
                    raise RuntimeError(f"ModelArk {path} unreachable after {attempt + 1} attempts: {type(e).__name__} {e}") from e
                last_error = f"{type(e).__name__} {e}".strip()
            delay = min(3 * 2 ** attempt + random.uniform(0, 2), 45)
            print(f"[BytePlus] {path} (stream) attempt {attempt + 1} failed ({last_error}); retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
        raise RuntimeError(f"unreachable: {last_error}")

    @staticmethod
    async def _assemble(response: httpx.Response, model: str) -> dict:
        """The chunks of a streamed chat completion as one reply: the answer (not the reasoning), the finish reason and
        the usage the last chunk carries."""
        content: list[str] = []
        finish, usage = "", {}
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            chunk = line[6:].strip()
            if chunk == "[DONE]":
                break
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if data.get("error"):
                raise BytePlusAPIError(500, json.dumps(data["error"])[:500])
            model = data.get("model") or model
            usage = data.get("usage") or usage
            for choice in data.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                finish = choice.get("finish_reason") or finish
        return {"model": model, "choices": [{"message": {"content": "".join(content)}, "finish_reason": finish}], "usage": usage}

    # ----------------------------------------------------------------- text
    @staticmethod
    def _image_part(img: ImageContent) -> dict:
        if img.is_url:
            return {"type": "image_url", "image_url": {"url": img.data}}
        return {"type": "image_url", "image_url": {"url": f"data:{img.mime_type};base64,{img.data}"}}

    def _build_messages(self, messages: list[Message]) -> list[dict]:
        result = []
        for msg in messages:
            if msg.images:
                content = [self._image_part(img) for img in msg.images]
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                result.append({"role": msg.role.value, "content": content})
            else:
                result.append({"role": msg.role.value, "content": msg.content})
        return result

    def _text_payload(self, messages: list[Message], config: Optional[GenerationConfig]) -> dict:
        config = config or GenerationConfig()
        thinking = getattr(config, "thinking", None)
        if thinking is None:
            thinking = self.default_thinking
        payload = {
            "model": getattr(config, "model", None) or self.model,
            "messages": self._build_messages(messages),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
        effort = getattr(config, "reasoning_effort", None) or self.reasoning_effort
        if thinking and effort:
            payload["reasoning_effort"] = effort
        if getattr(config, "prompt_cache_key", None):
            payload["prompt_cache_key"] = config.prompt_cache_key
        if config.stop_sequences:
            payload["stop"] = config.stop_sequences
        response_format = getattr(config, "response_format", None)
        if response_format:
            payload["response_format"] = response_format
        return payload

    async def generate_text(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None,
    ) -> TextResponse:
        payload = self._text_payload(messages, config)
        long = payload["thinking"]["type"] == "enabled" or int(payload.get("max_tokens") or 0) >= STREAM_FROM_TOKENS
        data = await (self._post_stream if long else self._post)("/chat/completions", payload)
        choice = data["choices"][0]
        message = choice.get("message", {})
        usage = data.get("usage", {}) or {}
        counted = {
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
        }
        self._count(payload["model"], counted)
        if choice.get("finish_reason") == "content_filter" and not (message.get("content") or "").strip():
            raise ContentFilteredError(payload["model"])
        return TextResponse(
            content=message.get("content") or "",
            model=data.get("model", payload["model"]),
            usage=counted,
            finish_reason=choice.get("finish_reason", ""),
        )

    def _count(self, model: str, usage: dict) -> None:
        mine = self.tokens.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0})
        for key in mine:
            mine[key] += int(usage.get(key, 0) or 0)

    async def generate_text_stream(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        payload = self._text_payload(messages, config)
        payload["stream"] = True
        client = await self._get_client()
        async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise BytePlusAPIError(response.status_code, body.decode(errors="replace"))
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                text = delta.get("content")
                if text:
                    yield text

    # ---------------------------------------------------------------- image
    async def generate_image(
        self,
        prompt: str,
        config: Optional[ImageGenerationConfig] = None,
        reference_images: list[ImageContent] = None,
    ) -> ImageResponse:
        """Generate exactly one image with Seedream.

        ``config.width/height`` are clamped into Seedream's pixel range. Optional per-call knobs are read
        from ``config.extra`` (dict): ``fast`` (bool), ``output_format`` ("png"|"jpeg"), ``watermark`` (bool),
        ``size`` (explicit override such as "2K").
        """
        config = config or ImageGenerationConfig()
        extra = getattr(config, "extra", None) or {}

        fmt = str(extra.get("output_format") or "jpeg").lower().strip(".")
        payload = {
            "model": self.image_model,
            "prompt": prompt,
            "size": extra.get("size") or normalize_size(config.width, config.height),
            "response_format": "b64_json",
            "watermark": bool(extra.get("watermark", False)),
            "output_format": "png" if fmt == "png" else "jpeg",
        }
        if extra.get("fast"):
            payload["optimize_prompt_options"] = {"mode": "fast"}

        refs = []
        for img in (reference_images or [])[:SEEDREAM_MAX_REFERENCE_IMAGES]:
            if img.is_url:
                refs.append(img.data)
            else:
                refs.append(f"data:{img.mime_type};base64,{img.data}")
        if refs:
            payload["image"] = refs if len(refs) > 1 else refs[0]

        data = await self._post("/images/generations", payload)
        if data.get("error"):
            raise RuntimeError(f"Seedream error: {data['error']}")

        images = []
        for item in data.get("data", []):
            fmt = item.get("output_format") or payload["output_format"]
            mime = "image/png" if fmt == "png" else "image/jpeg"
            if item.get("b64_json"):
                images.append(ImageContent.from_base64(item["b64_json"], mime))
            elif item.get("url"):
                images.append(ImageContent.from_url(item["url"], mime))
        if not images:
            raise ValueError(f"Seedream returned no image: {str(data)[:300]}")

        response = ImageResponse(images=images, model=data.get("model", self.image_model), prompt=prompt)
        # expose usage for cost accounting without changing the dataclass contract
        setattr(response, "usage", data.get("usage", {}))
        setattr(response, "size", data["data"][0].get("size"))
        return response

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
