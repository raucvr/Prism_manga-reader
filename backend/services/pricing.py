"""
Prices of the ModelArk models Prism uses, for cost estimates in job reports and dry runs.

USD per million tokens (input, cached input, output) on the standard tier, BytePlus ModelArk, Oct 2026
(https://ai.byteplus.com/ark/region:ap-southeast-1/docs/model-pricing.md). Seed 2.0 Pro and Lite cost twice as much for
prompts over 128K tokens; Prism's prompts stay below that (the paper text is capped at 120k characters), so the lower
tier is used. Seedream: per image.
"""

from __future__ import annotations

TEXT_PRICES = {               # model id prefix → (input, cached input, output) per million tokens
    "seed-2-0-pro": (0.50, 0.10, 3.00),
    "seed-2-0-lite": (0.25, 0.05, 2.00),
    "seed-2-0-mini": (0.10, 0.02, 0.40),
    "dola-seed-2-1-turbo": (0.50, 0.10, 2.50),
    "deepseek-v4-pro": (1.32, 0.044, 3.96),
    "deepseek-v4-flash": (0.44, 0.014, 1.32),
    "deepseek-v4-1-flash": (0.30, 0.006, 1.20),   # peak hours (off-peak is half)
    "glm-5-2": (1.40, 0.26, 4.40),
}
DEFAULT_TEXT_PRICE = TEXT_PRICES["seed-2-0-pro"]
IMAGE_SMALL = 0.045           # Seedream 5.0 Pro, one output image ≤ 2.61 MP
IMAGE_LARGE = 0.09
IMAGE_REF = 0.003             # each reference image from the second one


def text_price(model: str) -> tuple[float, float, float]:
    """The (input, cached input, output) price of a model id ("deepseek-v4-pro-ga-260813"); Seed 2.0 Pro's when unknown."""
    model = str(model or "").split("/")[-1]
    best = max((p for p in TEXT_PRICES if model.startswith(p)), key=len, default=None)
    return TEXT_PRICES[best] if best else DEFAULT_TEXT_PRICE


def text_usd(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
    """Cost of a model's tokens; cached prompt tokens (a prefix the provider already holds) at the cached price."""
    price_in, price_cached, price_out = text_price(model)
    cached = min(int(cached_tokens or 0), int(prompt_tokens or 0))
    return ((int(prompt_tokens or 0) - cached) * price_in + cached * price_cached + int(completion_tokens or 0) * price_out) / 1_000_000


def usage_usd(by_model: dict) -> float:
    """Total cost of {model: {"prompt_tokens", "completion_tokens", "cached_tokens"}}."""
    return sum(text_usd(m, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("cached_tokens", 0))
               for m, u in (by_model or {}).items())
