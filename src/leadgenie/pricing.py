"""Token pricing. Cost is computed from the API's usage block, including prompt-cache tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens, Anthropic first-party list prices."""

    input: float
    output: float
    cache_read: float

    @property
    def cache_write(self) -> float:
        return self.input * 1.25  # 5-minute cache writes bill at 1.25x base input


PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(4.00, 20.00, 0.20),
    "claude-opus-4-8": Price(5.00, 25.00, 0.50),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.20),
    "claude-sonnet-5": Price(2.00, 10.00, 0.20),
    "claude-haiku-4-5": Price(1.00, 5.00, 0.10),
}


def price_for(model: str) -> Price:
    # An unknown model must fail loudly: silently pricing it at $0 would make
    # every cost budget and cost metric downstream meaningless.
    if model not in PRICES:
        raise KeyError(f"no price for model {model!r}; add it to PRICES")
    return PRICES[model]


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_api(cls, usage: Any) -> TokenUsage:
        def get(name: str) -> int:
            return int(getattr(usage, name, 0) or 0)

        return cls(
            get("input_tokens"),
            get("output_tokens"),
            get("cache_read_input_tokens"),
            get("cache_creation_input_tokens"),
        )

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
        )


def cost_usd(model: str, usage: TokenUsage) -> float:
    p = price_for(model)
    return (
        usage.input_tokens * p.input
        + usage.output_tokens * p.output
        + usage.cache_read_input_tokens * p.cache_read
        + usage.cache_creation_input_tokens * p.cache_write
    ) / 1_000_000
