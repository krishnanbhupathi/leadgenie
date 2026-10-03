"""A scripted stand-in for anthropic.AsyncAnthropic. Responses are real SDK BetaMessage
objects, so the agent code under test handles exactly the types it gets in production."""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from anthropic.types.beta import BetaMessage

_ids = itertools.count(1)


def tool_use(name: str, **input: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": f"toolu_{next(_ids):04d}", "name": name, "input": input}


def text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


def message(
    *blocks: dict[str, Any],
    stop_reason: str | None = None,
    input_tokens: int = 1000,
    output_tokens: int = 100,
    model: str = "claude-opus-5-5",
) -> BetaMessage:
    if stop_reason is None:
        stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    return BetaMessage.model_validate(
        {
            "id": f"msg_{next(_ids):04d}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": list(blocks),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
    )


Script = Callable[[dict[str, Any]], BetaMessage]


class FakeMessages:
    def __init__(self, script: Script | list[BetaMessage | Exception]) -> None:
        self.calls: list[dict[str, Any]] = []
        if isinstance(script, list):
            queue = list(script)

            def pop(_: dict[str, Any]) -> BetaMessage:
                item = queue.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

            self._script: Script = pop
        else:
            self._script = script

    async def create(self, **kwargs: Any) -> BetaMessage:
        # Snapshot: the agent appends to its message list after the call returns.
        self.calls.append({**kwargs, "messages": copy.copy(kwargs["messages"])})
        return self._script(kwargs)


class FakeAnthropic:
    def __init__(self, script: Script | list[BetaMessage | Exception]) -> None:
        self.messages = FakeMessages(script)
        self.beta = SimpleNamespace(messages=self.messages)


def sourced(value: Any, source: str = "inferred", evidence: str = "", confidence: float = 0.5):
    return {"value": value, "source": source, "evidence": evidence, "confidence": confidence}


def submission(**overrides: Any) -> dict[str, Any]:
    """A valid, well-sourced submission for the Acme fixture in tests."""
    url = "https://acme.example/"
    payload = {
        "company": sourced("Acme Robotics", url, "Acme Robotics", 0.95),
        "domain": sourced("acme.example", url, "Acme Robotics", 0.9),
        "role": sourced("CTO", "input", "", 0.9),
        "seniority": sourced("c_level", "inferred", "", 0.8),
        "industry": sourced("logistics", url, "autonomous forklifts", 0.85),
        "accepts_email": sourced(True, "dns:mx:acme.example", "mx1.acme.example", 0.95),
        "outreach": {
            "text": "Ada, autonomous forklifts for warehouses is a sharp bet. How is Acme "
            "handling mixed human-robot aisles?",
            "source": url,
            "evidence": "autonomous forklifts",
        },
    }
    payload.update(overrides)
    return payload
