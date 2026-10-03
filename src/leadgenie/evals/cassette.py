"""Record/replay of model calls, so evals can be re-run without an API key or spend.

Responses are keyed by a hash of the full request (model, system prompt, tools, every
message). In the offline fixture world, tool results are deterministic, so a replayed run
re-issues byte-identical requests and gets the recorded responses back. Any change to
the prompt, tools or agent logic changes the requests and the replay fails loudly with
CassetteMiss: re-record with a real key instead of silently scoring stale behaviour.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from anthropic.types import Message
from anthropic.types.beta import BetaMessage

Mode = Literal["record", "replay"]


class CassetteMiss(KeyError):
    pass


def _canonical(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _canonical(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def request_key(namespace: str, kwargs: dict[str, Any]) -> str:
    blob = json.dumps({"ns": namespace, **_canonical(kwargs)}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class _Messages:
    def __init__(self, cassette: Cassette, namespace: str, inner: Any) -> None:
        self._cassette = cassette
        self._ns = namespace
        self._inner = inner

    async def create(self, **kwargs: Any) -> Any:
        return await self._cassette.call(self._ns, self._inner, kwargs)


class Cassette:
    """Wraps an AsyncAnthropic-like client (record) or stands in for one (replay)."""

    def __init__(self, path: str | Path, mode: Mode, inner: Any = None) -> None:
        if mode == "record" and inner is None:
            raise ValueError("record mode needs a real client to wrap")
        self.path = Path(path)
        self.mode = mode
        self.hits = 0
        self.recorded = 0
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                entry = json.loads(line)
                self._entries[entry["key"]] = entry
        self.messages = _Messages(self, "messages", inner.messages if inner else None)
        self.beta = SimpleNamespace(
            messages=_Messages(self, "beta", inner.beta.messages if inner else None)
        )

    async def call(self, namespace: str, inner: Any, kwargs: dict[str, Any]) -> Any:
        key = request_key(namespace, kwargs)
        model_cls = BetaMessage if namespace == "beta" else Message
        if key in self._entries:
            self.hits += 1
            return model_cls.model_validate(self._entries[key]["response"])
        if self.mode == "replay":
            raise CassetteMiss(
                f"no recorded response for this request ({namespace}, model="
                f"{kwargs.get('model')}); the prompt, tools or agent changed since recording"
            )
        response = await inner.create(**kwargs)
        entry = {"key": key, "ns": namespace, "response": response.model_dump(mode="json")}
        self._entries[key] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.recorded += 1
        return model_cls.model_validate(entry["response"])
