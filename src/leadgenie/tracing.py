"""Tracing: one span per lead, per model call and per tool call.

A span records inputs, outputs, tokens, cost, latency and errors. Spans nest
(lead → model_call / tool_call) via parent_id. They go to SQLite through the Store,
and can be exported to JSONL. The per-run report (report.py) is computed from spans,
so it reflects what actually happened rather than what the code meant to log.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from leadgenie.pricing import TokenUsage

SpanKind = Literal["lead", "model_call", "tool_call"]
MAX_PAYLOAD_CHARS = 4_000


def _clip(value: Any) -> Any:
    """Keep traces bounded: long strings are truncated, with the original length noted."""
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    if len(text) <= MAX_PAYLOAD_CHARS:
        return value
    return f"{text[:MAX_PAYLOAD_CHARS]}…[truncated, {len(text)} chars]"


@dataclass
class Span:
    run_id: str
    kind: SpanKind
    name: str
    lead_id: str | None = None
    parent_id: str | None = None
    step: int | None = None
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    latency_ms: int | None = None
    input: Any = None
    output: Any = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def set_usage(self, model: str, usage: TokenUsage, cost: float) -> None:
        self.model = model
        self.input_tokens = usage.input_tokens
        self.output_tokens = usage.output_tokens
        self.cache_read_tokens = usage.cache_read_input_tokens
        self.cache_write_tokens = usage.cache_creation_input_tokens
        self.cost_usd = cost

    def fail(self, error: str) -> None:
        self.status = "error"
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["input"] = _clip(d["input"])
        d["output"] = _clip(d["output"])
        return d


class SpanSink(Protocol):
    def write_span(self, span: dict[str, Any]) -> None: ...


class MemorySink:
    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []

    def write_span(self, span: dict[str, Any]) -> None:
        self.spans.append(span)


class Tracer:
    def __init__(self, run_id: str, sinks: list[SpanSink]) -> None:
        self.run_id = run_id
        self.sinks = sinks

    @contextmanager
    def span(
        self,
        kind: SpanKind,
        name: str,
        *,
        lead_id: str | None = None,
        parent: Span | None = None,
        step: int | None = None,
        input: Any = None,
    ) -> Iterator[Span]:
        span = Span(
            run_id=self.run_id,
            kind=kind,
            name=name,
            lead_id=lead_id or (parent.lead_id if parent else None),
            parent_id=parent.span_id if parent else None,
            step=step,
            input=input,
        )
        start = time.monotonic()
        try:
            yield span
        except BaseException as err:
            span.fail(f"{type(err).__name__}: {err}")
            raise
        finally:
            span.latency_ms = int((time.monotonic() - start) * 1000)
            record = span.to_dict()
            for sink in self.sinks:
                sink.write_span(record)


class JsonlSink:
    def __init__(self, path: str) -> None:
        self.path = path

    def write_span(self, span: dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(span, default=str, ensure_ascii=False) + "\n")
