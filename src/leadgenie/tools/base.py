"""Tool plumbing shared by every tool: the result type, the registry, and the evidence log.

Every tool returns a ToolResult. Besides the text the model sees, a result carries
`documents`: the URL → text the tool actually retrieved. The registry accumulates these
per lead into an EvidenceLog, which validate.py later uses to check that every cited
source URL was really retrieved and that the quoted evidence really appears in it.
That is what makes "every field carries a source URL" verifiable rather than decorative.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

ToolHandler = Callable[[dict[str, Any]], Awaitable["ToolResult"]]


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    documents: dict[str, str] = field(default_factory=dict)

    @classmethod
    def json(cls, payload: Any, documents: dict[str, str] | None = None) -> ToolResult:
        return cls(json.dumps(payload, ensure_ascii=False), documents=documents or {})

    @classmethod
    def error(cls, message: str) -> ToolResult:
        return cls(message, is_error=True)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def to_param(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": True,
        }


def _json_unescape(text: str) -> str:
    try:
        value = json.loads(f'"{text}"')
    except json.JSONDecodeError:
        return text
    return value if isinstance(value, str) else text


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class EvidenceLog:
    """URL → retrieved text for one lead's run."""

    documents: dict[str, str] = field(default_factory=dict)

    def add(self, documents: dict[str, str]) -> None:
        for url, text in documents.items():
            # Keep every retrieval of the same URL (e.g. a search snippet and a full fetch).
            self.documents[url] = (self.documents.get(url, "") + "\n" + text).strip()

    def has_url(self, url: str) -> bool:
        return url in self.documents

    def supports(self, url: str, quote: str) -> bool:
        """True if `quote` (whitespace/case-normalized) appears in the text retrieved from url.

        Tool results reach the model as JSON, so a faithful quote may carry JSON string
        escapes (\\" or \\u2014); the unescaped form is accepted too.
        """
        if not quote.strip() or url not in self.documents:
            return False
        doc = _normalize(self.documents[url])
        return any(_normalize(q) in doc for q in (quote, _json_unescape(quote)))


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def params(self) -> list[dict[str, Any]]:
        # Deterministic order keeps the tools block byte-identical → prompt-cache friendly.
        return [self._tools[n].to_param() for n in sorted(self._tools)]

    async def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error(f"unknown tool {name!r}; available: {sorted(self._tools)}")
        try:
            return await tool.handler(args)
        except Exception as err:  # a tool failure is data for the agent, not a crash
            return ToolResult.error(f"{name} failed: {type(err).__name__}: {err}")
