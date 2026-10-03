"""Agent tools. Each tool has an offline fixture backend (evals, CI) and a live backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from leadgenie.store import Store
from leadgenie.tools.base import EvidenceLog, Tool, ToolRegistry, ToolResult
from leadgenie.tools.dns import FixtureMX, LiveMX, MXResolver, mx_tool
from leadgenie.tools.prior import prior_tool
from leadgenie.tools.search import FixtureSearch, SearchProvider, UnconfiguredSearch, search_tool
from leadgenie.tools.web import Fetcher, FixtureFetcher, LiveFetcher, fetch_tool

__all__ = [
    "Backends",
    "EvidenceLog",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "World",
    "build_registry",
    "fixture_backends",
    "live_backends",
]


@dataclass
class World:
    """The offline web used by evals: pages by URL and MX records by domain."""

    pages: dict[str, dict[str, Any]] = field(default_factory=dict)
    mx: dict[str, list[str] | None] = field(default_factory=dict)


@dataclass
class Backends:
    fetcher: Fetcher
    search: SearchProvider
    mx: MXResolver


def fixture_backends(world: World) -> Backends:
    return Backends(
        fetcher=FixtureFetcher(world.pages),
        search=FixtureSearch(world.pages),
        mx=FixtureMX(world.mx),
    )


def live_backends() -> Backends:
    return Backends(fetcher=LiveFetcher(), search=UnconfiguredSearch(), mx=LiveMX())


def build_registry(backends: Backends, store: Store) -> ToolRegistry:
    return ToolRegistry(
        [
            fetch_tool(backends.fetcher),
            search_tool(backends.search),
            mx_tool(backends.mx),
            prior_tool(store),
        ]
    )
