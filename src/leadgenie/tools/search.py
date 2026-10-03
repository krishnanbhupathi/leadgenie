"""web_search: find candidate pages for a company.

The provider is pluggable. The offline FixtureSearch ranks the eval world's pages by
token overlap, which is enough to make the agent choose between look-alike companies.
No live provider is wired up yet: every option is a paid service or needs a new API
key, and that is the repo owner's call. Until then the tool returns a clear error so
the agent falls back to fetching a guessed domain.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from leadgenie.tools.base import Tool, ToolResult

MAX_RESULTS = 5
SNIPPET_CHARS = 240


class SearchProvider(Protocol):
    async def search(self, query: str) -> list[dict[str, str]]:
        """Returns [{"url", "title", "snippet"}, ...], best first."""
        ...


class SearchUnavailable(Exception):
    pass


class UnconfiguredSearch:
    async def search(self, query: str) -> list[dict[str, str]]:
        raise SearchUnavailable(
            "web search is not configured in this deployment; fetch the company's site "
            "directly (guess the domain from the company name) instead"
        )


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}


class FixtureSearch:
    def __init__(self, pages: dict[str, dict[str, str]]) -> None:
        self._pages = pages

    async def search(self, query: str) -> list[dict[str, str]]:
        q = _tokens(query)
        if not q:
            return []
        scored = []
        for url, page in self._pages.items():
            if page.get("unlisted"):  # reachable by fetch but not indexed (like a real /team page)
                continue
            title = page.get("title", "")
            body = f"{page.get('description', '')} {page.get('text', '')}"
            # Title and URL matches count double: that is roughly how real engines rank.
            score = 2 * len(q & _tokens(f"{title} {url}")) + len(q & _tokens(body))
            if score:
                snippet = (page.get("description") or page.get("text", ""))[:SNIPPET_CHARS]
                scored.append((score, url, {"url": url, "title": title, "snippet": snippet}))
        scored.sort(key=lambda s: (-s[0], s[1]))  # URL tie-break keeps results deterministic
        return [r for _, _, r in scored[:MAX_RESULTS]]


def search_tool(provider: SearchProvider) -> Tool:
    async def handler(args: dict[str, Any]) -> ToolResult:
        try:
            results = await provider.search(args["query"])
        except SearchUnavailable as err:
            return ToolResult.error(str(err))
        docs = {r["url"]: f"{r['title']} {r['snippet']}" for r in results}
        return ToolResult.json({"query": args["query"], "results": results}, documents=docs)

    return Tool(
        name="web_search",
        description=(
            "Search the public web. Returns up to 5 results with url, title and snippet. "
            "Use it to find a company's official website when the name is ambiguous or "
            "messy; then fetch the page to confirm. Snippets are short; fetch a page before "
            "relying on details from it."
        ),
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query."}},
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=handler,
    )
