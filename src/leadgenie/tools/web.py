"""fetch_company_site: retrieve a company's public web page as plain text.

Live fetching is deliberately conservative:
  * a hard denylist of sites whose terms forbid scraping (LinkedIn and friends) — never fetched
  * robots.txt is honoured for our user agent
  * only http(s), only public IP addresses (no SSRF into localhost / private ranges)
  * redirects followed manually so every hop gets the same checks
  * per-host rate limit, response size cap, short timeout
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from html.parser import HTMLParser
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx2

from leadgenie.ratelimit import KeyedLimiter
from leadgenie.tools.base import Tool, ToolResult

USER_AGENT = "LeadGenieBot/0.2 (+https://github.com/; research prototype)"
MAX_BYTES = 2_000_000
MAX_TEXT_CHARS = 8_000
MAX_REDIRECTS = 3

# Sites whose terms of service forbid automated access. Matched on the registrable suffix,
# so subdomains (de.linkedin.com, m.facebook.com) are blocked too.
DENYLIST = (
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "glassdoor.com",
    "indeed.com",
    "zoominfo.com",
    "crunchbase.com",
    "apollo.io",
    "rocketreach.co",
)


class FetchBlocked(Exception):
    """The URL is not allowed to be fetched (policy, not a transient failure)."""


def is_denylisted(host: str) -> bool:
    host = host.lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in DENYLIST)


class _TextExtractor(HTMLParser):
    SKIP = frozenset({"script", "style", "noscript", "svg", "template"})

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.description = ""
        self._chunks: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            a = dict(attrs)
            if (a.get("name") or a.get("property") or "").lower() in (
                "description",
                "og:description",
            ):
                self.description = self.description or (a.get("content") or "")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
        elif data.strip():
            self._chunks.append(data.strip())

    @property
    def text(self) -> str:
        return " ".join(self._chunks)


def html_to_text(html: str) -> dict[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    return {
        "title": parser.title.strip(),
        "description": parser.description.strip(),
        "text": parser.text[:MAX_TEXT_CHARS],
    }


class Fetcher(Protocol):
    async def fetch(self, url: str) -> dict[str, str]:
        """Returns {"url", "title", "description", "text"}; raises FetchBlocked or other errors."""
        ...


def normalize_url(url: str) -> str:
    url = url.strip()
    if "://" not in url:
        url = "https://" + url
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchBlocked(f"only http(s) URLs with a host are allowed: {url!r}")
    return parts._replace(fragment="").geturl()


async def _assert_public_host(host: str) -> None:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise FetchBlocked(f"{host} resolves to non-public address {ip}")


class LiveFetcher:
    def __init__(self, per_host_rate: float = 0.5, timeout: float = 10.0) -> None:
        self._client = httpx2.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=False,
        )
        self._limiter = KeyedLimiter(rate=per_host_rate)
        self._robots: dict[str, RobotFileParser] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _robots_allows(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            rp = RobotFileParser()
            try:
                resp = await self._client.get(origin + "/robots.txt")
                if resp.status_code in (401, 403):
                    rp.disallow_all = True
                elif resp.status_code >= 400:
                    rp.allow_all = True
                else:
                    rp.parse(resp.text.splitlines())
            except httpx2.HTTPError:
                rp.allow_all = True  # unreachable robots.txt → no stated restriction
            self._robots[origin] = rp
        return self._robots[origin].can_fetch(USER_AGENT, url)

    async def fetch(self, url: str) -> dict[str, str]:
        url = normalize_url(url)
        for _ in range(MAX_REDIRECTS + 1):
            host = urlsplit(url).hostname or ""
            if is_denylisted(host):
                raise FetchBlocked(f"{host} forbids automated access; not fetched")
            await _assert_public_host(host)
            if not await self._robots_allows(url):
                raise FetchBlocked(f"robots.txt disallows {url}")
            await self._limiter.for_key(host).acquire()
            async with self._client.stream("GET", url) as resp:
                if resp.is_redirect and "location" in resp.headers:
                    url = normalize_url(urljoin(url, resp.headers["location"]))
                    continue
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                if "html" not in ctype and "text" not in ctype:
                    raise FetchBlocked(f"unsupported content-type {ctype!r}")
                body = b""
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) > MAX_BYTES:
                        break
                html = body.decode(resp.encoding or "utf-8", errors="replace")
                return {"url": str(resp.url), **html_to_text(html)}
        raise FetchBlocked(f"too many redirects for {url}")


class FixtureFetcher:
    """Serves pages from the offline eval world. Unknown URLs behave like a 404."""

    def __init__(self, pages: dict[str, dict[str, str]]) -> None:
        self._pages = {self._key(u): {"url": u, **p} for u, p in pages.items()}

    @staticmethod
    def _key(url: str) -> str:
        parts = urlsplit(normalize_url(url))
        path = parts.path.rstrip("/") or "/"
        return f"{(parts.hostname or '').removeprefix('www.')}{path}"

    async def fetch(self, url: str) -> dict[str, str]:
        url = normalize_url(url)
        host = urlsplit(url).hostname or ""
        if is_denylisted(host):
            raise FetchBlocked(f"{host} forbids automated access; not fetched")
        page = self._pages.get(self._key(url))
        if page is None:
            raise LookupError(f"404 Not Found: {url}")
        return {
            "url": page["url"],
            "title": page.get("title", ""),
            "description": page.get("description", ""),
            "text": page.get("text", "")[:MAX_TEXT_CHARS],
        }


def fetch_tool(fetcher: Fetcher) -> Tool:
    async def handler(args: dict[str, Any]) -> ToolResult:
        try:
            page = await fetcher.fetch(args["url"])
        except FetchBlocked as err:
            return ToolResult.error(f"blocked: {err}")
        doc = " ".join(v for k, v in page.items() if k != "url" and v)
        return ToolResult.json(page, documents={page["url"]: doc})

    return Tool(
        name="fetch_company_site",
        description=(
            "Fetch a company's public web page (homepage, /about, /careers, /team) and return "
            "its title, meta description and visible text. Use it to verify the company's "
            "identity, what it does, and people's roles. Social networks and data brokers "
            "(LinkedIn, Facebook, ZoomInfo, ...) are blocked by policy. Returns an error for "
            "pages that do not exist."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL or bare domain, e.g. 'https://acme.example/about'.",
                }
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        handler=handler,
    )
