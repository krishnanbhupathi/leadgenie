"""check_mx: does an email domain accept mail? (DNS MX lookup — no mail is ever sent.)"""

from __future__ import annotations

import re
from typing import Any, Protocol

import dns.asyncresolver
import dns.exception
import dns.resolver

from leadgenie.tools.base import Tool, ToolResult

_DOMAIN = re.compile(r"^(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")


class MXResolver(Protocol):
    async def lookup(self, domain: str) -> dict[str, Any]:
        """Returns {"domain", "exists", "has_mx", "mx_hosts"}."""
        ...


class LiveMX:
    def __init__(self, timeout: float = 5.0) -> None:
        self._resolver = dns.asyncresolver.Resolver()
        self._resolver.lifetime = timeout

    async def lookup(self, domain: str) -> dict[str, Any]:
        try:
            answer = await self._resolver.resolve(domain, "MX")
        except dns.resolver.NXDOMAIN:
            return {"domain": domain, "exists": False, "has_mx": False, "mx_hosts": []}
        except dns.resolver.NoAnswer:
            return {"domain": domain, "exists": True, "has_mx": False, "mx_hosts": []}
        hosts = sorted((r.preference, str(r.exchange).rstrip(".")) for r in answer)
        # A single "." exchange is a null MX (RFC 7505): the domain explicitly takes no mail.
        mx_hosts = [h for _, h in hosts if h]
        return {"domain": domain, "exists": True, "has_mx": bool(mx_hosts), "mx_hosts": mx_hosts}


class FixtureMX:
    def __init__(self, records: dict[str, list[str] | None]) -> None:
        # domain → list of MX hosts; [] = exists without MX; missing key = NXDOMAIN
        self._records = records

    async def lookup(self, domain: str) -> dict[str, Any]:
        if domain not in self._records:
            return {"domain": domain, "exists": False, "has_mx": False, "mx_hosts": []}
        hosts = self._records[domain] or []
        return {"domain": domain, "exists": True, "has_mx": bool(hosts), "mx_hosts": hosts}


def mx_tool(resolver: MXResolver) -> Tool:
    async def handler(args: dict[str, Any]) -> ToolResult:
        domain = args["domain"].strip().lower().rstrip(".")
        domain = domain.split("@")[-1].removeprefix("www.")
        if not _DOMAIN.match(domain):
            return ToolResult.error(f"not a valid domain name: {args['domain']!r}")
        try:
            result = await resolver.lookup(domain)
        except dns.exception.Timeout:
            return ToolResult.error(f"DNS timeout for {domain}; result unknown")
        return ToolResult.json(result, documents={f"dns:mx:{domain}": str(result)})

    return Tool(
        name="check_mx",
        description=(
            "Check whether a domain can receive email by looking up its DNS MX records. "
            "Returns exists (domain resolves), has_mx and the MX hosts. No email is sent. "
            "Use the company's website domain."
        ),
        input_schema={
            "type": "object",
            "properties": {"domain": {"type": "string", "description": "e.g. 'acme.example'"}},
            "required": ["domain"],
            "additionalProperties": False,
        },
        handler=handler,
    )
