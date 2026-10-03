"""The agent: a tool-use loop that researches one lead and submits a sourced enrichment.

Loop per lead:
  model turn → run the requested tools (in parallel) → feed results back → repeat,
until the model calls submit_enrichment with a valid payload, or a budget runs out.

Budgets are per lead and enforced in code, not left to the model:
  * max_steps     — model turns
  * max_cost_usd  — computed from the usage block of every response
Before the last step, or once 75% of the cost budget is spent, the agent is told to
submit what it has. If it still doesn't, the lead goes to review with the stop reason.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic
from pydantic import ValidationError

from leadgenie.models import SUBMIT_SCHEMA, Enrichment, Lead
from leadgenie.pricing import PRICES, TokenUsage, cost_usd
from leadgenie.ratelimit import TokenBucket
from leadgenie.tools.base import EvidenceLog, ToolRegistry, ToolResult
from leadgenie.tracing import Span, Tracer

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
MAX_TOKENS = 16_000
SUBMIT_TOOL = "submit_enrichment"

# output_config.effort is rejected by Haiku 4.5; server-side refusal fallbacks are only
# offered on the newest models.
EFFORT_MODELS = {"claude-opus-5-5", "claude-opus-4-8", "claude-sonnet-5-5", "claude-sonnet-5"}
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

BASE_SYSTEM_PROMPT = """\
You are a lead-enrichment research agent for a B2B sales-prospecting tool.

For one messy lead (name, raw company string, maybe a job title) you research the company \
with your tools and then call submit_enrichment exactly once.

How to research:
- Identify the company's official website. Use web_search when it is available; otherwise \
guess the domain from the company name and fetch it. Beware of look-alike companies with \
similar names: confirm identity from the page content, not the name alone.
- Fetch the homepage and, if useful, /about or /team pages to confirm what the company does \
and, when possible, the person's role.
- Call check_mx on the company's domain.
- lookup_prior_results may already hold an approved answer for the same company.
- Call independent tools in parallel. You have a small step budget: don't re-fetch pages \
or run near-duplicate searches.

Provenance rules (enforced by code after you submit):
- Every field has a `source`: either the exact URL (or prior:/dns:mx: ref) that a tool \
returned in this session, or "input" if the value comes straight from the lead row, or \
"inferred" if it is your own judgement with no retrieved evidence.
- For a retrieved source, `evidence` must be a short verbatim quote (under 200 characters) \
copied from that tool result. For "input" and "inferred", evidence is "".
- Never cite a URL you did not retrieve. Never invent facts, funding rounds, launches or news.

Confidence: for each field, the probability that the value is correct. Inferred values \
should rarely exceed 0.6. If you cannot identify the company, still submit, with low \
confidence; a vague but honest answer beats a confident hallucination.

Outreach: one opening line, at most 30 words, that references one specific fact from a page \
you retrieved, and cites that page. Never generic ("I hope this finds you well")."""


class AsyncMessages(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class AsyncBeta(Protocol):
    @property
    def messages(self) -> AsyncMessages: ...


class ModelClient(Protocol):
    """The subset of anthropic.AsyncAnthropic the agent uses (so tests can fake it)."""

    @property
    def beta(self) -> AsyncBeta: ...


@dataclass(frozen=True)
class Budget:
    max_steps: int = 8
    max_cost_usd: float = 0.25
    warn_cost_fraction: float = 0.75


@dataclass(frozen=True)
class Lesson:
    """A few-shot example distilled from a human correction (see improve.py)."""

    situation: str
    guidance: str


@dataclass(frozen=True)
class PromptConfig:
    """Everything about the prompt that the self-improving loop is allowed to change."""

    base: str = BASE_SYSTEM_PROMPT
    lessons: tuple[Lesson, ...] = ()
    addendum: str = ""

    def render(self) -> str:
        parts = [self.base]
        if self.lessons:
            parts.append(
                "Lessons from human review of earlier leads:\n"
                + "\n".join(f"- {ls.situation} → {ls.guidance}" for ls in self.lessons)
            )
        if self.addendum:
            parts.append(self.addendum)
        return "\n\n".join(parts)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.render().encode()).hexdigest()[:12]


@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    budget: Budget = field(default_factory=Budget)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    fallbacks: bool = True


@dataclass
class AgentOutcome:
    enrichment: Enrichment | None
    evidence: EvidenceLog
    stop: str  # submitted | budget_steps | budget_cost | refusal | api_error
    steps: int
    usage: TokenUsage
    cost_usd: float
    error: str | None = None


def submit_tool_param() -> dict[str, Any]:
    return {
        "name": SUBMIT_TOOL,
        "description": (
            "Submit the final enrichment for this lead. Call exactly once, after research. "
            "If the payload is invalid you will get an error explaining what to fix."
        ),
        "input_schema": SUBMIT_SCHEMA,
        "strict": True,
    }


def render_lead(lead: Lead) -> str:
    return (
        "Enrich this lead:\n"
        f"- name: {lead.name}\n"
        f"- raw_company: {lead.raw_company}\n"
        f"- title: {lead.title or '(missing)'}"
    )


def _summarize_content(content: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in content:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "name": b.name, "input": b.input})
        else:
            out.append({"type": b.type})
    return out


class LeadAgent:
    def __init__(
        self,
        client: ModelClient,
        registry: ToolRegistry,
        tracer: Tracer,
        config: AgentConfig | None = None,
        api_limiter: TokenBucket | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.tracer = tracer
        self.config = config or AgentConfig()
        self.api_limiter = api_limiter
        self._tools = [*registry.params(), submit_tool_param()]
        self._system = self.config.prompt.render()

    def _request(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        cfg = self.config
        req: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": MAX_TOKENS,
            "system": self._system,
            "tools": self._tools,
            "messages": messages,
            # Each step re-sends the whole conversation; caching makes that prefix cheap.
            "cache_control": {"type": "ephemeral"},
        }
        if cfg.model in EFFORT_MODELS:
            req["output_config"] = {"effort": cfg.effort}
        if cfg.fallbacks and cfg.model in FALLBACK_MODELS:
            req["betas"] = [FALLBACK_BETA]
            req["fallbacks"] = "default"
        return req

    def _price(self, response: Any, usage: TokenUsage, span: Span) -> float:
        served = getattr(response, "model", None) or self.config.model
        if served not in PRICES:
            # A refusal fallback can serve the turn on another model; price it as the
            # requested model rather than as $0, and make the substitution visible.
            span.attrs["priced_as"] = self.config.model
            served = self.config.model
        cost = cost_usd(served, usage)
        span.set_usage(served, usage, cost)
        return cost

    async def _call_model(
        self, messages: list[dict[str, Any]], lead_span: Span, step: int
    ) -> tuple[Any, float]:
        with self.tracer.span(
            "model_call",
            "messages.create",
            parent=lead_span,
            step=step,
            input={"step": step, "n_messages": len(messages)},
        ) as span:
            if self.api_limiter:
                await self.api_limiter.acquire()
            # Transient failures (429, 5xx, connection) are retried with backoff by the SDK
            # client (max_retries); what reaches us here is final.
            response = await self.client.beta.messages.create(**self._request(messages))
            cost = self._price(response, TokenUsage.from_api(response.usage), span)
            span.attrs["stop_reason"] = response.stop_reason
            span.output = _summarize_content(response.content)
            return response, cost

    async def _run_tool(self, block: Any, lead_span: Span, step: int) -> ToolResult:
        with self.tracer.span(
            "tool_call", block.name, parent=lead_span, step=step, input=block.input
        ) as span:
            result = await self.registry.call(block.name, dict(block.input))
            span.output = result.content
            if result.is_error:
                span.fail(result.content)
            return result

    def _check_submission(self, payload: dict[str, Any]) -> tuple[Enrichment | None, str]:
        try:
            return Enrichment.model_validate(payload), ""
        except ValidationError as err:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in err.errors()
            )
            return None, f"Invalid submission, fix and resubmit: {problems}"

    async def run(self, lead: Lead) -> AgentOutcome:
        budget = self.config.budget
        evidence = EvidenceLog()
        messages: list[dict[str, Any]] = [{"role": "user", "content": render_lead(lead)}]
        usage = TokenUsage()
        spent = 0.0
        stop = "budget_steps"
        enrichment: Enrichment | None = None
        error: str | None = None
        step = 0

        with self.tracer.span("lead", "enrich", lead_id=lead.id, input=lead.model_dump()) as span:
            for step in range(1, budget.max_steps + 1):
                try:
                    response, cost = await self._call_model(messages, span, step)
                except anthropic.APIError as err:
                    stop, error = "api_error", f"{type(err).__name__}: {err}"
                    break
                usage = usage + TokenUsage.from_api(response.usage)
                spent += cost
                if response.stop_reason == "refusal":
                    stop, error = "refusal", str(getattr(response, "stop_details", None))
                    break

                # Append-only history: the assistant turn goes back exactly as received.
                messages.append({"role": "assistant", "content": response.content})
                tool_uses = [b for b in response.content if b.type == "tool_use"]
                if not tool_uses:
                    messages.append(
                        {"role": "user", "content": f"Call {SUBMIT_TOOL} to finish this lead."}
                    )
                    continue

                results: dict[str, ToolResult] = {}
                research = [b for b in tool_uses if b.name != SUBMIT_TOOL]
                outputs = await asyncio.gather(*(self._run_tool(b, span, step) for b in research))
                for block, result in zip(research, outputs, strict=True):
                    evidence.add(result.documents)
                    results[block.id] = result
                for block in (b for b in tool_uses if b.name == SUBMIT_TOOL):
                    enrichment, problem = self._check_submission(dict(block.input))
                    results[block.id] = (
                        ToolResult("accepted") if enrichment else ToolResult.error(problem)
                    )
                if enrichment is not None:
                    stop = "submitted"
                    break

                if spent >= budget.max_cost_usd:
                    stop = "budget_cost"
                    break

                content: list[dict[str, Any]] = [
                    {
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": results[b.id].content,
                        "is_error": results[b.id].is_error,
                    }
                    for b in tool_uses
                ]
                last_step = step == budget.max_steps - 1
                if last_step or spent >= budget.warn_cost_fraction * budget.max_cost_usd:
                    content.append(
                        {
                            "type": "text",
                            "text": (
                                "Budget nearly exhausted. Call submit_enrichment now with "
                                "what you have; mark anything unverified as inferred with "
                                "low confidence."
                            ),
                        }
                    )
                messages.append({"role": "user", "content": content})

            span.attrs.update(
                {"stop": stop, "steps": step, "cost": round(spent, 6), "model": self.config.model}
            )
            span.output = enrichment.model_dump() if enrichment else {"error": error}

        return AgentOutcome(
            enrichment=enrichment,
            evidence=evidence,
            stop=stop,
            steps=step,
            usage=usage,
            cost_usd=spent,
            error=error,
        )
