"""Run the agent over many leads concurrently: dedupe, skip done, enrich, route, record."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from leadgenie.agent import AgentConfig, AgentOutcome, LeadAgent, ModelClient
from leadgenie.models import Enrichment, Lead
from leadgenie.ratelimit import TokenBucket
from leadgenie.store import ResultRow, Store
from leadgenie.tools import Backends, build_registry
from leadgenie.tracing import SpanSink, Tracer
from leadgenie.validate import DEFAULT_THRESHOLD, route, routing_confidence

ENRICHED_FIELDS = [
    "lead_id",
    "name",
    "company",
    "domain",
    "role",
    "seniority",
    "industry",
    "accepts_email",
    "outreach_line",
    "confidence",
    "company_source",
    "role_source",
    "industry_source",
    "outreach_source",
]
REVIEW_FIELDS = [*ENRICHED_FIELDS, "reasons"]

# Agent stop reasons that mean "the system failed" (→ error) rather than "the agent
# ran out of budget before it was sure" (→ review). Both end up in the review queue.
ERROR_STOPS = {"api_error", "refusal"}


def flatten_enrichment(e: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in Enrichment.SOURCED_FIELDS:
        out[name] = e[name]["value"]
        out[f"{name}_source"] = e[name]["source"]
    out["outreach_line"] = e["outreach"]["text"]
    out["outreach_source"] = e["outreach"]["source"]
    return out


@dataclass
class PipelineConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    threshold: float = DEFAULT_THRESHOLD
    concurrency: int = 4
    requests_per_minute: float = 50.0


@dataclass
class LeadResult:
    lead: Lead
    row: ResultRow
    outcome: AgentOutcome


@dataclass
class RunResult:
    run_id: str
    results: list[LeadResult]
    skipped: int


def to_row(lead: Lead, run_id: str, outcome: AgentOutcome, threshold: float) -> ResultRow:
    if outcome.enrichment is None:
        status = "error" if outcome.stop in ERROR_STOPS else "review"
        return ResultRow(
            lead_id=lead.id,
            run_id=run_id,
            name=lead.name,
            raw_company=lead.raw_company,
            title=lead.title,
            status=status,
            reasons=[outcome.stop],
            confidence=None,
            enrichment=None,
            error=outcome.error,
        )
    status, reasons = route(outcome.enrichment, outcome.evidence, threshold)
    return ResultRow(
        lead_id=lead.id,
        run_id=run_id,
        name=lead.name,
        raw_company=lead.raw_company,
        title=lead.title,
        status=status,
        reasons=reasons,
        confidence=routing_confidence(outcome.enrichment),
        enrichment=outcome.enrichment.model_dump(),
    )


async def run_pipeline(
    leads: Iterable[Lead],
    *,
    client: ModelClient,
    store: Store,
    backends: Backends,
    config: PipelineConfig,
    extra_sinks: list[SpanSink] | None = None,
    run_id: str | None = None,
    on_result: Any = None,
) -> RunResult:
    run_id = run_id or uuid.uuid4().hex[:12]
    tracer = Tracer(run_id, [store, *(extra_sinks or [])])
    limiter = TokenBucket(rate=config.requests_per_minute / 60, burst=config.concurrency)
    agent = LeadAgent(
        client,
        build_registry(backends, store, current_run_id=run_id),
        tracer,
        config.agent,
        api_limiter=limiter,
    )
    store.start_run(
        run_id,
        config.agent.model,
        {
            "threshold": config.threshold,
            "effort": config.agent.effort,
            "max_steps": config.agent.budget.max_steps,
            "max_cost_usd": config.agent.budget.max_cost_usd,
            "prompt": config.agent.prompt.fingerprint,
            "concurrency": config.concurrency,
        },
    )

    todo: dict[str, Lead] = {}
    skipped = 0
    for lead in leads:
        if lead.id in todo or lead.id in store:
            skipped += 1  # duplicate row in the input, or already done in an earlier run
        else:
            todo[lead.id] = lead

    semaphore = asyncio.Semaphore(config.concurrency)
    results: list[LeadResult] = []

    async def worker(lead: Lead) -> None:
        async with semaphore:
            outcome = await agent.run(lead)
        row = to_row(lead, run_id, outcome, config.threshold)
        # The result row is the done-marker: written in one transaction, so a crash
        # leaves the lead either fully recorded or untouched, never half-done.
        store.record_result(row)
        result = LeadResult(lead, row, outcome)
        results.append(result)
        if on_result:
            on_result(result)

    await asyncio.gather(*(worker(lead) for lead in todo.values()))
    store.finish_run(run_id, {"processed": len(results), "skipped": skipped})
    return RunResult(run_id, results, skipped)
