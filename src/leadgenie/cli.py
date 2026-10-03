"""LeadGenie CLI.

Usage:
    leadgenie run leads.csv                           # live web, real model
    leadgenie run leads.csv --world evals/world.json  # offline fixture web
    leadgenie run leads.csv --model claude-haiku-4-5 --threshold 0.8
    leadgenie report <run_id>
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import anthropic

from leadgenie.agent import DEFAULT_EFFORT, DEFAULT_MODEL, AgentConfig, Budget
from leadgenie.models import Lead
from leadgenie.pipeline import (
    ENRICHED_FIELDS,
    REVIEW_FIELDS,
    LeadResult,
    PipelineConfig,
    flatten_enrichment,
    run_pipeline,
)
from leadgenie.report import build_report, render_markdown, write_report
from leadgenie.store import Store
from leadgenie.tools import World, fixture_backends, live_backends
from leadgenie.tracing import JsonlSink
from leadgenie.validate import DEFAULT_THRESHOLD


def read_leads(path: str) -> Iterator[Lead]:
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            lead = Lead.from_row(row)
            if lead.name and lead.raw_company:
                yield lead


def load_world(path: str) -> World:
    data = json.loads(Path(path).read_text())
    return World(pages=data["pages"], mx=data["mx"])


def _print_result(r: LeadResult) -> None:
    o = r.outcome
    if r.row.status == "approved":
        flag = "APPROVED"
    else:
        flag = f"{r.row.status.upper()} ({';'.join(r.row.reasons)})"
    conf = f"{r.row.confidence:.2f}" if r.row.confidence is not None else "n/a"
    print(
        f"[{r.lead.id}] {r.lead.name} @ {r.lead.raw_company!r}: conf={conf} "
        f"steps={o.steps} ${o.cost_usd:.4f} → {flag}"
    )


def add_agent_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high"])
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--max-steps", type=int, default=Budget.max_steps)
    p.add_argument("--max-cost", type=float, default=Budget.max_cost_usd, help="USD per lead")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--rpm", type=float, default=50.0, help="model requests per minute")
    p.add_argument("--no-fallbacks", action="store_true", help="disable refusal fallbacks")


def pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        agent=AgentConfig(
            model=args.model,
            effort=args.effort,
            budget=Budget(max_steps=args.max_steps, max_cost_usd=args.max_cost),
            fallbacks=not args.no_fallbacks,
        ),
        threshold=args.threshold,
        concurrency=args.concurrency,
        requests_per_minute=args.rpm,
    )


async def cmd_run(args: argparse.Namespace) -> int:
    store = Store(args.db)
    backends = fixture_backends(load_world(args.world)) if args.world else live_backends()
    run = await run_pipeline(
        read_leads(args.csv_path),
        client=anthropic.AsyncAnthropic(max_retries=4),
        store=store,
        backends=backends,
        config=pipeline_config(args),
        extra_sinks=[JsonlSink(args.trace_jsonl)] if args.trace_jsonl else None,
        on_result=_print_result,
    )
    store.export_csv("enriched.csv", ["approved"], ENRICHED_FIELDS, flatten_enrichment)
    store.export_csv("review_queue.csv", ["review", "error"], REVIEW_FIELDS, flatten_enrichment)
    report = build_report(run.run_id, store.spans(run.run_id), store.run_results(run.run_id))
    json_path, md_path = write_report(report, args.reports_dir)
    print(f"\nskipped {run.skipped} already-processed or duplicate lead(s)")
    print(render_markdown(report))
    print(f"report: {md_path} / {json_path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    store = Store(args.db)
    report = build_report(args.run_id, store.spans(args.run_id), store.run_results(args.run_id))
    print(render_markdown(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="leadgenie", description="Agentic lead enrichment.")
    parser.add_argument("--db", default="leadgenie.db", help="SQLite database path")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="enrich a CSV of leads")
    run.add_argument("csv_path", nargs="?", default="leads.csv")
    run.add_argument("--world", help="offline fixture world JSON (default: live web)")
    run.add_argument("--reports-dir", default="reports")
    run.add_argument("--trace-jsonl", help="also append spans to this JSONL file")
    add_agent_args(run)

    rep = sub.add_parser("report", help="print the report for a past run")
    rep.add_argument("run_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return asyncio.run(cmd_run(args))
    if args.command == "report":
        return cmd_report(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
