"""LeadGenie CLI.

Usage:
    leadgenie run leads.csv                           # live web, real model
    leadgenie run leads.csv --world evals/world.json  # offline fixture web
    leadgenie run leads.csv --model claude-haiku-4-5 --threshold 0.8
    leadgenie report <run_id>
    leadgenie eval --split test [--replay | --record] [--no-judge] [--gate evals/gate.json]
    leadgenie judge-check
    leadgenie review --reviewer you@example.com
    leadgenie improve --iterations 3 [--record | --replay]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import anthropic

from leadgenie.agent import DEFAULT_EFFORT, DEFAULT_MODEL, AgentConfig, Budget, ModelClient
from leadgenie.evals.cassette import Cassette
from leadgenie.evals.gate import check_gate, run_judge_check
from leadgenie.evals.outreach import JUDGE_MODEL
from leadgenie.evals.run import EvalConfig, run_eval, write_results
from leadgenie.improve import DEFAULT_PROMPT_PATH, ImproveConfig, Improver, load_prompt
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
from leadgenie.review import review_queue
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
    p.add_argument(
        "--prompt",
        default=str(DEFAULT_PROMPT_PATH),
        help="prompt config JSON kept by `improve` (base prompt if the file is missing)",
    )


def pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        agent=AgentConfig(
            model=args.model,
            effort=args.effort,
            budget=Budget(max_steps=args.max_steps, max_cost_usd=args.max_cost),
            fallbacks=not args.no_fallbacks,
            prompt=load_prompt(Path(args.prompt)),
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
        client=live_client(),
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


def live_client() -> ModelClient:
    # AsyncAnthropic satisfies ModelClient structurally at runtime; its overloaded,
    # fully-typed create() signature just doesn't unify with the Protocol's **kwargs.
    return cast(ModelClient, anthropic.AsyncAnthropic(max_retries=4))


def model_client(args: argparse.Namespace) -> Any:
    """Live client, or a cassette that records (live + saved) or replays (no key needed)."""
    if args.replay:
        return Cassette(args.cassette, "replay")
    live = live_client()
    return Cassette(args.cassette, "record", inner=live) if args.record else live


async def cmd_eval(args: argparse.Namespace) -> int:
    client = model_client(args)
    cfg = EvalConfig(
        split=args.split,
        pipeline=pipeline_config(args),
        judge=not args.no_judge,
        judge_model=args.judge_model,
        limit=args.limit,
    )
    results = await run_eval(cfg, client, judge_client=client)
    path = write_results(results, args.out, tag=args.tag)
    print(Path(path).with_suffix(".md").read_text())
    print(f"results: {path}")
    if isinstance(client, Cassette):
        print(f"cassette {client.path}: {client.hits} replayed, {client.recorded} recorded")
    if args.gate:
        failures = check_gate(results, json.loads(Path(args.gate).read_text()))
        for f in failures:
            print(f"GATE FAIL  {f}")
        if failures:
            return 1
        print("gate: pass")
    return 0


async def cmd_judge_check(args: argparse.Namespace) -> int:
    result = await run_judge_check(model_client(args), model=args.judge_model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "judge-check.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"judge {result['model']}: agreement {result['agreement']} "
        f"kappa {result['cohen_kappa']} on n={result['n']} (false pass {result['false_pass']}, "
        f"false fail {result['false_fail']}), ${result['cost_usd']}"
    )
    print(f"results: {path}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    n = review_queue(Store(args.db), args.reviewer, limit=args.limit)
    print(f"\nrecorded {n} correction(s)")
    return 0


async def cmd_improve(args: argparse.Namespace) -> int:
    cfg = ImproveConfig(
        iterations=args.iterations,
        min_delta=args.min_delta,
        pipeline=pipeline_config(args),
        prompt_path=Path(args.prompt),
        out_dir=Path(args.out) / "improve",
        log_path=Path(args.out) / "improve_log.jsonl",
    )
    summary = await Improver(cfg, model_client(args), Store(args.db)).run()
    print(json.dumps(summary, indent=2))
    return 0


def add_cassette_args(p: argparse.ArgumentParser, default: str) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--record", action="store_true", help="call the API and save responses")
    g.add_argument("--replay", action="store_true", help="replay saved responses; no API key")
    p.add_argument("--cassette", default=default)
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--out", default="results")


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

    ev = sub.add_parser("eval", help="score the agent on the golden set (offline world)")
    ev.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    ev.add_argument("--no-judge", action="store_true", help="skip the LLM outreach judge")
    ev.add_argument("--gate", help="JSON thresholds; exit 1 if any fails")
    ev.add_argument("--tag", default="", help="suffix for the results filename")
    ev.add_argument("--limit", type=int, help="only the first N leads of the split")
    add_agent_args(ev)
    add_cassette_args(ev, "evals/cassettes/eval.jsonl")

    rv = sub.add_parser("review", help="correct review-queue leads (feeds `improve`)")
    rv.add_argument("--reviewer", required=True)
    rv.add_argument("--limit", type=int)

    im = sub.add_parser("improve", help="self-improving loop over corrections + evals")
    im.add_argument("--iterations", type=int, default=3)
    im.add_argument("--min-delta", type=float, default=0.03)
    add_agent_args(im)
    add_cassette_args(im, "evals/cassettes/improve.jsonl")

    jc = sub.add_parser("judge-check", help="measure judge agreement with hand labels")
    add_cassette_args(jc, "evals/cassettes/judge-check.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return asyncio.run(cmd_run(args))
    if args.command == "report":
        return cmd_report(args)
    if args.command == "eval":
        return asyncio.run(cmd_eval(args))
    if args.command == "review":
        return cmd_review(args)
    if args.command == "improve":
        return asyncio.run(cmd_improve(args))
    if args.command == "judge-check":
        return asyncio.run(cmd_judge_check(args))
    return 2


if __name__ == "__main__":
    sys.exit(main())
