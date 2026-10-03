"""Run the eval: the full agent pipeline over a golden split in the offline world, scored.

    leadgenie eval --split test               # live model calls (costs money)
    leadgenie eval --split test --replay      # replay recorded calls (free, no key)

Writes results/eval-<split>-<timestamp>.json (metrics + every prediction) and a .md
summary next to it. Each results file records the git commit it was produced from.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from leadgenie.agent import AgentConfig, ModelClient
from leadgenie.evals.golden import GoldenLead, load_golden
from leadgenie.evals.metrics import (
    Prediction,
    by_scenario,
    calibration_pairs,
    per_field,
    reliability,
    review_metrics,
)
from leadgenie.evals.outreach import (
    CRITERIA,
    JUDGE_MODEL,
    JudgeClient,
    deterministic_checks,
    judge_inputs,
    judge_line,
)
from leadgenie.models import Lead
from leadgenie.pipeline import LeadResult, PipelineConfig, run_pipeline
from leadgenie.report import build_report, percentile
from leadgenie.store import Store
from leadgenie.tools import World, fixture_backends

GOLDEN_PATH = Path("evals/golden.jsonl")
WORLD_PATH = Path("evals/world.json")


@dataclass
class EvalConfig:
    split: str = "test"  # train | dev | test | all
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    judge: bool = True
    judge_model: str = JUDGE_MODEL
    judge_concurrency: int = 4
    limit: int | None = None  # first N leads of the split (by lead id), for cheap smoke runs


def git_revision() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{rev}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def select(golden: list[GoldenLead], split: str) -> list[GoldenLead]:
    chosen = golden if split == "all" else [g for g in golden if g.split == split]
    if not chosen:
        raise ValueError(f"no golden leads in split {split!r}")
    return chosen


def _lead_latency(spans: list[dict[str, Any]]) -> dict[str, int | None]:
    return {s["lead_id"]: s["latency_ms"] for s in spans if s["kind"] == "lead"}


async def _judge_all(
    results: list[LeadResult], client: JudgeClient, cfg: EvalConfig
) -> dict[str, dict[str, Any]]:
    sem = asyncio.Semaphore(cfg.judge_concurrency)

    async def one(r: LeadResult) -> tuple[str, dict[str, Any]]:
        assert r.outcome.enrichment is not None
        async with sem:
            j = await judge_line(
                client,
                **judge_inputs(r.outcome.enrichment, r.outcome.evidence, r.lead.name),
                model=cfg.judge_model,
            )
        return r.lead.id, {**asdict(j)}

    pairs = await asyncio.gather(*(one(r) for r in results if r.outcome.enrichment))
    return dict(pairs)


def _outreach_summary(
    preds: list[dict[str, Any]], judged: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    checked = [p["outreach_checks"] for p in preds if p["outreach_checks"]]
    summary: dict[str, Any] = {"n": len(checked)}
    if checked:
        summary["deterministic_pass_rate"] = {
            k: round(sum(c[k] for c in checked) / len(checked), 3) for k in checked[0]
        }
    if judged:
        js = list(judged.values())
        summary["judge"] = {
            "n": len(js),
            "pass_rate": round(sum(j["passed"] for j in js) / len(js), 3),
            "mean_scores": {
                c: round(sum(j["scores"][c] for j in js) / len(js), 2) for c in CRITERIA
            },
            "cost_usd": round(sum(j["cost_usd"] for j in js), 6),
        }
    return summary


async def run_eval(
    cfg: EvalConfig,
    client: ModelClient,
    judge_client: JudgeClient | None = None,
    golden_path: Path = GOLDEN_PATH,
    world_path: Path = WORLD_PATH,
) -> dict[str, Any]:
    golden = select(load_golden(golden_path), cfg.split)[: cfg.limit]
    world_data = json.loads(world_path.read_text())
    world = World(pages=world_data["pages"], mx=world_data["mx"])
    by_id = {g.lead_id: g for g in golden}

    store = Store(":memory:")  # each eval starts from a clean slate: no prior results leak in
    run = await run_pipeline(
        [Lead.from_row(g.row()) for g in golden],
        client=client,
        store=store,
        backends=fixture_backends(world),
        config=cfg.pipeline,
    )
    spans = store.spans(run.run_id)
    latency = _lead_latency(spans)
    results = sorted(run.results, key=lambda r: r.lead.id)

    judged: dict[str, dict[str, Any]] = {}
    if cfg.judge and judge_client is not None:
        judged = await _judge_all(results, judge_client, cfg)

    preds: list[Prediction] = []
    rows: list[dict[str, Any]] = []
    for r in results:
        g = by_id[r.lead.id]
        e = r.outcome.enrichment
        preds.append(
            Prediction(
                g,
                r.row.status,
                e,
                r.outcome.cost_usd,
                latency.get(r.lead.id),
                r.outcome.steps,
                r.outcome.stop,
            )
        )
        checks = (
            deterministic_checks(
                e.outreach.text,
                e.outreach.source,
                e.outreach.evidence,
                r.outcome.evidence,
                r.lead.name.split()[0],
                e.company.value,
            )
            if e
            else None
        )
        rows.append(
            {
                "lead_id": r.lead.id,
                "scenario": g.scenario,
                "input": g.row(),
                "labels": asdict(g.labels),
                "status": r.row.status,
                "reasons": r.row.reasons,
                "routing_confidence": r.row.confidence,
                "stop": r.outcome.stop,
                "steps": r.outcome.steps,
                "cost_usd": round(r.outcome.cost_usd, 6),
                "latency_ms": latency.get(r.lead.id),
                "enrichment": e.model_dump() if e else None,
                "outreach_checks": checks,
                "judge": judged.get(r.lead.id),
            }
        )

    ece, table = reliability(calibration_pairs(preds))
    costs = [p.cost_usd for p in preds]
    lats = [p.latency_ms for p in preds if p.latency_ms is not None]
    agent_cfg: AgentConfig = cfg.pipeline.agent
    report = build_report(run.run_id, spans, store.run_results(run.run_id))
    return {
        "meta": {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git": git_revision(),
            "split": cfg.split,
            "n": len(golden),
            "model": agent_cfg.model,
            "effort": agent_cfg.effort,
            "prompt_fingerprint": agent_cfg.prompt.fingerprint,
            "lessons": len(agent_cfg.prompt.lessons),
            "threshold": cfg.pipeline.threshold,
            "budget": asdict(agent_cfg.budget),
            "judge_model": cfg.judge_model if judged else None,
        },
        "fields": per_field(preds),
        "calibration": {"ece": ece, "reliability": table},
        "review": review_metrics(preds),
        "scenarios": by_scenario(preds),
        "cost_usd": {
            "agent_total": round(sum(costs), 6),
            "per_lead_mean": round(sum(costs) / len(costs), 6) if costs else None,
            "per_lead_p95": percentile(costs, 95),
        },
        "latency_ms": {"per_lead_p50": percentile(lats, 50), "per_lead_p95": percentile(lats, 95)},
        "steps_mean": round(sum(p.steps for p in preds) / len(preds), 2) if preds else None,
        "stops": report["agent_stop_reasons"],
        "tools": report["tools"],
        "outreach": _outreach_summary(rows, judged),
        "predictions": rows,
    }


def render_markdown(r: dict[str, Any]) -> str:
    m = r["meta"]
    lines = [
        f"# Eval: split `{m['split']}` (n={m['n']})",
        "",
        f"model `{m['model']}` (effort {m['effort']}), prompt `{m['prompt_fingerprint']}` "
        f"({m['lessons']} lessons), git `{m['git']}`, {m['created_at']}",
        "",
        "## Per-field accuracy",
        "",
        "| field | n | accuracy | coverage |",
        "|---|---|---|---|",
        *(
            f"| {k} | {v['n']} | {v['accuracy']} | {v['coverage']} |"
            for k, v in r["fields"].items()
        ),
        "",
        f"## Calibration (ECE = {r['calibration']['ece']})",
        "",
        "| confidence bin | n | mean confidence | accuracy | gap |",
        "|---|---|---|---|---|",
        *(
            f"| {b['bin']} | {b['n']} | {b['mean_confidence']} | {b['accuracy']} | {b['gap']} |"
            for b in r["calibration"]["reliability"]
        ),
        "",
        "## Review queue",
        "",
        "| precision | recall | F1 | review rate | unsafe approval rate |",
        "|---|---|---|---|---|",
        "| {precision} | {recall} | {f1} | {review_rate} | {unsafe_approval_rate} |".format(
            **r["review"]
        ),
        "",
        "## Cost and latency per lead",
        "",
        f"mean ${r['cost_usd']['per_lead_mean']}, p95 ${r['cost_usd']['per_lead_p95']}; "
        f"latency p50 {r['latency_ms']['per_lead_p50']} ms, p95 {r['latency_ms']['per_lead_p95']} "
        f"ms; {r['steps_mean']} model calls per lead; agent total ${r['cost_usd']['agent_total']}",
        "",
        "## Outreach",
        "",
        "```json",
        json.dumps(r["outreach"], indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_results(r: dict[str, Any], out_dir: str | Path, tag: str = "") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = r["meta"]["created_at"].replace(":", "").replace("-", "").replace("+0000", "Z")
    stem = f"eval-{r['meta']['split']}-{stamp}{('-' + tag) if tag else ''}"
    path = out / f"{stem}.json"
    path.write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
    (out / f"{stem}.md").write_text(render_markdown(r))
    return path
