"""Per-run summary report, computed from trace spans and stored results."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in [0, 100]); None for an empty sequence."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def _reason_kind(reason: str) -> str:
    return reason.split(":", 1)[0]


def build_report(
    run_id: str, spans: Iterable[Mapping[str, Any]], results: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    spans = list(spans)
    results = list(results)
    lead_spans = [s for s in spans if s["kind"] == "lead"]
    model_spans = [s for s in spans if s["kind"] == "model_call"]
    tool_spans = [s for s in spans if s["kind"] == "tool_call"]

    cost_by_lead: dict[str, float] = defaultdict(float)
    steps_by_lead: Counter[str] = Counter()
    for s in model_spans:
        cost_by_lead[s["lead_id"]] += s["cost_usd"]
        steps_by_lead[s["lead_id"]] += 1
    lead_costs = [cost_by_lead[s["lead_id"]] for s in lead_spans]
    lead_latency = [s["latency_ms"] for s in lead_spans if s["latency_ms"] is not None]

    tokens = {
        k: sum(s[k] for s in model_spans)
        for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
    }
    prompt_tokens = (
        tokens["input_tokens"] + tokens["cache_read_tokens"] + tokens["cache_write_tokens"]
    )

    tools: dict[str, dict[str, Any]] = {}
    for name in sorted({s["name"] for s in tool_spans}):
        calls = [s for s in tool_spans if s["name"] == name]
        errors = [s for s in calls if s["status"] == "error"]
        tools[name] = {
            "calls": len(calls),
            "errors": len(errors),
            "error_rate": round(len(errors) / len(calls), 3),
            "p50_latency_ms": percentile([s["latency_ms"] or 0 for s in calls], 50),
        }

    statuses = Counter(r["status"] for r in results)
    reasons: Counter[str] = Counter()
    for r in results:
        raw = r["reasons"]
        for reason in json.loads(raw) if isinstance(raw, str) else raw:
            reasons[_reason_kind(reason)] += 1
    stop_reasons = Counter((s.get("attrs") or {}).get("stop", "unknown") for s in lead_spans)
    n = len(results)

    return {
        "run_id": run_id,
        "leads": n,
        "status": {k: statuses.get(k, 0) for k in ("approved", "review", "error")},
        "approved_pct": round(100 * statuses.get("approved", 0) / n, 1) if n else None,
        "cost_usd": {
            "total": round(sum(s["cost_usd"] for s in model_spans), 6),
            "per_lead_mean": round(sum(lead_costs) / len(lead_costs), 6) if lead_costs else None,
            "per_lead_p95": percentile(lead_costs, 95),
        },
        "latency_ms_per_lead": {
            "p50": percentile(lead_latency, 50),
            "p95": percentile(lead_latency, 95),
        },
        "steps_per_lead": {
            "mean": round(sum(steps_by_lead.values()) / len(lead_spans), 2) if lead_spans else None,
            "max": max(steps_by_lead.values(), default=0),
        },
        "tokens": tokens,
        "cache_hit_ratio": round(tokens["cache_read_tokens"] / prompt_tokens, 3)
        if prompt_tokens
        else None,
        "model_call_errors": sum(1 for s in model_spans if s["status"] == "error"),
        "tools": tools,
        "agent_stop_reasons": dict(stop_reasons.most_common()),
        "review_reasons": dict(reasons.most_common()),
    }


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}" if v < 1 else f"{v:,.2f}"
    return str(v)


def render_markdown(r: Mapping[str, Any]) -> str:
    lines = [
        f"# Run report `{r['run_id']}`",
        "",
        f"Leads: **{r['leads']}** — approved {r['status']['approved']}, "
        f"review {r['status']['review']}, error {r['status']['error']}",
        "",
        "| metric | value |",
        "|---|---|",
        f"| total cost (USD) | {_fmt(r['cost_usd']['total'])} |",
        f"| cost / lead, mean (USD) | {_fmt(r['cost_usd']['per_lead_mean'])} |",
        f"| cost / lead, p95 (USD) | {_fmt(r['cost_usd']['per_lead_p95'])} |",
        f"| latency / lead, p50 (ms) | {_fmt(r['latency_ms_per_lead']['p50'])} |",
        f"| latency / lead, p95 (ms) | {_fmt(r['latency_ms_per_lead']['p95'])} |",
        f"| model calls / lead, mean | {_fmt(r['steps_per_lead']['mean'])} |",
        f"| prompt-cache hit ratio | {_fmt(r['cache_hit_ratio'])} |",
        f"| model call errors | {r['model_call_errors']} |",
        "",
        "## Tools",
        "",
        "| tool | calls | errors | error rate | p50 latency (ms) |",
        "|---|---|---|---|---|",
        *(
            f"| `{name}` | {t['calls']} | {t['errors']} | {t['error_rate']:.1%} | "
            f"{_fmt(t['p50_latency_ms'])} |"
            for name, t in r["tools"].items()
        ),
        "",
        "## Why the agent stopped",
        "",
        *(f"- `{k}`: {v}" for k, v in r["agent_stop_reasons"].items()),
        "",
        "## Review reasons",
        "",
        *(f"- `{k}`: {v}" for k, v in r["review_reasons"].items()),
        "",
    ]
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"run-{report['run_id']}.json"
    md_path = out / f"run-{report['run_id']}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(render_markdown(report))
    return json_path, md_path
