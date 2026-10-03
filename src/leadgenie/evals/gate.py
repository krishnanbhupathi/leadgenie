"""Eval gate (CI) and judge-agreement check."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from leadgenie.evals.outreach import JUDGE_MODEL, JudgeClient, judge_line

# Each threshold names a dotted path into the eval results and a bound.
# "min" fails if the value is below it, "max" fails if above.


def _get(results: dict[str, Any], dotted: str) -> Any:
    value: Any = results
    for part in dotted.split("."):
        value = value[part]
    return value


def check_gate(results: dict[str, Any], gate: dict[str, dict[str, float]]) -> list[str]:
    failures = []
    for path, bound in gate.items():
        value = _get(results, path)
        if value is None:
            failures.append(f"{path}: no value")
        elif "min" in bound and value < bound["min"]:
            failures.append(f"{path}: {value} < min {bound['min']}")
        elif "max" in bound and value > bound["max"]:
            failures.append(f"{path}: {value} > max {bound['max']}")
    return failures


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if expected == 1 else (observed - expected) / (1 - expected)


async def run_judge_check(
    client: JudgeClient, path: str | Path = "evals/judge_check.jsonl", model: str = JUDGE_MODEL
) -> dict[str, Any]:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines()]
    sem = asyncio.Semaphore(4)

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            j = await judge_line(
                client,
                row["line"],
                row["recipient"],
                row["role"],
                row["company"],
                row["source"],
                row["source_text"],
                model=model,
            )
        return {
            "line": row["line"],
            "expected_pass": row["expected_pass"],
            "judge_pass": j.passed,
            "scores": j.scores,
            "rationale": j.rationale,
            "cost_usd": j.cost_usd,
        }

    judged = await asyncio.gather(*(one(r) for r in rows))
    expected = [r["expected_pass"] for r in judged]
    got = [r["judge_pass"] for r in judged]
    kappa = cohen_kappa(expected, got)
    return {
        "model": model,
        "n": len(judged),
        "agreement": round(
            sum(e == g for e, g in zip(expected, got, strict=True)) / len(judged), 3
        ),
        "cohen_kappa": round(kappa, 3) if kappa is not None else None,
        "false_pass": sum(g and not e for e, g in zip(expected, got, strict=True)),
        "false_fail": sum(e and not g for e, g in zip(expected, got, strict=True)),
        "cost_usd": round(sum(r["cost_usd"] for r in judged), 6),
        "rows": judged,
    }
