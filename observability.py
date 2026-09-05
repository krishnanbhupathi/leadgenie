"""Per-lead observability: model, tokens, cost, latency, confidence, outcome → runs.jsonl.

This is how I watch the agent: every lead leaves a structured trace, and the run ends
with a summary that answers "what did this cost and can I trust it?"
"""
import json
from datetime import datetime, timezone
from typing import List, Optional

# USD per 1M tokens (input, output)
PRICES = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def make_record(lead, model: str, status: str, reasons: List[str],
                enriched=None, usage=None, latency_ms: Optional[int] = None,
                error: Optional[str] = None) -> dict:
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "lead_id": lead.id,
        "name": lead.name,
        "model": model,
        "status": status,  # approved | review | error
        "reasons": reasons,
        "confidence": enriched.confidence if enriched else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd(model, input_tokens, output_tokens), 6),
        "latency_ms": latency_ms,
        "error": error,
    }


def log_record(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def print_summary(records: List[dict], skipped: int) -> None:
    if not records and not skipped:
        print("Nothing to do.")
        return
    approved = [r for r in records if r["status"] == "approved"]
    review = [r for r in records if r["status"] == "review"]
    errors = [r for r in records if r["status"] == "error"]
    confidences = [r["confidence"] for r in records if r["confidence"] is not None]
    total_cost = sum(r["cost_usd"] for r in records)
    latencies = [r["latency_ms"] for r in records if r["latency_ms"] is not None]
    n = len(records)

    print("\n" + "=" * 52)
    print("RUN SUMMARY")
    print("=" * 52)
    print(f"  processed:       {n}  (skipped {skipped} already done)")
    if n:
        print(f"  auto-approved:   {len(approved)}  ({100 * len(approved) / n:.0f}%)")
        print(f"  human review:    {len(review)}  ({100 * len(review) / n:.0f}%)")
        print(f"  errors:          {len(errors)}")
    if confidences:
        print(f"  avg confidence:  {sum(confidences) / len(confidences):.2f}")
    if latencies:
        print(f"  avg latency:     {sum(latencies) / len(latencies) / 1000:.1f}s per lead")
    print(f"  total cost:      ${total_cost:.4f}")
    print("=" * 52)
