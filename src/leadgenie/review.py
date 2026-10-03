"""Human review of the review queue. Every edit is stored as a Correction — the input to
the self-improving loop (improve.py)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from leadgenie.store import Correction, Store

REVIEW_FIELDS = ("company", "role", "seniority", "industry", "accepts_email")
VERDICT = "_verdict"  # pseudo-field recording that a human looked at the lead


def _parse(raw: str, current: Any) -> Any:
    raw = raw.strip()
    if raw == "-":
        return None
    if isinstance(current, bool) or (current is None and raw.lower() in ("true", "false")):
        return raw.lower() in ("true", "yes", "y")
    return raw


def pending(store: Store) -> list[Any]:
    reviewed = {c.lead_id for c in store.corrections("human")}
    queue = store.results("review") + store.results("error")
    return [r for r in queue if r["lead_id"] not in reviewed]


def review_queue(
    store: Store,
    reviewer: str,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
    limit: int | None = None,
) -> int:
    """Walk pending review items; returns the number of corrections recorded."""
    recorded = 0
    for row in pending(store)[:limit]:
        enrichment = json.loads(row["enrichment"]) if row["enrichment"] else {}
        say(f"\n[{row['lead_id']}] {row['name']} @ {row['raw_company']!r} title={row['title']!r}")
        say(f"  reasons: {', '.join(json.loads(row['reasons']))}")
        for name in REVIEW_FIELDS:
            item = enrichment.get(name) or {}
            current = item.get("value")
            say(
                f"  {name:<14} {current!r:<32} "
                f"src={item.get('source', 'n/a')} conf={item.get('confidence', 'n/a')}"
            )
            answer = ask(f"  {name} [Enter=ok, '-'=unknown, or correct value]: ")
            if not answer.strip():
                continue
            corrected = _parse(answer, current)
            if corrected == current:
                continue
            note = ask("  why? (one line, optional): ").strip()
            store.add_correction(
                Correction(
                    lead_id=row["lead_id"],
                    raw_company=row["raw_company"],
                    title=row["title"],
                    field=name,
                    predicted=current,
                    predicted_source=item.get("source"),
                    corrected=corrected,
                    note=note,
                    origin="human",
                    reviewer=reviewer,
                )
            )
            recorded += 1
        verdict = ask("  verdict [a=usable after corrections / r=reject]: ").strip().lower()
        store.add_correction(
            Correction(
                lead_id=row["lead_id"],
                raw_company=row["raw_company"],
                title=row["title"],
                field=VERDICT,
                predicted=row["status"],
                predicted_source=None,
                corrected="rejected" if verdict.startswith("r") else "approved",
                note="",
                origin="human",
                reviewer=reviewer,
            )
        )
    return recorded
