"""Scoring: per-field accuracy, confidence calibration, review-queue precision/recall.

Definitions (also stated in the README so the numbers can be read correctly):

* Field accuracy — over leads whose label for that field is known. A lead the agent
  produced no enrichment for (budget exhausted, API error) counts as wrong; coverage is
  reported separately.
* Calibration — every (field confidence, field correct) pair with a known label. ECE is
  the bin-weighted mean |accuracy - mean confidence| over 10 equal-width bins.
* Needs review (ground truth) — the label says so (unidentifiable company, no MX,
  unknowable role), OR the agent got a key field (company, role, industry) wrong. A wrong
  answer is exactly what the review queue exists to catch.
* Review precision/recall — predicted positive = routed to review or error.
* Unsafe approval rate — share of auto-approved leads that needed review. This is the
  number that matters most for a sales team: a confidently wrong lead in the CRM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from leadgenie.evals.golden import GoldenLead
from leadgenie.models import Enrichment

FIELDS = ("company", "domain", "role", "seniority", "industry", "accepts_email")
KEY_FIELDS = ("company", "role", "industry")

_LEGAL = {"inc", "ltd", "llc", "gmbh", "pty", "sas", "bv", "sl", "llp", "co", "plc", "corp"}
_ROLE_STOP = {"of", "the", "and"}
_ROLE_EXPAND = {"sr": "senior", "eng": "engineering", "mgr": "manager", "vice": "vp"}


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def norm_company(s: str) -> str:
    return " ".join(w for w in _words(s) if w not in _LEGAL)


def norm_role(s: str) -> str:
    words = [_ROLE_EXPAND.get(w, w) for w in _words(s) if w not in _ROLE_STOP]
    return " ".join(w for w in words if w != "president" or "vp" not in words)


def norm_domain(s: str) -> str:
    s = s.strip().lower().removeprefix("https://").removeprefix("http://")
    return s.split("/")[0].removeprefix("www.")


def field_correct(name: str, predicted: Any, g: GoldenLead) -> bool | None:
    """None when the label is unknown (field excluded from accuracy/calibration)."""
    label = getattr(g.labels, name)
    if label is None:
        return None
    if predicted is None:
        return False
    if name == "company":
        return norm_company(str(predicted)) == norm_company(label)
    if name == "domain":
        return norm_domain(str(predicted)) == norm_domain(label)
    if name == "role":
        accepted = {norm_role(label), *(norm_role(a) for a in g.labels.role_aliases)}
        return norm_role(str(predicted)) in accepted
    return bool(predicted == label)


@dataclass
class Prediction:
    golden: GoldenLead
    status: str  # approved | review | error
    enrichment: Enrichment | None
    cost_usd: float
    latency_ms: int | None
    steps: int
    stop: str


def per_field(preds: list[Prediction]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in FIELDS:
        labelled = [p for p in preds if getattr(p.golden.labels, name) is not None]
        correct = 0
        covered = 0
        for p in labelled:
            value = p.enrichment.field(name).value if p.enrichment else None
            covered += p.enrichment is not None
            correct += bool(field_correct(name, value, p.golden))
        out[name] = {
            "n": len(labelled),
            "accuracy": round(correct / len(labelled), 3) if labelled else None,
            "coverage": round(covered / len(labelled), 3) if labelled else None,
        }
    return out


def calibration_pairs(preds: list[Prediction]) -> list[tuple[float, bool]]:
    pairs = []
    for p in preds:
        if p.enrichment is None:
            continue
        for name in FIELDS:
            item = p.enrichment.field(name)
            ok = field_correct(name, item.value, p.golden)
            if ok is not None:
                pairs.append((item.confidence, ok))
    return pairs


def reliability(
    pairs: list[tuple[float, bool]], bins: int = 10
) -> tuple[float | None, list[dict[str, Any]]]:
    """Returns (ECE, table). Bin i covers [i/bins, (i+1)/bins); 1.0 goes in the last bin."""
    if not pairs:
        return None, []
    table = []
    ece = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        members = [(c, ok) for c, ok in pairs if lo <= c < hi or (i == bins - 1 and c == 1.0)]
        if not members:
            continue
        conf = sum(c for c, _ in members) / len(members)
        acc = sum(ok for _, ok in members) / len(members)
        ece += len(members) / len(pairs) * abs(acc - conf)
        table.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": len(members),
                "mean_confidence": round(conf, 3),
                "accuracy": round(acc, 3),
                "gap": round(acc - conf, 3),
            }
        )
    return round(ece, 4), table


def needs_review(p: Prediction) -> bool:
    if p.golden.labels.should_review:
        return True
    if p.enrichment is None:
        return True  # nothing usable was produced
    return any(field_correct(n, p.enrichment.field(n).value, p.golden) is False for n in KEY_FIELDS)


def review_metrics(preds: list[Prediction]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for p in preds:
        predicted = p.status in ("review", "error")
        truth = needs_review(p)
        tp += predicted and truth
        fp += predicted and not truth
        fn += truth and not predicted
        tn += not predicted and not truth
    approved = tn + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "f1": round(f1, 3) if f1 is not None else None,
        "review_rate": round((tp + fp) / len(preds), 3) if preds else None,
        "unsafe_approval_rate": round(fn / approved, 3) if approved else None,
    }


def by_scenario(preds: list[Prediction]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for scenario in sorted({p.golden.scenario for p in preds}):
        group = [p for p in preds if p.golden.scenario == scenario]
        out[scenario] = {
            "n": len(group),
            "routed_to_review": sum(p.status != "approved" for p in group),
            "needs_review": sum(needs_review(p) for p in group),
            "key_fields_all_correct": sum(
                p.enrichment is not None
                and all(
                    field_correct(n, p.enrichment.field(n).value, p.golden) is not False
                    for n in KEY_FIELDS
                )
                for p in group
            ),
        }
    return out
