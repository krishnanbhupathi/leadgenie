"""Business-rule validation + confidence routing.

The tool schema guarantees the *shape* of a submission; this layer enforces what a schema
can't: that cited sources were really retrieved, that quoted evidence really appears in
them, domain rules, and the human-in-the-loop threshold. A lead never silently enters
results — it is either approved by these rules or lands in the review queue with
machine-readable reasons attached.
"""

from __future__ import annotations

from typing import Any, get_args

from leadgenie.models import (
    INFERRED,
    INPUT,
    Enrichment,
    Industry,
    Outreach,
    Sourced,
    is_retrieved_source,
)
from leadgenie.tools.base import EvidenceLog

DEFAULT_THRESHOLD = 0.75

# An unsourced ("inferred") value can never carry enough confidence on its own to skip
# review: its effective confidence is capped below the default threshold.
INFERRED_CONFIDENCE_CAP = 0.6

MAX_OUTREACH_CHARS = 220

# Derived from the schema enum — kept as a runtime check anyway (defense in depth:
# if the schema constraint ever loosens, this layer still catches drift).
ALLOWED_INDUSTRIES = set(get_args(Industry))

GENERIC_PHRASES = (
    "i hope this finds you well",
    "i came across your profile",
    "quick question",
    "i wanted to reach out",
    "touch base",
)


def provenance_violations(enrichment: Enrichment, evidence: EvidenceLog) -> list[str]:
    """Every cited source must have been retrieved in this run, and every quote found in it."""
    violations = []
    cited: list[tuple[str, Sourced[Any] | Outreach]] = [
        (name, enrichment.field(name)) for name in Enrichment.SOURCED_FIELDS
    ]
    cited.append(("outreach", enrichment.outreach))
    for name, item in cited:
        source = item.source.strip()
        if source in (INFERRED, INPUT):
            continue
        if not is_retrieved_source(source):
            violations.append(f"bad_source:{name}")
        elif not evidence.has_url(source):
            violations.append(f"uncited_source:{name}")  # cites a URL no tool returned
        elif not evidence.supports(source, item.evidence):
            violations.append(f"evidence_mismatch:{name}")  # quote not found in that page
    return violations


def business_rule_violations(enrichment: Enrichment) -> list[str]:
    violations = []
    if not enrichment.role.value.strip():
        violations.append("role_empty")
    if not enrichment.company.value.strip():
        violations.append("company_empty")
    if enrichment.industry.value not in ALLOWED_INDUSTRIES:
        violations.append(f"industry_not_allowed:{enrichment.industry.value}")
    if enrichment.accepts_email.value is False:
        violations.append("domain_rejects_mail")
    line = enrichment.outreach.text.strip()
    if not line:
        violations.append("outreach_empty")
    elif len(line) > MAX_OUTREACH_CHARS:
        violations.append("outreach_too_long")
    elif any(p in line.lower() for p in GENERIC_PHRASES):
        violations.append("outreach_generic")
    if line and not is_retrieved_source(enrichment.outreach.source):
        violations.append("outreach_ungrounded")  # personalization must point at a real fact
    return violations


def effective_confidence(enrichment: Enrichment, name: str) -> float:
    item = enrichment.field(name)
    if item.source == INFERRED:
        return min(item.confidence, INFERRED_CONFIDENCE_CAP)
    return item.confidence


def routing_confidence(enrichment: Enrichment) -> float:
    return min(effective_confidence(enrichment, n) for n in Enrichment.KEY_FIELDS)


def route(
    enrichment: Enrichment, evidence: EvidenceLog, threshold: float = DEFAULT_THRESHOLD
) -> tuple[str, list[str]]:
    """Returns ("approved" | "review", reasons)."""
    reasons = provenance_violations(enrichment, evidence) + business_rule_violations(enrichment)
    confidence = routing_confidence(enrichment)
    if confidence < threshold:
        reasons.append(f"low_confidence:{confidence:.2f}<{threshold}")
    return ("review" if reasons else "approved"), reasons
