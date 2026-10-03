"""Business-rule validation + confidence routing.

The API guarantees the *shape* of the output; this layer enforces what the schema can't:
domain rules, and the human-in-the-loop threshold. A lead never silently enters results —
it is either approved by these rules or lands in the review queue with reasons attached.
"""

from typing import get_args

from leadgenie.models import EnrichedLead, Industry

DEFAULT_THRESHOLD = 0.75

# Derived from the schema enum — kept as a runtime check anyway (defense in depth:
# if the schema constraint ever loosens, this layer still catches drift).
ALLOWED_INDUSTRIES = set(get_args(Industry))

GENERIC_PHRASES = ("i hope this finds you well", "i came across your profile", "quick question")


def business_rule_violations(enriched: EnrichedLead) -> list[str]:
    violations = []
    if not enriched.role.strip():
        violations.append("role_empty")
    if not enriched.normalized_company.strip():
        violations.append("company_empty")
    if enriched.industry not in ALLOWED_INDUSTRIES:
        violations.append(f"industry_not_allowed:{enriched.industry}")
    line = enriched.outreach_line.strip()
    if not line:
        violations.append("outreach_empty")
    elif len(line) > 220:
        violations.append("outreach_too_long")
    elif any(p in line.lower() for p in GENERIC_PHRASES):
        violations.append("outreach_generic")
    return violations


def route(enriched: EnrichedLead, threshold: float = DEFAULT_THRESHOLD) -> tuple[str, list[str]]:
    """Returns ("approved" | "review", reasons)."""
    reasons = business_rule_violations(enriched)
    if enriched.confidence < threshold:
        reasons.append(f"low_confidence:{enriched.confidence:.2f}<{threshold}")
    return ("review" if reasons else "approved"), reasons
