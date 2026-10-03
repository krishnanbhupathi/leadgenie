"""Pydantic schemas — input leads, and the per-field-sourced enrichment the agent must submit."""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field


class Lead(BaseModel):
    """One messy input row from leads.csv."""

    id: str
    name: str
    raw_company: str
    title: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Lead:
        name = (row.get("name") or "").strip()
        raw_company = (row.get("raw_company") or "").strip()
        title = (row.get("title") or "").strip() or None
        # Stable ID from the row content itself → idempotency across re-runs
        digest = hashlib.sha1(f"{name}|{raw_company}".lower().encode()).hexdigest()[:12]
        return cls(id=digest, name=name, raw_company=raw_company, title=title)


Seniority = Literal["founder", "c_level", "vp", "director", "manager", "senior_ic", "ic", "unknown"]

# Single source of truth for industry labels. Baking this into the schema means the API
# *cannot* return an off-list label — v1 left industry as free text and the model drifted
# ("enterprise software", "food delivery"), inflating the human-review rate.
Industry = Literal[
    "saas",
    "fintech",
    "ecommerce",
    "healthcare",
    "agency",
    "manufacturing",
    "education",
    "real_estate",
    "logistics",
    "delivery",
    "media",
    "consulting",
    "cybersecurity",
    "ai",
    "hr_tech",
    "legal",
    "energy",
    "hospitality",
    "dev_tools",
    "it_services",
    "other",
]

# Where a value came from. Exactly one of:
#   https://...        a page a tool retrieved during this lead's run
#   prior:<lead_id>    an approved earlier result (lookup_prior_results)
#   dns:mx:<domain>    an MX lookup (check_mx)
#   input              copied/normalized from the lead row itself
#   inferred           the model's own judgement, with no retrieved evidence
INFERRED = "inferred"
INPUT = "input"
RETRIEVED_PREFIXES = ("http://", "https://", "prior:", "dns:mx:")


def is_retrieved_source(source: str) -> bool:
    return source.startswith(RETRIEVED_PREFIXES)


class Sourced[T](BaseModel):
    value: T
    source: str = Field(description="Retrieved URL/ref, 'input', or 'inferred'.")
    evidence: str = Field(
        default="", description="Short verbatim quote from the source; empty for input/inferred."
    )
    confidence: float = Field(ge=0.0, le=1.0)


class Outreach(BaseModel):
    text: str
    source: str
    evidence: str = ""


class Enrichment(BaseModel):
    company: Sourced[str]
    domain: Sourced[str | None]
    role: Sourced[str]
    seniority: Sourced[Seniority]
    industry: Sourced[Industry]
    accepts_email: Sourced[bool | None]
    outreach: Outreach

    SOURCED_FIELDS: ClassVar[tuple[str, ...]] = (
        "company",
        "domain",
        "role",
        "seniority",
        "industry",
        "accepts_email",
    )
    # Routing confidence is the weakest of these: a lead is only as good as its least
    # certain identifying field.
    KEY_FIELDS: ClassVar[tuple[str, ...]] = ("company", "role", "industry")

    def field(self, name: str) -> Sourced[Any]:
        value = getattr(self, name)
        assert isinstance(value, Sourced)
        return value


# --- JSON schema for the submit_enrichment tool --------------------------------------
# Hand-written (rather than Enrichment.model_json_schema()) so it satisfies strict tool
# use: every property required, additionalProperties false, no $refs. Numeric ranges
# are enforced by Pydantic after the call; a violation goes back to the model as a
# tool error it can fix.


def _sourced_schema(value_schema: dict[str, Any], what: str) -> dict[str, Any]:
    return {
        "type": "object",
        "description": what,
        "properties": {
            "value": value_schema,
            "source": {"type": "string"},
            "evidence": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["value", "source", "evidence", "confidence"],
        "additionalProperties": False,
    }


SENIORITY_VALUES = list(Seniority.__args__)
INDUSTRY_VALUES = list(Industry.__args__)

SUBMIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "company": _sourced_schema(
            {"type": "string"}, "Canonical company name (no legal suffix, typos fixed)."
        ),
        "domain": _sourced_schema(
            {"type": ["string", "null"]},
            "Company website domain, e.g. 'acme.example'; null if unknown.",
        ),
        "role": _sourced_schema({"type": "string"}, "The person's job function."),
        "seniority": _sourced_schema({"type": "string", "enum": SENIORITY_VALUES}, "Seniority."),
        "industry": _sourced_schema(
            {"type": "string", "enum": INDUSTRY_VALUES}, "Closest industry; 'other' if none fits."
        ),
        "accepts_email": _sourced_schema(
            {"type": ["boolean", "null"]},
            "Does the company domain have MX records? null if unchecked.",
        ),
        "outreach": {
            "type": "object",
            "description": "One personalized opening line grounded in a retrieved fact.",
            "properties": {
                "text": {"type": "string"},
                "source": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["text", "source", "evidence"],
            "additionalProperties": False,
        },
    },
    "required": ["company", "domain", "role", "seniority", "industry", "accepts_email", "outreach"],
    "additionalProperties": False,
}
