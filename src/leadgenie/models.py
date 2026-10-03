"""Pydantic schemas — every LLM output must parse into EnrichedLead or it is rejected."""
import hashlib
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Lead(BaseModel):
    """One messy input row from leads.csv."""

    id: str
    name: str
    raw_company: str
    title: Optional[str] = None

    @classmethod
    def from_row(cls, row: dict) -> "Lead":
        name = (row.get("name") or "").strip()
        raw_company = (row.get("raw_company") or "").strip()
        title = (row.get("title") or "").strip() or None
        # Stable ID from the row content itself → idempotency across re-runs
        digest = hashlib.sha1(f"{name}|{raw_company}".lower().encode()).hexdigest()[:12]
        return cls(id=digest, name=name, raw_company=raw_company, title=title)


Seniority = Literal[
    "founder", "c_level", "vp", "director", "manager", "senior_ic", "ic", "unknown"
]

# Single source of truth for industry labels. Baking this into the schema means the API
# *cannot* return an off-list label — v1 left industry as free text and the model drifted
# ("enterprise software", "food delivery"), inflating the human-review rate.
Industry = Literal[
    "saas", "fintech", "ecommerce", "healthcare", "agency", "manufacturing",
    "education", "real_estate", "logistics", "delivery", "media", "consulting",
    "cybersecurity", "ai", "hr_tech", "legal", "energy", "hospitality",
    "dev_tools", "it_services", "other",
]


class EnrichedLead(BaseModel):
    """The contract the agent must fulfil. The API enforces the shape (JSON schema);
    Pydantic enforces the constraints the API can't (e.g. confidence in [0, 1])."""

    normalized_company: str = Field(
        description="Clean, canonical company name (no legal suffixes, typos fixed)."
    )
    role: str = Field(description="The person's inferred job function, e.g. 'Head of Sales'.")
    seniority: Seniority
    industry: Industry = Field(
        description="Closest matching industry label; use 'other' only if nothing fits."
    )
    outreach_line: str = Field(
        description="One personalized, non-generic cold-outreach opening line (max ~30 words)."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Calibrated confidence in this enrichment. Below 0.75 means a human should review it.",
    )
