"""Outreach-line evaluation: deterministic checks + a rubric-based LLM judge.

Deterministic checks are cheap, exact and run on every line. The judge covers what code
can't (is this specific? relevant to the role?) and is itself checked against a small
hand-labelled set (evals/judge_check.jsonl) so its agreement with a human is a measured
number, not an assumption.

The judge defaults to a different model family tier than the agent (Sonnet judging
Opus output) to reduce self-preference bias.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from leadgenie.models import Enrichment, is_retrieved_source
from leadgenie.pricing import TokenUsage, cost_usd
from leadgenie.tools.base import EvidenceLog
from leadgenie.validate import GENERIC_PHRASES, MAX_OUTREACH_CHARS

JUDGE_MODEL = "claude-sonnet-5-5"
MAX_WORDS = 30
SOURCE_CHARS = 3_000


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d[\d,.]*\d|\d", text))


def deterministic_checks(
    line: str,
    source: str,
    evidence_quote: str,
    evidence: EvidenceLog,
    first_name: str,
    company: str,
) -> dict[str, bool]:
    low = line.lower()
    source_text = evidence.documents.get(source, "")
    grounded = is_retrieved_source(source) and evidence.supports(source, evidence_quote)
    checks = {
        "length_ok": 0 < len(line) <= MAX_OUTREACH_CHARS,
        "word_count_ok": len(line.split()) <= MAX_WORDS,
        "not_generic": not any(p in low for p in GENERIC_PHRASES),
        "personalized": first_name.lower() in low or company.lower() in low,
        "grounded": grounded,
        # Any figure in the line ("40 clinics", "$3M") must appear in the cited page.
        "no_unsupported_numbers": _numbers(line) <= _numbers(source_text),
        "no_placeholders": not re.search(r"[\[\]{}<>]|TODO|XXX", line),
    }
    checks["all_pass"] = all(checks.values())
    return checks


JUDGE_SYSTEM = """\
You grade one cold-outreach opening line written by a sales-research agent.
Score each criterion from 1 (bad) to 5 (excellent):

- specificity: refers to a concrete fact about this company or role that could not be \
pasted into an email to a different company.
- grounding: every factual claim in the line is supported by the SOURCE TEXT. Score 1 if \
anything is invented or contradicts it; 5 if all claims are clearly supported.
- relevance: connects the fact to the recipient's role in a way that invites a reply.
- tone: concise, natural, respectful, not pushy or salesy clichés.

Judge only what is given; do not assume facts not present in the source text. \
Give a one-sentence rationale."""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "specificity": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "grounding": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "relevance": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "tone": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "rationale": {"type": "string"},
    },
    "required": ["specificity", "grounding", "relevance", "tone", "rationale"],
    "additionalProperties": False,
}

CRITERIA = ("specificity", "grounding", "relevance", "tone")


def judge_pass(scores: dict[str, int]) -> bool:
    """Grounding and specificity are what make a line usable; the rest must not be poor."""
    return (
        scores["grounding"] >= 4
        and scores["specificity"] >= 4
        and min(scores[c] for c in CRITERIA) >= 3
    )


class JudgeClient(Protocol):
    @property
    def messages(self) -> Any: ...


@dataclass
class JudgeResult:
    scores: dict[str, int]
    rationale: str
    passed: bool
    cost_usd: float


def judge_prompt(
    line: str, recipient: str, role: str, company: str, source: str, source_text: str
) -> str:
    return (
        f"RECIPIENT: {recipient}, {role} at {company}\n"
        f"LINE: {line}\n"
        f"CITED SOURCE: {source}\n"
        f"SOURCE TEXT:\n{source_text[:SOURCE_CHARS] or '(nothing was retrieved from this source)'}"
    )


async def judge_line(
    client: JudgeClient,
    line: str,
    recipient: str,
    role: str,
    company: str,
    source: str,
    source_text: str,
    model: str = JUDGE_MODEL,
) -> JudgeResult:
    response = await client.messages.create(
        model=model,
        max_tokens=4_000,
        system=JUDGE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": judge_prompt(line, recipient, role, company, source, source_text),
            }
        ],
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": JUDGE_SCHEMA},
        },
    )
    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    scores = {c: int(data[c]) for c in CRITERIA}
    cost = cost_usd(model, TokenUsage.from_api(response.usage))
    return JudgeResult(scores, data["rationale"], judge_pass(scores), cost)


def judge_inputs(enrichment: Enrichment, evidence: EvidenceLog, recipient: str) -> dict[str, str]:
    o = enrichment.outreach
    return {
        "line": o.text,
        "recipient": recipient,
        "role": enrichment.role.value,
        "company": enrichment.company.value,
        "source": o.source,
        "source_text": evidence.documents.get(o.source, ""),
    }
