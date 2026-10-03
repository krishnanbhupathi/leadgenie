"""The agent: one lead in → one validated EnrichedLead out, with retries + backoff."""

import time

import anthropic
from pydantic import ValidationError

from leadgenie.models import EnrichedLead, Lead

DEFAULT_MODEL = "claude-opus-4-8"
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """You are a lead-enrichment agent for a B2B sales prospecting tool.

Given one messy lead (name, raw company string, maybe a job title) you must:
1. Normalize the company name — fix casing/typos, drop legal suffixes (Inc, GmbH, Ltd, ...).
2. Infer the person's role, seniority, and the company's industry.
3. Write ONE personalized cold-outreach opening line. It must reference something specific
   about the company or role — never a generic line that could be sent to anyone.
4. Report a calibrated confidence in [0, 1]:
   - Only give >= 0.75 when the company is clearly identifiable and the role is explicit.
   - If you are guessing the company identity, industry, or role, confidence MUST be below 0.75.
   - Never invent facts to make an outreach line more specific. A vaguer line with honest
     confidence beats a confident hallucination.

Base everything only on the input text and general knowledge. Do not fabricate funding rounds,
product launches, or news you are not sure about."""


class EnrichmentError(Exception):
    """Raised when the agent cannot produce a valid EnrichedLead after all retries."""


def _build_prompt(lead: Lead) -> str:
    return (
        "Enrich this lead:\n"
        f"- name: {lead.name}\n"
        f"- raw_company: {lead.raw_company}\n"
        f"- title: {lead.title or '(missing)'}"
    )


def enrich_lead(client: anthropic.Anthropic, lead: Lead, model: str = DEFAULT_MODEL):
    """Returns (enriched, usage, latency_ms). Raises EnrichmentError after MAX_ATTEMPTS.

    Structured output: the API is forced to emit JSON matching EnrichedLead's schema,
    and the SDK parses it into the Pydantic model — free text is impossible by construction.
    """
    last_err = None
    for attempt in range(MAX_ATTEMPTS):
        start = time.monotonic()
        try:
            response = client.messages.parse(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _build_prompt(lead)}],
                output_format=EnrichedLead,
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            enriched = response.parsed_output
            if enriched is None:
                raise ValueError(f"no parsed output (stop_reason={response.stop_reason})")
            return enriched, response.usage, latency_ms
        except (
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            ValidationError,
            ValueError,
        ) as err:
            last_err = err
            delay = 2**attempt  # 1s, 2s, 4s
            print(f"    retry {attempt + 1}/{MAX_ATTEMPTS} for {lead.id} in {delay}s: {err}")
            time.sleep(delay)
    raise EnrichmentError(f"lead {lead.id} failed after {MAX_ATTEMPTS} attempts: {last_err}")
