"""A scripted 'oracle' agent for harness tests: it reads the golden labels (cheating on
purpose) and drives the real tools. If the harness is wired correctly, the oracle must
score perfectly on every labelled field and route exactly the should-review leads."""

from __future__ import annotations

import json
import re
from typing import Any

from leadgenie.agent import SUBMIT_TOOL
from leadgenie.evals.golden import GoldenLead
from tests.fakes import message, sourced, tool_use


def _lead_name(req: dict[str, Any]) -> str:
    first = req["messages"][0]["content"]
    return re.search(r"- name: (.+)", first).group(1).strip()


def make_oracle(
    golden: list[GoldenLead],
    world: dict[str, Any],
    confidence: float = 0.9,
    fooled_unless: str | None = None,
):
    """fooled_unless: if set, look-alike leads resolve to the *wrong* twin company unless
    this text appears in the system prompt. Used to test that the improve loop finds and
    keeps a prompt change that fixes a real failure."""
    by_name = {g.name: g for g in golden}

    def script(req: dict[str, Any]):
        g = by_name[_lead_name(req)]
        if fooled_unless and g.scenario == "lookalike" and fooled_unless not in req["system"]:
            g = _twin(g)
        lab = g.labels
        if len(req["messages"]) == 1:
            if lab.domain is None:
                return message(tool_use("web_search", query=g.raw_company))
            return message(
                tool_use("fetch_company_site", url=f"https://{lab.domain}/"),
                tool_use("fetch_company_site", url=f"https://{lab.domain}/team"),
                tool_use("check_mx", domain=lab.domain),
            )
        return message(tool_use(SUBMIT_TOOL, **_submission(g, world, confidence)))

    return script


def _twin(g: GoldenLead) -> GoldenLead:
    from dataclasses import replace

    from leadgenie.evals.golden import LOOKALIKES, slug

    name, industry, _ = LOOKALIKES[g.labels.company]
    labels = replace(g.labels, company=name, domain=f"{slug(name)}.example", industry=industry)
    return replace(g, labels=labels)


def _submission(g: GoldenLead, world: dict[str, Any], conf: float) -> dict[str, Any]:
    lab = g.labels
    first = g.name.split()[0]
    if lab.domain is None:  # unidentifiable: honest low-confidence guess
        return {
            "company": sourced(g.raw_company, "input", "", 0.2),
            "domain": sourced(None, "inferred", "", 0.1),
            "role": sourced(lab.role, "input", "", 0.8),
            "seniority": sourced(lab.seniority, "inferred", "", 0.6),
            "industry": sourced("other", "inferred", "", 0.1),
            "accepts_email": sourced(None, "inferred", "", 0.1),
            "outreach": {"text": f"{first}, quick note.", "source": "inferred", "evidence": ""},
        }
    home = f"https://{lab.domain}/"
    team = f"{home}team"
    desc = world["pages"][home]["description"]
    blurb = desc.removeprefix(lab.company).strip(" .")
    if lab.role is not None and g.title:
        role = sourced(lab.role, "input", "", conf)
        seniority = sourced(lab.seniority, "inferred", "", 0.7)
    elif lab.role is not None:
        role = sourced(lab.role, team, f"{g.name}, {lab.role}", conf)
        seniority = sourced(lab.seniority, "inferred", "", 0.7)
    else:
        role = sourced("unknown", "inferred", "", 0.3)
        seniority = sourced("unknown", "inferred", "", 0.3)
    mx = json.dumps(bool(lab.accepts_email)).capitalize()
    return {
        "company": sourced(lab.company, home, lab.company, conf),
        "domain": sourced(lab.domain, home, f"hello@{lab.domain}", conf),
        "role": role,
        "seniority": seniority,
        "industry": sourced(lab.industry, home, blurb, conf),
        "accepts_email": sourced(
            lab.accepts_email, f"dns:mx:{lab.domain}", f"'has_mx': {mx}", conf
        ),
        "outreach": {
            "text": f"{first}, {lab.company} {blurb.split(' for ')[0]}: what is hardest to scale?",
            "source": home,
            "evidence": blurb,
        },
    }
