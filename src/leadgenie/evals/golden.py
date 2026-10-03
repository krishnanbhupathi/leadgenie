"""Deterministic generator for the synthetic golden set and its offline web ("world").

Every company, person and page here is fictional; all domains are under the reserved
.example TLD, so nothing can collide with a real organisation or resolve on the internet.

Each lead belongs to one scenario, which fixes what the right answer *is*:

  clean          company and title clear; the person is listed on the team page
  messy_company  typos / casing / legal suffixes in the raw company string
  title_on_site  no title in the row, but the company's /team page lists the person
  lookalike      a similarly named company in another industry also exists
  no_mx          real company whose domain has no MX record → must go to review
  unidentifiable "stealth startup", "acme corp", ... → no site exists → must go to review
  title_unknown  no title in the row and the person is not on the site → must go to review

Labels are None where the truth is unknowable from the world; those fields are excluded
from accuracy/calibration and the lead is instead expected in the review queue.

Run:  python -m leadgenie.evals.golden  (rewrites evals/golden.jsonl and evals/world.json)
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SEED = 20261003
SPLITS = {"train": 30, "dev": 15, "test": 15}

# name, legal suffix, industry, what they do (one sentence, used on the homepage)
COMPANIES: list[tuple[str, str, str, str]] = [
    (
        "Brightwave Ledger",
        "Inc.",
        "fintech",
        "automates accounts-payable reconciliation for mid-market finance teams",
    ),
    (
        "Corvid Shield",
        "Ltd",
        "cybersecurity",
        "detects credential-stuffing attacks against customer login pages",
    ),
    (
        "Tallowmere Freight",
        "GmbH",
        "logistics",
        "runs cross-border LTL freight brokerage between Germany and Poland",
    ),
    (
        "Pinecrest Health Systems",
        "LLC",
        "healthcare",
        "builds scheduling software for outpatient physiotherapy clinics",
    ),
    (
        "Quillfeather Learning",
        "Inc.",
        "education",
        "makes adaptive maths practice for secondary-school students",
    ),
    (
        "Saltmarsh Commerce",
        "Pty Ltd",
        "ecommerce",
        "operates a marketplace for independent ceramic studios",
    ),
    ("Orbitline Devtools", "Inc.", "dev_tools", "offers a CI cache that cuts monorepo build times"),
    (
        "Hearthstone Realty Labs",
        "LLC",
        "real_estate",
        "provides valuation models for small multifamily properties",
    ),
    (
        "Juniper Loop",
        "SAS",
        "hr_tech",
        "runs structured onboarding programs for distributed engineering teams",
    ),
    (
        "Vellum & Vane",
        "LLP",
        "legal",
        "provides contract-review services for venture-backed startups",
    ),
    (
        "Sundial Grid",
        "Inc.",
        "energy",
        "forecasts rooftop-solar output for regional grid operators",
    ),
    (
        "Copperleaf Kitchens",
        "Ltd",
        "hospitality",
        "runs a group of ghost kitchens serving three city centres",
    ),
    ("Nightjar Media", "Inc.", "media", "publishes a newsletter network covering climate policy"),
    (
        "Granite Peak Consulting",
        "LLC",
        "consulting",
        "advises manufacturers on lean production rollouts",
    ),
    (
        "Mosswood Robotics",
        "Inc.",
        "manufacturing",
        "builds collaborative welding robots for small fabrication shops",
    ),
    ("Lanternfish AI", "Inc.", "ai", "trains document-understanding models for insurance claims"),
    ("Driftwood Delivery", "BV", "delivery", "operates same-day grocery delivery by cargo bike"),
    (
        "Kestrel Cloudworks",
        "Inc.",
        "saas",
        "sells a field-service scheduling app for HVAC contractors",
    ),
    (
        "Bramblegate Agency",
        "Ltd",
        "agency",
        "runs performance-marketing campaigns for DTC skincare brands",
    ),
    (
        "Ironbark IT Partners",
        "Pty Ltd",
        "it_services",
        "provides managed IT and helpdesk for dental practices",
    ),
    ("Foxglove Payments", "Ltd", "fintech", "issues virtual cards for travel-expense management"),
    ("Starling Notes", "Inc.", "saas", "makes a meeting-notes tool for customer-success teams"),
    (
        "Riverstone Clinics",
        "LLC",
        "healthcare",
        "operates urgent-care clinics with same-day telehealth follow-up",
    ),
    ("Thistle Analytics", "Ltd", "ai", "builds demand-forecasting models for regional bakeries"),
    ("Wavecrest Security", "Inc.", "cybersecurity", "runs phishing simulations for hospital staff"),
    (
        "Amberline Logistics",
        "LLC",
        "logistics",
        "manages cold-chain warehousing for pharmaceutical distributors",
    ),
    (
        "Clearbrook Learning",
        "Ltd",
        "education",
        "teaches data-literacy courses to public-sector employees",
    ),
    (
        "Halcyon Stays",
        "SL",
        "hospitality",
        "manages boutique short-stay apartments in Lisbon and Porto",
    ),
    ("Ravenmoor Studio", "Ltd", "agency", "designs brand identities for craft breweries"),
    (
        "Tidewater Energy Labs",
        "Inc.",
        "energy",
        "develops battery-storage controllers for commercial buildings",
    ),
]

# Look-alike twins: (twin name, twin industry, twin description), keyed by the original.
LOOKALIKES = {
    "Kestrel Cloudworks": (
        "Kestrel Cloud Partners",
        "consulting",
        "advises banks on cloud migration",
    ),
    "Lanternfish AI": ("Lanternfish Aquatics", "ecommerce", "sells aquarium supplies online"),
    "Corvid Shield": ("Corvid Shields & Signs", "manufacturing", "fabricates metal shop signage"),
    "Starling Notes": ("Starling Note Co.", "ecommerce", "sells handmade paper notebooks"),
    "Sundial Grid": ("Sundial Grid Events", "hospitality", "runs rooftop event venues"),
}

UNIDENTIFIABLE = [
    "stealth startup",
    "acme corp",
    "xyz holdings",
    "confidential",
    "self-employed",
    "tbd",
]

# raw title variants → canonical role, seniority, accepted role aliases
TITLES: list[tuple[list[str], str, str, list[str]]] = [
    (["CTO", "cto", "Chief Technology Officer"], "Chief Technology Officer", "c_level", ["cto"]),
    (
        ["CEO & Founder", "founder/ceo", "Co-founder & CEO"],
        "Chief Executive Officer",
        "founder",
        ["ceo", "founder", "co-founder"],
    ),
    (
        ["vp eng", "VP Engineering", "VP of Engineering"],
        "VP of Engineering",
        "vp",
        ["vp engineering", "vice president engineering"],
    ),
    (["Head of Growth", "head of growth"], "Head of Growth", "director", ["growth lead"]),
    (
        ["Sr. AE", "Senior Account Executive"],
        "Senior Account Executive",
        "senior_ic",
        ["account executive", "ae"],
    ),
    (
        ["eng manager", "Engineering Manager"],
        "Engineering Manager",
        "manager",
        ["engineering manager"],
    ),
    (["Product Manager", "PM"], "Product Manager", "ic", ["product manager"]),
    (["Director of Sales", "sales director"], "Director of Sales", "director", ["sales director"]),
    (["COO", "Chief Operating Officer"], "Chief Operating Officer", "c_level", ["coo"]),
    (["Data Scientist", "data scientist"], "Data Scientist", "ic", ["data scientist"]),
    (["Head of Marketing", "marketing lead"], "Head of Marketing", "director", ["marketing lead"]),
    (["Ops Manager", "Operations Manager"], "Operations Manager", "manager", ["ops manager"]),
]

FIRST = [
    "Ada",
    "Bram",
    "Chiara",
    "Dmitri",
    "Esi",
    "Farah",
    "Goran",
    "Hana",
    "Ilse",
    "Jonas",
    "Keiko",
    "Luca",
    "Mira",
    "Nikhil",
    "Oona",
    "Pavel",
    "Quinn",
    "Rosa",
    "Soren",
    "Tamsin",
    "Uma",
    "Viktor",
    "Wen",
    "Xavi",
    "Yara",
    "Zeno",
]
LAST = [
    "Abernathy",
    "Brightwell",
    "Castellan",
    "Dunmore",
    "Ellery",
    "Fairbairn",
    "Galloway",
    "Hollins",
    "Ingram",
    "Jessop",
    "Kerrigan",
    "Lindqvist",
    "Marchetti",
    "Nakamura",
    "Okafor",
    "Pemberton",
    "Quayle",
    "Rasmussen",
    "Sterling",
    "Thorne",
    "Underhill",
    "Valdez",
    "Whitlock",
    "Yeoman",
    "Zielinski",
]

SCENARIO_COUNTS = {
    "clean": 24,
    "messy_company": 10,
    "title_on_site": 6,
    "lookalike": 5,
    "no_mx": 4,
    "unidentifiable": 6,
    "title_unknown": 5,
}


@dataclass
class Labels:
    company: str | None
    domain: str | None
    role: str | None
    role_aliases: list[str]
    seniority: str | None
    industry: str | None
    accepts_email: bool | None
    should_review: bool


@dataclass
class GoldenLead:
    lead_id: str
    split: str
    scenario: str
    name: str
    raw_company: str
    title: str | None
    labels: Labels
    notes: str = ""

    def row(self) -> dict[str, Any]:
        return {"name": self.name, "raw_company": self.raw_company, "title": self.title or ""}


@dataclass
class _Company:
    name: str
    suffix: str
    industry: str
    blurb: str
    domain: str
    team: list[tuple[str, str]] = field(default_factory=list)  # (person, display title)
    has_mx: bool = True


def slug(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def _messy(name: str, suffix: str, rng: random.Random) -> str:
    variants = [
        f"{name.lower()} {suffix.lower()}",
        f"{name.upper()} {suffix.upper()}",
        name.replace(" ", "").lower(),
        f"{name} ({suffix})",
    ]
    # one adjacent-letter swap inside the longest word
    words = name.split()
    i = max(range(len(words)), key=lambda k: len(words[k]))
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    variants.append(" ".join(words).lower())
    return rng.choice(variants)


def _pages(c: _Company, founded: int, city: str) -> dict[str, dict[str, Any]]:
    base = f"https://{c.domain}/"
    team_lines = "; ".join(f"{p}, {t}" for p, t in c.team) or "Our team page is being updated."
    return {
        base: {
            "title": f"{c.name} | {c.blurb.split(' for ')[0].capitalize()}",
            "description": f"{c.name} {c.blurb}.",
            "text": (
                f"{c.name} {c.blurb}. Founded in {founded} and based in {city}. "
                f"Contact: hello@{c.domain}. See /about and /team."
            ),
        },
        f"{base}about": {
            "title": f"About {c.name}",
            "text": (
                f"About us: {c.name} was founded in {founded} in {city}. "
                f"We {c.blurb.split(' ', 1)[1]}."
            ),
        },
        f"{base}team": {
            "title": f"Team | {c.name}",
            "text": f"Leadership and team at {c.name}: {team_lines}.",
            "unlisted": True,  # fetchable, but not in the search index
        },
    }


def generate(seed: int = SEED) -> tuple[list[GoldenLead], dict[str, Any]]:
    rng = random.Random(seed)
    people = [f"{first} {last}" for first in FIRST for last in LAST]
    rng.shuffle(people)

    companies = [_Company(n, s, ind, blurb, f"{slug(n)}.example") for n, s, ind, blurb in COMPANIES]
    rng.shuffle(companies)
    by_name = {c.name: c for c in companies}
    pool = [c for c in companies if c.name not in LOOKALIKES]
    # Companies are reused across leads, so a no-MX company must be reserved for no_mx
    # leads only; otherwise flipping its MX would contradict other leads' labels.
    no_mx_pool, pool = pool[: SCENARIO_COUNTS["no_mx"]], pool[SCENARIO_COUNTS["no_mx"] :]
    for c in no_mx_pool:
        c.has_mx = False

    scenarios = [s for s, n in SCENARIO_COUNTS.items() for _ in range(n)]
    leads: list[GoldenLead] = []
    used: set[str] = set()

    def next_company() -> _Company:
        # Spread leads across companies; reuse is fine once every company is used.
        unused = [c for c in pool if c.name not in used] or pool
        c = rng.choice(unused)
        used.add(c.name)
        return c

    lookalike_names = list(LOOKALIKES)
    for i, scenario in enumerate(scenarios):
        person = people[i]
        variants, role, seniority, aliases = rng.choice(TITLES)
        raw_title = rng.choice(variants)
        notes = ""

        if scenario == "unidentifiable":
            raw_company = UNIDENTIFIABLE[len([x for x in leads if x.scenario == scenario])]
            labels = Labels(None, None, role, aliases, seniority, None, None, True)
            leads.append(
                GoldenLead(
                    "",
                    "",
                    scenario,
                    person,
                    raw_company,
                    raw_title,
                    labels,
                    "no website exists for this company in the world",
                )
            )
            continue

        if scenario == "lookalike":
            c = by_name[lookalike_names.pop(0)]
        elif scenario == "no_mx":
            c = no_mx_pool.pop()
        else:
            c = next_company()

        title: str | None = raw_title
        on_team = True
        if scenario in ("title_on_site", "title_unknown"):
            title = None
            on_team = scenario == "title_on_site"
        if on_team:
            c.team.append((person, role))

        raw_company = (
            _messy(c.name, c.suffix, rng) if scenario == "messy_company" else f"{c.name.lower()}"
        )
        if scenario == "lookalike":
            notes = f"look-alike: {LOOKALIKES[c.name][0]} ({LOOKALIKES[c.name][1]})"

        unknown_role = scenario == "title_unknown"
        labels = Labels(
            company=c.name,
            domain=c.domain,
            role=None if unknown_role else role,
            role_aliases=[] if unknown_role else aliases,
            seniority=None if unknown_role else seniority,
            industry=c.industry,
            accepts_email=c.has_mx,
            should_review=scenario in ("no_mx", "title_unknown"),
        )
        leads.append(GoldenLead("", "", scenario, person, raw_company, title, labels, notes))

    # World: real companies, look-alike twins, MX records.
    cities = ["Leeds", "Rotterdam", "Porto", "Gdansk", "Adelaide", "Lyon", "Austin", "Tartu"]
    pages: dict[str, dict[str, Any]] = {}
    mx: dict[str, list[str] | None] = {}
    for c in companies:
        pages.update(_pages(c, rng.randint(2012, 2023), rng.choice(cities)))
        mx[c.domain] = [f"mx1.{c.domain}", f"mx2.{c.domain}"] if c.has_mx else []
    for k, (twin, industry, blurb) in enumerate(LOOKALIKES.values()):
        t = _Company(twin, "Ltd", industry, blurb, f"{slug(twin)}.example")
        t.team = [(people[-1 - k], "Managing Director")]  # people[-k] are never lead names
        pages.update(_pages(t, rng.randint(2005, 2020), rng.choice(cities)))
        mx[t.domain] = [f"mx.{t.domain}"]

    # Stratified split: deal each scenario's leads round-robin into train/dev/test.
    order = [s for s, n in SPLITS.items() for _ in range(n)]
    rng.shuffle(leads)
    leads.sort(key=lambda g: g.scenario)
    slots = dict(SPLITS)
    cycle = ["train", "dev", "train", "test"]
    k = 0
    for g in leads:
        while True:
            s = cycle[k % len(cycle)]
            k += 1
            if slots[s]:
                slots[s] -= 1
                g.split = s
                break
    assert sorted(g.split for g in leads) == sorted(order)

    from leadgenie.models import Lead  # stable IDs, identical to what the pipeline computes

    for g in leads:
        g.lead_id = Lead.from_row(g.row()).id
    assert len({g.lead_id for g in leads}) == len(leads), "lead id collision"
    leads.sort(key=lambda g: (g.split, g.lead_id))
    return leads, {"pages": dict(sorted(pages.items())), "mx": dict(sorted(mx.items()))}


def load_golden(path: str | Path) -> list[GoldenLead]:
    out = []
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        d["labels"] = Labels(**d["labels"])
        out.append(GoldenLead(**d))
    return out


def write(out_dir: str | Path = "evals") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    leads, world = generate()
    with open(out / "golden.jsonl", "w") as f:
        for g in leads:
            f.write(json.dumps(asdict(g), ensure_ascii=False) + "\n")
    (out / "world.json").write_text(json.dumps(world, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(leads)} leads and {len(world['pages'])} pages to {out}/")


if __name__ == "__main__":
    write()
