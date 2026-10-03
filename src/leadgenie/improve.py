"""The self-improving loop: corrections → candidate prompt → eval → keep only if better.

    leadgenie improve --iterations 3 --record

Per iteration:
  1. Gather corrections: human ones from `leadgenie review`, plus simulated ones produced
     by auditing the *train* split against its labels (no human reviewers exist for the
     synthetic set; these are marked origin='simulated').
  2. Propose one candidate, alternating between two kinds:
       lessons  — few-shot examples distilled from the most frequent correction clusters
       revision — an LLM rewrites the prompt addendum from the correction patterns
  3. Evaluate the candidate on the *dev* split.
  4. Accept only if the objective improves by ≥ min_delta AND no guardrail regresses
     (unsafe approval rate, ECE, cost per lead). Otherwise keep the current prompt.
  5. Log the iteration (results/improve_log.jsonl) with links to the eval result files.
At the end the baseline and final prompts are both scored on the held-out *test* split,
which no correction or acceptance decision ever saw.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from leadgenie.agent import Lesson, ModelClient, PromptConfig
from leadgenie.evals.metrics import KEY_FIELDS, field_correct
from leadgenie.evals.run import EvalConfig, run_eval, write_results
from leadgenie.pipeline import PipelineConfig
from leadgenie.pricing import TokenUsage, cost_usd
from leadgenie.review import REVIEW_FIELDS
from leadgenie.store import Correction, Store

DEFAULT_PROMPT_PATH = Path("prompts/current.json")

# What a reviewer would write when fixing each kind of mistake. The simulated reviewer
# picks the note by scenario; a human reviewer types their own.
REVIEWER_NOTES = {
    "lookalike": "A similarly named company exists in another industry; confirm identity "
    "on the candidate's /team or /about page before choosing.",
    "title_on_site": "The title was missing from the row but listed on the company's /team "
    "page; fetch /team when the title is missing.",
    "title_unknown": "The person is not on the site and no title was given; the role is "
    "unknown and must be marked inferred with low confidence.",
    "no_mx": "check_mx showed no MX records, so accepts_email must be false.",
    "unidentifiable": "No company with this name could be found; do not guess a company.",
    "messy_company": "The raw company string had typos or a legal suffix; normalize it to "
    "the name shown on the official site.",
    "clean": "The answer was on the company's own site.",
}


@dataclass
class ImproveConfig:
    iterations: int = 3
    max_lessons: int = 6
    min_delta: float = 0.03
    max_unsafe_increase: float = 0.0
    max_ece_increase: float = 0.05
    max_cost_ratio: float = 1.25
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    proposer_model: str = "claude-opus-5-5"
    out_dir: Path = Path("results/improve")
    log_path: Path = Path("results/improve_log.jsonl")
    prompt_path: Path = DEFAULT_PROMPT_PATH


# --- prompt persistence --------------------------------------------------------------


def load_prompt(path: Path) -> PromptConfig:
    if not path.exists():
        return PromptConfig()
    data = json.loads(path.read_text())
    return PromptConfig(
        lessons=tuple(Lesson(**ls) for ls in data.get("lessons", [])),
        addendum=data.get("addendum", ""),
    )


def save_prompt(prompt: PromptConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "fingerprint": prompt.fingerprint,
        "lessons": [asdict(ls) for ls in prompt.lessons],
        "addendum": prompt.addendum,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


# --- scoring -------------------------------------------------------------------------


def objective(results: dict[str, Any]) -> float:
    """Half key-field accuracy, half review-queue F1: right answers, and the wrong ones caught."""
    key_acc = sum(results["fields"][f]["accuracy"] or 0.0 for f in KEY_FIELDS) / len(KEY_FIELDS)
    f1 = results["review"]["f1"] or 0.0
    return round(0.5 * key_acc + 0.5 * f1, 4)


def decide(
    candidate: dict[str, Any], incumbent: dict[str, Any], cfg: ImproveConfig
) -> tuple[bool, list[str]]:
    reasons = []
    delta = objective(candidate) - objective(incumbent)
    if delta < cfg.min_delta:
        reasons.append(f"objective +{delta:.4f} < min_delta {cfg.min_delta}")
    unsafe_new = candidate["review"]["unsafe_approval_rate"] or 0.0
    unsafe_old = incumbent["review"]["unsafe_approval_rate"] or 0.0
    if unsafe_new - unsafe_old > cfg.max_unsafe_increase:
        reasons.append(f"unsafe approval rate {unsafe_old} → {unsafe_new}")
    ece_new, ece_old = candidate["calibration"]["ece"], incumbent["calibration"]["ece"]
    if ece_new is not None and ece_old is not None and ece_new - ece_old > cfg.max_ece_increase:
        reasons.append(f"ECE {ece_old} → {ece_new}")
    cost_new = candidate["cost_usd"]["per_lead_mean"] or 0.0
    cost_old = incumbent["cost_usd"]["per_lead_mean"] or 0.0
    if cost_old and cost_new / cost_old > cfg.max_cost_ratio:
        reasons.append(f"cost/lead ${cost_old:.4f} → ${cost_new:.4f}")
    return not reasons, reasons


# --- corrections ---------------------------------------------------------------------


def simulate_corrections(train_results: dict[str, Any]) -> list[Correction]:
    """Audit train-split predictions against labels, as a reviewer would."""
    from leadgenie.evals.golden import GoldenLead, Labels

    if train_results["meta"]["split"] != "train":
        raise ValueError("simulated corrections may only come from the train split")
    out = []
    for p in train_results["predictions"]:
        labels = Labels(**p["labels"])
        g = GoldenLead(p["lead_id"], "train", p["scenario"], "", "", None, labels)
        enrichment = p["enrichment"] or {}
        for name in REVIEW_FIELDS:
            item = enrichment.get(name) or {}
            predicted = item.get("value")
            ok = field_correct(name, predicted, g)
            if ok is not False:
                continue
            out.append(
                Correction(
                    lead_id=p["lead_id"],
                    raw_company=p["input"]["raw_company"],
                    title=p["input"]["title"] or None,
                    field=name,
                    predicted=predicted,
                    predicted_source=item.get("source"),
                    corrected=getattr(labels, name),
                    note=REVIEWER_NOTES.get(p["scenario"], ""),
                    origin="simulated",
                    reviewer="simulated:train-labels",
                )
            )
    return out


def lessons_from(corrections: list[Correction], max_lessons: int) -> tuple[Lesson, ...]:
    """One few-shot lesson per (field, note) cluster, most frequent clusters first."""
    clusters: dict[tuple[str, str], list[Correction]] = defaultdict(list)
    for c in corrections:
        if c.field.startswith("_"):
            continue
        clusters[(c.field, c.note)].append(c)
    ranked = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    lessons = []
    for (name, note), members in ranked[:max_lessons]:
        ex = members[-1]
        title = ex.title or "missing"
        lessons.append(
            Lesson(
                situation=(
                    f"raw company {ex.raw_company!r}, title {title!r}: you answered "
                    f"{name}={ex.predicted!r} (source {ex.predicted_source or 'none'})"
                ),
                guidance=f"a reviewer corrected it to {ex.corrected!r}. {note}".strip(),
            )
        )
    return tuple(lessons)


REVISION_SYSTEM = """\
You improve the instructions of a lead-enrichment research agent. You are given the \
current extra instructions (may be empty) and clusters of reviewer corrections showing \
where the agent went wrong. Write a replacement for the extra instructions: at most 120 \
words of general, actionable rules that would prevent these errors. Do not mention any \
specific company, person or domain. Do not loosen the provenance or honesty rules."""

REVISION_SCHEMA = {
    "type": "object",
    "properties": {"addendum": {"type": "string"}, "rationale": {"type": "string"}},
    "required": ["addendum", "rationale"],
    "additionalProperties": False,
}


async def propose_revision(
    client: ModelClient, current: PromptConfig, corrections: list[Correction], model: str
) -> tuple[PromptConfig, str, float]:
    clusters = Counter((c.field, c.note) for c in corrections if not c.field.startswith("_"))
    summary = "\n".join(
        f"- {n}x field={name}: {note or '(no note)'}" for (name, note), n in clusters.most_common()
    )
    response = await client.messages.create(  # type: ignore[attr-defined]
        model=model,
        max_tokens=4_000,
        system=REVISION_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"CURRENT EXTRA INSTRUCTIONS:\n{current.addendum or '(none)'}\n\n"
                f"CORRECTION CLUSTERS:\n{summary}",
            }
        ],
        output_config={"format": {"type": "json_schema", "schema": REVISION_SCHEMA}},
    )
    data = json.loads(next(b.text for b in response.content if b.type == "text"))
    addendum = data["addendum"].strip()
    leaked = sorted(
        {c.raw_company for c in corrections if c.raw_company.lower() in addendum.lower()}
    )
    if leaked:
        raise ValueError(f"proposed revision names corrected companies: {leaked}")
    cost = cost_usd(model, TokenUsage.from_api(response.usage))
    return replace(current, addendum=addendum), data["rationale"], cost


# --- the loop ------------------------------------------------------------------------


class Improver:
    def __init__(self, cfg: ImproveConfig, client: ModelClient, store: Store) -> None:
        self.cfg = cfg
        self.client = client
        self.store = store
        self.spent = 0.0

    def _log(self, entry: dict[str, Any]) -> None:
        self.cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **entry}
        with open(self.cfg.log_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def _eval(self, prompt: PromptConfig, split: str, tag: str) -> tuple[dict[str, Any], str]:
        pipeline = replace(self.cfg.pipeline, agent=replace(self.cfg.pipeline.agent, prompt=prompt))
        results = await run_eval(
            EvalConfig(split=split, pipeline=pipeline, judge=False), self.client
        )
        self.spent += results["cost_usd"]["agent_total"]
        path = write_results(results, self.cfg.out_dir, tag=tag)
        return results, str(path)

    def _refresh_simulated(self, train_results: dict[str, Any]) -> list[Correction]:
        self.store.clear_corrections("simulated")
        for c in simulate_corrections(train_results):
            self.store.add_correction(c)
        return self.store.corrections()

    @staticmethod
    def _metrics(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "objective": objective(r),
            "key_accuracy": {f: r["fields"][f]["accuracy"] for f in KEY_FIELDS},
            "review_f1": r["review"]["f1"],
            "unsafe_approval_rate": r["review"]["unsafe_approval_rate"],
            "ece": r["calibration"]["ece"],
            "cost_per_lead": r["cost_usd"]["per_lead_mean"],
        }

    async def run(self) -> dict[str, Any]:
        cfg = self.cfg
        baseline = current = load_prompt(cfg.prompt_path)
        dev, dev_path = await self._eval(current, "dev", "i0-dev")
        train, train_path = await self._eval(current, "train", "i0-train")
        corrections = self._refresh_simulated(train)
        self._log(
            {
                "iteration": 0,
                "kind": "baseline",
                "prompt": current.fingerprint,
                "dev": self._metrics(dev),
                "train_objective": objective(train),
                "corrections": len(corrections),
                "results": [dev_path, train_path],
            }
        )

        for i in range(1, cfg.iterations + 1):
            kind = "lessons" if i % 2 else "revision"
            note = ""
            if kind == "lessons":
                candidate = replace(current, lessons=lessons_from(corrections, cfg.max_lessons))
            else:
                candidate, note, cost = await propose_revision(
                    self.client, current, corrections, cfg.proposer_model
                )
                self.spent += cost
            if candidate.fingerprint == current.fingerprint:
                self._log({"iteration": i, "kind": kind, "decision": "skipped", "why": "no change"})
                continue

            cand_dev, cand_path = await self._eval(candidate, "dev", f"i{i}-dev")
            accepted, why = decide(cand_dev, dev, cfg)
            entry: dict[str, Any] = {
                "iteration": i,
                "kind": kind,
                "prompt": candidate.fingerprint,
                "incumbent": current.fingerprint,
                "corrections_used": len(corrections),
                "dev": self._metrics(cand_dev),
                "incumbent_dev": self._metrics(dev),
                "decision": "accepted" if accepted else "rejected",
                "why": why,
                "proposer_rationale": note,
                "results": [cand_path],
            }
            if accepted:
                current, dev = candidate, cand_dev
                train, train_path = await self._eval(current, "train", f"i{i}-train")
                corrections = self._refresh_simulated(train)
                entry["results"].append(train_path)
            self._log(entry)

        base_test, base_path = await self._eval(baseline, "test", "final-baseline-test")
        final_test, final_path = base_test, base_path
        if current.fingerprint != baseline.fingerprint:
            final_test, final_path = await self._eval(current, "test", "final-test")
            save_prompt(current, cfg.prompt_path)
        summary = {
            "iteration": "final",
            "baseline_prompt": baseline.fingerprint,
            "final_prompt": current.fingerprint,
            "test_baseline": self._metrics(base_test),
            "test_final": self._metrics(final_test),
            "spent_usd": round(self.spent, 4),
            "results": sorted({base_path, final_path}),
        }
        self._log(summary)
        return summary
