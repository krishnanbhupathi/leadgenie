# LeadGenie — an agentic lead-enrichment & outreach pipeline

Takes a messy CSV of leads → an LLM agent normalizes the company, infers role/seniority/industry,
drafts one personalized outreach line, and reports a **calibrated confidence**. The system then
validates the output, routes low-confidence leads to a human queue, logs cost/quality per lead,
and never processes the same lead twice.

Built with Claude Code, ~300 lines of Python. Designed around one belief: **agents fail silently,
so the "I'm not sure" path has to be designed first.**

```
leads.csv ──▶ agent.py ──▶ validate.py ──▶ enriched.csv        (auto-approved)
              │                        └▶ review_queue.csv     (human-in-the-loop)
              │
              ├──▶ observability.py ──▶ runs.jsonl + run summary
              └──▶ store.py            (idempotency — re-runs skip done leads)
```

## Modules

| File | Responsibility |
|---|---|
| `models.py` | Pydantic schemas. `Lead` (input) and `EnrichedLead` (output contract). Invalid output = rejected. |
| `agent.py` | The agent. Structured output via `messages.parse()` — the API is forced to emit JSON matching the schema. Retries with exponential backoff. |
| `validate.py` | Business rules the schema can't express (industry allowlist, generic-line detection) + confidence threshold routing. |
| `observability.py` | Per-lead trace (model, tokens, cost, latency, confidence, outcome) → `runs.jsonl` + end-of-run summary. |
| `store.py` | Idempotency. Crash-safe processed-ID set — re-runs never double-spend. |
| `main.py` | CLI orchestration: read → check cache → enrich → validate → route → log → write. |

## The 5 reliability patterns

1. **Structured output + Pydantic validation** — the model never returns free text. The API enforces
   the JSON shape; Pydantic enforces what the schema can't (confidence ∈ [0,1]).
2. **Confidence threshold → human-in-the-loop** — `confidence < 0.75` (or any rule violation) routes
   to `review_queue.csv` with machine-readable reasons. Failures are routed, never dropped.
3. **Observability** — every lead leaves a JSONL trace; the summary answers "what did this cost and
   can I trust it?" (approved %, flagged %, avg confidence, total $).
4. **Idempotency + retries** — stable content-hash lead IDs; the processed set is flushed to disk
   after every lead, so a crash mid-run resumes exactly where it stopped.
5. **Model routing** — `--model claude-haiku-4-5` runs enrichment ~5x cheaper; the observability log
   makes the cost/quality trade-off measurable instead of a guess.

## Run it

```bash
pip install anthropic pydantic
export ANTHROPIC_API_KEY=sk-ant-...

python3 main.py leads.csv                                  # default: claude-opus-4-8, threshold 0.75
python3 main.py leads.csv --model claude-haiku-4-5         # cheap-model routing
python3 main.py leads.csv --threshold 0.85                 # stricter human-review gate
python3 main.py leads.csv                                  # re-run → everything skipped (idempotent)
```

Outputs: `enriched.csv`, `review_queue.csv`, `runs.jsonl`, `.processed.json` (delete it to force reprocessing).

## A real iteration, caught by the pipeline itself

On the first live run, 55% of leads landed in the review queue — and `runs.jsonl` showed why:
the model was inventing industry labels outside the allowlist ("enterprise software",
"food delivery"), so high-confidence leads (0.82–0.83) were flagged for label drift, not bad
enrichment. The fix: the industry enum moved *into the output schema* (`Industry` Literal in
`models.py`), so the API now physically cannot return an off-list label. The runtime allowlist
check stays as defense in depth. This is the observability loop working as designed — the trace
told me the review rate was inflated by a contract gap, not model quality.

## Measured: model routing on the same 20 leads (post-fix)

| | claude-opus-4-8 | claude-haiku-4-5 |
|---|---|---|
| Auto-approved | 70% | 75% |
| Human review | 30% | 25% |
| Avg confidence | 0.73 | 0.76 |
| Avg latency / lead | 3.5s | 2.8s |
| Total cost (20 leads) | $0.185 | $0.028 |

Both models flag the same genuinely ambiguous leads ("acme corp", "stealth startup", missing
titles). The one disagreement: Opus held `glovoapp23` at 0.70 (review) where Haiku said 0.78
(approve) — Opus is slightly more conservative near the threshold. Haiku's confidences also
cluster at exact values (0.82, 0.92), suggesting coarser calibration; with dirtier real-world
data that's the number I'd watch. For this task shape, the cheap model wins: ~6.6x cheaper with
the same quality gate outcome.

## Where the model fails (honestly)

Silent overconfidence: the model returns a clean, well-formed, plausible — and wrong — company or
role. The schema can't catch that, which is exactly why confidence calibration is prompted
explicitly ("a vaguer line with honest confidence beats a confident hallucination"), why business
rules run after the schema, and why anything below the threshold goes to a human.

Still figuring out: proper evals for outreach-line quality (currently a heuristic generic-phrase
filter), and driving the human-review % down without letting bad leads through — the `runs.jsonl`
history is the raw material for that loop.

Uses fake sample leads only — no scraped personal data.
