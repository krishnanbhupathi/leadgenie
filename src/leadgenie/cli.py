"""LeadGenie CLI — read leads.csv, enrich each lead, validate, route, log, write outputs.

Usage:
    leadgenie leads.csv
    leadgenie leads.csv --threshold 0.8 --model claude-haiku-4-5
"""

import argparse
import csv
import uuid

import anthropic

from leadgenie.agent import DEFAULT_MODEL, EnrichmentError, enrich_lead
from leadgenie.models import Lead
from leadgenie.observability import log_record, make_record, print_summary
from leadgenie.store import ResultRow, Store
from leadgenie.validate import DEFAULT_THRESHOLD, route

ENRICHED_FIELDS = [
    "lead_id",
    "name",
    "normalized_company",
    "role",
    "seniority",
    "industry",
    "outreach_line",
    "confidence",
]
REVIEW_FIELDS = [*ENRICHED_FIELDS, "reasons"]


def read_leads(path: str):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            lead = Lead.from_row(row)
            if lead.name and lead.raw_company:
                yield lead


def result_row(lead: Lead, run_id: str, status: str, reasons, enriched=None, error=None):
    return ResultRow(
        lead_id=lead.id,
        run_id=run_id,
        name=lead.name,
        raw_company=lead.raw_company,
        title=lead.title,
        status=status,
        reasons=list(reasons),
        confidence=enriched.confidence if enriched else None,
        enrichment=enriched.model_dump() if enriched else None,
        error=error,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich messy leads with an LLM agent.")
    parser.add_argument("csv_path", nargs="?", default="leads.csv")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="confidence below this → human review queue",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--db", default="leadgenie.db", help="SQLite database path")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    store = Store(args.db)
    run_id = uuid.uuid4().hex[:12]
    store.start_run(run_id, args.model, {"threshold": args.threshold, "input": args.csv_path})
    records, skipped = [], 0

    for lead in read_leads(args.csv_path):
        if lead.id in store:
            skipped += 1
            continue

        print(f"[{lead.id}] {lead.name} @ {lead.raw_company!r}")
        try:
            enriched, usage, latency_ms = enrich_lead(client, lead, model=args.model)
        except EnrichmentError as err:
            # Failures are routed, never dropped: the lead goes to the review queue.
            row = result_row(lead, run_id, "error", ["enrichment_failed"], error=str(err))
            record = make_record(lead, args.model, "error", ["enrichment_failed"], error=str(err))
            print(f"    ERROR → review queue ({err})")
        else:
            status, reasons = route(enriched, args.threshold)
            row = result_row(lead, run_id, status, reasons, enriched=enriched)
            record = make_record(
                lead,
                args.model,
                status,
                reasons,
                enriched=enriched,
                usage=usage,
                latency_ms=latency_ms,
            )
            flag = "APPROVED" if status == "approved" else f"REVIEW ({';'.join(reasons)})"
            print(f"    conf={enriched.confidence:.2f} → {flag}")

        # The result row is the done-marker: written in one transaction, so a crash
        # leaves the lead either fully recorded or untouched — never half-done.
        store.record_result(row)
        log_record("runs.jsonl", record)
        records.append(record)

    store.finish_run(run_id, {"processed": len(records), "skipped": skipped})
    store.export_csv("enriched.csv", ["approved"], ENRICHED_FIELDS)
    store.export_csv("review_queue.csv", ["review", "error"], REVIEW_FIELDS)
    print_summary(records, skipped)


if __name__ == "__main__":
    main()
