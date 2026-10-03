"""LeadGenie CLI — read leads.csv, enrich each lead, validate, route, log, write outputs.

Usage:
    python3 main.py leads.csv
    python3 main.py leads.csv --threshold 0.8 --model claude-haiku-4-5
"""
import argparse
import csv
import os

import anthropic

from leadgenie.agent import DEFAULT_MODEL, EnrichmentError, enrich_lead
from leadgenie.models import Lead
from leadgenie.observability import log_record, make_record, print_summary
from leadgenie.store import ProcessedStore
from leadgenie.validate import DEFAULT_THRESHOLD, route

ENRICHED_FIELDS = [
    "lead_id", "name", "normalized_company", "role", "seniority",
    "industry", "outreach_line", "confidence",
]
REVIEW_FIELDS = ENRICHED_FIELDS + ["reasons"]


def read_leads(path: str):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            lead = Lead.from_row(row)
            if lead.name and lead.raw_company:
                yield lead


def append_csv(path: str, fields, row: dict) -> None:
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def enriched_row(lead: Lead, enriched) -> dict:
    return {
        "lead_id": lead.id,
        "name": lead.name,
        "normalized_company": enriched.normalized_company,
        "role": enriched.role,
        "seniority": enriched.seniority,
        "industry": enriched.industry,
        "outreach_line": enriched.outreach_line,
        "confidence": f"{enriched.confidence:.2f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich messy leads with an LLM agent.")
    parser.add_argument("csv_path", nargs="?", default="leads.csv")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="confidence below this → human review queue")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    client = anthropic.Anthropic()
    store = ProcessedStore()
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
            append_csv("review_queue.csv", REVIEW_FIELDS, {
                "lead_id": lead.id, "name": lead.name, "reasons": "enrichment_failed",
            })
            record = make_record(lead, args.model, "error", ["enrichment_failed"], error=str(err))
            print(f"    ERROR → review queue ({err})")
        else:
            status, reasons = route(enriched, args.threshold)
            row = enriched_row(lead, enriched)
            if status == "approved":
                append_csv("enriched.csv", ENRICHED_FIELDS, row)
            else:
                append_csv("review_queue.csv", REVIEW_FIELDS, {**row, "reasons": ";".join(reasons)})
            record = make_record(lead, args.model, status, reasons,
                                 enriched=enriched, usage=usage, latency_ms=latency_ms)
            flag = "APPROVED" if status == "approved" else f"REVIEW ({';'.join(reasons)})"
            print(f"    conf={enriched.confidence:.2f} → {flag}")

        log_record("runs.jsonl", record)
        records.append(record)
        store.add(lead.id)

    print_summary(records, skipped)


if __name__ == "__main__":
    main()
