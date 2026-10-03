"""SQLite store — idempotency, results, review queue, and (later) traces and corrections.

v1 kept a JSON set of processed IDs next to two append-only CSVs. A crash between
"append the CSV row" and "mark the ID done" re-processed (and re-paid for) that lead
and duplicated its row on the next run. Here the result row *is* the done-marker:
one INSERT in one transaction, keyed by lead_id, so a lead is either fully recorded
or not recorded at all. The CSVs become exports of the database, written at the end.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    model       TEXT NOT NULL,
    config      TEXT NOT NULL,          -- JSON
    summary     TEXT                    -- JSON, filled at the end of the run
);

CREATE TABLE IF NOT EXISTS results (
    lead_id     TEXT PRIMARY KEY,       -- one row per lead == idempotency
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    name        TEXT NOT NULL,
    raw_company TEXT NOT NULL,
    title       TEXT,
    status      TEXT NOT NULL CHECK (status IN ('approved', 'review', 'error')),
    reasons     TEXT NOT NULL,          -- JSON list
    confidence  REAL,
    enrichment  TEXT,                   -- JSON, NULL when status = 'error'
    error       TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS results_status ON results(status);
"""


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ResultRow:
    lead_id: str
    run_id: str
    name: str
    raw_company: str
    title: str | None
    status: str
    reasons: list[str]
    confidence: float | None
    enrichment: dict[str, Any] | None
    error: str | None = None


class Store:
    def __init__(self, path: str | Path = "leadgenie.db") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.conn:  # commits on success, rolls back on exception
            yield self.conn

    # --- runs -------------------------------------------------------------

    def start_run(self, run_id: str, model: str, config: dict[str, Any]) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO runs (run_id, started_at, model, config) VALUES (?, ?, ?, ?)",
                (run_id, utcnow(), model, json.dumps(config, sort_keys=True)),
            )

    def finish_run(self, run_id: str, summary: dict[str, Any]) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE runs SET finished_at = ?, summary = ? WHERE run_id = ?",
                (utcnow(), json.dumps(summary, sort_keys=True), run_id),
            )

    # --- results / idempotency -------------------------------------------

    def is_processed(self, lead_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM results WHERE lead_id = ?", (lead_id,)).fetchone()
        return row is not None

    def __contains__(self, lead_id: object) -> bool:
        return isinstance(lead_id, str) and self.is_processed(lead_id)

    def record_result(self, r: ResultRow) -> bool:
        """Atomically record a lead's outcome. Returns False if it was already recorded
        (e.g. a concurrent worker got there first) — the existing row is never overwritten."""
        with self.transaction() as c:
            cur = c.execute(
                # ON CONFLICT, not INSERT OR IGNORE: OR IGNORE would also swallow
                # CHECK-constraint violations and silently drop a malformed row.
                """INSERT INTO results
                   (lead_id, run_id, name, raw_company, title, status, reasons,
                    confidence, enrichment, error, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(lead_id) DO NOTHING""",
                (
                    r.lead_id,
                    r.run_id,
                    r.name,
                    r.raw_company,
                    r.title,
                    r.status,
                    json.dumps(r.reasons),
                    r.confidence,
                    json.dumps(r.enrichment) if r.enrichment is not None else None,
                    r.error,
                    utcnow(),
                ),
            )
            return cur.rowcount == 1

    def results(self, status: str | None = None) -> list[sqlite3.Row]:
        if status is None:
            return self.conn.execute("SELECT * FROM results ORDER BY created_at").fetchall()
        return self.conn.execute(
            "SELECT * FROM results WHERE status = ? ORDER BY created_at", (status,)
        ).fetchall()

    def find_prior(self, company: str, limit: int = 3) -> list[sqlite3.Row]:
        """Earlier *approved* results whose raw or normalized company name matches.

        Only approved rows are returned: a review-queue row is by definition something we
        were not sure about, and feeding it back to the agent would launder a guess into
        a "prior result".
        """
        needle = company.strip().lower()
        if not needle:
            return []
        return self.conn.execute(
            """SELECT * FROM results
               WHERE status = 'approved'
                 AND (lower(raw_company) = ?
                      OR lower(json_extract(enrichment, '$.company.value')) = ?)
               ORDER BY created_at DESC LIMIT ?""",
            (needle, needle, limit),
        ).fetchall()

    # --- exports ----------------------------------------------------------

    def export_csv(self, path: str | Path, statuses: Sequence[str], fields: Sequence[str]) -> int:
        """Rewrite `path` from the database. Returns the number of rows written."""
        rows = [row for s in statuses for row in self.results(s)]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                enrichment = json.loads(row["enrichment"]) if row["enrichment"] else {}
                writer.writerow(
                    {
                        **enrichment,
                        "lead_id": row["lead_id"],
                        "name": row["name"],
                        "confidence": (
                            f"{row['confidence']:.2f}" if row["confidence"] is not None else ""
                        ),
                        "reasons": ";".join(json.loads(row["reasons"])),
                    }
                )
        return len(rows)
