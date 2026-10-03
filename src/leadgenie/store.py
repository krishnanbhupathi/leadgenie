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

CREATE TABLE IF NOT EXISTS spans (
    span_id     TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    parent_id   TEXT,
    lead_id     TEXT,
    kind        TEXT NOT NULL,          -- lead | model_call | tool_call
    name        TEXT NOT NULL,
    step        INTEGER,
    started_at  TEXT NOT NULL,
    latency_ms  INTEGER,
    status      TEXT NOT NULL,
    error       TEXT,
    model       TEXT,
    input_tokens       INTEGER NOT NULL DEFAULT 0,
    output_tokens      INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens  INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd    REAL NOT NULL DEFAULT 0,
    input       TEXT,                   -- JSON
    output      TEXT,                   -- JSON
    attrs       TEXT                    -- JSON
);

CREATE INDEX IF NOT EXISTS spans_run ON spans(run_id, kind);
"""

SPAN_COLUMNS = (
    "span_id",
    "run_id",
    "parent_id",
    "lead_id",
    "kind",
    "name",
    "step",
    "started_at",
    "latency_ms",
    "status",
    "error",
    "model",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cost_usd",
    "input",
    "output",
    "attrs",
)
_JSON_SPAN_COLUMNS = ("input", "output", "attrs")


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

    # --- spans (tracing sink) --------------------------------------------

    def write_span(self, span: dict[str, Any]) -> None:
        values = [
            json.dumps(span.get(c), default=str, ensure_ascii=False)
            if c in _JSON_SPAN_COLUMNS
            else span.get(c)
            for c in SPAN_COLUMNS
        ]
        with self.transaction() as c:
            c.execute(
                f"INSERT INTO spans ({', '.join(SPAN_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(SPAN_COLUMNS))})",
                values,
            )

    def spans(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM spans WHERE run_id = ? ORDER BY started_at", (run_id,)
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            for col in _JSON_SPAN_COLUMNS:
                d[col] = json.loads(d[col]) if d[col] is not None else None
            out.append(d)
        return out

    def run_results(self, run_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM results WHERE run_id = ? ORDER BY created_at", (run_id,)
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
