import csv
import sqlite3

import pytest

from leadgenie.store import ResultRow, Store


def _row(lead_id: str = "abc", status: str = "approved", **kw) -> ResultRow:
    base = dict(
        lead_id=lead_id,
        run_id="r1",
        name="Ada Example",
        raw_company="example co",
        title="CTO",
        status=status,
        reasons=[],
        confidence=0.9,
        enrichment={"normalized_company": "Example", "role": "CTO"},
    )
    base.update(kw)
    return ResultRow(**base)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    s.start_run("r1", "claude-opus-5-5", {})
    yield s
    s.close()


def test_recorded_lead_is_processed(store):
    assert "abc" not in store
    assert store.record_result(_row())
    assert "abc" in store


def test_second_record_is_ignored_not_overwritten(store):
    store.record_result(_row(confidence=0.9))
    assert store.record_result(_row(confidence=0.1)) is False
    assert store.results()[0]["confidence"] == 0.9


def test_persists_across_reopen(tmp_path):
    s = Store(tmp_path / "t.db")
    s.start_run("r1", "m", {})
    s.record_result(_row())
    s.close()
    assert "abc" in Store(tmp_path / "t.db")


def test_crash_inside_transaction_leaves_lead_unprocessed(store):
    with pytest.raises(RuntimeError), store.transaction() as c:
        c.execute(
            "INSERT INTO results (lead_id, run_id, name, raw_company, status, reasons, created_at)"
            " VALUES ('abc', 'r1', 'n', 'c', 'approved', '[]', 'now')"
        )
        raise RuntimeError("simulated crash")
    assert "abc" not in store


def test_invalid_status_rejected(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.record_result(_row(status="maybe"))


def test_export_csv_splits_by_status(store, tmp_path):
    store.record_result(_row("a"))
    store.record_result(_row("b", status="review", reasons=["low_confidence:0.50<0.75"]))
    store.record_result(_row("c", status="error", enrichment=None, confidence=None, error="x"))
    out = tmp_path / "review.csv"
    n = store.export_csv(out, ["review", "error"], ["lead_id", "role", "confidence", "reasons"])
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert n == 2
    assert [r["lead_id"] for r in rows] == ["b", "c"]
    assert rows[0]["reasons"] == "low_confidence:0.50<0.75"
    assert rows[1]["confidence"] == ""
