from leadgenie.review import VERDICT, pending, review_queue
from leadgenie.store import ResultRow, Store
from tests.fakes import submission


def _store(tmp_path) -> Store:
    s = Store(tmp_path / "t.db")
    s.start_run("r", "m", {})
    s.record_result(
        ResultRow(
            lead_id="L1",
            run_id="r",
            name="Ada Example",
            raw_company="acme robotics",
            title=None,
            status="review",
            reasons=["low_confidence:0.30<0.75"],
            confidence=0.3,
            enrichment=submission(),
        )
    )
    return s


def test_review_records_corrections_and_verdict(tmp_path):
    store = _store(tmp_path)
    answers = iter(
        [
            "",  # company ok
            "VP of Engineering",  # role corrected
            "title was on the /team page",  # note
            "vp",  # seniority corrected
            "",  # note
            "",  # industry ok
            "false",  # accepts_email corrected
            "",  # note
            "a",  # verdict
        ]
    )
    n = review_queue(store, "tester", ask=lambda _: next(answers), say=lambda _: None)
    assert n == 3
    by_field = {c.field: c for c in store.corrections("human")}
    assert by_field["role"].corrected == "VP of Engineering"
    assert by_field["role"].predicted == "CTO" and by_field["role"].predicted_source == "input"
    assert by_field["role"].note == "title was on the /team page"
    assert by_field["accepts_email"].corrected is False
    assert by_field[VERDICT].corrected == "approved"
    assert pending(store) == []  # reviewed leads leave the queue


def test_unknown_marker_and_same_value_are_handled(tmp_path):
    store = _store(tmp_path)
    answers = iter(["Acme Robotics", "-", "", "", "", "", "r"])
    review_queue(store, "tester", ask=lambda _: next(answers), say=lambda _: None)
    fields = {c.field: c.corrected for c in store.corrections("human")}
    assert "company" not in fields  # unchanged value is not a correction
    assert fields["role"] is None and fields[VERDICT] == "rejected"
