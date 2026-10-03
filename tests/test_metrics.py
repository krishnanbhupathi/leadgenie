import pytest

from leadgenie.evals.golden import GoldenLead, Labels
from leadgenie.evals.metrics import (
    Prediction,
    calibration_pairs,
    field_correct,
    needs_review,
    norm_company,
    norm_role,
    per_field,
    reliability,
    review_metrics,
)
from leadgenie.models import Enrichment
from tests.fakes import sourced, submission


def golden(scenario="clean", **labels) -> GoldenLead:
    base = dict(
        company="Acme Robotics",
        domain="acme.example",
        role="Chief Technology Officer",
        role_aliases=["cto"],
        seniority="c_level",
        industry="logistics",
        accepts_email=True,
        should_review=False,
    )
    base.update(labels)
    return GoldenLead("id", "test", scenario, "Ada", "acme", "CTO", Labels(**base))


def pred(g, status="approved", **overrides) -> Prediction:
    e = Enrichment.model_validate(submission(**overrides))
    return Prediction(g, status, e, 0.01, 1000, 2, "submitted")


@pytest.mark.parametrize(
    ("a", "b"),
    [("Acme Robotics Inc.", "acme robotics"), ("ACME ROBOTICS LTD", "Acme Robotics")],
)
def test_company_normalization(a, b):
    assert norm_company(a) == norm_company(b)


@pytest.mark.parametrize(
    ("predicted", "ok"),
    [
        ("CTO", True),
        ("Chief Technology Officer", True),
        ("chief technology officer", True),
        ("Head of Engineering", False),
    ],
)
def test_role_matching_uses_aliases(predicted, ok):
    assert field_correct("role", predicted, golden()) is ok


def test_role_normalization_expands_abbreviations():
    assert norm_role("Sr. AE") == "senior ae"
    assert norm_role("VP of Engineering") == norm_role("vp eng")
    assert norm_role("Vice President, Engineering") == "vp engineering"


def test_unknown_label_is_excluded():
    assert field_correct("role", "CTO", golden(role=None)) is None


def test_per_field_counts_missing_enrichment_as_wrong():
    g = golden()
    preds = [pred(g), Prediction(g, "review", None, 0.0, None, 8, "budget_steps")]
    company = per_field(preds)["company"]
    assert company == {"n": 2, "accuracy": 0.5, "coverage": 0.5}


def test_reliability_perfect_and_overconfident():
    perfect = [(0.95, True)] * 19 + [(0.95, False)]  # 95% conf, 95% acc
    ece, table = reliability(perfect)
    assert ece == pytest.approx(0.0) and table[0]["bin"] == "0.9-1.0"
    over = [(0.9, False)] * 10
    assert reliability(over)[0] == pytest.approx(0.9)
    assert reliability([]) == (None, [])
    # 1.0 lands in the top bin, not off the end
    assert reliability([(1.0, True)])[1][0]["bin"] == "0.9-1.0"


def test_calibration_pairs_skip_unknown_labels():
    g = golden(role=None, seniority=None)
    pairs = calibration_pairs([pred(g)])
    assert len(pairs) == 4  # company, domain, industry, accepts_email


def test_needs_review_truth():
    assert needs_review(pred(golden(should_review=True)))
    assert not needs_review(pred(golden()))
    wrong = pred(golden(), company=sourced("Acme Bakery", "inferred", "", 0.9))
    assert needs_review(wrong)


def test_review_metrics_confusion_matrix():
    ok, wrong = golden(), golden(should_review=True)
    preds = [
        pred(ok, "approved"),  # TN
        pred(ok, "review"),  # FP
        pred(wrong, "review"),  # TP
        pred(wrong, "approved"),  # FN → unsafe approval
    ]
    m = review_metrics(preds)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["unsafe_approval_rate"] == 0.5
