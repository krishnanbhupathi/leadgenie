import pytest

from leadgenie.models import Enrichment
from leadgenie.tools.base import EvidenceLog
from leadgenie.validate import route, routing_confidence
from tests.fakes import sourced, submission


@pytest.fixture
def evidence():
    log = EvidenceLog()
    log.add({"https://acme.example/": "Acme Robotics builds autonomous forklifts."})
    log.add({"dns:mx:acme.example": "{'mx_hosts': ['mx1.acme.example']}"})
    return log


def enrichment(**overrides) -> Enrichment:
    return Enrichment.model_validate(submission(**overrides))


def test_clean_submission_is_approved(evidence):
    assert route(enrichment(), evidence) == ("approved", [])


def test_inferred_key_field_is_capped_below_threshold(evidence):
    e = enrichment(industry=sourced("logistics", "inferred", "", 0.95))
    assert routing_confidence(e) == 0.6
    status, reasons = route(e, evidence)
    assert status == "review" and reasons == ["low_confidence:0.60<0.75"]


def test_inferred_non_key_field_does_not_block(evidence):
    # seniority is inferred in the base submission; it is not a key field
    assert route(enrichment(), evidence)[0] == "approved"


def test_weakest_key_field_drives_routing(evidence):
    e = enrichment(role=sourced("CTO", "input", "", 0.4))
    assert routing_confidence(e) == 0.4


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"company": sourced("Acme", "made-up-ref", "", 0.9)}, "bad_source:company"),
        ({"company": sourced("Acme", "https://acme.example/about", "x", 0.9)}, "uncited_source"),
        ({"company": sourced("Acme", "https://acme.example/", "", 0.9)}, "evidence_mismatch"),
        ({"accepts_email": sourced(False, "dns:mx:acme.example", "mx1", 0.9)}, "domain_rejects"),
    ],
)
def test_provenance_and_business_rules(evidence, override, reason):
    status, reasons = route(enrichment(**override), evidence)
    assert status == "review"
    assert any(r.startswith(reason) for r in reasons), reasons


@pytest.mark.parametrize(
    ("line", "source", "reason"),
    [
        ("", "https://acme.example/", "outreach_empty"),
        ("x" * 221, "https://acme.example/", "outreach_too_long"),
        ("Hi Ada, I hope this finds you well.", "https://acme.example/", "outreach_generic"),
        ("Ada, how are robots going?", "inferred", "outreach_ungrounded"),
    ],
)
def test_outreach_rules(evidence, line, source, reason):
    e = enrichment(outreach={"text": line, "source": source, "evidence": "autonomous forklifts"})
    assert reason in route(e, evidence)[1]
