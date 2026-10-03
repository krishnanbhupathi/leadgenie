import json
from pathlib import Path

import pytest
from anthropic.types import Message

from leadgenie.evals.outreach import (
    JUDGE_MODEL,
    deterministic_checks,
    judge_line,
    judge_pass,
)
from leadgenie.tools.base import EvidenceLog
from tests.fakes import FakeAnthropic

URL = "https://acme.example/"
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def evidence():
    log = EvidenceLog()
    log.add({URL: "Acme Robotics builds autonomous forklifts for 40 warehouses."})
    return log


def check(line, evidence, source=URL, quote="autonomous forklifts"):
    return deterministic_checks(line, source, quote, evidence, "Ada", "Acme")


def test_good_line_passes_all_checks(evidence):
    c = check(
        "Ada, autonomous forklifts across 40 warehouses: how do you handle shift changes?", evidence
    )
    assert c["all_pass"], c


@pytest.mark.parametrize(
    ("line", "failed"),
    [
        ("Ada, I hope this finds you well.", "not_generic"),
        ("Hi there, autonomous forklifts are neat.", "personalized"),
        ("Ada, congrats on 300 warehouses!", "no_unsupported_numbers"),
        ("Ada, [COMPANY] forklifts?", "no_placeholders"),
        ("Ada " + "very " * 40, "word_count_ok"),
    ],
)
def test_each_check_can_fail(evidence, line, failed):
    c = check(line, evidence)
    assert not c[failed] and not c["all_pass"]


def test_ungrounded_when_quote_missing_or_source_not_retrieved(evidence):
    assert not check("Ada, forklifts?", evidence, quote="self-driving trucks")["grounded"]
    assert not check("Ada, forklifts?", evidence, source="https://x.example/")["grounded"]
    assert not check("Ada, forklifts?", evidence, source="inferred")["grounded"]


@pytest.mark.parametrize(
    ("scores", "ok"),
    [
        ({"specificity": 5, "grounding": 5, "relevance": 3, "tone": 4}, True),
        ({"specificity": 5, "grounding": 3, "relevance": 5, "tone": 5}, False),
        ({"specificity": 3, "grounding": 5, "relevance": 5, "tone": 5}, False),
        ({"specificity": 5, "grounding": 5, "relevance": 2, "tone": 5}, False),
    ],
)
def test_judge_pass_rule(scores, ok):
    assert judge_pass(scores) is ok


async def test_judge_line_parses_structured_output():
    payload = {"specificity": 5, "grounding": 4, "relevance": 4, "tone": 5, "rationale": "good"}
    reply = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": JUDGE_MODEL,
            "content": [{"type": "text", "text": json.dumps(payload)}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 500, "output_tokens": 50},
        }
    )
    client = FakeAnthropic([reply])
    result = await judge_line(client, "line", "Ada", "CTO", "Acme", URL, "text")
    assert result.passed and result.scores["grounding"] == 4
    assert result.cost_usd == pytest.approx((500 * 2 + 50 * 10) / 1e6)
    req = client.messages.calls[0]
    assert req["model"] == JUDGE_MODEL
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert "SOURCE TEXT" in req["messages"][0]["content"]


def test_judge_check_set_is_balanced():
    rows = [json.loads(x) for x in (ROOT / "evals/judge_check.jsonl").read_text().splitlines()]
    assert len(rows) >= 10
    assert 0.3 < sum(r["expected_pass"] for r in rows) / len(rows) < 0.7
