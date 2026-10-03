import json
from pathlib import Path

import pytest
from anthropic.types import Message

from leadgenie.agent import AgentConfig
from leadgenie.evals.cassette import Cassette, CassetteMiss
from leadgenie.evals.gate import check_gate, cohen_kappa, run_judge_check
from leadgenie.evals.golden import load_golden
from leadgenie.evals.run import EvalConfig, render_markdown, run_eval, write_results
from leadgenie.pipeline import PipelineConfig
from tests.fakes import FakeAnthropic
from tests.oracle import make_oracle

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "evals/golden.jsonl"
WORLD = ROOT / "evals/world.json"


def judge_reply(passing: bool = True) -> Message:
    s = 5 if passing else 2
    payload = {"specificity": s, "grounding": s, "relevance": 4, "tone": 4, "rationale": "r"}
    return Message.model_validate(
        {
            "id": "msg_j",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": json.dumps(payload)}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 400, "output_tokens": 60},
        }
    )


class OracleClient(FakeAnthropic):
    """Agent calls go to the oracle (beta namespace); judge calls get a fixed reply."""

    def __init__(self, split: str) -> None:
        golden = load_golden(GOLDEN)
        super().__init__(make_oracle(golden, json.loads(WORLD.read_text())))
        self.judge = FakeAnthropic(lambda _: judge_reply())
        self.messages = self.judge.messages


def cfg(split="test", judge=True) -> EvalConfig:
    return EvalConfig(
        split=split,
        pipeline=PipelineConfig(agent=AgentConfig(), requests_per_minute=1e6),
        judge=judge,
    )


async def _oracle_eval(split="test", judge=True):
    client = OracleClient(split)
    return await run_eval(cfg(split, judge), client, client, GOLDEN, WORLD), client


async def test_oracle_scores_perfectly_on_labelled_fields():
    r, _ = await _oracle_eval()
    assert r["meta"]["n"] == 15
    for name, f in r["fields"].items():
        assert f["accuracy"] == 1.0, (name, f)
    assert r["review"]["precision"] == 1.0 and r["review"]["recall"] == 1.0
    assert r["review"]["unsafe_approval_rate"] == 0.0
    # oracle says 0.9 on fields that are always right: under-confident by ~0.1
    assert 0.0 < r["calibration"]["ece"] < 0.2


async def test_oracle_routes_each_must_review_scenario():
    r, _ = await _oracle_eval(split="all", judge=False)
    for scenario in ("no_mx", "unidentifiable", "title_unknown"):
        s = r["scenarios"][scenario]
        assert s["routed_to_review"] == s["n"] == s["needs_review"], scenario
    assert r["scenarios"]["clean"]["routed_to_review"] == 0


async def test_results_include_cost_latency_outreach_and_predictions(tmp_path):
    r, client = await _oracle_eval()
    assert r["cost_usd"]["agent_total"] > 0 and r["latency_ms"]["per_lead_p50"] is not None
    assert r["outreach"]["judge"]["pass_rate"] == 1.0
    assert r["outreach"]["deterministic_pass_rate"]["grounded"] > 0.5
    assert len(r["predictions"]) == 15 and r["meta"]["git"]
    assert len(client.judge.messages.calls) == r["outreach"]["judge"]["n"]
    path = write_results(r, tmp_path, tag="t")
    assert json.loads(path.read_text())["meta"]["split"] == "test"
    assert "Per-field accuracy" in path.with_suffix(".md").read_text()
    assert "| company |" in render_markdown(r)


async def test_cassette_record_then_replay_is_identical(tmp_path):
    tape = tmp_path / "tape.jsonl"
    recording = Cassette(tape, "record", inner=OracleClient("dev"))
    first = await run_eval(cfg("dev"), recording, recording, GOLDEN, WORLD)
    assert recording.recorded > 0 and recording.hits == 0

    replay = Cassette(tape, "replay")
    second = await run_eval(cfg("dev"), replay, replay, GOLDEN, WORLD)
    assert replay.recorded == 0 and replay.hits == recording.recorded

    def strip(r):
        r = {**r, "meta": {**r["meta"], "created_at": None}}
        r["predictions"] = [{**p, "latency_ms": None} for p in r["predictions"]]
        r.pop("latency_ms")
        return r

    assert strip(first) == strip(second)


async def test_replay_fails_loudly_when_prompt_changes(tmp_path):
    from leadgenie.agent import PromptConfig

    tape = tmp_path / "tape.jsonl"
    rec = Cassette(tape, "record", inner=OracleClient("dev"))
    await run_eval(cfg("dev", judge=False), rec, None, GOLDEN, WORLD)
    changed = cfg("dev", judge=False)
    changed.pipeline.agent.prompt = PromptConfig(addendum="Be terse.")
    # A miss must propagate, not be swallowed as a per-lead error and scored.
    with pytest.raises(CassetteMiss):
        await run_eval(changed, Cassette(tape, "replay"), None, GOLDEN, WORLD)


def test_gate():
    results = {"review": {"recall": 0.8, "unsafe_approval_rate": 0.2}, "x": {"y": None}}
    gate = {"review.recall": {"min": 0.9}, "review.unsafe_approval_rate": {"max": 0.1}}
    assert len(check_gate(results, gate)) == 2
    assert check_gate(results, {"review.recall": {"min": 0.5}}) == []
    assert check_gate(results, {"x.y": {"min": 0}}) == ["x.y: no value"]


def test_cohen_kappa():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohen_kappa([], []) is None


async def test_judge_check_measures_agreement():
    rows = [json.loads(x) for x in (ROOT / "evals/judge_check.jsonl").read_text().splitlines()]
    # a judge that always says "pass" agrees exactly on the expected-pass rows
    always_pass = FakeAnthropic(lambda _: judge_reply(True))
    r = await run_judge_check(always_pass, ROOT / "evals/judge_check.jsonl")
    n_pass = sum(x["expected_pass"] for x in rows)
    assert r["agreement"] == round(n_pass / len(rows), 3)
    assert r["false_pass"] == len(rows) - n_pass and r["false_fail"] == 0
    assert r["cohen_kappa"] == 0.0
