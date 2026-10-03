import json
from pathlib import Path

import pytest
from anthropic.types import Message

from leadgenie.agent import AgentConfig, Lesson, PromptConfig
from leadgenie.evals.golden import load_golden
from leadgenie.improve import (
    ImproveConfig,
    Improver,
    decide,
    lessons_from,
    load_prompt,
    objective,
    propose_revision,
    save_prompt,
    simulate_corrections,
)
from leadgenie.pipeline import PipelineConfig
from leadgenie.store import Correction, Store
from tests.fakes import FakeAnthropic
from tests.oracle import make_oracle

ROOT = Path(__file__).resolve().parent.parent
TRIGGER = "similarly named company"  # appears in the lookalike reviewer note


def revision_reply(addendum: str) -> Message:
    return Message.model_validate(
        {
            "id": "msg_r",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [
                {"type": "text", "text": json.dumps({"addendum": addendum, "rationale": "r"})}
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 300, "output_tokens": 80},
        }
    )


def flawed_client(addendum: str = "Be thorough.") -> FakeAnthropic:
    golden = load_golden(ROOT / "evals/golden.jsonl")
    world = json.loads((ROOT / "evals/world.json").read_text())
    client = FakeAnthropic(make_oracle(golden, world, fooled_unless=TRIGGER))
    client.messages = FakeAnthropic(lambda _: revision_reply(addendum)).messages
    return client


def improve_cfg(tmp_path, **kw) -> ImproveConfig:
    return ImproveConfig(
        pipeline=PipelineConfig(agent=AgentConfig(), requests_per_minute=1e6),
        out_dir=tmp_path / "results",
        log_path=tmp_path / "improve_log.jsonl",
        prompt_path=tmp_path / "prompt.json",
        min_delta=0.01,
        **kw,
    )


async def test_loop_accepts_the_fix_then_stops_without_feedback(tmp_path):
    cfg = improve_cfg(tmp_path, iterations=2)
    store = Store(tmp_path / "t.db")
    summary = await Improver(cfg, flawed_client(), store).run()

    log = [json.loads(x) for x in cfg.log_path.read_text().splitlines()]
    assert [e["iteration"] for e in log] == [0, 1, 2, "final"]
    baseline, lessons, revision, _final = log
    assert baseline["corrections"] > 0
    assert lessons["kind"] == "lessons" and lessons["decision"] == "accepted", lessons["why"]
    assert lessons["dev"]["objective"] > lessons["incumbent_dev"]["objective"]
    # the fix removed every train error, so there is nothing left to learn from
    assert revision["decision"] == "skipped" and revision["why"] == "no corrections"
    # held-out test: the fixed prompt beats the baseline and is persisted
    assert summary["test_final"]["objective"] > summary["test_baseline"]["objective"]
    assert (
        summary["test_final"]["unsafe_approval_rate"]
        < summary["test_baseline"]["unsafe_approval_rate"]
    )
    assert load_prompt(cfg.prompt_path).fingerprint == summary["final_prompt"]
    for entry in log:
        for path in entry.get("results", []):
            assert Path(path).exists()


async def test_useless_candidates_are_rejected_and_prompt_kept(tmp_path):
    cfg = improve_cfg(tmp_path, iterations=2)
    golden = load_golden(ROOT / "evals/golden.jsonl")
    world = json.loads((ROOT / "evals/world.json").read_text())
    # fooled unless a phrase no candidate will ever contain → nothing can fix it
    client = FakeAnthropic(make_oracle(golden, world, fooled_unless="xyzzy-never"))
    client.messages = FakeAnthropic(lambda _: revision_reply("Be thorough.")).messages
    summary = await Improver(cfg, client, Store(tmp_path / "t.db")).run()
    log = [json.loads(x) for x in cfg.log_path.read_text().splitlines()]
    assert [(e["kind"], e["decision"]) for e in log[1:3]] == [
        ("lessons", "rejected"),
        ("revision", "rejected"),
    ]
    assert summary["final_prompt"] == summary["baseline_prompt"]
    assert not cfg.prompt_path.exists()


async def test_simulated_corrections_only_from_train(tmp_path):
    with pytest.raises(ValueError):
        simulate_corrections({"meta": {"split": "dev"}, "predictions": []})


def test_lessons_cluster_by_field_and_note():
    def c(field, note, company):
        return Correction(
            company, company, None, field, "x", "inferred", "y", note, "simulated", "s"
        )

    corrections = [
        c("company", "look-alike", "a"),
        c("company", "look-alike", "b"),
        c("role", "on /team", "c"),
        c("_verdict", "", "d"),
    ]
    lessons = lessons_from(corrections, max_lessons=5)
    assert len(lessons) == 2
    assert "look-alike" in lessons[0].guidance and "'b'" in lessons[0].situation
    assert len(lessons_from(corrections, max_lessons=1)) == 1


def _results(acc=0.8, f1=0.8, unsafe=0.1, ece=0.1, cost=0.05):
    return {
        "fields": {f: {"accuracy": acc} for f in ("company", "role", "industry")},
        "review": {"f1": f1, "unsafe_approval_rate": unsafe},
        "calibration": {"ece": ece},
        "cost_usd": {"per_lead_mean": cost},
    }


def test_decide_requires_improvement_and_no_regressions(tmp_path):
    cfg = improve_cfg(tmp_path)
    base = _results()
    assert objective(base) == 0.8
    assert decide(_results(acc=0.9), base, cfg)[0]
    assert not decide(_results(acc=0.805), base, cfg)[0]  # below min_delta
    ok, why = decide(_results(acc=0.95, unsafe=0.2), base, cfg)
    assert not ok and "unsafe" in why[0]
    assert not decide(_results(acc=0.95, ece=0.3), base, cfg)[0]
    assert not decide(_results(acc=0.95, cost=0.10), base, cfg)[0]


async def test_revision_that_names_a_corrected_company_is_refused():
    leaky = FakeAnthropic(lambda _: revision_reply("For acme robotics, use the team page."))
    c = Correction("L", "acme robotics", None, "role", "x", "inferred", "y", "n", "simulated", "s")
    with pytest.raises(ValueError, match="names corrected companies"):
        await propose_revision(leaky, PromptConfig(), [c], "claude-opus-5-5")


async def test_no_corrections_means_no_candidate(tmp_path):
    cfg = improve_cfg(tmp_path, iterations=2)
    # the oracle is never fooled without fooled_unless → train audit yields no corrections
    golden = load_golden(ROOT / "evals/golden.jsonl")
    world = json.loads((ROOT / "evals/world.json").read_text())
    client = FakeAnthropic(make_oracle(golden, world))
    proposer = FakeAnthropic(lambda _: revision_reply("anything"))
    client.messages = proposer.messages
    summary = await Improver(cfg, client, Store(tmp_path / "t.db")).run()
    log = [json.loads(x) for x in cfg.log_path.read_text().splitlines()]
    assert [e.get("decision") for e in log[1:3]] == ["skipped", "skipped"]
    assert proposer.messages.calls == []  # the LLM was never asked to rewrite blind
    assert summary["final_prompt"] == summary["baseline_prompt"]


def test_prompt_roundtrip(tmp_path):
    p = PromptConfig(lessons=(Lesson("s", "g"),), addendum="a")
    save_prompt(p, tmp_path / "p.json")
    assert load_prompt(tmp_path / "p.json") == p
    assert load_prompt(tmp_path / "missing.json") == PromptConfig()
