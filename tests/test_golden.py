import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from leadgenie.evals.golden import SPLITS, generate, load_golden

ROOT = Path(__file__).resolve().parent.parent


def test_generation_is_deterministic():
    a, wa = generate()
    b, wb = generate()
    assert [asdict(x) for x in a] == [asdict(y) for y in b] and wa == wb


def test_committed_files_match_generator():
    leads, world = generate()
    committed = load_golden(ROOT / "evals/golden.jsonl")
    assert [asdict(x) for x in committed] == [asdict(y) for y in leads]
    assert json.loads((ROOT / "evals/world.json").read_text()) == world


def test_size_and_splits():
    leads, _ = generate()
    assert len(leads) >= 50
    assert Counter(g.split for g in leads) == SPLITS
    assert len({g.lead_id for g in leads}) == len(leads)
    # every scenario that must be reviewed appears in the held-out test split
    test_scenarios = {g.scenario for g in leads if g.split == "test"}
    assert {"no_mx", "unidentifiable", "title_unknown", "lookalike"} <= test_scenarios


def test_labels_are_consistent_with_world():
    leads, world = generate()
    for g in leads:
        lab = g.labels
        if g.scenario == "unidentifiable":
            assert lab.company is None and lab.should_review
            continue
        assert f"https://{lab.domain}/" in world["pages"]
        assert lab.accepts_email == bool(world["mx"][lab.domain])
        team = world["pages"][f"https://{lab.domain}/team"]["text"]
        assert (g.name in team) == (g.scenario != "title_unknown")
        assert lab.should_review == (g.scenario in ("no_mx", "title_unknown"))


def test_only_reserved_example_domains():
    _, world = generate()
    hosts = {u.split("/")[2] for u in world["pages"]}
    assert all(h.endswith(".example") for h in hosts)
