import asyncio
import csv
import json

from leadgenie.agent import SUBMIT_TOOL, AgentConfig
from leadgenie.cli import main
from leadgenie.models import Lead
from leadgenie.pipeline import (
    ENRICHED_FIELDS,
    PipelineConfig,
    flatten_enrichment,
    run_pipeline,
)
from leadgenie.store import Store
from leadgenie.tools import World, fixture_backends
from tests.fakes import FakeAnthropic, message, sourced, submission, tool_use

WORLD = World(
    pages={
        "https://acme.example/": {
            "title": "Acme Robotics",
            "description": "Acme Robotics builds autonomous forklifts for warehouses.",
        }
    },
    mx={"acme.example": ["mx1.acme.example"]},
)


def lead(i: int) -> Lead:
    return Lead.from_row({"name": f"Person {i}", "raw_company": f"acme {i}", "title": "CTO"})


def scripted(payload_for=lambda _: submission()):
    """Fetch, then submit; the payload may depend on the lead text."""

    def script(req):
        msgs = req["messages"]
        if len(msgs) == 1:
            return message(
                tool_use("fetch_company_site", url="acme.example"),
                tool_use("check_mx", domain="acme.example"),
            )
        return message(tool_use(SUBMIT_TOOL, **payload_for(msgs[0]["content"])))

    return script


async def _run(store, client, leads, **cfg):
    return await run_pipeline(
        leads,
        client=client,
        store=store,
        backends=fixture_backends(WORLD),
        config=PipelineConfig(agent=AgentConfig(), **{"requests_per_minute": 1e6, **cfg}),
    )


async def test_rerun_skips_processed_leads(tmp_path):
    store = Store(tmp_path / "t.db")
    first = await _run(store, FakeAnthropic(scripted()), [lead(1), lead(2)])
    assert len(first.results) == 2 and first.skipped == 0
    client = FakeAnthropic(scripted())
    second = await _run(store, client, [lead(1), lead(2), lead(3)])
    assert second.skipped == 2 and [r.lead.id for r in second.results] == [lead(3).id]
    assert len(store.results()) == 3


async def test_duplicate_input_rows_are_processed_once(tmp_path):
    store = Store(tmp_path / "t.db")
    run = await _run(store, FakeAnthropic(scripted()), [lead(1), lead(1)])
    assert len(run.results) == 1 and run.skipped == 1


async def test_routing_mix_and_records(tmp_path):
    store = Store(tmp_path / "t.db")

    def payload(lead_text):
        if "Person 2" in lead_text:  # unsure → low confidence → review
            return submission(role=sourced("CTO", "input", "", 0.3))
        return submission()

    run = await _run(store, FakeAnthropic(scripted(payload)), [lead(1), lead(2)])
    by_status = {r.row.status for r in run.results}
    assert by_status == {"approved", "review"}
    review = store.results("review")[0]
    assert json.loads(review["reasons"]) == ["low_confidence:0.30<0.75"]
    spans = store.spans(run.run_id)
    assert {s["kind"] for s in spans} == {"lead", "model_call", "tool_call"}


async def test_budget_exhaustion_lands_in_review_not_error(tmp_path):
    store = Store(tmp_path / "t.db")
    loop_forever = lambda _: message(tool_use("check_mx", domain="acme.example"))  # noqa: E731
    run = await _run(store, FakeAnthropic(loop_forever), [lead(1)])
    row = run.results[0].row
    assert row.status == "review" and row.reasons == ["budget_steps"]


async def test_concurrency_limit_is_respected(tmp_path):
    store = Store(tmp_path / "t.db")
    active = peak = 0
    inner = scripted()

    client = FakeAnthropic(inner)
    original = client.messages.create

    async def tracked(**kw):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        try:
            return await original(**kw)
        finally:
            active -= 1

    client.beta.messages.create = tracked
    await _run(store, client, [lead(i) for i in range(8)], concurrency=3)
    assert peak == 3


def test_flatten_enrichment_keeps_sources():
    flat = flatten_enrichment(submission())
    assert flat["company"] == "Acme Robotics"
    assert flat["company_source"] == "https://acme.example/"
    assert set(ENRICHED_FIELDS) - {"lead_id", "name", "confidence"} <= set(flat)


def test_cli_run_end_to_end_offline(tmp_path, monkeypatch):
    world = tmp_path / "world.json"
    world.write_text(json.dumps({"pages": WORLD.pages, "mx": WORLD.mx}))
    leads_csv = tmp_path / "leads.csv"
    leads_csv.write_text("name,raw_company,title\nAda Example,acme robotics,CTO\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "leadgenie.cli.anthropic.AsyncAnthropic", lambda **_: FakeAnthropic(scripted())
    )
    argv = ["--db", "t.db", "run", str(leads_csv), "--world", str(world), "--rpm", "1e6"]
    assert main(argv) == 0
    with open("enriched.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["company"] == "Acme Robotics"
    assert len(list((tmp_path / "reports").glob("run-*.md"))) == 1
