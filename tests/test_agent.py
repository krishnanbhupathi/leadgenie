import anthropic
import httpx2
import pytest

from leadgenie.agent import SUBMIT_TOOL, AgentConfig, Budget, LeadAgent, Lesson, PromptConfig
from leadgenie.models import Lead
from leadgenie.store import Store
from leadgenie.tools import World, build_registry, fixture_backends
from leadgenie.tracing import MemorySink, Tracer
from leadgenie.validate import route
from tests.fakes import FakeAnthropic, message, sourced, submission, text, tool_use

WORLD = World(
    pages={
        "https://acme.example/": {
            "title": "Acme Robotics",
            "description": "Acme Robotics builds autonomous forklifts for warehouses.",
            "text": "Team: Ada Example, CTO.",
        }
    },
    mx={"acme.example": ["mx1.acme.example"]},
)
LEAD = Lead(id="L1", name="Ada Example", raw_company="acme robotics inc", title="CTO")


@pytest.fixture
def harness(tmp_path):
    store = Store(tmp_path / "t.db")
    sink = MemorySink()
    registry = build_registry(fixture_backends(WORLD), store)

    def make(script, **cfg):
        client = FakeAnthropic(script)
        agent = LeadAgent(client, registry, Tracer("run", [sink]), AgentConfig(**cfg))
        return agent, client

    yield make, sink
    store.close()


def research_turn():
    return message(
        tool_use("fetch_company_site", url="acme.example"),
        tool_use("check_mx", domain="acme.example"),
    )


async def test_happy_path_researches_then_submits(harness):
    make, sink = harness
    agent, client = make([research_turn(), message(tool_use(SUBMIT_TOOL, **submission()))])
    out = await agent.run(LEAD)

    assert out.stop == "submitted" and out.steps == 2
    assert out.enrichment.company.value == "Acme Robotics"
    assert route(out.enrichment, out.evidence) == ("approved", [])
    # second request carries both tool results back in ONE user message
    results = client.messages.calls[1]["messages"][-1]["content"]
    assert [r["type"] for r in results] == ["tool_result", "tool_result"]
    assert not any(r["is_error"] for r in results)
    # 2 model calls x (1000 in, 100 out) at opus-5-5 prices
    assert out.cost_usd == pytest.approx(2 * (1000 * 4 + 100 * 20) / 1e6)
    kinds = [s["kind"] for s in sink.spans]
    assert kinds.count("model_call") == 2 and kinds.count("tool_call") == 2
    assert sink.spans[-1]["kind"] == "lead" and sink.spans[-1]["attrs"]["stop"] == "submitted"


async def test_request_shape(harness):
    make, _ = harness
    agent, client = make([message(tool_use(SUBMIT_TOOL, **submission()))])
    await agent.run(LEAD)
    req = client.messages.calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"] == {"effort": "medium"}
    assert req["cache_control"] == {"type": "ephemeral"}
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert "tool_choice" not in req  # forced tool choice is a 400 on Opus 5.5
    assert all(t["strict"] for t in req["tools"])
    assert {t["name"] for t in req["tools"]} >= {SUBMIT_TOOL, "fetch_company_site", "check_mx"}


async def test_haiku_gets_no_effort_or_fallbacks(harness):
    make, _ = harness
    agent, client = make([message(tool_use(SUBMIT_TOOL, **submission()))], model="claude-haiku-4-5")
    await agent.run(LEAD)
    req = client.messages.calls[0]
    assert "output_config" not in req and "fallbacks" not in req


async def test_hallucinated_citation_is_routed_to_review(harness):
    make, _ = harness
    fake_url = "https://acme.example/press/series-b"
    bad = submission(company=sourced("Acme Robotics", fake_url, "raised $40M", 0.99))
    agent, _ = make([research_turn(), message(tool_use(SUBMIT_TOOL, **bad))])
    out = await agent.run(LEAD)
    status, reasons = route(out.enrichment, out.evidence)
    assert status == "review" and "uncited_source:company" in reasons


async def test_quote_not_in_page_is_routed_to_review(harness):
    make, _ = harness
    bad = submission(industry=sourced("fintech", "https://acme.example/", "payments", 0.9))
    agent, _ = make([research_turn(), message(tool_use(SUBMIT_TOOL, **bad))])
    out = await agent.run(LEAD)
    assert "evidence_mismatch:industry" in route(out.enrichment, out.evidence)[1]


async def test_invalid_submission_gets_error_feedback_and_resubmits(harness):
    make, _ = harness
    invalid = submission(role=sourced("CTO", "input", "", 1.7))
    agent, client = make(
        [
            message(tool_use(SUBMIT_TOOL, **invalid)),
            message(tool_use(SUBMIT_TOOL, **submission())),
        ]
    )
    out = await agent.run(LEAD)
    assert out.stop == "submitted" and out.steps == 2
    feedback = client.messages.calls[1]["messages"][-1]["content"][0]
    assert feedback["is_error"] and "role.confidence" in feedback["content"]


async def test_step_budget_stops_and_warns_before_last_step(harness):
    make, _ = harness
    agent, client = make(
        lambda _: message(tool_use("fetch_company_site", url="acme.example")),
        budget=Budget(max_steps=3, max_cost_usd=10),
    )
    out = await agent.run(LEAD)
    assert out.stop == "budget_steps" and out.steps == 3 and out.enrichment is None
    assert len(client.messages.calls) == 3
    warned = client.messages.calls[2]["messages"][-1]["content"][-1]
    assert warned["type"] == "text" and "Budget nearly exhausted" in warned["text"]
    first = client.messages.calls[1]["messages"][-1]["content"]
    assert all(b["type"] == "tool_result" for b in first)  # no warning early on


async def test_cost_budget_is_enforced_from_usage(harness):
    make, _ = harness
    pricey = lambda _: message(  # noqa: E731
        tool_use("check_mx", domain="acme.example"), input_tokens=50_000, output_tokens=0
    )
    agent, client = make(pricey, budget=Budget(max_steps=10, max_cost_usd=0.5))
    out = await agent.run(LEAD)
    # each call costs $0.20 → stop after the 3rd crosses $0.50
    assert out.stop == "budget_cost" and len(client.messages.calls) == 3
    assert out.cost_usd == pytest.approx(0.6)


async def test_text_only_turn_is_nudged_to_submit(harness):
    make, _ = harness
    agent, client = make(
        [message(text("I think it's Acme.")), message(tool_use(SUBMIT_TOOL, **submission()))]
    )
    out = await agent.run(LEAD)
    assert out.stop == "submitted"
    assert SUBMIT_TOOL in client.messages.calls[1]["messages"][-1]["content"]


async def test_api_error_ends_lead_as_api_error(harness):
    make, sink = harness
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    agent, _ = make([anthropic.APIConnectionError(request=req)])
    out = await agent.run(LEAD)
    assert out.stop == "api_error" and out.enrichment is None
    assert sink.spans[0]["kind"] == "model_call" and sink.spans[0]["status"] == "error"


async def test_refusal_stops_the_loop(harness):
    make, _ = harness
    agent, client = make([message(text(""), stop_reason="refusal")])
    out = await agent.run(LEAD)
    assert out.stop == "refusal" and len(client.messages.calls) == 1


async def test_tool_failure_is_returned_to_model_not_raised(harness):
    make, _ = harness
    agent, client = make(
        [
            message(tool_use("fetch_company_site", url="https://acme.example/missing")),
            message(tool_use(SUBMIT_TOOL, **submission())),
        ]
    )
    out = await agent.run(LEAD)
    assert out.stop == "submitted"
    result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] and "404" in result["content"]


def test_prompt_config_renders_lessons_and_changes_fingerprint():
    base = PromptConfig()
    tuned = PromptConfig(lessons=(Lesson("title missing", "use seniority 'unknown'"),))
    assert "title missing → use seniority 'unknown'" in tuned.render()
    assert base.fingerprint != tuned.fingerprint
