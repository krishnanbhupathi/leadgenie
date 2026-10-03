import json

import pytest

from leadgenie.pricing import TokenUsage, cost_usd, price_for
from leadgenie.report import build_report, percentile, render_markdown, write_report
from leadgenie.store import Store
from leadgenie.tracing import MAX_PAYLOAD_CHARS, JsonlSink, MemorySink, Tracer


def test_cost_includes_cache_tokens():
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )
    # opus-5-5: 4 in + 20 out + 0.20 cache read + 5.00 cache write (1.25x)
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(29.20)


def test_unknown_model_price_raises():
    with pytest.raises(KeyError):
        price_for("gpt-whatever")


def test_usage_from_api_tolerates_missing_fields():
    class U:
        input_tokens = 10
        output_tokens = 5
        cache_read_input_tokens = None

    assert TokenUsage.from_api(U()) == TokenUsage(10, 5, 0, 0)


def test_spans_nest_and_record_errors():
    sink = MemorySink()
    tracer = Tracer("run1", [sink])
    with tracer.span("lead", "enrich", lead_id="L1") as lead:
        with tracer.span("tool_call", "check_mx", parent=lead, input={"domain": "a.example"}):
            pass
        with pytest.raises(ValueError), tracer.span("model_call", "messages.create", parent=lead):
            raise ValueError("boom")
    tool, model, lead_rec = sink.spans
    assert tool["parent_id"] == lead_rec["span_id"] and tool["lead_id"] == "L1"
    assert model["status"] == "error" and "boom" in model["error"]
    assert lead_rec["status"] == "ok" and lead_rec["latency_ms"] is not None


def test_large_payloads_are_clipped():
    sink = MemorySink()
    with Tracer("r", [sink]).span("tool_call", "t", input="x" * (MAX_PAYLOAD_CHARS + 50)):
        pass
    assert sink.spans[0]["input"].endswith(f"[truncated, {MAX_PAYLOAD_CHARS + 50} chars]")


def test_store_and_jsonl_sinks_roundtrip(tmp_path):
    store = Store(tmp_path / "t.db")
    jsonl = tmp_path / "spans.jsonl"
    tracer = Tracer("run1", [store, JsonlSink(str(jsonl))])
    with tracer.span("tool_call", "web_search", lead_id="L", input={"query": "q"}) as s:
        s.output = {"results": []}
        s.attrs["n"] = 0
    [row] = store.spans("run1")
    assert row["input"] == {"query": "q"} and row["attrs"] == {"n": 0}
    assert json.loads(jsonl.read_text())["name"] == "web_search"


def test_percentile_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([5, 1, 3], 50) == 3
    assert percentile(list(range(1, 101)), 95) == 95


def _span(kind, name, lead_id, **kw):
    base = dict(
        kind=kind,
        name=name,
        lead_id=lead_id,
        latency_ms=100,
        status="ok",
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        attrs={},
    )
    base.update(kw)
    return base


def test_report_aggregates(tmp_path):
    spans = [
        _span("lead", "enrich", "A", latency_ms=1000, attrs={"stop": "submitted"}),
        _span("lead", "enrich", "B", latency_ms=3000, attrs={"stop": "budget_steps"}),
        _span("model_call", "m", "A", cost_usd=0.01, input_tokens=100, cache_read_tokens=300),
        _span("model_call", "m", "A", cost_usd=0.01),
        _span("model_call", "m", "B", cost_usd=0.04, status="error"),
        _span("tool_call", "check_mx", "A"),
        _span("tool_call", "check_mx", "B", status="error"),
    ]
    results = [
        {"status": "approved", "reasons": "[]"},
        {"status": "review", "reasons": json.dumps(["low_confidence:0.5<0.75", "budget_steps"])},
    ]
    r = build_report("run1", spans, results)
    assert r["status"] == {"approved": 1, "review": 1, "error": 0}
    assert r["cost_usd"]["total"] == pytest.approx(0.06)
    assert r["cost_usd"]["per_lead_mean"] == pytest.approx(0.03)
    assert r["steps_per_lead"] == {"mean": 1.5, "max": 2}
    assert r["cache_hit_ratio"] == 0.75
    assert r["tools"]["check_mx"]["error_rate"] == 0.5
    assert r["review_reasons"] == {"low_confidence": 1, "budget_steps": 1}
    assert r["agent_stop_reasons"] == {"submitted": 1, "budget_steps": 1}
    assert "check_mx" in render_markdown(r)
    json_path, md_path = write_report(r, tmp_path)
    assert json.loads(json_path.read_text())["run_id"] == "run1" and md_path.exists()
