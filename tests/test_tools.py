import json

import httpx2
import pytest

from leadgenie.store import ResultRow, Store
from leadgenie.tools import EvidenceLog, World, build_registry, fixture_backends
from leadgenie.tools import web as web_mod
from leadgenie.tools.search import UnconfiguredSearch, search_tool
from leadgenie.tools.web import FetchBlocked, LiveFetcher, html_to_text, is_denylisted

WORLD = World(
    pages={
        "https://acme.example/": {
            "title": "Acme Robotics — warehouse robots",
            "description": "Acme builds autonomous forklifts for logistics warehouses.",
            "text": "Founded 2019. Team: Ada Example, CTO.",
        },
        "https://acme.example/team": {
            "title": "Team",
            "text": "Ada Example — Chief Technology Officer",
            "unlisted": True,
        },
        "https://acme-bakery.example/": {
            "title": "Acme Bakery",
            "text": "Fresh bread daily.",
        },
    },
    mx={"acme.example": ["mx1.acme.example"], "nomail.example": []},
)


@pytest.fixture
def registry(tmp_path):
    store = Store(tmp_path / "t.db")
    yield build_registry(fixture_backends(WORLD), store), store
    store.close()


async def test_fetch_returns_text_and_document(registry):
    reg, _ = registry
    res = await reg.call("fetch_company_site", {"url": "acme.example"})
    assert not res.is_error
    page = json.loads(res.content)
    assert "autonomous forklifts" in page["description"]
    assert "https://acme.example/" in res.documents


async def test_fetch_missing_page_is_tool_error_not_exception(registry):
    reg, _ = registry
    res = await reg.call("fetch_company_site", {"url": "https://acme.example/nope"})
    assert res.is_error and "404" in res.content


@pytest.mark.parametrize(
    "url", ["https://www.linkedin.com/in/someone", "https://de.linkedin.com/company/x"]
)
async def test_denylisted_sites_are_never_fetched(registry, url):
    reg, _ = registry
    res = await reg.call("fetch_company_site", {"url": url})
    assert res.is_error and res.content.startswith("blocked")


def test_denylist_matches_subdomains_only_on_label_boundary():
    assert is_denylisted("linkedin.com")
    assert is_denylisted("uk.linkedin.com")
    assert not is_denylisted("notlinkedin.com")


async def test_search_ranks_and_hides_unlisted(registry):
    reg, _ = registry
    res = await reg.call("web_search", {"query": "acme robotics warehouse"})
    results = json.loads(res.content)["results"]
    assert results[0]["url"] == "https://acme.example/"
    assert all("team" not in r["url"] for r in results)


async def test_unconfigured_search_tells_agent_what_to_do_instead():
    res = await search_tool(UnconfiguredSearch()).handler({"query": "x"})
    assert res.is_error and "fetch" in res.content


async def test_mx_lookup_cases(registry):
    reg, _ = registry
    ok = json.loads((await reg.call("check_mx", {"domain": "ACME.example."})).content)
    none = json.loads((await reg.call("check_mx", {"domain": "nomail.example"})).content)
    missing = json.loads((await reg.call("check_mx", {"domain": "ghost.example"})).content)
    assert ok["has_mx"] and ok["domain"] == "acme.example"
    assert none["exists"] and not none["has_mx"]
    assert not missing["exists"]


async def test_mx_rejects_garbage(registry):
    reg, _ = registry
    res = await reg.call("check_mx", {"domain": "not a domain"})
    assert res.is_error


async def test_prior_results_only_returns_approved(registry):
    reg, store = registry
    store.start_run("r", "m", {})
    for lead_id, status in [("a", "approved"), ("b", "review")]:
        store.record_result(
            ResultRow(
                lead_id=lead_id,
                run_id="r",
                name="n",
                raw_company="acme inc",
                title=None,
                status=status,
                reasons=[],
                confidence=0.9,
                enrichment={"company": {"value": "Acme Robotics"}},
            )
        )
    res = await reg.call("lookup_prior_results", {"company": "Acme Robotics"})
    refs = [r["ref"] for r in json.loads(res.content)["results"]]
    assert refs == ["prior:a"]
    assert "prior:a" in res.documents


async def test_unknown_tool_and_handler_crash_become_errors(registry):
    reg, _ = registry
    assert (await reg.call("nope", {})).is_error
    assert (await reg.call("check_mx", {})).is_error  # KeyError inside handler


def test_tool_params_are_strict_and_sorted(registry):
    reg, _ = registry
    params = reg.params()
    assert [p["name"] for p in params] == sorted(p["name"] for p in params)
    assert all(p["strict"] and not p["input_schema"]["additionalProperties"] for p in params)


def test_evidence_log_checks_quotes_against_retrieved_text():
    log = EvidenceLog()
    log.add({"https://acme.example/": "Ada Example —  Chief Technology\nOfficer"})
    assert log.supports("https://acme.example/", "chief technology officer")
    assert not log.supports("https://acme.example/", "CEO")
    assert not log.supports("https://other.example/", "Ada")
    assert not log.supports("https://acme.example/", "  ")


def test_html_to_text_strips_scripts_and_reads_meta():
    html = (
        "<html><head><title>Acme</title><meta name='description' content='Robots'>"
        "<script>var x = 'secret';</script></head><body><p>Hello</p></body></html>"
    )
    out = html_to_text(html)
    assert out == {"title": "Acme", "description": "Robots", "text": "Hello"}


# --- live fetcher policy, exercised against a mock transport (no network) -------------


def _live_fetcher(handler, monkeypatch, public=True):
    async def fake_public(host):
        if not public:
            raise FetchBlocked(f"{host} resolves to non-public address 10.0.0.1")

    monkeypatch.setattr(web_mod, "_assert_public_host", fake_public)
    f = LiveFetcher(per_host_rate=1000)
    f._client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return f


async def test_live_fetch_respects_robots(monkeypatch):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx2.Response(200, text="User-agent: *\nDisallow: /private")
        return httpx2.Response(200, html="<title>ok</title>")

    f = _live_fetcher(handler, monkeypatch)
    assert (await f.fetch("https://acme.example/"))["title"] == "ok"
    with pytest.raises(FetchBlocked, match="robots"):
        await f.fetch("https://acme.example/private/page")


async def test_live_fetch_rechecks_policy_on_redirect(monkeypatch):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx2.Response(404)
        return httpx2.Response(302, headers={"location": "https://www.linkedin.com/company/x"})

    f = _live_fetcher(handler, monkeypatch)
    with pytest.raises(FetchBlocked, match="linkedin"):
        await f.fetch("https://acme.example/")


async def test_live_fetch_refuses_private_addresses(monkeypatch):
    f = _live_fetcher(lambda r: httpx2.Response(200), monkeypatch, public=False)
    with pytest.raises(FetchBlocked, match="non-public"):
        await f.fetch("http://internal.example/")


async def test_live_fetch_rejects_non_http_schemes():
    with pytest.raises(FetchBlocked):
        await LiveFetcher().fetch("file:///etc/passwd")


async def test_mx_evidence_matches_what_the_model_sees(registry):
    # Regression: the document was a Python repr ('has_mx': True) while the model saw
    # JSON ("has_mx": true), so faithful quotes failed verification.
    reg, _ = registry
    res = await reg.call("check_mx", {"domain": "acme.example"})
    log = EvidenceLog()
    log.add(res.documents)
    assert log.supports("dns:mx:acme.example", '"has_mx": true')


def test_evidence_accepts_json_escaped_quotes():
    log = EvidenceLog()
    log.add({"https://a.example/": 'We say "hello" — always.'})
    assert log.supports("https://a.example/", 'say \\"hello\\" \\u2014 always')
    assert log.supports("https://a.example/", 'say "hello"')
