import httpx
import pytest

from app.services.scholarly_apis import (
    ScholarlyAPIError,
    ScholarlyClient,
    ScholarlyRateLimitError,
    SearchFilters,
    reconstruct_abstract,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


S2_RESPONSE = {"total": 1, "offset": 0, "data": [{
    "paperId": "abc123", "title": "Vitamin D and fractures: a meta-analysis",
    "abstract": None, "tldr": {"model": "tldr@v2.0.0", "text": "No reduction in fractures."},
    "year": 2024, "authors": [{"authorId": "1", "name": "A. Author"}],
    "externalIds": {"DOI": "10.1007/s11606-024-08933-1", "PubMed": "38997531"},
    "url": "https://www.semanticscholar.org/paper/abc123", "venue": "J Gen Intern Med",
    "citationCount": 14, "publicationTypes": ["JournalArticle", "Review"],
}]}
OPENALEX_RESPONSE = {"meta": {"count": 1}, "results": [{
    "id": "https://openalex.org/W1", "doi": "https://doi.org/10.1001/jamanetworkopen.2019.17789",
    "display_name": "Vitamin D and Calcium for the Prevention of Fracture",
    "publication_year": 2019, "type": "review", "cited_by_count": 300,
    "abstract_inverted_index": {"Vitamin": [0], "D": [1], "helps?": [2]},
    "authorships": [{"author": {"display_name": "B. Author"}}],
    "primary_location": {"landing_page_url": "https://jamanetwork.com/x", "source": {"display_name": "JAMA Netw Open"}},
}]}
EPMC_RESPONSE = {"hitCount": 1, "resultList": {"result": [{
    "id": "38997531", "source": "MED", "pmid": "38997531", "doi": "10.1007/s11606-024-08933-1",
    "title": "Vitamin D <i>supplementation</i>.", "abstractText": "<h4>Background</h4>Text here.",
    "authorString": "Lee H, Kim J.", "pubYear": "2024", "citedByCount": 3,
    "journalInfo": {"journal": {"title": "J Gen Intern Med"}},
    "pubTypeList": {"pubType": ["Journal Article", "Meta-Analysis"]},
}]}}


def client_with(handler) -> ScholarlyClient:
    return ScholarlyClient(semantic_scholar_api_key="s2-key", transport=httpx.MockTransport(handler))


async def test_semantic_scholar_parsing_and_params() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"], seen["headers"] = dict(request.url.params), request.headers
        return httpx.Response(200, json=S2_RESPONSE)

    async with client_with(handler) as c:
        pubs = await c.search_semantic_scholar(
            "vitamin D fractures", SearchFilters(year_from=2015, reviews_only=True), limit=3,
            fields_of_study=["Medicine"],
        )
    assert seen["params"]["year"] == "2015-"
    assert seen["params"]["publicationTypes"] == "Review,MetaAnalysis"
    assert seen["params"]["fieldsOfStudy"] == "Medicine" and seen["params"]["limit"] == "3"
    assert seen["headers"]["x-api-key"] == "s2-key"
    p = pubs[0]
    assert (p.provider, p.doi, p.year, p.authors) == ("Semantic Scholar", "10.1007/s11606-024-08933-1", 2024, ["A. Author"])
    assert p.abstract is None and p.summary == "No reduction in fractures."


async def test_openalex_parsing_and_filter() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, json=OPENALEX_RESPONSE)

    async with client_with(handler) as c:
        pubs = await c.search_openalex("vitamin D", SearchFilters(year_from=2018, year_to=2024, reviews_only=True))
    assert seen["filter"] == "is_retracted:false,publication_year:2018-2024,type:review"
    assert "api_key" not in seen
    p = pubs[0]
    assert p.doi == "10.1001/jamanetworkopen.2019.17789" and p.provider_id == "W1"
    assert p.abstract == "Vitamin D helps?" and p.venue == "JAMA Netw Open"
    assert p.url == "https://jamanetwork.com/x"


async def test_europe_pmc_parsing_and_query_syntax() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, json=EPMC_RESPONSE)

    async with client_with(handler) as c:
        pubs = await c.search_europe_pmc("vitamin D", SearchFilters(year_to=2020, reviews_only=True))
    assert seen["query"] == '(vitamin D) AND PUB_YEAR:[1800 TO 2020] AND PUB_TYPE:"review"'
    assert seen["resultType"] == "core" and seen["format"] == "json"
    p = pubs[0]
    assert p.title == "Vitamin D supplementation ." and p.abstract == "Background Text here."
    assert p.authors == ["Lee H", "Kim J"] and p.url == "https://europepmc.org/article/MED/38997531"
    assert p.publication_types == ["Journal Article", "Meta-Analysis"]


@pytest.mark.parametrize("method,body", [
    ("search_semantic_scholar", {"total": 0, "offset": 0}),  # S2 omits "data" when empty
    ("search_openalex", {"meta": {}, "results": []}),
    ("search_europe_pmc", {"hitCount": 0, "resultList": {"result": []}}),
])
async def test_empty_results(method: str, body: dict) -> None:
    async with client_with(lambda r: httpx.Response(200, json=body)) as c:
        assert await getattr(c, method)("nothing") == []


async def test_items_without_title_are_skipped() -> None:
    body = {"data": [{"paperId": "x"}, S2_RESPONSE["data"][0]]}
    async with client_with(lambda r: httpx.Response(200, json=body)) as c:
        assert len(await c.search_semantic_scholar("q")) == 1


@pytest.mark.parametrize("response", [
    httpx.Response(200, text="<html>not json</html>"),
    httpx.Response(200, json={"results": "not a list"}),
    httpx.Response(200, json=["unexpected"]),
])
async def test_malformed_responses_raise(response: httpx.Response) -> None:
    async with client_with(lambda r: response) as c:
        with pytest.raises(ScholarlyAPIError):
            await c.search_openalex("q")


async def test_rate_limit_and_http_errors() -> None:
    async with client_with(lambda r: httpx.Response(429, json={"code": "429"})) as c:
        with pytest.raises(ScholarlyRateLimitError):
            await c.search_semantic_scholar("q")
    async with client_with(lambda r: httpx.Response(500, text="boom")) as c:
        with pytest.raises(ScholarlyAPIError, match="HTTP 500"):
            await c.search_europe_pmc("q")


async def test_timeout_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async with client_with(handler) as c:
        with pytest.raises(ScholarlyAPIError, match="timed out"):
            await c.search_openalex("q")


def test_reconstruct_abstract_handles_bad_input() -> None:
    assert reconstruct_abstract({"b": [1], "a": [0]}) == "a b"
    assert reconstruct_abstract(None) is None and reconstruct_abstract({}) is None
