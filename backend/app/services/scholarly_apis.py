"""Async REST clients for three scholarly search APIs.

Each provider returns different data; each client normalizes it into `Publication`
and only fills fields the provider actually returned.

Semantic Scholar  GET https://api.semanticscholar.org/graph/v1/paper/search
    params: query, fields, limit, year ("2018-2024", "2018-", "-2020"),
            publicationTypes (e.g. "Review"), fieldsOfStudy (e.g. "Medicine").
    Optional key: header `x-api-key`. Without a key, requests share a public rate
    limit and often get HTTP 429.
OpenAlex          GET https://api.openalex.org/works
    params: search, per_page, filter ("publication_year:2018-2024,type:review").
    Abstracts come as an inverted index (word -> positions). Optional `api_key`;
    without it, requests share a small free daily budget per IP.
Europe PMC        GET https://www.ebi.ac.uk/europepmc/webservices/rest/search
    params: query (supports PUB_YEAR:[a TO b], PUB_TYPE:"review"), format=json,
            resultType=core (includes abstracts), pageSize. No key required.
"""

import logging
import re
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_URL = "https://api.openalex.org/works"
EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

S2_FIELDS = "title,abstract,year,authors,externalIds,url,venue,citationCount,tldr,publicationTypes"
OPENALEX_SELECT = (
    "id,doi,display_name,publication_year,authorships,abstract_inverted_index,"
    "primary_location,cited_by_count,type"
)


class ScholarlyAPIError(Exception):
    """A provider request failed (HTTP error, timeout or malformed response)."""


class ScholarlyRateLimitError(ScholarlyAPIError):
    """The provider returned HTTP 429."""


class SearchFilters(BaseModel):
    """Filters every provider supports (mapped to each provider's own syntax)."""

    year_from: int | None = Field(default=None, ge=1800, le=2100)
    year_to: int | None = Field(default=None, ge=1800, le=2100)
    reviews_only: bool = False  # reviews / meta-analyses only


class Publication(BaseModel):
    provider: str  # "Semantic Scholar" | "OpenAlex" | "Europe PMC"
    provider_id: str
    title: str
    abstract: str | None = None
    summary: str | None = None  # Semantic Scholar's machine-generated TLDR, if any
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    doi: str | None = None  # bare DOI, e.g. "10.1001/jama.2019.1"
    url: str | None = None
    citation_count: int | None = None
    publication_types: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        """Identity used to deduplicate across providers."""
        return f"doi:{self.doi.lower()}" if self.doi else f"{self.provider}:{self.provider_id}"


class ScholarlyClient:
    """Holds one shared httpx.AsyncClient. Use as `async with ScholarlyClient(...) as c`."""

    def __init__(
        self,
        semantic_scholar_api_key: str | None = None,
        openalex_api_key: str | None = None,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,  # for tests
    ) -> None:
        self._s2_key = semantic_scholar_api_key or None
        self._openalex_key = openalex_api_key or None
        self._http = httpx.AsyncClient(
            timeout=timeout, transport=transport, headers={"User-Agent": "TruthLens/0.1"}
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._http.aclose()

    # --- Semantic Scholar -------------------------------------------------------

    async def search_semantic_scholar(
        self, query: str, filters: SearchFilters | None = None, limit: int = 5,
        fields_of_study: list[str] | None = None,
    ) -> list[Publication]:
        filters = filters or SearchFilters()
        params: dict[str, Any] = {"query": query, "limit": limit, "fields": S2_FIELDS}
        if filters.year_from or filters.year_to:  # S2 accepts "2018-2024", "2018-", "-2020"
            params["year"] = f"{filters.year_from or ''}-{filters.year_to or ''}"
        if filters.reviews_only:
            params["publicationTypes"] = "Review,MetaAnalysis"
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        headers = {"x-api-key": self._s2_key} if self._s2_key else {}
        data = await self._get("Semantic Scholar", SEMANTIC_SCHOLAR_URL, params, headers)
        return [p for item in _as_list(data, "data") if (p := parse_semantic_scholar(item))]

    # --- OpenAlex ---------------------------------------------------------------

    async def search_openalex(
        self, query: str, filters: SearchFilters | None = None, limit: int = 5
    ) -> list[Publication]:
        filters = filters or SearchFilters()
        params: dict[str, Any] = {"search": query, "per_page": limit, "select": OPENALEX_SELECT}
        parts = ["is_retracted:false"]
        if filters.year_from or filters.year_to:  # OpenAlex needs a closed range
            parts.append(f"publication_year:{filters.year_from or 1800}-{filters.year_to or 2100}")
        if filters.reviews_only:
            parts.append("type:review")
        params["filter"] = ",".join(parts)
        if self._openalex_key:
            params["api_key"] = self._openalex_key
        data = await self._get("OpenAlex", OPENALEX_URL, params)
        return [p for item in _as_list(data, "results") if (p := parse_openalex(item))]

    # --- Europe PMC -------------------------------------------------------------

    async def search_europe_pmc(
        self, query: str, filters: SearchFilters | None = None, limit: int = 5
    ) -> list[Publication]:
        filters = filters or SearchFilters()
        q = f"({query})"
        if filters.year_from or filters.year_to:
            q += f" AND PUB_YEAR:[{filters.year_from or 1800} TO {filters.year_to or 2100}]"
        if filters.reviews_only:
            q += ' AND PUB_TYPE:"review"'
        params = {"query": q, "format": "json", "resultType": "core", "pageSize": limit}
        data = await self._get("Europe PMC", EUROPE_PMC_URL, params)
        result_list = data.get("resultList") if isinstance(data, dict) else None
        return [p for item in _as_list(result_list, "result") if (p := parse_europe_pmc(item))]

    # --- HTTP -------------------------------------------------------------------

    async def _get(
        self, provider: str, url: str, params: dict[str, Any], headers: dict[str, str] | None = None
    ) -> Any:
        try:
            response = await self._http.get(url, params=params, headers=headers)
        except httpx.TimeoutException as exc:
            raise ScholarlyAPIError(f"{provider}: request timed out") from exc
        except httpx.HTTPError as exc:
            raise ScholarlyAPIError(f"{provider}: request failed ({type(exc).__name__})") from exc
        if response.status_code == 429:
            raise ScholarlyRateLimitError(f"{provider}: rate limited (HTTP 429)")
        if response.status_code >= 400:
            raise ScholarlyAPIError(f"{provider}: HTTP {response.status_code} {response.text[:200]}")
        try:
            return response.json()
        except ValueError as exc:
            raise ScholarlyAPIError(f"{provider}: response is not valid JSON") from exc


# --- Parsers (one per provider) -------------------------------------------------

def parse_semantic_scholar(item: Any) -> Publication | None:
    if not isinstance(item, dict) or not item.get("title") or not item.get("paperId"):
        return None
    ids = item.get("externalIds") or {}
    tldr = item.get("tldr") or {}
    return Publication(
        provider="Semantic Scholar",
        provider_id=str(item["paperId"]),
        title=item["title"].strip(),
        abstract=_clean(item.get("abstract")),
        summary=_clean(tldr.get("text")) if isinstance(tldr, dict) else None,
        year=_int(item.get("year")),
        authors=[a["name"] for a in item.get("authors") or [] if isinstance(a, dict) and a.get("name")],
        venue=_clean(item.get("venue")),
        doi=_clean(ids.get("DOI")) if isinstance(ids, dict) else None,
        url=item.get("url"),
        citation_count=_int(item.get("citationCount")),
        publication_types=[t for t in item.get("publicationTypes") or [] if isinstance(t, str)],
    )


def parse_openalex(item: Any) -> Publication | None:
    if not isinstance(item, dict) or not item.get("display_name") or not item.get("id"):
        return None
    doi = item.get("doi")
    if isinstance(doi, str):
        doi = doi.removeprefix("https://doi.org/")
    location = item.get("primary_location") or {}
    source = location.get("source") or {} if isinstance(location, dict) else {}
    authors = [
        a["author"]["display_name"]
        for a in item.get("authorships") or []
        if isinstance(a, dict) and isinstance(a.get("author"), dict) and a["author"].get("display_name")
    ]
    return Publication(
        provider="OpenAlex",
        provider_id=str(item["id"]).removeprefix("https://openalex.org/"),
        title=item["display_name"].strip(),
        abstract=reconstruct_abstract(item.get("abstract_inverted_index")),
        year=_int(item.get("publication_year")),
        authors=authors,
        venue=_clean(source.get("display_name")) if isinstance(source, dict) else None,
        doi=doi or None,
        url=(location.get("landing_page_url") if isinstance(location, dict) else None) or item["id"],
        citation_count=_int(item.get("cited_by_count")),
        publication_types=[item["type"]] if isinstance(item.get("type"), str) else [],
    )


def parse_europe_pmc(item: Any) -> Publication | None:
    if not isinstance(item, dict) or not item.get("title") or not item.get("id"):
        return None
    source, pid = item.get("source", ""), str(item["id"])
    journal = (item.get("journalInfo") or {}).get("journal") or {}
    pub_types = (item.get("pubTypeList") or {}).get("pubType") or []
    return Publication(
        provider="Europe PMC",
        provider_id=f"{source}:{pid}",
        title=_strip_tags(item["title"]),
        abstract=_strip_tags(item["abstractText"]) if item.get("abstractText") else None,
        year=_int(item.get("pubYear")),
        authors=[a.strip() for a in (item.get("authorString") or "").rstrip(".").split(",") if a.strip()],
        venue=_clean(journal.get("title")) if isinstance(journal, dict) else None,
        doi=_clean(item.get("doi")),
        url=f"https://europepmc.org/article/{source}/{pid}" if source else None,
        citation_count=_int(item.get("citedByCount")),
        publication_types=[t for t in pub_types if isinstance(t, str)] if isinstance(pub_types, list) else [],
    )


def reconstruct_abstract(index: Any) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not isinstance(index, dict) or not index:
        return None
    positions: dict[int, str] = {}
    for word, places in index.items():
        if isinstance(places, list):
            for p in places:
                if isinstance(p, int):
                    positions[p] = word
    return " ".join(positions[i] for i in sorted(positions)) or None


def _as_list(data: Any, key: str) -> list[Any]:
    if not isinstance(data, dict):
        raise ScholarlyAPIError(f"Malformed response: expected an object with '{key}'")
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ScholarlyAPIError(f"Malformed response: '{key}' is not a list")
    return value


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _strip_tags(text: str) -> str:
    """Europe PMC titles/abstracts may contain simple HTML tags like <i> or <h4>."""
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())
