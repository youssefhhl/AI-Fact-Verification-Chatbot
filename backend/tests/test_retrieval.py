import hashlib

import pytest
from qdrant_client import QdrantClient

from app.services.retrieval_service import RetrievalService, resolve_device
from scripts.ingest import ingest, point_id

DIM = 16


class FakeEmbedder:
    """Deterministic bag-of-words embedder: shared words -> similar vectors."""

    dimension = DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * DIM
            for word in text.lower().replace(".", " ").split():
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
            out.append(v)
        return out


DOCS = [
    {"doc_id": "fever:Paris:0", "source": "fever", "title": "Paris",
     "text": "Paris is the capital of France", "page_id": "Paris", "sentence_id": 0},
    {"doc_id": "fever:Moon:0", "source": "fever", "title": "Moon",
     "text": "The Moon orbits the Earth", "page_id": "Moon", "sentence_id": 0},
    {"doc_id": "scifact:42", "source": "scifact", "title": "Vitamin D",
     "text": "Vitamin D supplementation reduces fractures", "scifact_doc_id": 42},
]


@pytest.fixture
def service() -> RetrievalService:
    return RetrievalService(QdrantClient(":memory:"), FakeEmbedder(), "test")


def test_ingest_and_search_returns_metadata(service: RetrievalService) -> None:
    assert ingest(service, iter(DOCS), batch_size=2) == 3

    hits = service.search_sync("capital of France Paris", top_k=1)

    assert hits[0].id == "fever:Paris:0"
    assert hits[0].source == "fever"
    assert hits[0].text == "Paris is the capital of France"
    assert hits[0].metadata == {"page_id": "Paris", "sentence_id": 0}
    assert -1.0 <= hits[0].score <= 1.0


def test_ingest_is_idempotent(service: RetrievalService) -> None:
    ingest(service, iter(DOCS))
    ingest(service, iter(DOCS))
    assert service.client.count("test").count == 3


def test_source_filter(service: RetrievalService) -> None:
    ingest(service, iter(DOCS))
    hits = service.search_sync("capital of France", top_k=5, source="scifact")
    assert [h.id for h in hits] == ["scifact:42"]


@pytest.mark.anyio
async def test_async_search(service: RetrievalService) -> None:
    ingest(service, iter(DOCS))
    hits = await service.search("Moon Earth", top_k=2)
    assert hits[0].id == "fever:Moon:0"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_point_id_is_deterministic() -> None:
    assert point_id("scifact:42") == point_id("scifact:42") != point_id("scifact:43")


def test_resolve_device(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    assert resolve_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cuda") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


# --- search_evidence (tool entry point) -------------------------------------

from unittest.mock import AsyncMock  # noqa: E402

from app.schemas.evidence import Evidence  # noqa: E402
from app.services.retrieval_service import (  # noqa: E402
    RetrievalError,
    RetrievedPassage,
    source_url,
    to_evidence,
)

FEVER_HIT = RetrievedPassage(
    id="fever:Oliver_Reed:0", source="fever", title="Oliver Reed",
    text="Oliver Reed was an English actor.", score=0.77,
    metadata={"page_id": "Oliver_Reed", "sentence_id": 0},
)
SCIFACT_HIT = RetrievedPassage(
    id="scifact:42", source="scifact", title="Vitamin D trial",
    text="Vitamin D did not reduce fractures.", score=0.55, metadata={"scifact_doc_id": 42},
)
WEAK_HIT = FEVER_HIT.model_copy(update={"id": "fever:X:0", "text": "Unrelated.", "score": 0.1})


@pytest.mark.anyio
async def test_search_evidence_returns_evidence_without_stance(service: RetrievalService) -> None:
    service.search = AsyncMock(return_value=[FEVER_HIT, SCIFACT_HIT, WEAK_HIT])

    evidence = await service.search_evidence("  Oliver Reed was an actor  ", top_k=3)

    service.search.assert_awaited_once_with("Oliver Reed was an actor", 3, None)
    assert all(isinstance(e, Evidence) and e.stance is None for e in evidence)
    assert [e.content for e in evidence] == [FEVER_HIT.text, SCIFACT_HIT.text]  # weak hit dropped
    assert evidence[0].source == "FEVER (Wikipedia): Oliver Reed"
    assert evidence[0].relevance == 0.77
    assert str(evidence[1].url) == "https://api.semanticscholar.org/CorpusID:42"


@pytest.mark.anyio
async def test_search_evidence_end_to_end_in_memory(service: RetrievalService) -> None:
    ingest(service, iter(DOCS))
    service.min_score = 0.0
    evidence = await service.search_evidence("capital of France Paris", top_k=1)
    assert str(evidence[0].url) == "https://en.wikipedia.org/wiki/Paris"


@pytest.mark.anyio
async def test_search_evidence_wraps_backend_errors(service: RetrievalService) -> None:
    service.search = AsyncMock(side_effect=ConnectionError("qdrant down"))
    with pytest.raises(RetrievalError):
        await service.search_evidence("claim")


@pytest.mark.anyio
async def test_search_evidence_rejects_empty_claim(service: RetrievalService) -> None:
    with pytest.raises(ValueError):
        await service.search_evidence("   ")


def test_fever_url_restores_brackets() -> None:
    hit = FEVER_HIT.model_copy(update={"metadata": {"page_id": "Gladiator_-LRB-2000_film-RRB-"}})
    assert source_url(hit) == "https://en.wikipedia.org/wiki/Gladiator_%282000_film%29"


def test_relevance_is_clamped() -> None:
    assert to_evidence(FEVER_HIT.model_copy(update={"score": 1.0000002})).relevance == 1.0
    assert to_evidence(FEVER_HIT.model_copy(update={"score": -0.2})).relevance == 0.0
