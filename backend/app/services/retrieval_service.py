"""Local knowledge-base retrieval (RAG): Sentence Transformers + Qdrant.

This is a retrieval tool, not a reasoning agent. It embeds a query, finds similar
FEVER/SciFact passages in Qdrant and returns them as Evidence. It does not call the
LLM or judge stance; the central Fact-Checking Agent does that.

Sentence Transformers turns text into vectors; Qdrant stores and searches them.
The same Embedder is used by offline ingestion and by runtime queries.
"""

import asyncio
import logging
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from app.config import Settings, get_settings
from app.schemas.evidence import Evidence

logger = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> str:
    """Return "cuda" when requested/available, otherwise "cpu"."""
    import torch

    if preference == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if preference == "cuda":
        logger.warning("CUDA requested but not available, falling back to CPU")
    return "cpu"


class Embedder:
    """Wraps a Sentence Transformers model. Loads it once."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 128) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, done lazily

        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=self.device)
        self.dimension: int = self._model.get_embedding_dimension()
        logger.info("Loaded %s on %s (dim=%d)", model_name, self.device, self.dimension)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Normalized vectors -> cosine similarity == dot product.
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


class RetrievedPassage(BaseModel):
    id: str
    source: str  # "fever" or "scifact"
    title: str
    text: str
    score: float  # cosine similarity
    metadata: dict[str, Any]


class RetrievalError(Exception):
    """Raised when the knowledge base cannot be searched (e.g. Qdrant is down)."""


class RetrievalService:
    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        collection: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.top_k = top_k
        self.min_score = min_score

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RetrievalService":
        settings = settings or get_settings()
        embedder = Embedder(
            settings.embedding_model, settings.embedding_device, settings.embedding_batch_size
        )
        return cls(
            QdrantClient(url=settings.qdrant_url),
            embedder,
            settings.qdrant_collection,
            top_k=settings.rag_top_k,
            min_score=settings.rag_min_score,
        )

    def ensure_collection(self) -> None:
        """Create the collection if missing (safe to call repeatedly)."""
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(
                size=self.embedder.dimension, distance=models.Distance.COSINE
            ),
        )
        self.client.create_payload_index(
            self.collection, "source", models.PayloadSchemaType.KEYWORD
        )

    def search_sync(
        self, query: str, top_k: int = 5, source: str | None = None
    ) -> list[RetrievedPassage]:
        vector = self.embedder.embed([query])[0]
        query_filter = None
        if source:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
            )
        hits = self.client.query_points(
            self.collection, query=vector, limit=top_k, query_filter=query_filter,
            with_payload=True,
        ).points
        return [self._to_passage(hit) for hit in hits]

    async def search(
        self, query: str, top_k: int = 5, source: str | None = None
    ) -> list[RetrievedPassage]:
        """Async wrapper: embedding is CPU/GPU-bound, so run it off the event loop."""
        return await asyncio.to_thread(self.search_sync, query, top_k, source)

    async def search_evidence(
        self, claim: str, top_k: int | None = None, source: str | None = None
    ) -> list[Evidence]:
        """Tool entry point: relevant knowledge-base passages for a claim, as Evidence.

        Passages below `min_score` are dropped. Stance is left unset (None).
        """
        claim = claim.strip()
        if not claim:
            raise ValueError("claim must not be empty")
        try:
            passages = await self.search(claim, top_k or self.top_k, source)
        except Exception as exc:  # Qdrant down, collection missing, ...
            raise RetrievalError(f"Knowledge base search failed: {exc}") from exc
        return [to_evidence(p) for p in passages if p.score >= self.min_score]

    @staticmethod
    def _to_passage(hit: models.ScoredPoint) -> RetrievedPassage:
        payload = dict(hit.payload or {})
        return RetrievedPassage(
            id=payload.pop("doc_id"),
            source=payload.pop("source"),
            title=payload.pop("title", ""),
            text=payload.pop("text"),
            score=hit.score,
            metadata=payload,
        )


SOURCE_LABELS = {"fever": "FEVER (Wikipedia)", "scifact": "SciFact"}


def source_url(passage: RetrievedPassage) -> str | None:
    if passage.source == "fever" and "page_id" in passage.metadata:
        page = passage.metadata["page_id"]
        for token, char in {"-LRB-": "(", "-RRB-": ")", "-COLON-": ":"}.items():
            page = page.replace(token, char)
        return f"https://en.wikipedia.org/wiki/{quote(page)}"
    if passage.source == "scifact" and "scifact_doc_id" in passage.metadata:
        # SciFact doc IDs are Semantic Scholar corpus IDs.
        return f"https://api.semanticscholar.org/CorpusID:{passage.metadata['scifact_doc_id']}"
    return None


def to_evidence(passage: RetrievedPassage) -> Evidence:
    label = SOURCE_LABELS.get(passage.source, passage.source)
    return Evidence(
        source=f"{label}: {passage.title}" if passage.title else label,
        url=source_url(passage),
        content=passage.text,
        relevance=min(max(passage.score, 0.0), 1.0),
    )
