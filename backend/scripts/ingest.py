"""Embed FEVER passages and SciFact abstracts and store them in Qdrant (offline).

Safe to rerun: each point ID is derived from the document ID (UUID5), so a rerun
overwrites the same points instead of creating duplicates.

Usage (from backend/, with Qdrant running):
    uv run python -m scripts.ingest
"""

import json
import logging
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from qdrant_client import models

from app.services.retrieval_service import RetrievalService

DATA = Path(__file__).resolve().parent.parent / "data"
FEVER_FILE = DATA / "fever" / "fever_passages.jsonl"
SCIFACT_FILE = DATA / "scifact" / "raw" / "data" / "corpus.jsonl"
UPSERT_BATCH = 512

Document = dict[str, Any]  # payload: doc_id, source, title, text, + metadata


def load_fever(path: Path = FEVER_FILE) -> Iterator[Document]:
    with path.open() as f:
        for line in f:
            p = json.loads(line)
            yield {
                "doc_id": p["id"],
                "source": "fever",
                "title": p["title"],
                "text": p["text"],
                "page_id": p["page_id"],
                "sentence_id": p["sentence_id"],
                "claim_ids": p["claim_ids"],
            }


def load_scifact(path: Path = SCIFACT_FILE) -> Iterator[Document]:
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            yield {
                "doc_id": f"scifact:{d['doc_id']}",
                "source": "scifact",
                "title": d["title"],
                "text": " ".join(s.strip() for s in d["abstract"]),
                "scifact_doc_id": d["doc_id"],
                "structured": d["structured"],
            }


def point_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


def embedding_text(doc: Document) -> str:
    # Title gives FEVER sentences their subject ("He was..." -> which page?).
    return f"{doc['title']}. {doc['text']}"


def ingest(service: RetrievalService, docs: Iterator[Document], batch_size: int = UPSERT_BATCH) -> int:
    service.ensure_collection()
    total = 0
    batch: list[Document] = []

    def flush() -> None:
        vectors = service.embedder.embed([embedding_text(d) for d in batch])
        service.client.upsert(
            service.collection,
            points=[
                models.PointStruct(id=point_id(d["doc_id"]), vector=v, payload=d)
                for d, v in zip(batch, vectors, strict=True)
            ],
        )

    for doc in docs:
        batch.append(doc)
        if len(batch) == batch_size:
            flush()
            total += len(batch)
            batch = []
            logging.info("upserted %d", total)
    if batch:
        flush()
        total += len(batch)
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    service = RetrievalService.from_settings()
    for name, loader in [("fever", load_fever), ("scifact", load_scifact)]:
        start = time.perf_counter()
        n = ingest(service, loader())
        logging.info("%s: %d documents in %.1fs", name, n, time.perf_counter() - start)
    count = service.client.count(service.collection, exact=True).count
    logging.info("collection %s now holds %d points", service.collection, count)


if __name__ == "__main__":
    main()
