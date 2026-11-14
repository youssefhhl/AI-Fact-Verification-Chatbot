# TruthLens — AI Fact Verification Chatbot

TruthLens is a conversational fact-verification system. Users submit general,
scientific or technical claims and receive a verdict (TRUE, FALSE, PARTIALLY TRUE,
NOT ENOUGH EVIDENCE) with an explanation and cited evidence, and can ask follow-up
questions in the same conversation.

Planned stack: FastAPI, LangGraph, Groq LLM, Tavily (web), RAG over
FEVER/SciFact (Sentence Transformers + Qdrant), scientific-publication search,
PostgreSQL, React.

## RAG architecture (planned)

- **Embeddings:** Sentence Transformers, running locally, model
  `sentence-transformers/all-MiniLM-L6-v2`. It only turns text into vectors.
- **Vector store:** Qdrant, the only vector database. It stores the vectors, the
  original text and the metadata, and performs similarity search.
- **Knowledge sources:** FEVER and SciFact.
- LangChain is not used unless a concrete benefit is identified.

Offline ingestion (run once): download FEVER and SciFact, inspect and preprocess
them, extract evidence documents with their metadata, embed them, and store
vectors + text + metadata in Qdrant. Dataset embeddings are persisted and are
not regenerated per question.

Runtime: embed the user's question with the same model, search Qdrant for
relevant evidence, return the documents to the RAG Agent, and let the LLM
analyze them.

**Current status:** Phase 2 done. FastAPI app with a health endpoint, shared
Pydantic schemas (`Evidence`, `VerificationResult`, `ChatRequest`, `ChatResponse`)
and an async Groq LLM service (`app/services/llm_service.py`).

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
cd backend
cp .env.example .env     # adjust values if needed
uv sync                  # creates .venv and installs locked dependencies
```

## Configuration

Set these in `backend/.env` (see `.env.example`):

- `GROQ_API_KEY`: your Groq API key. Only needed when the LLM is called; tests do not need it.
- `GROQ_MODEL`: Groq model ID (default `llama-3.3-70b-versatile`).
- `LLM_TIMEOUT_SECONDS`: request timeout (default 30).

## Datasets (not committed)

```bash
cd backend
# FEVER (fever.ai) — ~1.7 GB, wiki dump is read directly from the zip
mkdir -p data/fever/raw && cd data/fever/raw
curl -LO https://fever.ai/download/fever/train.jsonl
curl -LO https://fever.ai/download/fever/shared_task_dev.jsonl
curl -LO https://fever.ai/download/fever/wiki-pages.zip
cd ../../..
# SciFact (AllenAI)
mkdir -p data/scifact/raw && cd data/scifact/raw
curl -LO https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz && tar -xzf data.tar.gz
cd ../../..
# Build the ~20k FEVER passage subset -> data/fever/fever_passages.jsonl
uv run python scripts/prepare_fever.py
```

## Run the application

```bash
cd backend
uv run uvicorn app.main:app --reload
```

Health check: <http://localhost:8000/health>. Interactive docs: <http://localhost:8000/docs>.

## Run the tests

```bash
cd backend
uv run pytest
```

## Dependencies

```bash
uv add <package>          # runtime
uv add --dev <package>    # development
```
