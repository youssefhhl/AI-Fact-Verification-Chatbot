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
relevant evidence, and return it as `Evidence`. RAG is a retrieval tool, not an
agent: the central Fact-Checking Agent analyzes the evidence.

## Architecture

One central **Fact-Checking Agent** reasons and orchestrates. It resolves the
conversation context, decides whether to reuse evidence or research, calls the
research components, then judges stance, resolves contradictions and produces
the verdict. Two specialized **research agents** (Web via Tavily MCP, scientific
publications) do their own research. **RAG** over FEVER/SciFact is a retrieval
tool, not an agent. LangGraph runs the workflow.

**Current status:** Phase 4.1 done (Web Agent over Tavily MCP). Phase 3 done (embeddings, Qdrant, ingestion, retrieval tool). Phase 2 done: FastAPI app with a health endpoint, shared
Pydantic schemas (`Evidence`, `VerificationResult`, `ChatRequest`, `ChatResponse`)
and an async Groq LLM service (`app/services/llm_service.py`).

## Web Agent

`app/agents/web_agent.py` is an LLM-driven research loop. The Groq LLM picks the next
action by tool calling (`web_search`, `extract_pages`, `finish_research`). A Python
controller runs each tool through an MCP client connected to Tavily's remote MCP
server (`tavily_search`, `tavily_extract`), validates the arguments, and enforces
hard limits: 3 searches, 5 extracted URLs and 5 LLM turns. It only extracts URLs
found by its own searches and never extracts the same URL twice. It returns
`Evidence` with `stance=None`; judging the claim is the Fact-Checking Agent's job.

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
- `EMBEDDING_MODEL`: Sentence Transformers model (default `all-MiniLM-L6-v2`).
- `EMBEDDING_DEVICE`: `auto` (CUDA GPU if available, else CPU), `cuda` or `cpu`.
- `QDRANT_URL`: default `http://localhost:6335`. `QDRANT_COLLECTION`: default `truthlens_knowledge`.
- `TAVILY_API_KEY`: Tavily key for the Web Agent (only needed for real web research).
- `TAVILY_MCP_URL`: default `https://mcp.tavily.com/mcp/` (Tavily's remote MCP server).

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

## Vector database and ingestion

Qdrant runs through Docker Compose with a persistent volume. It is exposed on host
port **6335** to avoid clashing with other local Qdrant instances on 6333.

```bash
cd backend
docker compose up -d qdrant          # dashboard: http://localhost:6335/dashboard
uv run python -m scripts.ingest      # embeds FEVER + SciFact and upserts into Qdrant
```

Ingestion is safe to rerun: point IDs are derived from document IDs, so a rerun
overwrites the same points. It uses the GPU if CUDA is available (~1 min on a
GTX 1650), otherwise the CPU.

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
