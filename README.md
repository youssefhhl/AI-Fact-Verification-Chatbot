# TruthLens — AI Fact Verification Chatbot

TruthLens is a conversational fact-verification system. Users submit general,
scientific or technical claims and receive a verdict (TRUE, FALSE, PARTIALLY TRUE,
NOT ENOUGH EVIDENCE) with an explanation and cited evidence, and can ask follow-up
questions in the same conversation.

Planned stack: FastAPI, LangGraph, Groq LLM, Tavily (web), RAG over
FEVER/SciFact (Sentence Transformers + Qdrant), scientific-publication search,
PostgreSQL, React.

## Screenshots

A claim is checked against several sources and gets a verdict, a confidence score, an
explanation, and its supporting and contradicting evidence with links:

![Verification of a claim](docs/screenshots/verification.png)

Follow-up questions stay in the same conversation. Here the agent recognizes that the
evidence it already collected answers the question, so it reuses it instead of searching again:

![Follow-up question that reuses evidence](docs/screenshots/follow-up.png)

| Research in progress | Mobile layout |
|---|---|
| ![Research in progress](docs/screenshots/researching.png) | ![Mobile layout](docs/screenshots/mobile.png) |

*These screenshots show the real frontend with a sample response. The papers are real
results from OpenAlex and the TruthLens knowledge base; the verdict and explanation
text were written for the example.*

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

**Current status:** Phase 9 done (React frontend). Phases 6–8 done (LangGraph workflow, PostgreSQL persistence, chat API). Phase 5 done (central Fact-Checking Agent). Phase 4 done (Web Agent over Tavily MCP, Research Agent over scholarly APIs). Phase 3 done (embeddings, Qdrant, ingestion, retrieval tool). Phase 2 done: FastAPI app with a health endpoint, shared
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

## Research Agent

`app/agents/research_agent.py` follows the same pattern for scientific literature.
The LLM chooses which provider to search, with which academic query and filters
(year range, reviews only), via tool calling: `search_europe_pmc`,
`search_semantic_scholar`, `search_openalex`, then `finish_research`. The REST
clients live in `app/services/scholarly_apis.py` (async httpx). The controller
validates calls, skips repeated searches, enforces the limits above, deduplicates
papers by DOI across providers, and keeps going when one provider fails (a
rate-limited provider is removed for the rest of the run). Evidence keeps the title,
year, authors, venue, abstract and DOI link exactly as the provider returned them;
`stance` and `relevance` are `None`.

## Fact-Checking Agent

`app/agents/fact_checking_agent.py` is the central agent and the only one that judges
claims. `handle_message(message, state)` runs one conversation turn:

1. **Context resolution** (LLM). It classifies the message as `NEW_VERIFICATION`,
   `REUSE_CONTEXT` (a follow-up the stored evidence already covers) or
   `TARGETED_RESEARCH` (a follow-up that needs more evidence), picks the earlier topic
   it refers to, and rewrites it as a standalone claim ("What about elderly people?" →
   "Vitamin D prevents fractures in elderly people").
2. **Research**, skipped for `REUSE_CONTEXT`. The local knowledge base is searched
   first because it is cheap. The LLM then decides by tool calling whether to run the
   Research Agent, the Web Agent, both, or neither, based on the claim's domain and
   what is still missing. Agents requested in the same step run concurrently.
3. **Synthesis** (LLM). It marks every evidence item as relevant or not and as
   SUPPORTS / CONTRADICTS / NEUTRAL, weighs source reliability, resolves contradictions,
   and returns a verdict, explanation, answer and remaining evidence gaps.
4. **Guards** (Python). A verdict needs at least one relevant item on its side,
   otherwise it becomes `NOT_ENOUGH_EVIDENCE`. Confidence is capped by the number of
   independent sources backing it (1 → 0.6, 2 → 0.8, 3+ → 0.95, minus 0.15 when
   independent sources disagree). Several sentences from one website count as one source.

Failures are handled per component: if RAG, one agent or the planner fails, the turn
continues with what is available, and the failure is listed in `limitations`. When
every source fails, the verdict is `NOT_ENOUGH_EVIDENCE` with an explanation that the
failure does not mean that no evidence exists. Budgets are configurable:
`FACT_CHECK_MAX_EXTERNAL_RUNS` (2), `FACT_CHECK_MAX_RAG_CALLS` (2),
`FACT_CHECK_MAX_RESEARCH_ITERATIONS` (3), `FACT_CHECK_MAX_SYNTHESIS_EVIDENCE` (20),
`FACT_CHECK_HISTORY_TURNS` (6).

The agent stores nothing itself. It takes a `ConversationState` (topics with their
evidence, verdicts and gaps, plus past turns; see `app/schemas/conversation.py`) and
returns an updated copy. Evidence is stored
without stance, because stance depends on the exact claim and is re-assessed on every turn.

## LangGraph workflow

The agent's steps run as a LangGraph `StateGraph` (`app/graph/`). Nodes are thin
wrappers around `FactCheckingAgent` methods, so the reasoning stays in one place.

```text
START → resolve_context ──REUSE_CONTEXT──────────────────────────────→ synthesize → END
              │                                                            ↑
              └→ knowledge_base → plan_research ──finished─────────────────┘
                                    │   ↑
            Send (parallel):        └→ knowledge_base | research_agent | web_agent
```

- `resolve_context` decides `NEW_VERIFICATION` / `REUSE_CONTEXT` / `TARGETED_RESEARCH`.
  Reuse goes straight to synthesis. Research turns start with the knowledge base.
- `plan_research` records the results that came back, then asks the LLM what to do
  next. The conditional edge fans out with `Send` to the chosen components: one more
  RAG search, the Research Agent, the Web Agent, or both agents in parallel. When the
  planner finishes, or a budget or the iteration limit is reached, it goes to `synthesize`.
- Budgets are reserved in Python before execution, so parallel branches cannot overspend.

The graph state (`app/graph/state.py`) holds the persistent `ConversationState` plus
per-turn working data, which is reset at the start of every turn. With a checkpointer,
the conversation id is the LangGraph `thread_id`, so each conversation resumes its own
memory and never sees another conversation's state.

## Persistence

Two separate stores, linked by the conversation id:

| Store | Tables | Content | Written by |
|---|---|---|---|
| Conversation history | `conversations`, `messages` (SQLAlchemy + Alembic) | user and assistant messages, timestamps, decision, claim, verification JSON, limitations | `ConversationService` |
| Agent memory | `checkpoints*` (langgraph-checkpoint-postgres) | graph state: topics, evidence, verdicts, gaps | LangGraph checkpointer |

A chat turn runs in two transactions: (1) create the conversation if needed and store
the user message, (2) store the assistant message with its verification. If the agent
fails between them, the user message stays and the API returns 502. Deleting a
conversation removes its rows and its checkpoints in one transaction. Turns of the same
conversation are serialized with a per-conversation lock (one process).

## Chat API

| Method | Path | Description |
|---|---|---|
| POST | `/api/chat` | `{"message", "conversation_id"?}` → `{conversation_id, answer, verification, decision, limitations}`. Omit `conversation_id` to start a new conversation. Unknown ids return 404; they are never created. |
| GET | `/api/conversations?limit=50&offset=0` | Conversations, most recent first, with `created_at`, `updated_at`, `message_count`. |
| GET | `/api/conversations/{id}` | Conversation with its full message history. |
| DELETE | `/api/conversations/{id}` | Deletes messages, conversation and LangGraph checkpoints (204). |

Errors always look like `{"error": {"code", "message", "details"}}`:
`validation_error` (422), `conversation_not_found` (404), `agent_failed` (502),
`agent_unavailable` (503, e.g. no `GROQ_API_KEY`), `database_unavailable` (503),
`database_not_configured` (503, no `DATABASE_URL`). There is no authentication (MVP).

## Frontend

`frontend/` is a React + TypeScript + Vite app (Axios, TanStack React Query). It has a
chat view with message bubbles, a sidebar with past conversations (new / open), and a
result card per verification: verdict, confidence bar, explanation, and evidence grouped
into supporting, contradicting and related items with links to the sources. Follow-ups
reuse the open conversation's id. Opening a conversation loads its history from
`GET /api/conversations/{id}`. API errors are shown with the backend's message and a
Retry button. On screens below 768 px the sidebar becomes a drawer.

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Node.js 20+ and npm (frontend)

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
- `SEMANTIC_SCHOLAR_API_KEY`, `OPENALEX_API_KEY`: optional. Without them Semantic Scholar
  often returns HTTP 429 and OpenAlex uses a small shared daily budget. Europe PMC needs no key.
- `RESEARCH_MAX_API_CALLS` (6), `RESEARCH_MAX_ITERATIONS` (5), `RESEARCH_RESULTS_PER_CALL` (5),
  `RESEARCH_HTTP_TIMEOUT_SECONDS` (15): Research Agent limits.
- `FACT_CHECK_*`: Fact-Checking Agent budgets (see above).
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`: used by Docker Compose to create the database.
- `DATABASE_URL`: `postgresql+psycopg://user:password@localhost:5433/truthlens`. Without it the
  app starts but the chat endpoints return 503.
- `TEST_DATABASE_URL`: optional, the `truthlens_test` database for the PostgreSQL integration tests.

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

## Database setup

PostgreSQL runs in the same Docker Compose file as Qdrant, on host port **5433**
(5432 is often taken). It creates `truthlens` and, on first start, `truthlens_test`.

```bash
cd backend
docker compose up -d postgres
uv run python -m scripts.init_db   # Alembic migrations + LangGraph checkpoint tables
```

`init_db` is explicit and idempotent; the app never migrates on startup. To add a
schema change later: `uv run alembic revision --autogenerate -m "..."`, then rerun `init_db`.

## Run the application

```bash
cd backend
uv run uvicorn app.main:app --reload
```

Health check: <http://localhost:8000/health>. Interactive docs: <http://localhost:8000/docs>.

### Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

The Vite dev server proxies `/api` to `http://localhost:8000`, so start the backend
first (with `DATABASE_URL` and `GROQ_API_KEY` set). To use another backend address in
development: `VITE_BACKEND_URL=http://host:port npm run dev`. For a production build
(`npm run build`, output in `frontend/dist/`), set `VITE_API_URL` to the backend origin
and add the frontend origin to the backend's `CORS_ORIGINS`.

## Run the tests

```bash
cd backend
uv run pytest
```

Groq, Tavily and the research APIs are always mocked. API tests use SQLite and an
in-memory checkpointer. `tests/test_postgres.py` runs against `TEST_DATABASE_URL`
(migrations, PostgreSQL checkpointer, restart survival) and is skipped when that
database is not reachable.

## Dependencies

```bash
uv add <package>          # runtime
uv add --dev <package>    # development
```
