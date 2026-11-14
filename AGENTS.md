# AGENTS.md — TruthLens Project Instructions

## 1. Project Overview

**TruthLens** is a conversational AI fact-verification application.

Its purpose is to verify claims using evidence collected from multiple independent sources, including:
- General web sources
- Scientific publications
- A local knowledge base containing FEVER and SciFact datasets

Users submit claims through a chat interface. TruthLens researches the claims, evaluates the collected evidence, and returns a structured verdict with explanations and citations.

### Supported verdicts

- `TRUE`
- `FALSE`
- `PARTIALLY_TRUE`
- `NOT_ENOUGH_EVIDENCE`

Every verification should provide:
- Verdict
- Explanation
- Confidence score
- Supporting and contradicting evidence
- Source references and citations

The application must also support follow-up questions within the same conversation.

### MVP Scope

The initial version supports text-based chat only.

Do not implement:
- PDF, DOCX, or image uploads
- YouTube verification
- Browser extensions
- Other features outside the agreed MVP

---

## 2. Technology Stack

Use the following technologies.

| Component | Technology |
|---|---|
| Backend | Python 3.12 |
| API | FastAPI |
| Package management | uv |
| Agent orchestration | LangGraph |
| LLM provider | Groq |
| Web research | Tavily MCP |
| Scientific research | Scientific publication APIs (to be selected and verified) |
| RAG datasets | FEVER and SciFact |
| Embeddings | Sentence Transformers |
| Vector database | Qdrant |
| Relational database | PostgreSQL |
| Frontend | React |

### Important technology decisions

**Embeddings and vector storage**
- Sentence Transformers generates vector embeddings.
- Qdrant stores and retrieves those vectors.
- Use the same embedding model for ingestion and runtime retrieval.
- Initial embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Generate dataset embeddings during offline ingestion, not on every user request.
- Qdrant is the only vector database.

**LLM**
- Use Groq through its official Python SDK.
- Keep the provider and model configurable.
- Avoid coupling business logic directly to a specific LLM provider.

**Agent orchestration**
- Use LangGraph for workflow orchestration, state management, parallel execution, and checkpointing.
- Do not introduce additional orchestration frameworks.

### Explicitly excluded technologies

Do not introduce these without explicit approval:

- FAISS
- Another vector database
- LangChain
- Docker before the DevOps phase
- Kubernetes before the DevOps phase
- Additional LLM providers without a concrete requirement

LangChain is not required for the current architecture. Prefer direct integrations with Sentence Transformers, Qdrant, Groq, and LangGraph.

---

## 3. Target Architecture

TruthLens has **one central Fact-Checking Agent** that reasons and orchestrates,
two **specialized research agents**, and one **retrieval tool** (RAG).

| Component | Kind | Reasons with the LLM? | Decides the verdict? |
|---|---|---|---|
| Fact-Checking Agent | Central agent | Yes | **Yes** |
| Web Agent | Specialized research agent | Yes (its own research) | No |
| Research Agent | Specialized research agent | Yes (its own research) | No |
| Knowledge-base retrieval (RAG) | Tool / service | No | No |

### 3.1 Fact-Checking Agent (central)

The only component that decides verdicts. Responsibilities:
- Receive the user's message and understand it in the conversation context.
- Decide whether the message is a new claim or a follow-up (Context Resolution).
- Decide whether previous evidence can be reused or new research is needed.
- Decide which research agents and tools to invoke.
- Collect and analyze the returned evidence: relevance, reliability, duplicates.
- Classify evidence stance (supporting / contradicting / neutral).
- Detect and resolve contradictions between sources.
- Produce the final verdict, confidence, explanation and citations
  (`VerificationResult`) and the user-facing answer.

It absorbs the responsibilities of the former Evidence Reducer, which no longer
exists as a separate component.

### 3.2 Web Agent (specialized research agent)

Implemented in `app/agents/web_agent.py`, with the MCP client in `app/services/tavily_mcp.py`.
- Bounded tool loop: the LLM chooses tools; the Python controller enforces
  3 searches, 5 extracted URLs, 5 iterations, no duplicate or unseen URLs.
- Research the claim on the web using Tavily MCP.
- Uses its own reasoning to plan queries, select sources and extract evidence.
- Returns structured `Evidence` with source URLs and metadata.
- Must not determine the final verdict.

### 3.3 Research Agent (specialized research agent)

- Search scientific publications through a verified scientific API.
- Uses its own reasoning to plan queries, select papers and extract evidence.
- Returns structured `Evidence` with references.
- The provider/API must be investigated and verified before implementation.
  Do not invent API capabilities.
- Must not determine the final verdict.

### 3.4 Knowledge-base retrieval — RAG (tool)

Not an agent. A deterministic retrieval tool in
`app/services/retrieval_service.py`:
- Embeds the claim with Sentence Transformers.
- Searches the FEVER + SciFact passages in Qdrant.
- Returns `Evidence` (text, source, URL, relevance = similarity score) via
  `RetrievalService.search_evidence()`.
- Does not call the LLM and leaves `stance` unset (`None`); the Fact-Checking
  Agent assesses stance.

### 3.5 LangGraph Workflow

LangGraph orchestrates the process around the Fact-Checking Agent.

Expected high-level workflow:

```text
                 User Message
                      |
                      v
        Fact-Checking Agent: context resolution
        (new claim? follow-up? reuse evidence?)
                      |
              Research required?
           no /            \ yes
             /     +--------+---------+
            |      |        |         |
            |      v        v         v
            |   Web Agent  RAG tool  Research Agent
            |      |        |         |
            |      +--------+---------+
            |               |
            v               v
        Fact-Checking Agent: analysis and synthesis
        (stance, contradictions, verdict, citations)
                      |
                      v
                 User Response
```

The research components run in parallel when their evidence is required. The
Fact-Checking Agent decides which ones are needed.

LangGraph manages:
- Shared workflow state
- Node execution
- Parallel research
- Error handling
- Checkpointing

Do not implement the entire workflow before the individual components are ready and tested.

---

## 4. Conversation Management

TruthLens is a conversational application, not a collection of independent verification requests.

It must support:
- Conversation history
- Conversation identifiers
- Follow-up questions
- Context resolution
- Evidence reuse

### Context Resolution

The Fact-Checking Agent performs Context Resolution (as a LangGraph node) and must determine whether a new user message:
1. Introduces a new claim.
2. Asks a follow-up about a previously verified claim.
3. Requests clarification or additional analysis.
4. Requires new research.

When previous evidence is still relevant, reuse it instead of unnecessarily repeating all research operations.

When the new question requires additional information, retrieve new evidence as needed.

### Persistence

Use:
- PostgreSQL for application data and conversation persistence.
- LangGraph checkpointing for workflow state.

Keep conversation persistence and LangGraph checkpointing conceptually separate. They have different responsibilities.

---

## 5. Backend Project Structure

Target structure:

```text
backend/
├── app/
│   ├── main.py
│   ├── config.py
│   │
│   ├── api/
│   │   ├── chat.py
│   │   └── health.py
│   │
│   ├── agents/
│   │   ├── fact_checking_agent.py
│   │   ├── web_agent.py
│   │   └── research_agent.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   │
│   ├── services/
│   │   ├── llm_service.py
│   │   ├── retrieval_service.py
│   │   └── conversation_service.py
│   │
│   ├── schemas/
│   │   ├── chat.py
│   │   ├── evidence.py
│   │   └── verification.py
│   │
│   └── database/
│       ├── models.py
│       └── repository.py
│
├── data/
│   ├── fever/
│   └── scifact/
│
├── tests/
│
├── .env
├── .env.example
├── pyproject.toml
├── uv.lock
└── .python-version
```

This is the target structure, not a requirement to create all files immediately.

Create modules only when they are needed for the current phase.

### Module responsibilities

- `api/`: HTTP endpoints and request/response handling.
- `agents/`: The central Fact-Checking Agent and the specialized research agents.
- `graph/`: LangGraph state, nodes, and workflow configuration.
- `services/`: Reusable integrations and application services, including the RAG retrieval tool.
- `schemas/`: Shared Pydantic models.
- `database/`: Database models and persistence operations.
- `tests/`: Automated tests.

Avoid unnecessary abstractions and excessive layering.

---

## 6. Development Roadmap

Follow these phases in order.

**Do not start a future phase unless explicitly instructed.**

### Phase 1 — Project Setup

Status: COMPLETED

Objectives:
- Initialize the backend project.
- Configure Python 3.12.
- Initialize uv.
- Set up FastAPI.
- Configure Pydantic Settings.
- Create the health endpoint.
- Configure pytest and HTTPX.
- Initialize Git.
- Create the README.
- Configure environment variables.

Docker is intentionally postponed.

### Phase 2 — Schemas and LLM Service

Status: COMPLETED

Objectives:

1. Create shared Pydantic schemas:
   - Evidence
   - VerificationResult
   - ChatRequest
   - ChatResponse

2. Implement the LLM service:
   - Integrate the official Groq SDK.
   - Configure the API key through environment variables.
   - Configure the model through environment variables.
   - Support asynchronous calls.
   - Handle API errors and timeouts.
   - Keep the integration reusable.

3. Add tests:
   - Schema validation.
   - Invalid input handling.
   - LLM service behavior using mocks.
   - Missing configuration handling.

Expected result: A stable data contract and reusable LLM integration.

### Phase 3 — RAG Pipeline

Status: COMPLETED

Objectives:

1. Investigate FEVER and SciFact datasets.
2. Understand their formats, fields, and licensing.
3. Download and inspect the datasets.
4. Preprocess documents and preserve useful metadata.
5. Generate embeddings using Sentence Transformers.
6. Configure Qdrant.
7. Implement offline ingestion.
8. Store vectors, original text, and metadata in Qdrant.
9. Implement a retrieval service.
10. Expose retrieval as a tool returning `Evidence` (`search_evidence`). RAG is a tool, not an agent.
11. Test retrieval quality and metadata preservation.

Important:

- Dataset ingestion must be separate from runtime retrieval.
- Do not regenerate all dataset embeddings for each request.
- Use Qdrant as the only vector database.
- Do not introduce FAISS.

Expected result: A working local knowledge base and a retrieval tool that returns `Evidence`.

### Phase 4 — External Research Agents

Objectives:

**Web Agent** — Status: COMPLETED (Phase 4.1)
- Verify Tavily MCP integration.
- Implement web search.
- Retrieve useful source content.
- Extract structured evidence.
- Preserve source URLs and metadata.

**Research Agent**
- Investigate suitable scientific publication APIs.
- Compare available providers.
- Verify API capabilities, limits, and access requirements.
- Select an appropriate source before implementation.
- Retrieve relevant scientific publications.
- Extract structured evidence and references.

Both agents must return the shared Evidence schema.

Expected result: Two external research agents capable of gathering evidence independently.

### Phase 5 — Fact-Checking Agent

Objectives:
- Implement the central Fact-Checking Agent.
- Decide which research agents/tools to invoke for a claim.
- Normalize, combine and deduplicate evidence.
- Evaluate source relevance and reliability.
- Classify evidence stance and detect conflicting information.
- Use the LLM to analyze the evidence and resolve contradictions.
- Generate the final verification result and user-facing answer.

The agent must follow the shared VerificationResult schema.

Expected result: A central agent capable of producing structured, evidence-based verdicts.

### Phase 6 — LangGraph Workflow

Objectives:
- Define the shared graph state.
- Implement workflow nodes.
- Integrate the Fact-Checking Agent, the two research agents and the RAG tool.
- Execute the selected research components in parallel.
- Handle agent failures and partial results.
- Add LangGraph checkpointing.
- Test the complete workflow.

Expected result: A working end-to-end verification workflow.

### Phase 7 — Conversation Management and PostgreSQL

Objectives:
- Configure PostgreSQL.
- Design conversation and message persistence.
- Implement database models and repositories.
- Persist conversation history.
- Integrate LangGraph checkpointing.
- Implement Context Resolution in the Fact-Checking Agent.
- Support follow-up questions.
- Reuse previous evidence when appropriate.
- Determine when additional research is necessary.

Expected result: Persistent conversations with contextual follow-up support.

### Phase 8 — Chat API

Objectives:
- Implement the chat endpoint.
- Connect HTTP requests to the LangGraph workflow.
- Handle conversation identifiers.
- Return structured verification responses.
- Handle errors and invalid requests.
- Add API integration tests.

Expected result: A complete backend API ready for frontend integration.

### Phase 9 — React Frontend

Objectives:
- Initialize the React application.
- Implement the chat interface.
- Integrate the backend API.
- Display verification verdicts.
- Display explanations, confidence, and citations.
- Support conversation history and follow-up questions.
- Handle loading and error states.

Expected result: A usable conversational fact-verification application.

### Phase 10 — DevOps and Deployment

Only begin this phase when explicitly requested.

Potential objectives:
- Dockerize the backend and frontend.
- Configure PostgreSQL and Qdrant deployment.
- Create Docker Compose configuration.
- Build CI/CD pipelines.
- Integrate code quality and security tools.
- Add automated tests to the pipeline.
- Configure deployment.
- Add monitoring and logging.

Do not introduce these tools during earlier phases.

---

## 7. Development Rules

### Follow the agreed architecture

- Do not silently change the architecture.
- Do not add technologies without explaining why they are needed.
- Do not introduce unnecessary agents.
- Do not duplicate responsibilities between modules.
- Do not add abstractions without a concrete benefit.
- If a design decision needs to change, explain the reason and request approval.

### Package management

Use `uv` for all Python dependency management.

Commands must be executed from the `backend/` directory.

```bash
uv sync
uv add <package>
uv add --dev <package>
uv run uvicorn app.main:app --reload
uv run pytest
```

Never use pip or create a requirements.txt file.

### Environment variables

- Store local secrets in `.env`.
- Never commit `.env`.
- Keep `.env.example` updated.
- Use Pydantic Settings for application configuration.
- Never hardcode API keys or credentials.

### Testing

- Write tests for implemented functionality.
- Mock external APIs and services.
- Automated tests must not depend on real external services.
- Test error cases as well as successful execution.
- Do not claim a feature works without verification.

### Code quality

- Use type hints.
- Prefer clear, readable, maintainable code.
- Keep functions focused.
- Use asynchronous code where appropriate.
- Handle exceptions explicitly.
- Avoid unnecessary dependencies.

### Documentation

Update the README when implementation changes affect:
- Installation
- Configuration
- Running the application
- Testing
- Architecture

Keep documentation consistent with the actual implementation.

---

## 8. Git Rules

- Do not commit unless explicitly requested.
- Stage only the files relevant to the requested work.
- Never push without permission.
- Never reset, force-push, or rewrite Git history without permission.
- Do not overwrite unrelated user changes.

---

## 9. Agent Working Procedure

Before implementing any task:

1. Read this entire AGENTS.md file.
2. Inspect the existing project structure.
3. Check the current implementation and tests.
4. Identify the current development phase.
5. Understand the requested scope.
6. Explain important design decisions when necessary.
7. Implement only the requested phase or task.

After implementation:

1. Run relevant tests.
2. Verify that the application still starts when applicable.
3. Review changed files.
4. Update documentation when needed.
5. Report what was implemented.
6. Report dependencies added.
7. Report tests and their results.
8. Mention assumptions, limitations, and unfinished work.

Do not move to the next phase automatically.

---

## 10. Definition of Done

A task is complete only when:

- The requested functionality is implemented.
- The implementation respects this architecture.
- Relevant tests pass.
- No unrelated functionality was added.
- Configuration and documentation are updated when necessary.
- Remaining limitations are clearly reported.

**Final rule: The agreed TruthLens architecture and roadmap are authoritative. This file guides implementation; it does not authorize changing the project plan independently.**