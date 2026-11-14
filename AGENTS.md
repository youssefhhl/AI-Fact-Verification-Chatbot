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

TruthLens uses three independent research agents.

### 3.1 Web Agent

Responsibilities:
- Research the submitted claim on the web.
- Use Tavily MCP for web search.
- Retrieve relevant sources and their content.
- Extract evidence and source references.
- Return structured evidence.

The Web Agent must not determine the final verdict.

### 3.2 RAG Agent

Responsibilities:
- Search the local FEVER and SciFact knowledge base.
- Use Sentence Transformers to embed the user's claim.
- Retrieve relevant documents from Qdrant.
- Return relevant evidence with metadata.

The RAG Agent must not determine the final verdict.

### 3.3 Research Agent

Responsibilities:
- Search scientific publications.
- Retrieve relevant scientific papers and abstracts.
- Extract relevant evidence and references.
- Return structured evidence.

The scientific research provider/API must be investigated and verified before implementation. Do not invent API capabilities.

The Research Agent must not determine the final verdict.

### 3.4 Evidence Reducer

The Evidence Reducer receives evidence from all research agents.

Responsibilities:
- Combine collected evidence.
- Remove duplicates where appropriate.
- Evaluate evidence relevance and reliability.
- Identify supporting and contradicting evidence.
- Detect contradictions between sources.
- Ask the LLM to analyze the combined evidence.
- Produce the final verdict, confidence, explanation, and citations.

The Evidence Reducer is the component responsible for producing the final verification result.

### 3.5 LangGraph Workflow

LangGraph orchestrates the complete verification process.

Expected high-level workflow:

```text
                 User Claim
                      |
                      v
              Context Resolution
                      |
                      v
              Research Required?
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Web Agent   RAG Agent   Research Agent
          |           |           |
          +-----------+-----------+
                      |
                      v
               Evidence Reducer
                      |
                      v
              Verification Result
                      |
                      v
                 User Response
```

The three research agents should execute in parallel when their evidence is required.

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

A Context Resolution node must determine whether a new user message:
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
│   │   ├── web_agent.py
│   │   ├── rag_agent.py
│   │   └── research_agent.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   │
│   ├── reducer/
│   │   └── evidence_reducer.py
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
- `agents/`: Evidence collection from individual sources.
- `graph/`: LangGraph state, nodes, and workflow configuration.
- `reducer/`: Evidence analysis and final verification.
- `services/`: Reusable integrations and application services.
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
10. Implement the RAG Agent.
11. Test retrieval quality and metadata preservation.

Important:

- Dataset ingestion must be separate from runtime retrieval.
- Do not regenerate all dataset embeddings for each request.
- Use Qdrant as the only vector database.
- Do not introduce FAISS.

Expected result: A working local knowledge base and RAG Agent.

### Phase 4 — External Research Agents

Objectives:

**Web Agent**
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

### Phase 5 — Evidence Reducer

Objectives:
- Receive evidence from all agents.
- Normalize and combine evidence.
- Handle duplicates.
- Evaluate source relevance and reliability.
- Identify supporting and contradicting evidence.
- Detect conflicting information.
- Use the LLM to analyze the evidence.
- Generate the final verification result.

The reducer must follow the shared VerificationResult schema.

Expected result: A component capable of producing structured, evidence-based verdicts.

### Phase 6 — LangGraph Workflow

Objectives:
- Define the shared graph state.
- Implement workflow nodes.
- Integrate the three research agents.
- Execute independent research agents in parallel.
- Integrate the Evidence Reducer.
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
- Implement Context Resolution.
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