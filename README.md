# TruthLens — AI Fact Verification Chatbot

TruthLens is a conversational fact-verification system. Users submit general,
scientific or technical claims and receive a verdict (TRUE, FALSE, PARTIALLY TRUE,
NOT ENOUGH EVIDENCE) with an explanation and cited evidence, and can ask follow-up
questions in the same conversation.

Planned stack: FastAPI, LangGraph, Groq LLM, Tavily (web), FAISS RAG over
FEVER/SciFact, scientific-publication search, PostgreSQL, React.

**Current status:** Phase 1 (project setup). Only the FastAPI app, configuration
and a health endpoint exist so far.

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
cd backend
cp .env.example .env     # adjust values if needed
uv sync                  # creates .venv and installs locked dependencies
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
