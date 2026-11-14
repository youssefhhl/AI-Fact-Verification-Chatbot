from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# backend/.env, independent of the current working directory.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    app_name: str = "TruthLens"
    app_env: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    cors_origins: list[str] = []

    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    llm_timeout_seconds: float = 30.0

    # Embeddings: same model for ingestion and runtime queries.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "auto"  # "auto" (CUDA if available), "cuda" or "cpu"
    embedding_batch_size: int = 128

    qdrant_url: str = "http://localhost:6335"
    qdrant_collection: str = "truthlens_knowledge"

    # Knowledge-base retrieval (RAG): passages below rag_min_score (cosine similarity) are dropped.
    rag_top_k: int = 5
    rag_min_score: float = 0.3

    # Web Agent: Tavily remote MCP server (key passed as ?tavilyApiKey=).
    tavily_api_key: SecretStr | None = None
    tavily_mcp_url: str = "https://mcp.tavily.com/mcp/"

    # Research Agent: scholarly REST APIs. Keys are optional (higher rate limits).
    semantic_scholar_api_key: SecretStr | None = None
    openalex_api_key: SecretStr | None = None
    research_max_api_calls: int = 6  # provider searches per research run
    research_max_iterations: int = 5  # LLM turns per research run
    research_results_per_call: int = 5
    research_http_timeout_seconds: float = 15.0

    # Fact-Checking Agent (central): research budgets per user message.
    fact_check_max_external_runs: int = 2  # Research Agent + Web Agent runs in total
    fact_check_max_rag_calls: int = 2  # knowledge-base searches (the automatic first one included)
    fact_check_max_research_iterations: int = 3  # LLM planning turns
    fact_check_max_synthesis_evidence: int = 20  # newest items shown to the LLM for the verdict
    fact_check_history_turns: int = 6  # recent turns used for context resolution

    # PostgreSQL (application data + LangGraph checkpoints), SQLAlchemy URL with psycopg:
    # postgresql+psycopg://user:password@localhost:5433/truthlens
    database_url: SecretStr | None = None
    test_database_url: SecretStr | None = None  # separate database for integration tests


@lru_cache
def get_settings() -> Settings:
    return Settings()
