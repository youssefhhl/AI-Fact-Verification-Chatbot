import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, health
from app.api.errors import register_error_handlers
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the database engine and the checkpoint connection pool, if configured.

    Nothing is created or migrated here: run `uv run python -m scripts.init_db` first.
    The agent itself is built on the first chat request, so the app (and /health)
    starts without a Groq key.
    """
    settings: Settings = app.state.settings
    app.state.conversation_service = None
    if settings.database_url is None:
        logger.warning("DATABASE_URL is not set: chat endpoints are disabled")
        yield
        return

    from app.services.conversation_service import open_conversation_service

    async with open_conversation_service(settings.database_url.get_secret_value()) as service:
        app.state.conversation_service = service
        yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(chat.router)
    return app


app = create_app()
