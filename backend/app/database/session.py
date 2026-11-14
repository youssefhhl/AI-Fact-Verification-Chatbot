from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def create_engine(url: str) -> AsyncEngine:
    engine = create_async_engine(url, pool_pre_ping=True)
    if engine.dialect.name == "sqlite":  # used by tests: enforce ON DELETE CASCADE
        @event.listens_for(engine.sync_engine, "connect")
        def _fk_on(dbapi_connection, _record) -> None:  # noqa: ANN001
            dbapi_connection.execute("PRAGMA foreign_keys=ON")
    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


def checkpointer_conninfo(sqlalchemy_url: str) -> str:
    """postgresql+psycopg://... -> postgresql://... (libpq format for the LangGraph saver)."""
    return make_url(sqlalchemy_url).set(drivername="postgresql").render_as_string(hide_password=False)
