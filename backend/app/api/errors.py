"""One error format for every endpoint: {"error": {"code", "message", "details"}}."""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas.chat import ErrorBody, ErrorResponse
from app.services.conversation_service import (
    AgentFailedError,
    AgentUnavailableError,
    ConversationNotFoundError,
    DatabaseUnavailableError,
)


def error_response(status_code: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


class ServiceNotConfiguredError(Exception):
    """DATABASE_URL is not set, so the chat endpoints cannot work."""


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
        return error_response(status.HTTP_422_UNPROCESSABLE_CONTENT, "validation_error",
                              "The request is invalid.", {"errors": errors})

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return error_response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(ConversationNotFoundError)
    async def _not_found(_: Request, exc: ConversationNotFoundError) -> JSONResponse:
        return error_response(status.HTTP_404_NOT_FOUND, "conversation_not_found",
                              "Conversation not found.", {"conversation_id": str(exc)})

    @app.exception_handler(AgentUnavailableError)
    async def _agent_unavailable(_: Request, exc: AgentUnavailableError) -> JSONResponse:
        return error_response(status.HTTP_503_SERVICE_UNAVAILABLE, "agent_unavailable",
                              f"The fact-checking agent is not configured: {exc}")

    @app.exception_handler(AgentFailedError)
    async def _agent_failed(_: Request, exc: AgentFailedError) -> JSONResponse:
        return error_response(status.HTTP_502_BAD_GATEWAY, "agent_failed",
                              "The fact-checking agent failed to answer. Your message was saved; please retry.",
                              {"conversation_id": str(exc.conversation_id)})

    @app.exception_handler(DatabaseUnavailableError)
    async def _database(_: Request, exc: DatabaseUnavailableError) -> JSONResponse:
        return error_response(status.HTTP_503_SERVICE_UNAVAILABLE, "database_unavailable",
                              "The database is unavailable. Please retry later.")

    @app.exception_handler(ServiceNotConfiguredError)
    async def _not_configured(_: Request, exc: ServiceNotConfiguredError) -> JSONResponse:
        return error_response(status.HTTP_503_SERVICE_UNAVAILABLE, "database_not_configured",
                              "DATABASE_URL is not configured.")
