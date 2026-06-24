"""Exception handlers da API de recomendação."""

from datetime import UTC, datetime

from fastapi import Request
from fastapi.responses import JSONResponse

from src.api.schemas.errors import APIException
from src.api.utils.logger import get_logger

logger = get_logger()


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """Converte uma APIException em uma resposta JSON padronizada.

    Args:
        request: Requisição que originou o erro.
        exc: Exceção de negócio levantada pela aplicação.

    Returns:
        JSONResponse com error_code, message, details e timestamp.
    """
    logger.error(
        "api_exception_occurred",
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Converte uma exceção não tratada em uma resposta JSON genérica.

    Args:
        request: Requisição que originou o erro.
        exc: Exceção inesperada.

    Returns:
        JSONResponse com código 500 e mensagem genérica (sem detalhes internos).
    """
    logger.error("unexpected_error", path=request.url.path, exception=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "Erro interno do servidor",
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )
