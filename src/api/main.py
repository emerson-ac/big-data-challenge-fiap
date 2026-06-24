"""Aplicação FastAPI principal do serviço de recomendação."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI

from src.api.config import settings
from src.api.dependencies import get_recommendation_engine
from src.api.middleware.error_handler import (
    api_exception_handler,
    general_exception_handler,
)
from src.api.routes import health, recommendations
from src.api.schemas.errors import APIException
from src.api.utils.logger import configure_logging, get_logger

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Gerencia o ciclo de vida da aplicação: seed e carregamento do modelo.

    Args:
        app: Instância da aplicação FastAPI.
    """
    configure_logging(settings.log_level)
    np.random.seed(settings.random_seed)
    logger.info("application_startup", version=settings.app_version)
    engine_provider = app.dependency_overrides.get(
        get_recommendation_engine, get_recommendation_engine
    )
    engine_provider()
    yield
    logger.info("application_shutdown")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_exception_handler(APIException, api_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

app.include_router(health.router)
app.include_router(recommendations.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
