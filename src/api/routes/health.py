"""Rota de health check da API."""

from fastapi import APIRouter

from src.api.config import settings
from src.api.schemas.response import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/status", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Verifica se a API está operacional.

    Returns:
        HealthResponse com status, nome do serviço e versão.
    """
    return HealthResponse(
        status="healthy", service=settings.app_name, version=settings.app_version
    )
