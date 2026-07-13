"""Schemas de resposta da API de recomendação."""

from datetime import datetime

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """Item individual na recomendação.

    Attributes:
        product_id: ID do produto.
        score: Score de afinidade calculado pelo modelo.
        rank: Posição no ranking (1-based).
    """

    product_id: int
    score: float
    rank: int = Field(..., ge=1)


class RecommendationResponse(BaseModel):
    """Response de recomendações.

    Attributes:
        user_id: ID do usuário.
        is_cold_start: Indica se o usuário não tinha histórico conhecido.
        recommendations: Lista de recomendações ranqueadas.
        model_type: Modelo usado para gerar a recomendação.
        timestamp: Timestamp da resposta.
        processing_time_ms: Tempo de processamento em milissegundos.
    """

    user_id: int
    is_cold_start: bool
    recommendations: list[RecommendationItem]
    model_type: str
    timestamp: datetime
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Resposta do health check.

    Attributes:
        status: Estado da aplicação.
        service: Nome do serviço.
        version: Versão da aplicação.
    """

    status: str
    service: str
    version: str
