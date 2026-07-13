"""Modelos Pydantic de requisição/resposta da API."""

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """Um item recomendado, com posição, score e nome opcional."""

    product_id: int
    product_name: str | None = None
    score: float
    rank: int


class RecommendationResponse(BaseModel):
    """Resposta do endpoint de recomendações para um usuário."""

    user_id: int
    model_type: str
    cold_start: bool = Field(
        description="True quando o usuário é desconhecido (fallback de popularidade)."
    )
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    """Resposta do health check."""

    status: str
    model_type: str
