"""Schemas de requisição da API de recomendação."""

from pydantic import BaseModel, Field, field_validator


class RecommendationRequest(BaseModel):
    """Schema de requisição para recomendações.

    Attributes:
        user_id: ID do usuário (Instacart user_id).
        top_k: Número de recomendações desejadas.
        exclude_items: Lista de product_id a excluir do resultado.
        min_score_threshold: Score mínimo para uma recomendação ser incluída.
    """

    user_id: int = Field(..., description="ID do usuário")
    top_k: int = Field(default=10, ge=1, le=1000, description="Quantidade desejada")
    exclude_items: list[int] | None = Field(None, description="product_id a excluir")
    min_score_threshold: float | None = Field(None, ge=0.0, description="Score mínimo")

    @field_validator("exclude_items")
    @classmethod
    def validate_exclude_items(cls, value: list[int] | None) -> list[int] | None:
        """Normaliza a lista de exclusão removendo duplicatas.

        Args:
            value: Lista de product_id informada pelo cliente, ou None.

        Returns:
            Lista sem duplicatas, ou None se nada foi informado.
        """
        return list(dict.fromkeys(value)) if value else value
