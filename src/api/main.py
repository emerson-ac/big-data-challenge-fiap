"""Aplicação FastAPI: expõe o modelo Production como serviço de recomendação.

Uso:
    uv run uvicorn src.api.main:app --reload
"""

from fastapi import Depends, FastAPI, Query

from src.api.dependencies import get_engine, get_product_names
from src.api.schemas import HealthResponse, RecommendationResponse
from src.api.services.enrichment import select_strategy
from src.config import get_settings
from src.models.inference import RecommendationEngine

app = FastAPI(
    title="Sistema de Recomendação Instacart",
    description="Recomendações top-k servidas pelo modelo Production (item-based CF).",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check simples para readiness/liveness probes.

    Returns:
        Status do serviço e modelo ativo.
    """
    return HealthResponse(status="ok", model_type=get_settings().model_type)


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def recommend(
    user_id: int,
    k: int = Query(default=10, ge=1, le=100),
    enrich_names: bool = Query(default=True),
    engine: RecommendationEngine = Depends(get_engine),
) -> RecommendationResponse:
    """Gera o top-k de recomendações para um usuário.

    Args:
        user_id: ID externo do usuário (Instacart).
        k: Quantidade de itens recomendados.
        enrich_names: Se True, anexa o nome do produto (Strategy Pattern).
        engine: Engine de recomendação injetado.

    Returns:
        Recomendações ordenadas por score, com flag de cold-start.
    """
    strategy = select_strategy(enrich_names, get_product_names())
    recommendations = engine.recommend(user_id, k)
    return RecommendationResponse(
        user_id=user_id,
        model_type=get_settings().model_type,
        cold_start=not engine.is_known_user(user_id),
        recommendations=strategy.enrich(recommendations),
    )
