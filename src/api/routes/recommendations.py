"""Rota de geração de recomendações de produtos."""

import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from src.api.config import settings
from src.api.dependencies import get_recommendation_service
from src.api.schemas.request import RecommendationRequest
from src.api.schemas.response import RecommendationResponse
from src.api.services.recommendation_service import RecommendationService
from src.api.utils.logger import get_logger

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
logger = get_logger()


@router.post("/", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    """Gera recomendações de produtos para um usuário.

    Args:
        request: user_id e filtros opcionais (top_k, exclude_items, min_score).
        service: Serviço de recomendação injetado.

    Returns:
        RecommendationResponse com a lista ranqueada de produtos.
    """
    start_time = time.perf_counter()
    logger.info(
        "recommendation_request_received",
        user_id=request.user_id,
        top_k=request.top_k,
    )
    items, is_cold_start = service.get_recommendations(request)
    processing_time_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "recommendations_generated",
        user_id=request.user_id,
        count=len(items),
        is_cold_start=is_cold_start,
        processing_time_ms=round(processing_time_ms, 2),
    )
    return RecommendationResponse(
        user_id=request.user_id,
        is_cold_start=is_cold_start,
        recommendations=items,
        model_type="popularity" if is_cold_start else settings.recommender_type,
        timestamp=datetime.now(UTC),
        processing_time_ms=round(processing_time_ms, 2),
    )
