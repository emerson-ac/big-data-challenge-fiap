"""Injeção de dependências da API de recomendação."""

from fastapi import Depends

from src.api.config import settings
from src.api.schemas.errors import ModelNotLoadedError
from src.api.services.recommendation_service import RecommendationService
from src.models.inference import RecommendationEngine

_engine: RecommendationEngine | None = None


def get_recommendation_engine() -> RecommendationEngine:
    """Cria (uma única vez) e retorna o motor de predição carregado.

    Returns:
        Instância de RecommendationEngine pronta para uso.

    Raises:
        ModelNotLoadedError: Se os artefatos do modelo não puderem ser carregados.
    """
    global _engine
    if _engine is None:
        try:
            _engine = RecommendationEngine(
                model_type=settings.recommender_type,
                similarity_path=settings.similarity_path,
                interactions_path=settings.interactions_path,
                popularity_path=settings.popularity_path,
                vocab_path=settings.vocab_path,
            )
        except (FileNotFoundError, OSError) as exc:
            raise ModelNotLoadedError(str(exc)) from exc
    return _engine


def get_recommendation_service(
    engine: RecommendationEngine = Depends(get_recommendation_engine),
) -> RecommendationService:
    """Injeta o RecommendationService construído sobre o motor carregado.

    Args:
        engine: Motor de predição injetado por get_recommendation_engine.

    Returns:
        Instância de RecommendationService.
    """
    return RecommendationService(engine)
