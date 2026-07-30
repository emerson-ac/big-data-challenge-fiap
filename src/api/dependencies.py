"""Injeção de dependências da API de recomendação."""

from fastapi import Depends

from src.api.config import settings
from src.api.schemas.errors import ModelNotLoadedError
from src.api.services.recommendation_service import RecommendationService
from src.api.utils.logger import get_logger
from src.config import get_settings
from src.evaluation.promotion import promoted_model_from_csv
from src.models.inference import RecommendationEngine

logger = get_logger()

# Modelos que o modo local sabe servir a partir de artefatos em disco.
_LOCAL_SERVABLE = frozenset({"item_based_cf", "popularity"})

_engine: RecommendationEngine | None = None


def _resolve_local_model_type() -> str:
    """Resolve o modelo servido em modo local, seguindo a promoção do pipeline.

    Lê o modelo promovido (melhor recall@k) de ``metrics_comparison.csv`` — a
    mesma fonte usada pelo ``evaluate.py`` para registrar no MLflow. Se o
    promovido não for servível localmente ou o arquivo não existir, recorre ao
    ``recommender_type`` configurado.

    Returns:
        Nome do modelo a ser instanciado pela ModelFactory em modo local.
    """
    metrics_csv = get_settings().models_dir / "evaluation" / "metrics_comparison.csv"
    promoted = promoted_model_from_csv(metrics_csv)
    if promoted in _LOCAL_SERVABLE:
        return promoted
    if promoted is not None:
        logger.warning(
            "promoted_model_not_local_servable",
            promoted=promoted,
            fallback=settings.recommender_type,
        )
    return settings.recommender_type


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
                model_type=_resolve_local_model_type(),
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
