"""Service de recomendação: adapta o RecommendationEngine aos schemas da API."""

from src.api.schemas.request import RecommendationRequest
from src.api.schemas.response import RecommendationItem
from src.models.inference import Recommendation, RecommendationEngine

_CANDIDATE_MULTIPLIER = 5
_MAX_CANDIDATES = 500


class RecommendationService:
    """Gera recomendações para a API, aplicando filtros de negócio.

    Args:
        engine: Motor de predição já carregado (modelo Production + fallback).
    """

    def __init__(self, engine: RecommendationEngine) -> None:
        self._engine = engine

    def get_recommendations(
        self, request: RecommendationRequest
    ) -> tuple[list[RecommendationItem], bool]:
        """Gera as recomendações ranqueadas para uma requisição.

        Args:
            request: Requisição validada com user_id e filtros opcionais.

        Returns:
            Tupla (lista de RecommendationItem ranqueada, is_cold_start).
        """
        is_cold_start = not self._engine.is_known_user(request.user_id)
        candidate_k = self._candidate_pool_size(request)
        candidates = self._engine.recommend(request.user_id, k=candidate_k)
        filtered = self._apply_filters(candidates, request)
        items = self._to_schema(filtered[: request.top_k])
        return items, is_cold_start

    def _candidate_pool_size(self, request: RecommendationRequest) -> int:
        """Calcula quantos candidatos buscar para sobrar espaço aos filtros."""
        exclude_count = len(request.exclude_items or [])
        pool = request.top_k * _CANDIDATE_MULTIPLIER + exclude_count
        return min(pool, self._engine.catalog_size, _MAX_CANDIDATES)

    @staticmethod
    def _apply_filters(
        candidates: list[Recommendation], request: RecommendationRequest
    ) -> list[Recommendation]:
        """Remove itens excluídos e abaixo do score mínimo solicitado."""
        exclude_set = set(request.exclude_items or [])
        min_score = request.min_score_threshold or 0.0
        return [
            rec
            for rec in candidates
            if rec.product_id not in exclude_set and rec.score >= min_score
        ]

    @staticmethod
    def _to_schema(
        recommendations: list[Recommendation],
    ) -> list[RecommendationItem]:
        """Converte Recommendation (domínio) em RecommendationItem (schema)."""
        return [
            RecommendationItem(product_id=rec.product_id, score=rec.score, rank=rank)
            for rank, rec in enumerate(recommendations, 1)
        ]
