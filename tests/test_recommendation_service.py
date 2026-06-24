"""Testes do RecommendationService (filtros de negócio sobre o engine)."""

from src.api.schemas.request import RecommendationRequest
from src.api.services.recommendation_service import RecommendationService
from src.models.inference import RecommendationEngine


def test_known_user_returns_engine_recommendations_in_order(
    engine: RecommendationEngine,
) -> None:
    """Sem filtros, o resultado segue o ranking do engine para usuário conhecido."""
    service = RecommendationService(engine)
    request = RecommendationRequest(user_id=10, top_k=2)

    items, is_cold_start = service.get_recommendations(request)

    assert is_cold_start is False
    assert [item.product_id for item in items] == [100, 200]
    assert [item.rank for item in items] == [1, 2]


def test_unknown_user_is_flagged_as_cold_start(engine: RecommendationEngine) -> None:
    """Usuário fora do vocabulário de treino deve ser sinalizado como cold-start."""
    service = RecommendationService(engine)
    request = RecommendationRequest(user_id=999, top_k=2)

    _, is_cold_start = service.get_recommendations(request)

    assert is_cold_start is True


def test_exclude_items_removes_product_and_reranks(
    engine: RecommendationEngine,
) -> None:
    """Itens em exclude_items não devem aparecer e o ranking deve ser recalculado."""
    service = RecommendationService(engine)
    request = RecommendationRequest(user_id=10, top_k=2, exclude_items=[100])

    items, _ = service.get_recommendations(request)

    assert 100 not in [item.product_id for item in items]
    assert items[0].product_id == 200
    assert items[0].rank == 1
    assert len(items) == 2


def test_min_score_threshold_filters_low_score_items(
    engine: RecommendationEngine,
) -> None:
    """Itens com score abaixo do threshold não devem ser retornados."""
    service = RecommendationService(engine)
    request = RecommendationRequest(user_id=10, top_k=4, min_score_threshold=0.6)

    items, _ = service.get_recommendations(request)

    assert [item.product_id for item in items] == [100]


def test_top_k_limits_result_size_after_filtering(
    engine: RecommendationEngine,
) -> None:
    """O tamanho final do resultado nunca excede o top_k solicitado."""
    service = RecommendationService(engine)
    request = RecommendationRequest(user_id=999, top_k=1)

    items, _ = service.get_recommendations(request)

    assert len(items) == 1
