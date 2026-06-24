"""Testes do recomendador de popularidade (fallback de cold-start)."""

import numpy as np

from src.models.popularity import PopularityRecommender


def test_top_k_returns_first_k_items_in_order(
    toy_popularity_ranking: np.ndarray,
) -> None:
    """top_k retorna os k primeiros itens do ranking, na mesma ordem."""
    model = PopularityRecommender(toy_popularity_ranking)

    assert model.top_k(2) == [3, 2]


def test_top_k_full_ranking_when_k_equals_length(
    toy_popularity_ranking: np.ndarray,
) -> None:
    """Pedir k igual ao tamanho do ranking retorna todos os itens."""
    model = PopularityRecommender(toy_popularity_ranking)

    assert model.top_k(4) == [3, 2, 1, 0]


def test_load_round_trips_ranking_from_disk(model_artifacts: dict) -> None:
    """load() reconstrói o ranking persistido em pickle."""
    model = PopularityRecommender.load(model_artifacts["popularity_path"])

    assert model.top_k(2) == [3, 2]
