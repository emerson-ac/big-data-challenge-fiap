"""Testes do recomendador item-based CF."""

import numpy as np
import pytest
import scipy.sparse as sp

from src.models.item_based_cf import ItemBasedCFRecommender


def test_score_user_returns_similarity_row_for_single_purchase(
    toy_similarity: sp.csr_matrix, toy_interactions: sp.csr_matrix
) -> None:
    """Usuário que comprou só o item 0 recebe a linha de similaridade do item 0."""
    model = ItemBasedCFRecommender(toy_similarity, toy_interactions)

    scores = model.score_user(0)

    np.testing.assert_allclose(scores, [1.0, 0.5, 0.0, 0.0], atol=1e-6)


def test_score_user_zero_for_user_without_history(
    toy_similarity: sp.csr_matrix, toy_interactions: sp.csr_matrix
) -> None:
    """Usuário sem histórico de compras recebe score zero em todos os itens."""
    model = ItemBasedCFRecommender(toy_similarity, toy_interactions)

    scores = model.score_user(1)

    np.testing.assert_allclose(scores, [0.0, 0.0, 0.0, 0.0])


def test_score_user_raises_for_out_of_range_index(
    toy_similarity: sp.csr_matrix, toy_interactions: sp.csr_matrix
) -> None:
    """Índices fora do intervalo de usuários treinados levantam IndexError."""
    model = ItemBasedCFRecommender(toy_similarity, toy_interactions)

    with pytest.raises(IndexError):
        model.score_user(99)


def test_n_items_matches_similarity_shape(
    toy_similarity: sp.csr_matrix, toy_interactions: sp.csr_matrix
) -> None:
    """n_items reflete o tamanho do catálogo da matriz de similaridade."""
    model = ItemBasedCFRecommender(toy_similarity, toy_interactions)

    assert model.n_items == 4


def test_load_round_trips_artifacts_from_disk(model_artifacts: dict) -> None:
    """load() reconstrói um modelo funcionalmente igual ao original."""
    model = ItemBasedCFRecommender.load(
        model_artifacts["similarity_path"], model_artifacts["interactions_path"]
    )

    scores = model.score_user(0)

    np.testing.assert_allclose(scores, [1.0, 0.5, 0.0, 0.0], atol=1e-6)
