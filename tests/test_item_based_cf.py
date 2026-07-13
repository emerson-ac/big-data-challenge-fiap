"""Tests for the item-based CF recommender."""

import numpy as np
import pytest
import scipy.sparse as sp

from src.models.item_based_cf import ItemBasedCFRecommender


def test_score_user_returns_similarity_row_for_single_purchase(
    toy_similarity: sp.csr_matrix, toy_interactions: sp.csr_matrix
) -> None:
    """A user who bought only item 0 receives item 0's similarity row."""
    model = ItemBasedCFRecommender(toy_similarity, toy_interactions)

    scores = model.score_user(0)

    np.testing.assert_allclose(scores, [1.0, 0.5, 0.0, 0.0], atol=1e-6)


def test_score_user_zero_for_user_without_history(
    toy_similarity: sp.csr_matrix, toy_interactions: sp.csr_matrix
) -> None:
    """A user with no purchase history receives zero score on every item."""
    model = ItemBasedCFRecommender(toy_similarity, toy_interactions)

    scores = model.score_user(1)

    np.testing.assert_allclose(scores, [0.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_score_user_raises_for_out_of_range_index(
    toy_similarity: sp.csr_matrix, toy_interactions: sp.csr_matrix
) -> None:
    """Indices outside the trained user range raise IndexError."""
    model = ItemBasedCFRecommender(toy_similarity, toy_interactions)

    with pytest.raises(IndexError):
        model.score_user(99)


def test_n_items_matches_similarity_shape(
    toy_similarity: sp.csr_matrix, toy_interactions: sp.csr_matrix
) -> None:
    """n_items reflects the catalog size of the similarity matrix."""
    model = ItemBasedCFRecommender(toy_similarity, toy_interactions)

    assert model.n_items == 4


def test_load_round_trips_artifacts_from_disk(model_artifacts: dict) -> None:
    """load() reconstructs a functionally identical model."""
    model = ItemBasedCFRecommender.load(
        model_artifacts["similarity_path"], model_artifacts["interactions_path"]
    )

    scores = model.score_user(0)

    np.testing.assert_allclose(scores, [1.0, 0.5, 0.0, 0.0], atol=1e-6)
