"""Tests for the popularity recommender (cold-start fallback)."""

import numpy as np

from src.models.popularity import PopularityRecommender


def test_top_k_returns_first_k_items_in_order(
    toy_popularity_ranking: np.ndarray,
) -> None:
    """top_k returns the first k items of the ranking, in the same order."""
    model = PopularityRecommender(toy_popularity_ranking)

    assert model.top_k(2) == [3, 2]


def test_top_k_full_ranking_when_k_equals_length(
    toy_popularity_ranking: np.ndarray,
) -> None:
    """Asking for k equal to the ranking length returns every item."""
    model = PopularityRecommender(toy_popularity_ranking)

    assert model.top_k(4) == [3, 2, 1, 0]


def test_load_round_trips_ranking_from_disk(model_artifacts: dict) -> None:
    """load() reconstructs the ranking persisted in pickle."""
    model = PopularityRecommender.load(model_artifacts["popularity_path"])

    assert model.top_k(2) == [3, 2]


def test_score_user_ranks_items_by_popularity(
    toy_popularity_ranking: np.ndarray,
) -> None:
    """score_user gives the most popular item the highest (user-agnostic) score."""
    model = PopularityRecommender(toy_popularity_ranking)

    scores = model.score_user(user_idx=0)

    # ranking [3, 2, 1, 0] -> item 3 most popular; scores indexed by item_idx.
    assert np.array_equal(scores, np.array([1, 2, 3, 4], dtype=np.float32))
    assert int(scores.argmax()) == 3
