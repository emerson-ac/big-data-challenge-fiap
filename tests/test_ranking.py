"""Tests for the ranking helpers shared by training, evaluation and serving."""

import numpy as np

from src.evaluation.ranking import recommendations_from_score_matrix, top_k_from_scores


def test_top_k_returns_indices_sorted_by_descending_score() -> None:
    """Scores [0.1, 0.9, 0.5] give indices 1, 2 in that order."""
    scores = np.array([0.1, 0.9, 0.5])

    assert top_k_from_scores(scores, 2) == [1, 2]


def test_top_k_can_return_the_whole_catalog() -> None:
    """k equal to the vector length is valid and returns every index."""
    scores = np.array([0.1, 0.9, 0.5])

    assert top_k_from_scores(scores, 3) == [1, 2, 0]


def test_top_k_handles_ties_without_error() -> None:
    """Tied scores still yield k distinct indices."""
    scores = np.array([0.5, 0.5, 0.5])

    assert sorted(top_k_from_scores(scores, 2)) != []
    assert len(set(top_k_from_scores(scores, 2))) == 2


def test_recommendations_from_score_matrix_maps_rows_to_user_ids() -> None:
    """Row i of the matrix belongs to user_indices[i], not to index i."""
    users = [7, 9]
    scores = np.array([[0.1, 0.9], [0.9, 0.1]])

    recs = recommendations_from_score_matrix(users, scores, 1)

    assert recs == {7: [1], 9: [0]}


def test_recommendations_from_score_matrix_accepts_non_array_input() -> None:
    """Rows that are not ndarrays are coerced before ranking."""
    recs = recommendations_from_score_matrix([0], [[0.2, 0.8]], 1)

    assert recs == {0: [1]}
