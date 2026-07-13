"""Ranking utilities shared by the model notebooks."""

import numpy as np


def top_k_from_scores(scores: np.ndarray, k: int) -> list[int]:
    """Returns the top-k indices of a score vector, in descending order.

    Args:
        scores: Dense per-item score vector.
        k: Top-k size.

    Returns:
        List of the k highest-score indices, sorted descending.
    """
    top_k = np.argpartition(-scores, k - 1)[:k]
    return top_k[np.argsort(-scores[top_k])].tolist()


def recommendations_from_score_matrix(
    user_indices: list[int], score_matrix: np.ndarray, k: int
) -> dict[int, list[int]]:
    """Builds top-k recommendations from a dense score matrix.

    Args:
        user_indices: List of user_idx, in the same order as score_matrix rows.
        score_matrix: Matrix (len(user_indices), n_items) with per-user scores.
        k: Top-k recommendation size.

    Returns:
        Mapping user_idx -> list of recommended item_idx.
    """
    return {
        user_idx: top_k_from_scores(np.asarray(score_matrix[offset]).flatten(), k)
        for offset, user_idx in enumerate(user_indices)
    }
