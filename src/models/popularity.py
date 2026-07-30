"""Popularity recommender — cold-start fallback."""

import pickle
from pathlib import Path

import numpy as np


class PopularityRecommender:
    """Recommender based on the global popularity ranking.

    Used as a fallback for users with no interaction history (cold-start),
    since it does not depend on a specific user.

    Args:
        ranking: Array of item_idx ordered from most to least popular.
    """

    def __init__(self, ranking: np.ndarray) -> None:
        self._ranking = ranking

    @classmethod
    def load(cls, ranking_path: Path) -> "PopularityRecommender":
        """Loads the persisted popularity ranking.

        Args:
            ranking_path: Path to the pickle holding the item_idx ranking.

        Returns:
            Instance ready to recommend.
        """
        with open(ranking_path, "rb") as f:
            ranking = pickle.load(f)
        return cls(np.asarray(ranking))

    def top_k(self, k: int) -> list[int]:
        """Returns the k most popular items.

        Args:
            k: Top-k size.

        Returns:
            List of item_idx, from most to least popular.
        """
        return self._ranking[:k].tolist()

    def score_user(self, user_idx: int) -> np.ndarray:
        """Computes a per-item score vector from the global popularity ranking.

        The score is user-agnostic (same for everyone): the most popular item
        gets the highest score. Lets this model serve as the primary model, not
        only as the cold-start fallback.

        Args:
            user_idx: Internal user index (ignored; popularity is global).

        Returns:
            Numpy score vector, shape (max(ranking) + 1,).
        """
        scores = np.zeros(int(self._ranking.max()) + 1, dtype=np.float32)
        scores[self._ranking] = np.arange(len(self._ranking), 0, -1, dtype=np.float32)
        return scores
