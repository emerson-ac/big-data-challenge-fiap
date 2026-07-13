"""Item-based collaborative filtering recommender (Production model)."""

from pathlib import Path

import numpy as np
import scipy.sparse as sp


class ItemBasedCFRecommender:
    """Recommender based on item-item (cosine) similarity.

    Scores are computed by multiplying the user's purchase history (vector over
    the catalog) by the trained item-item similarity matrix.

    Args:
        item_similarity: Sparse (n_items, n_items) similarity matrix.
        interactions: Sparse (n_users, n_items) purchase-history matrix.
    """

    def __init__(
        self, item_similarity: sp.csr_matrix, interactions: sp.csr_matrix
    ) -> None:
        self._item_similarity = item_similarity
        self._interactions = interactions

    @classmethod
    def load(
        cls, similarity_path: Path, interactions_path: Path
    ) -> "ItemBasedCFRecommender":
        """Loads the recommender from the persisted artifacts.

        Args:
            similarity_path: Path to the .npz holding item-item similarity.
            interactions_path: Path to the .npz holding the purchase history.

        Returns:
            Instance ready to produce scores.
        """
        item_similarity = sp.load_npz(similarity_path)
        interactions = sp.load_npz(interactions_path)
        return cls(item_similarity, interactions)

    def score_user(self, user_idx: int) -> np.ndarray:
        """Computes the dense per-item score vector for a known user.

        Args:
            user_idx: Internal user index (row of the interaction matrix).

        Returns:
            Numpy score vector, shape (n_items,).

        Raises:
            IndexError: If user_idx is outside the trained range.
        """
        if not 0 <= user_idx < self._interactions.shape[0]:
            raise IndexError(f"user_idx {user_idx} outside the trained range")
        user_history = self._interactions[user_idx]
        return user_history.dot(self._item_similarity).toarray().flatten()

    @property
    def n_items(self) -> int:
        """Number of items in the model catalog."""
        return self._item_similarity.shape[0]
