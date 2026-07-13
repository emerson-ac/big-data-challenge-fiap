"""Common interface for recommendation models (Interface Segregation)."""

from typing import Protocol

import numpy as np


class Recommender(Protocol):
    """Minimal contract for a model that can score items for a user."""

    def score_user(self, user_idx: int) -> np.ndarray:
        """Computes the dense per-item score vector for a known user.

        Args:
            user_idx: Internal user index.

        Returns:
            Numpy score vector, shape (n_items,).
        """
        ...
