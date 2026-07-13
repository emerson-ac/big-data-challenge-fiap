"""Preprocessing strategies for recommendation interactions (Strategy pattern).

Each strategy implements the same ``PreprocessingStrategy`` interface so they can
be swapped or chained at runtime without changing the caller. The vocabulary
built by ``UserItemEncoderStrategy`` matches the artifacts consumed by
``src/models/inference.py`` (``user_id_to_idx``, ``idx_to_product_id``).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class PreprocessingStrategy(Protocol):
    """Common interface for interaction-frame preprocessing strategies."""

    def transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Applies the transformation and returns the processed frame.

        Args:
            interactions: Raw interaction frame (user_id, product_id, ...).

        Returns:
            Transformed interaction frame.
        """
        ...


class InteractionFilterStrategy:
    """Keeps the most popular products and the most active users.

    Args:
        top_n_products: Maximum catalog size after filtering by popularity.
        min_user_interactions: Minimum interactions required to keep a user.
    """

    def __init__(self, top_n_products: int, min_user_interactions: int = 1) -> None:
        self._top_n_products = top_n_products
        self._min_user_interactions = min_user_interactions
        self.kept_products_: np.ndarray | None = None

    def transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Filters interactions to the top-N products and active users."""
        filtered = self._filter_products(interactions)
        return self._filter_users(filtered)

    def _filter_products(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Keeps only the rows whose product is in the top-N most popular."""
        popular = (
            interactions["product_id"].value_counts().nlargest(self._top_n_products)
        )
        self.kept_products_ = popular.index.to_numpy()
        return interactions[interactions["product_id"].isin(self.kept_products_)]

    def _filter_users(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Keeps only the rows of users with enough interactions."""
        counts = interactions["user_id"].value_counts()
        active = counts[counts >= self._min_user_interactions].index
        return interactions[interactions["user_id"].isin(active)]


class UserItemEncoderStrategy:
    """Encodes raw ids into dense contiguous indices for embeddings.

    Builds the vocabulary consumed by inference (``user_id_to_idx``,
    ``idx_to_product_id``) and adds ``user_idx`` / ``item_idx`` columns.

    Args:
        user_col: Name of the raw user id column.
        item_col: Name of the raw product id column.
    """

    def __init__(self, user_col: str = "user_id", item_col: str = "product_id") -> None:
        self._user_col = user_col
        self._item_col = item_col
        self.vocabulary_: dict[str, object] = {}

    def transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Encodes user and item ids into contiguous indices."""
        encoded = interactions.copy()
        encoded = self._build_user_index(encoded)
        encoded = self._build_item_index(encoded)
        return encoded

    def _build_user_index(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Maps each user id to a contiguous index and stores the mapping."""
        user_ids = interactions[self._user_col].unique()
        user_id_to_idx = {int(uid): idx for idx, uid in enumerate(user_ids)}
        self.vocabulary_["user_id_to_idx"] = user_id_to_idx
        interactions["user_idx"] = interactions[self._user_col].map(user_id_to_idx)
        return interactions

    def _build_item_index(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Maps each product id to a contiguous index and stores the mapping."""
        product_ids = interactions[self._item_col].unique()
        product_id_to_idx = {int(pid): idx for idx, pid in enumerate(product_ids)}
        idx_to_product_id = np.array(
            sorted(product_id_to_idx, key=product_id_to_idx.get)
        )
        self.vocabulary_["product_id_to_idx"] = product_id_to_idx
        self.vocabulary_["idx_to_product_id"] = idx_to_product_id
        interactions["item_idx"] = interactions[self._item_col].map(product_id_to_idx)
        return interactions
