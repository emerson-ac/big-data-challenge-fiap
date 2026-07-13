"""Tests for the preprocessing strategies and the Preprocessor context."""

import numpy as np
import pandas as pd

from src.preprocessing import (
    InteractionFilterStrategy,
    Preprocessor,
    UserItemEncoderStrategy,
)


def _toy_interactions() -> pd.DataFrame:
    """Builds a small interaction frame with 5 products and uneven popularity."""
    rows = [
        (1, 100),
        (1, 200),
        (1, 300),
        (2, 100),
        (2, 100),
        (2, 300),
        (2, 400),
        (3, 100),
        (3, 300),
        (3, 500),
    ]
    return pd.DataFrame(rows, columns=["user_id", "product_id"])


def test_interaction_filter_keeps_top_n_products() -> None:
    """Filtering keeps exactly the N most popular products."""
    strategy = InteractionFilterStrategy(top_n_products=2)

    filtered = strategy.transform(_toy_interactions())

    assert strategy.kept_products_ is not None
    assert len(strategy.kept_products_) == 2
    assert set(filtered["product_id"]).issubset(set(strategy.kept_products_))


def test_interaction_filter_drops_inactive_users() -> None:
    """Users below the interaction threshold are removed."""
    strategy = InteractionFilterStrategy(top_n_products=5, min_user_interactions=3)

    filtered = strategy.transform(_toy_interactions())

    counts = filtered["user_id"].value_counts()
    assert (counts >= 3).all()


def test_user_item_encoder_builds_contiguous_indices() -> None:
    """Encoding produces zero-based contiguous user and item indices."""
    strategy = UserItemEncoderStrategy()

    encoded = strategy.transform(_toy_interactions())

    assert set(encoded["user_idx"]) == set(range(encoded["user_id"].nunique()))
    assert set(encoded["item_idx"]) == set(range(encoded["product_id"].nunique()))


def test_user_item_encoder_vocabulary_matches_inference_schema() -> None:
    """Vocabulary keys match the artifacts consumed by inference."""
    strategy = UserItemEncoderStrategy()

    strategy.transform(_toy_interactions())

    vocab = strategy.vocabulary_
    assert "user_id_to_idx" in vocab
    assert "idx_to_product_id" in vocab
    assert isinstance(vocab["idx_to_product_id"], np.ndarray)


def test_preprocessor_chains_strategies_in_order() -> None:
    """The context applies filter then encoder and returns encoded indices."""
    preprocessor = Preprocessor(
        [
            InteractionFilterStrategy(top_n_products=3),
            UserItemEncoderStrategy(),
        ]
    )

    result = preprocessor.run(_toy_interactions())

    assert {"user_idx", "item_idx"}.issubset(result.columns)
    assert result["item_idx"].between(0, 2).all()
