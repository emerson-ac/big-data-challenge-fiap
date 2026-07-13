"""Tests for the ModelFactory (Factory Pattern)."""

import pytest

from src.models.item_based_cf import ItemBasedCFRecommender
from src.models.model_loader import ModelFactory
from src.models.popularity import PopularityRecommender


def test_create_item_based_cf_returns_loaded_recommender(model_artifacts: dict) -> None:
    """create('item_based_cf', ...) returns a loaded ItemBasedCFRecommender."""
    model = ModelFactory.create(
        "item_based_cf",
        similarity_path=model_artifacts["similarity_path"],
        interactions_path=model_artifacts["interactions_path"],
    )

    assert isinstance(model, ItemBasedCFRecommender)


def test_create_popularity_returns_loaded_recommender(model_artifacts: dict) -> None:
    """create('popularity', ...) returns a loaded PopularityRecommender."""
    model = ModelFactory.create(
        "popularity", ranking_path=model_artifacts["popularity_path"]
    )

    assert isinstance(model, PopularityRecommender)


def test_create_unregistered_model_raises_value_error() -> None:
    """Requesting an unregistered model_type raises ValueError."""
    with pytest.raises(ValueError, match="is not registered"):
        ModelFactory.create("unknown_model")


def test_register_adds_new_model_type() -> None:
    """register() extends the factory without changing create() (Open/Closed)."""
    ModelFactory.register("echo", lambda value: value)

    assert ModelFactory.create("echo", value=42) == 42
