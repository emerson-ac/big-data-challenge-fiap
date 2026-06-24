"""Testes da ModelFactory (Factory Pattern)."""

import pytest

from src.models.item_based_cf import ItemBasedCFRecommender
from src.models.model_loader import ModelFactory
from src.models.popularity import PopularityRecommender


def test_create_item_based_cf_returns_loaded_recommender(model_artifacts: dict) -> None:
    """create('item_based_cf', ...) retorna um ItemBasedCFRecommender carregado."""
    model = ModelFactory.create(
        "item_based_cf",
        similarity_path=model_artifacts["similarity_path"],
        interactions_path=model_artifacts["interactions_path"],
    )

    assert isinstance(model, ItemBasedCFRecommender)


def test_create_popularity_returns_loaded_recommender(model_artifacts: dict) -> None:
    """create('popularity', ...) retorna um PopularityRecommender carregado."""
    model = ModelFactory.create(
        "popularity", ranking_path=model_artifacts["popularity_path"]
    )

    assert isinstance(model, PopularityRecommender)


def test_create_unregistered_model_raises_value_error() -> None:
    """Solicitar um model_type não registrado levanta ValueError."""
    with pytest.raises(ValueError, match="não registrado"):
        ModelFactory.create("unknown_model")


def test_register_adds_new_model_type() -> None:
    """register() permite estender a factory sem alterar create() (Open/Closed)."""
    ModelFactory.register("echo", lambda value: value)

    assert ModelFactory.create("echo", value=42) == 42
