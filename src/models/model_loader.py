"""Factory for creating/loading recommendation models."""

from typing import Any, Callable

from src.models.item_based_cf import ItemBasedCFRecommender
from src.models.popularity import PopularityRecommender
from src.models.registry_recommender import RegistryRecommender


class ModelFactory:
    """Factory Pattern to instantiate/load recommenders by name."""

    _builders: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str, builder: Callable[..., Any]) -> None:
        """Registers a builder (class or callable) under a name.

        Args:
            name: Identifier of the model type (e.g. "item_based_cf").
            builder: Callable accepting **kwargs and returning the loaded instance.
        """
        cls._builders[name] = builder

    @classmethod
    def create(cls, model_type: str, **kwargs: Any) -> Any:
        """Creates/loads a model registered by name.

        Args:
            model_type: Name of the registered model.
            **kwargs: Arguments forwarded to the model builder.

        Returns:
            The loaded model instance.

        Raises:
            ValueError: If model_type is not registered.
        """
        builder = cls._builders.get(model_type)
        if builder is None:
            raise ValueError(f"Model '{model_type}' is not registered")
        return builder(**kwargs)


ModelFactory.register("item_based_cf", ItemBasedCFRecommender.load)
# "registry" resolves whichever model type is behind the @production alias, so
# it is not tied to a specific algorithm.
ModelFactory.register("registry", RegistryRecommender.load)
ModelFactory.register("popularity", PopularityRecommender.load)
