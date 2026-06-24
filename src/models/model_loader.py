"""Factory para criação/carregamento de modelos de recomendação."""

from typing import Any, Callable

from src.models.item_based_cf import ItemBasedCFRecommender
from src.models.popularity import PopularityRecommender


class ModelFactory:
    """Factory Pattern para instanciar/carregar recomendadores pelo nome."""

    _builders: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str, builder: Callable[..., Any]) -> None:
        """Registra um builder (classe ou callable) sob um nome.

        Args:
            name: Identificador do tipo de modelo (ex.: "item_based_cf").
            builder: Callable que recebe **kwargs e retorna a instância carregada.
        """
        cls._builders[name] = builder

    @classmethod
    def create(cls, model_type: str, **kwargs: Any) -> Any:
        """Cria/carrega um modelo registrado pelo nome.

        Args:
            model_type: Nome do modelo registrado.
            **kwargs: Argumentos passados ao builder do modelo.

        Returns:
            Instância do modelo carregado.

        Raises:
            ValueError: Se model_type não estiver registrado.
        """
        builder = cls._builders.get(model_type)
        if builder is None:
            raise ValueError(f"Modelo '{model_type}' não registrado")
        return builder(**kwargs)


ModelFactory.register("item_based_cf", ItemBasedCFRecommender.load)
ModelFactory.register("popularity", PopularityRecommender.load)
