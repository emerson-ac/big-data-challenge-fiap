"""Dependências injetáveis da API (engine e mapa de nomes de produto)."""

from functools import lru_cache

from src.config import get_settings
from src.models.inference import RecommendationEngine, load_vocabularies


@lru_cache
def get_engine() -> RecommendationEngine:
    """Instancia (uma única vez) o engine de recomendação do modelo Production.

    Returns:
        Engine pronto para gerar recomendações.
    """
    return RecommendationEngine(model_type=get_settings().model_type)


@lru_cache
def get_product_names() -> dict[int, str]:
    """Carrega o mapa product_id -> nome do produto a partir dos vocabulários.

    Returns:
        Dicionário product_id -> nome.
    """
    vocab = load_vocabularies()
    return {
        int(pid): str(name)
        for pid, name in zip(vocab["idx_to_product_id"], vocab["idx_to_product_name"])
    }
