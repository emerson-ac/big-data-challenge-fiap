"""Strategy Pattern para enriquecer as recomendações antes da resposta.

Cada estratégia decide como transformar a lista crua de ``Recommendation``
em ``RecommendationItem`` — por exemplo, anexando (ou não) o nome do produto.
"""

from typing import Protocol

from src.api.schemas import RecommendationItem
from src.models.inference import Recommendation


class EnrichmentStrategy(Protocol):
    """Contrato de uma estratégia de enriquecimento de recomendações."""

    def enrich(self, recommendations: list[Recommendation]) -> list[RecommendationItem]:
        """Converte recomendações cruas em itens de resposta.

        Args:
            recommendations: Recomendações produzidas pelo engine.

        Returns:
            Itens de resposta prontos para serialização.
        """
        ...


class IdOnlyStrategy:
    """Estratégia leve: retorna apenas ids, sem consultar nomes."""

    def enrich(self, recommendations: list[Recommendation]) -> list[RecommendationItem]:
        """Mapeia cada recomendação para um item sem nome de produto."""
        return [
            RecommendationItem(product_id=r.product_id, score=r.score, rank=r.rank)
            for r in recommendations
        ]


class WithNamesStrategy:
    """Estratégia que anexa o nome do produto a partir de um mapa id->nome.

    Args:
        product_names: Mapa product_id -> nome do produto.
    """

    def __init__(self, product_names: dict[int, str]) -> None:
        self._names = product_names

    def enrich(self, recommendations: list[Recommendation]) -> list[RecommendationItem]:
        """Mapeia cada recomendação para um item com nome do produto."""
        return [
            RecommendationItem(
                product_id=r.product_id,
                product_name=self._names.get(r.product_id),
                score=r.score,
                rank=r.rank,
            )
            for r in recommendations
        ]


def select_strategy(
    enrich_names: bool, product_names: dict[int, str]
) -> EnrichmentStrategy:
    """Escolhe a estratégia conforme a flag de enriquecimento.

    Args:
        enrich_names: Se True, anexa nomes de produto.
        product_names: Mapa product_id -> nome.

    Returns:
        Instância da estratégia selecionada.
    """
    if enrich_names:
        return WithNamesStrategy(product_names)
    return IdOnlyStrategy()
