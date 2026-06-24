"""Recomendador de popularidade — fallback para cold-start."""

import pickle
from pathlib import Path

import numpy as np


class PopularityRecommender:
    """Recomendador baseado em ranking global de popularidade.

    Usado como fallback para usuários sem histórico de interações
    (cold-start), pois não depende de um usuário específico.

    Args:
        ranking: Array de item_idx ordenado do mais para o menos popular.
    """

    def __init__(self, ranking: np.ndarray) -> None:
        self._ranking = ranking

    @classmethod
    def load(cls, ranking_path: Path) -> "PopularityRecommender":
        """Carrega o ranking de popularidade persistido.

        Args:
            ranking_path: Caminho do pickle com o ranking de item_idx.

        Returns:
            Instância pronta para recomendar.
        """
        with open(ranking_path, "rb") as f:
            ranking = pickle.load(f)
        return cls(np.asarray(ranking))

    def top_k(self, k: int) -> list[int]:
        """Retorna os k itens mais populares.

        Args:
            k: Tamanho do top-k.

        Returns:
            Lista de item_idx, do mais para o menos popular.
        """
        return self._ranking[:k].tolist()
