"""Recomendador item-based collaborative filtering (modelo Production)."""

from pathlib import Path

import numpy as np
import scipy.sparse as sp


class ItemBasedCFRecommender:
    """Recomendador baseado em similaridade item-item (cosine).

    Calcula scores multiplicando o histórico de compras do usuário (vetor
    sobre o catálogo) pela matriz de similaridade item-item treinada.

    Args:
        item_similarity: Matriz esparsa (n_items, n_items) de similaridade.
        interactions: Matriz esparsa (n_users, n_items) de histórico de compras.
    """

    def __init__(
        self, item_similarity: sp.csr_matrix, interactions: sp.csr_matrix
    ) -> None:
        self._item_similarity = item_similarity
        self._interactions = interactions

    @classmethod
    def load(
        cls, similarity_path: Path, interactions_path: Path
    ) -> "ItemBasedCFRecommender":
        """Carrega o recomendador a partir dos artefatos persistidos.

        Args:
            similarity_path: Caminho do .npz com a similaridade item-item.
            interactions_path: Caminho do .npz com o histórico de compras.

        Returns:
            Instância pronta para gerar scores.
        """
        item_similarity = sp.load_npz(similarity_path)
        interactions = sp.load_npz(interactions_path)
        return cls(item_similarity, interactions)

    def score_user(self, user_idx: int) -> np.ndarray:
        """Calcula o vetor denso de scores por item para um usuário conhecido.

        Args:
            user_idx: Índice interno do usuário (linha da matriz de interações).

        Returns:
            Vetor numpy de scores, shape (n_items,).

        Raises:
            IndexError: Se user_idx estiver fora do intervalo treinado.
        """
        if not 0 <= user_idx < self._interactions.shape[0]:
            raise IndexError(f"user_idx {user_idx} fora do intervalo de treino")
        user_history = self._interactions[user_idx]
        return user_history.dot(self._item_similarity).toarray().flatten()

    @property
    def n_items(self) -> int:
        """Número de itens no catálogo do modelo."""
        return self._item_similarity.shape[0]
