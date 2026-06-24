"""Interface comum para modelos de recomendação (Interface Segregation)."""

from typing import Protocol

import numpy as np


class Recommender(Protocol):
    """Contrato mínimo de um modelo capaz de pontuar itens para um usuário."""

    def score_user(self, user_idx: int) -> np.ndarray:
        """Calcula o vetor denso de scores por item para um usuário conhecido.

        Args:
            user_idx: Índice interno do usuário.

        Returns:
            Vetor numpy de scores, shape (n_items,).
        """
        ...
