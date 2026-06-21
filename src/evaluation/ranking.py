"""Utilitários de ranking compartilhados pelos notebooks de modelo."""

import numpy as np


def top_k_from_scores(scores: np.ndarray, k: int) -> list[int]:
    """Retorna os índices do top-k de um vetor de scores, em ordem decrescente.

    Args:
        scores: Vetor denso de scores por item.
        k: Tamanho do top-k.

    Returns:
        Lista com os k índices de maior score, ordenados decrescentemente.
    """
    top_k = np.argpartition(-scores, k - 1)[:k]
    return top_k[np.argsort(-scores[top_k])].tolist()


def recommendations_from_score_matrix(
    user_indices: list[int], score_matrix: np.ndarray, k: int
) -> dict[int, list[int]]:
    """Constrói recomendações top-k a partir de uma matriz densa de scores.

    Args:
        user_indices: Lista de user_idx, na mesma ordem das linhas de score_matrix.
        score_matrix: Matriz (len(user_indices), n_items) com os scores por usuário.
        k: Tamanho do top-k recomendado.

    Returns:
        Mapeamento user_idx -> lista de item_idx recomendados.
    """
    return {
        user_idx: top_k_from_scores(np.asarray(score_matrix[offset]).flatten(), k)
        for offset, user_idx in enumerate(user_indices)
    }
