"""Métricas de ranking compartilhadas por todos os notebooks de modelo."""

import math

import pandas as pd


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calcula precision@k para uma lista de recomendações.

    Args:
        recommended: Itens recomendados, ordenados por score.
        relevant: Conjunto de itens relevantes (ground truth).
        k: Número de itens considerados no top-k.

    Returns:
        Valor de precision@k entre 0 e 1.
    """
    if k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calcula recall@k para uma lista de recomendações.

    Args:
        recommended: Itens recomendados, ordenados por score.
        relevant: Conjunto de itens relevantes (ground truth).
        k: Número de itens considerados no top-k.

    Returns:
        Valor de recall@k entre 0 e 1.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calcula NDCG@k (Normalized Discounted Cumulative Gain).

    Args:
        recommended: Itens recomendados, ordenados por score.
        relevant: Conjunto de itens relevantes (ground truth).
        k: Número de itens considerados no top-k.

    Returns:
        Valor de NDCG@k entre 0 e 1.
    """
    if not relevant:
        return 0.0
    hit_positions = (i for i, item in enumerate(recommended[:k], 1) if item in relevant)
    dcg = sum(1.0 / math.log2(i + 1) for i in hit_positions)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calcula average precision@k (componente do MAP@k).

    Args:
        recommended: Itens recomendados, ordenados por score.
        relevant: Conjunto de itens relevantes (ground truth).
        k: Número de itens considerados no top-k.

    Returns:
        Valor de average precision@k entre 0 e 1.
    """
    if not relevant:
        return 0.0
    score, hits = 0.0, 0
    for i, item in enumerate(recommended[:k], 1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def hit_rate_at_k(
    recommendations: dict[int, list[int]], ground_truth: dict[int, set[int]], k: int
) -> float:
    """Calcula a proporção de usuários com ao menos um acerto no top-k.

    Args:
        recommendations: Mapeamento user_idx -> lista de itens recomendados.
        ground_truth: Mapeamento user_idx -> conjunto de itens relevantes.
        k: Número de itens considerados no top-k.

    Returns:
        Hit rate@k entre 0 e 1.
    """
    if not ground_truth:
        return 0.0
    hits = sum(
        1
        for user_idx, relevant in ground_truth.items()
        if set(recommendations.get(user_idx, [])[:k]) & relevant
    )
    return hits / len(ground_truth)


def coverage_at_k(recommendations: dict[int, list[int]], n_items: int, k: int) -> float:
    """Calcula a fração do catálogo recomendada ao menos uma vez.

    Args:
        recommendations: Mapeamento user_idx -> lista de itens recomendados.
        n_items: Tamanho total do catálogo (vocabulário de itens).
        k: Número de itens considerados no top-k.

    Returns:
        Coverage@k entre 0 e 1.
    """
    if n_items == 0:
        return 0.0
    recommended_items: set[int] = set()
    for recs in recommendations.values():
        recommended_items.update(recs[:k])
    return len(recommended_items) / n_items


def pairs_to_ground_truth(pairs: pd.DataFrame) -> dict[int, set[int]]:
    """Agrupa pares positivos (user_idx, item_idx) em ground truth por usuário.

    Args:
        pairs: DataFrame com colunas user_idx e item_idx.

    Returns:
        Mapeamento user_idx -> conjunto de item_idx relevantes.
    """
    return pairs.groupby("user_idx")["item_idx"].apply(set).to_dict()


def evaluate_recommendations(
    recommendations: dict[int, list[int]], ground_truth: dict[int, set[int]], k: int
) -> dict[str, float]:
    """Agrega as 4 métricas oficiais (média sobre usuários) para um modelo.

    Args:
        recommendations: Mapeamento user_idx -> lista de itens recomendados.
        ground_truth: Mapeamento user_idx -> conjunto de itens relevantes.
        k: Número de itens considerados no top-k.

    Returns:
        Dicionário com precision_at_k, recall_at_k, ndcg_at_k e map_at_k.
    """
    rows = []
    for user_idx, relevant in ground_truth.items():
        recs = recommendations.get(user_idx, [])
        rows.append(
            (
                precision_at_k(recs, relevant, k),
                recall_at_k(recs, relevant, k),
                ndcg_at_k(recs, relevant, k),
                average_precision_at_k(recs, relevant, k),
            )
        )
    columns = ["precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k"]
    return pd.DataFrame(rows, columns=columns).mean().to_dict()
