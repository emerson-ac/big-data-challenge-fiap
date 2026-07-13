"""Ranking metrics shared by every model notebook."""

import math

import pandas as pd


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Computes precision@k for a recommendation list.

    Args:
        recommended: Recommended items, ordered by score.
        relevant: Set of relevant items (ground truth).
        k: Number of items considered in the top-k.

    Returns:
        precision@k value between 0 and 1.
    """
    if k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Computes recall@k for a recommendation list.

    Args:
        recommended: Recommended items, ordered by score.
        relevant: Set of relevant items (ground truth).
        k: Number of items considered in the top-k.

    Returns:
        recall@k value between 0 and 1.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Computes NDCG@k (Normalized Discounted Cumulative Gain).

    Args:
        recommended: Recommended items, ordered by score.
        relevant: Set of relevant items (ground truth).
        k: Number of items considered in the top-k.

    Returns:
        NDCG@k value between 0 and 1.
    """
    if not relevant:
        return 0.0
    hit_positions = (i for i, item in enumerate(recommended[:k], 1) if item in relevant)
    dcg = sum(1.0 / math.log2(i + 1) for i in hit_positions)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Computes average precision@k (the MAP@k component).

    Args:
        recommended: Recommended items, ordered by score.
        relevant: Set of relevant items (ground truth).
        k: Number of items considered in the top-k.

    Returns:
        average precision@k value between 0 and 1.
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
    """Computes the fraction of users with at least one hit in the top-k.

    Args:
        recommendations: Mapping user_idx -> list of recommended items.
        ground_truth: Mapping user_idx -> set of relevant items.
        k: Number of items considered in the top-k.

    Returns:
        Hit rate@k between 0 and 1.
    """
    if not ground_truth:
        return 0.0
    hits = _count_hit_users(recommendations, ground_truth, k)
    return hits / len(ground_truth)


def _count_hit_users(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int,
) -> int:
    """Counts users with at least one relevant item in their top-k."""
    return sum(
        1
        for user_idx, relevant in ground_truth.items()
        if set(recommendations.get(user_idx, [])[:k]) & relevant
    )


def coverage_at_k(recommendations: dict[int, list[int]], n_items: int, k: int) -> float:
    """Computes the fraction of the catalog recommended at least once.

    Args:
        recommendations: Mapping user_idx -> list of recommended items.
        n_items: Total catalog size (item vocabulary).
        k: Number of items considered in the top-k.

    Returns:
        Coverage@k between 0 and 1.
    """
    if n_items == 0:
        return 0.0
    recommended_items: set[int] = set()
    for recs in recommendations.values():
        recommended_items.update(recs[:k])
    return len(recommended_items) / n_items


def pairs_to_ground_truth(pairs: pd.DataFrame) -> dict[int, set[int]]:
    """Groups positive (user_idx, item_idx) pairs into per-user ground truth.

    Args:
        pairs: DataFrame with user_idx and item_idx columns.

    Returns:
        Mapping user_idx -> set of relevant item_idx.
    """
    return pairs.groupby("user_idx")["item_idx"].apply(set).to_dict()


def evaluate_recommendations(
    recommendations: dict[int, list[int]], ground_truth: dict[int, set[int]], k: int
) -> dict[str, float]:
    """Aggregates the 4 official metrics (user-averaged) for a model.

    Args:
        recommendations: Mapping user_idx -> list of recommended items.
        ground_truth: Mapping user_idx -> set of relevant items.
        k: Number of items considered in the top-k.

    Returns:
        Dictionary with precision_at_k, recall_at_k, ndcg_at_k and map_at_k.
    """
    rows = _collect_metric_rows(recommendations, ground_truth, k)
    columns = ["precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k"]
    return pd.DataFrame(rows, columns=columns).mean().to_dict()


def _collect_metric_rows(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int,
) -> list[tuple[float, float, float, float]]:
    """Collects the per-user metric tuple for every user in ground truth."""
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
    return rows
