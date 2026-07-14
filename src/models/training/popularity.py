"""Popularity baseline training (port of notebook 03)."""

from pathlib import Path

import numpy as np

from src.evaluation.metrics import evaluate_recommendations
from src.models.training.data import ProcessedData, save_metrics, save_pickle


def _recs_for(ranking: np.ndarray, ground_truth: dict, k: int) -> dict:
    """Recommends the same global top-k to every user in the ground truth."""
    top = ranking[:k].tolist()
    return {user_idx: top for user_idx in ground_truth}


def train(data: ProcessedData, k: int, out_dir: Path) -> dict:
    """Trains and evaluates the global popularity recommender.

    Args:
        data: Processed artifacts (interactions + ground truths).
        k: Top-k size used for evaluation.
        out_dir: Model output directory.

    Returns:
        Metrics payload ({k, validation, test}).
    """
    frequency = np.asarray(data.interactions.sum(axis=0)).flatten()
    ranking = np.argsort(-frequency)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(out_dir / "ranking.pkl", ranking)
    val = evaluate_recommendations(
        _recs_for(ranking, data.val_ground_truth, k), data.val_ground_truth, k
    )
    test = evaluate_recommendations(
        _recs_for(ranking, data.test_ground_truth, k), data.test_ground_truth, k
    )
    payload = {"k": k, "validation": val, "test": test}
    save_metrics(out_dir, payload)
    return payload
