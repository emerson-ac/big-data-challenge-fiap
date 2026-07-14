"""Loading of processed artifacts and shared training-time evaluation helpers."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp

from src.evaluation.metrics import evaluate_recommendations, pairs_to_ground_truth
from src.evaluation.ranking import recommendations_from_score_matrix


@dataclass
class ProcessedData:
    """Artifacts from ``data/processed`` consumed by the training stages.

    Attributes:
        interactions: Sparse (n_users, n_items) prior purchase-history matrix.
        val_ground_truth: Map user_idx -> relevant validation items.
        test_ground_truth: Map user_idx -> relevant test items.
        split_meta: Split metadata (includes the dataset_hash).
    """

    interactions: sp.csr_matrix
    val_ground_truth: dict[int, set[int]]
    test_ground_truth: dict[int, set[int]]
    split_meta: dict


def load_processed(processed_dir: Path) -> ProcessedData:
    """Loads the interaction matrix, ground truths and split metadata.

    Args:
        processed_dir: The ``data/processed`` directory.

    Returns:
        ProcessedData instance ready for training and evaluation.
    """
    import pandas as pd

    interactions = sp.load_npz(processed_dir / "interactions_prior.npz")
    val_gt = pairs_to_ground_truth(pd.read_pickle(processed_dir / "val_pairs.pkl"))
    test_gt = pairs_to_ground_truth(pd.read_pickle(processed_dir / "test_pairs.pkl"))
    with open(processed_dir / "split_meta.json", encoding="utf-8") as f:
        split_meta = json.load(f)
    return ProcessedData(interactions, val_gt, test_gt, split_meta)


def evaluate_score_fn(
    users: list[int],
    score_fn: Callable[[list[int]], np.ndarray],
    ground_truth: dict[int, set[int]],
    k: int,
) -> dict[str, float]:
    """Evaluates a model given a callable producing the score matrix.

    Args:
        users: user_idx values to evaluate.
        score_fn: Callable mapping users to a dense score matrix.
        ground_truth: Map user_idx -> relevant items.
        k: Top-k size.

    Returns:
        Dictionary with the four official metrics.
    """
    scores = score_fn(users)
    recs = recommendations_from_score_matrix(users, scores, k)
    return evaluate_recommendations(recs, ground_truth, k)


def save_metrics(out_dir: Path, payload: dict) -> None:
    """Persists a model's metrics.json.

    Args:
        out_dir: Model output directory.
        payload: Metrics dictionary to serialize.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def save_pickle(path: Path, obj: object) -> None:
    """Pickles an arbitrary object.

    Args:
        path: Destination path.
        obj: Object to persist.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)
