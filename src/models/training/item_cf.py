"""Item-based CF training (port of notebook 04, the Production model)."""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from src.models.training.data import ProcessedData, evaluate_score_fn, save_metrics


def truncate_similarity(similarity: np.ndarray, top_m: int) -> np.ndarray:
    """Keeps the top_m largest similarities per row, zeroing the rest.

    Args:
        similarity: Dense (n_items, n_items) similarity matrix.
        top_m: Number of neighbors kept per item.

    Returns:
        Truncated dense matrix.
    """
    if top_m >= similarity.shape[1]:
        return similarity
    truncated = np.zeros_like(similarity)
    for i in range(similarity.shape[0]):
        idx = np.argpartition(-similarity[i], top_m)[:top_m]
        truncated[i, idx] = similarity[i, idx]
    return truncated


def _score_fn(interactions: sp.csr_matrix, similarity: np.ndarray):
    """Returns a callable users -> dense score matrix."""
    return lambda users: interactions[users].dot(similarity)


def _run_search(data, similarity, val_users, k, cfg) -> list[dict]:
    """Random search over top_m_choices, scoring recall on validation."""
    rng = np.random.default_rng(42)
    choices = cfg["search"]["top_m_choices"]
    trials = rng.choice(
        choices, size=min(cfg["search"]["n_trials"], len(choices)), replace=False
    )
    results = []
    for top_m in trials:
        trunc = truncate_similarity(similarity, int(top_m))
        metrics = evaluate_score_fn(
            val_users, _score_fn(data.interactions, trunc), data.val_ground_truth, k
        )
        results.append({"top_m": int(top_m), **metrics})
    return results


def _finalize(data, similarity, best_top_m, k, out_dir) -> dict:
    """Truncates with the best top_m, evaluates and persists artifacts."""
    final = truncate_similarity(similarity, best_top_m)
    score_fn = _score_fn(data.interactions, final)
    val_users = list(data.val_ground_truth.keys())
    val = evaluate_score_fn(val_users, score_fn, data.val_ground_truth, k)
    test = evaluate_score_fn(
        list(data.test_ground_truth), score_fn, data.test_ground_truth, k
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "item_similarity.npz", sp.csr_matrix(final))
    return {"k": k, "top_m": best_top_m, "validation": val, "test": test}


def train(data: ProcessedData, k: int, cfg: dict, out_dir: Path) -> dict:
    """Trains item-based CF with a random search over top_m.

    Args:
        data: Processed artifacts.
        k: Top-k size used for evaluation.
        cfg: The ``item_based_cf`` config block.
        out_dir: Model output directory.

    Returns:
        Metrics payload ({k, top_m, search_results, validation, test}).
    """
    similarity = cosine_similarity(data.interactions.T, dense_output=True).astype(
        "float32"
    )
    val_users = list(data.val_ground_truth.keys())
    search = _run_search(data, similarity, val_users, k, cfg)
    best_top_m = max(search, key=lambda r: r["recall_at_k"])["top_m"]
    payload = _finalize(data, similarity, best_top_m, k, out_dir)
    payload["search_results"] = search
    save_metrics(out_dir, payload)
    return payload
