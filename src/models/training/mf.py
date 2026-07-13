"""Implicit Matrix Factorization training (port of notebook 06, TruncatedSVD)."""

from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

from src.models.training.data import ProcessedData, evaluate_score_fn, save_metrics


def _fit(interactions, n_components: int, seed: int) -> tuple:
    """Fits TruncatedSVD and returns (user_factors, item_factors, variance)."""
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    user_factors = svd.fit_transform(interactions.astype("float32"))
    item_factors = svd.components_.T
    return user_factors, item_factors, float(svd.explained_variance_ratio_.sum())


def _score_fn(user_factors: np.ndarray, item_factors: np.ndarray):
    """Returns a callable users -> dense score matrix (factor product)."""
    return lambda users: user_factors[users] @ item_factors.T


def _clamp(choices: list[int], n_items: int) -> list[int]:
    """Caps each n_components at n_items-1 (TruncatedSVD constraint)."""
    return sorted({min(c, n_items - 1) for c in choices if c > 0})


def _run_search(data, val_users, k, cfg, n_items, seed) -> tuple[list[dict], int]:
    """Random search over n_components_choices; returns (results, best)."""
    rng = np.random.default_rng(seed)
    choices = _clamp(cfg["search"]["n_components_choices"], n_items)
    trials = rng.choice(
        choices, size=min(cfg["search"]["n_trials"], len(choices)), replace=False
    )
    results = []
    for n_comp in trials:
        user_factors, item_factors, _ = _fit(data.interactions, int(n_comp), seed)
        metrics = evaluate_score_fn(
            val_users, _score_fn(user_factors, item_factors), data.val_ground_truth, k
        )
        results.append({"n_components": int(n_comp), **metrics})
    best = max(results, key=lambda r: r["recall_at_k"])["n_components"]
    return results, best


def _mf_payload(k, best, variance, search, val, test) -> dict:
    """Assembles the Matrix Factorization metrics payload."""
    return {
        "k": k,
        "n_components": best,
        "explained_variance_ratio": variance,
        "search_results": search,
        "validation": val,
        "test": test,
    }


def train(data: ProcessedData, k: int, cfg: dict, out_dir: Path, seed: int) -> dict:
    """Trains MF with a random search over n_components.

    Args:
        data: Processed artifacts.
        k: Top-k size used for evaluation.
        cfg: The ``matrix_factorization`` config block.
        out_dir: Model output directory.
        seed: Random seed.

    Returns:
        Metrics payload with search_results and explained_variance_ratio.
    """
    n_items = data.interactions.shape[1]
    val_users = list(data.val_ground_truth.keys())
    search, best = _run_search(data, val_users, k, cfg, n_items, seed)
    user_factors, item_factors, variance = _fit(data.interactions, best, seed)
    score_fn = _score_fn(user_factors, item_factors)
    val = evaluate_score_fn(val_users, score_fn, data.val_ground_truth, k)
    test = evaluate_score_fn(
        list(data.test_ground_truth), score_fn, data.test_ground_truth, k
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "user_factors.npy", user_factors)
    np.save(out_dir / "item_factors.npy", item_factors)
    payload = _mf_payload(k, best, variance, search, val, test)
    save_metrics(out_dir, payload)
    return payload
