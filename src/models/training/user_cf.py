"""User-based CF via KNN (port of notebook 05, sampled evaluation)."""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from src.evaluation.metrics import evaluate_recommendations
from src.evaluation.ranking import recommendations_from_score_matrix
from src.models.training.data import ProcessedData, save_metrics, save_pickle


def build_recommendations(
    users: list[int],
    knn,
    pool_matrix: sp.csr_matrix,
    interactions,
    n_neighbors: int,
    k: int,
) -> dict:
    """Generates weighted recommendations from the nearest neighbors (cosine).

    Args:
        users: user_idx values to recommend for.
        knn: NearestNeighbors fitted on the pool.
        pool_matrix: Interactions of the pool users.
        interactions: Full interaction matrix.
        n_neighbors: Neighbors considered per user.
        k: Top-k size.

    Returns:
        Map user_idx -> list of recommended items.
    """
    distances, neighbors = knn.kneighbors(interactions[users], n_neighbors=n_neighbors)
    similarities = 1.0 - distances
    rows = np.repeat(np.arange(len(users)), n_neighbors)
    weights = sp.csr_matrix(
        (similarities.flatten(), (rows, neighbors.flatten())),
        shape=(len(users), pool_matrix.shape[0]),
    )
    scores = weights.dot(pool_matrix).toarray()
    return recommendations_from_score_matrix(users, scores, k)


def _sample(rng, population: list[int], size: int) -> list[int]:
    """Samples without replacement up to ``size`` elements."""
    return rng.choice(
        population, size=min(size, len(population)), replace=False
    ).tolist()


def _eval(ground_truth, sample, knn, pool_matrix, interactions, n_neighbors, k) -> dict:
    """Evaluates user-based recommendations on a user sample."""
    recs = build_recommendations(
        sample,
        knn,
        pool_matrix,
        interactions,
        min(n_neighbors, pool_matrix.shape[0]),
        k,
    )
    gt = {u: ground_truth[u] for u in sample}
    return evaluate_recommendations(recs, gt, k)


def _eval_split(data, sample, knn, pool_matrix, best, k, split) -> dict:
    """Evaluates user-based recommendations on the val or test split."""
    gt = data.val_ground_truth if split == "val" else data.test_ground_truth
    return _eval(gt, sample, knn, pool_matrix, data.interactions, best, k)


def _payload(k, best, pool, sample_n, search, val, test) -> dict:
    """Assembles the user-based CF metrics payload."""
    return {
        "k": k,
        "n_neighbors": best,
        "neighbor_pool_size": int(len(pool)),
        "eval_sample_size": sample_n,
        "search_results": search,
        "validation": val,
        "test": test,
    }


def _run_search(
    data, knn, pool_matrix, val_sample, k, cfg, pool_size
) -> tuple[list[dict], int]:
    """Random search over n_neighbors_choices on the validation sample."""
    rng = np.random.default_rng(42)
    choices = [c for c in cfg["search"]["n_neighbors_choices"] if c <= pool_size]
    trials = rng.choice(
        choices, size=min(cfg["search"]["n_trials"], len(choices)), replace=False
    )
    results = []
    for n in trials:
        metrics = _eval(
            data.val_ground_truth,
            val_sample,
            knn,
            pool_matrix,
            data.interactions,
            int(n),
            k,
        )
        results.append({"n_neighbors": int(n), **metrics})
    best = max(results, key=lambda r: r["recall_at_k"])["n_neighbors"]
    return results, best


def _build_knn(data: ProcessedData, cfg: dict, seed: int) -> tuple:
    """Builds the KNN model over a sampled neighbor pool.

    Returns:
        Tuple (knn, pool_matrix, pool_indices, rng).
    """
    rng = np.random.default_rng(seed)
    n_users = data.interactions.shape[0]
    pool = rng.choice(
        n_users, size=min(cfg["neighbor_pool_size"], n_users), replace=False
    )
    pool_matrix = data.interactions[pool]
    knn = NearestNeighbors(metric="cosine", algorithm="brute").fit(pool_matrix)
    return knn, pool_matrix, pool, rng


def train(data: ProcessedData, k: int, cfg: dict, out_dir: Path, seed: int) -> dict:
    """Trains user-based CF with a random search over n_neighbors (sampled).

    Args:
        data: Processed artifacts.
        k: Top-k size used for evaluation.
        cfg: The ``user_based_cf`` config block.
        out_dir: Model output directory.
        seed: Random seed.

    Returns:
        Metrics payload with search_results.
    """
    knn, pool_matrix, pool, rng = _build_knn(data, cfg, seed)
    sample_n = cfg["eval_sample_size"]
    val_sample = _sample(rng, list(data.val_ground_truth), sample_n)
    test_sample = _sample(rng, list(data.test_ground_truth), sample_n)
    search, best = _run_search(data, knn, pool_matrix, val_sample, k, cfg, len(pool))
    val = _eval_split(data, val_sample, knn, pool_matrix, best, k, "val")
    test = _eval_split(data, test_sample, knn, pool_matrix, best, k, "test")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(out_dir / "knn_model.pkl", {"model": knn, "pool_indices": pool})
    payload = _payload(k, best, pool, sample_n, search, val, test)
    save_metrics(out_dir, payload)
    return payload
